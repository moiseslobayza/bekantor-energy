# Modos de gestión: interfaz, unidades y garantías

Los modos cambian qué objetivo optimiza una búsqueda; no cambian la batería,
el balance energético ni el modelo de demanda. No implementan resiliencia.

## Interfaz reutilizable

```python
from bekantor_energy.contracts import EnergyState
from bekantor_energy.energy.modes import ModePlanner, ObjectiveConfig
from bekantor_energy.energy.planner import PlannerConfig

# Elegir "economic", "autonomy" o "balanced".
objective = ObjectiveConfig(
    mode="balanced",
    cost_weight=0.5,
    cost_scale_sim=1.0,
    grid_scale_kwh=1.0,
)
planner = ModePlanner(PlannerConfig(storage_step_kwh=0.25), objective)
# planning_inputs sigue siendo el contrato existente: pronóstico, solar,
# tarifas y HomeConfig. stored_kwh_actual es el estado realizado disponible.
plan = planner.plan(planning_inputs, EnergyState(0, stored_kwh_actual))
```

`ModeEnergyPlan` extiende `EnergyPlan` sin cambiar la interpretación anterior:

| Campo | Significado |
|---|---|
| `predicted_cost` | Costo monetario bruto previsto, SIM |
| `predicted_grid_import_kwh` | Importación prevista, kWh AC |
| `objective_primary` | Valor optimizado: costo, importación o J, según modo |
| `objective_secondary` | Costo SIM de desempate en Autonomía; `None` en los otros |
| `objective_units` | `("SIM",)`, `("kWh", "SIM")` o `("dimensionless",)` |
| `heuristic_name` | Heurística efectivamente aplicada; `zero` en Autonomía |
| `actions`, `states` | Plan horario de acciones y energías almacenadas |

El valor J no es una factura y no se compara numéricamente con kWh o SIM.
La API recibe pronósticos causales; nunca necesita demanda realizada futura.
La evaluación ejecuta la primera acción y vuelve a planificar cada hora.

## Objetivos

Sea `G = Σ G_t` la importación AC en kWh y `C = Σ p_t G_t` el costo SIM.

**Económico:** minimiza C con las restricciones originales. No agrega una
valoración terminal al objetivo ni un objetivo secundario nuevo.

**Autonomía:** minimiza el par ordenado `(G, C)` lexicográficamente. C sólo
desempata a igualdad de G. No usa una combinación `G + epsilon*C`, que podría
aceptar más red por suficiente ahorro. No recompensa SOC alto.

Para preservar empates físicos, G se calcula racionalmente desde las ecuaciones,
antes de redondear importaciones en coma flotante. Con `ΔE = E_{t+1} − E_t`:

```text
si ΔE ≥ 0: G_t = max(0, L_t − PV_t + ΔE / eta_c)
si ΔE < 0: G_t = max(0, L_t − PV_t + ΔE * eta_d)
C_t = p_t * G_t
```

La demanda, solar, estados, eficiencias y tarifas representados por Python se
convierten a fracciones exactas. Se suman y comparan pares de fracciones sin
tolerancia de intercambio entre objetivos. El simulador sigue comprobando la
factibilidad y los balances con su tolerancia física original de 1e−9 kWh.
La exactitud aritmética no elimina la incertidumbre ni la precisión finita de
los datos de entrada. La importación racional y su replay flotante concuerdan
dentro de la tolerancia numérica declarada.

**Equilibrado:** minimiza el escalar adimensional
`J = w*C/C_ref + (1−w)*G/G_ref`. Se exige `0 ≤ w ≤ 1`, escalas positivas y
finitas. Cada término por transición es no negativo y aditivo.

La comparación de Fase 4.3 fija `w=0,5`, `C_ref=1 SIM`, `G_ref=1 kWh` antes
de observar resultados nuevos. Es equivalente en orden a minimizar
`C + 1 SIM/kWh * G`: admite 1 kWh adicional sólo si compensa más de 1 SIM en
costo. No es una tarifa efectiva ni un compromiso universalmente neutral.
Modificar escalas cambia el compromiso aunque w permanezca igual.
Con `0 < w < 1`, la tasa implícita general es
`lambda = (1−w)/w * C_ref/G_ref`, en SIM/kWh.

## Un único grafo y un único bucle de búsqueda

- `BatterySearchProblem.physical_successors` genera acciones factibles y sus
  ledgers físicos. Se reutilizan batería, eficiencias, potencia, reserva y SOC.
- `ModeSearchProblem` aplica el objetivo seleccionado a esas transiciones.
- `search.astar_search` mantiene frontera, etiquetas y reapertura, para costos
  escalares o pares lexicográficos. No hay tres motores independientes.
- `AStarPlanner` conserva la interfaz de referencia y usa el mismo bucle.
- `run_continuous_case` admite políticas inyectadas; mantiene un solo motor de
  evaluación continua para la referencia anterior y los nuevos modos.

El grafo es finito y acíclico: los estados son `(hora, energía de grilla)` y
cada transición avanza una hora. Se incluyen energía inicial exacta y límites,
sin redondear artificialmente el SOC realizado. Las restricciones físicas y
la reserva terminal mínima se mantienen. No se permite exportación remunerada
ni cortes conocidos en este planificador; no se añaden aristas negativas.

## Heurísticas y optimalidad

Defina sobre las horas restantes:

```text
D = suma max(L_t − PV_t, 0)
S = suma max(PV_t − L_t, 0)
U = eta_d * (E_actual − E_reserva)
B_G = max(0, D − U − eta_c*eta_d*S)
```

`B_G` relaja cronología, capacidad, potencia y reposición terminal. Es una
cota inferior de la importación necesaria. Si `p_min` es el menor precio
restante, `B_C = p_min*B_G` es una cota inferior de costo.

- Económico usa `h=B_C`, como la referencia.
- Equilibrado usa `h=w*B_C/C_ref+(1−w)*B_G/G_ref`. Una combinación no negativa
  de cotas inferiores es inferior al J de cualquier plan factible.
- Autonomía usa `(0,0)`: UCS lexicográfico exacto, un caso particular de A*.
  Se prioriza una garantía sencilla sobre acelerar con una cota vectorial.

Con costos no negativos, heurística admisible y reapertura de etiquetas
mejoradas, la primera meta extraída es óptima en el grafo/horizonte previsto.
La implementación escalar conserva la tolerancia de referencia de 1e−12 al
comparar etiquetas; las comprobaciones numéricas declaran esa precisión.
Si se alcanza el límite temporal o de nodos, se lanza `PlanningLimitError`;
no se devuelve un plan cuya optimalidad no fue establecida.

Las pruebas contrastan los tres modos contra UCS y enumeración independiente
en problemas pequeños, verifican la cota desde estados intermedios, extremos
de ponderación y un empate con eficiencia 0,95 que detecta errores de redondeo.
La optimalidad prevista **no garantiza** menor importación o costo realizados
que el reactivo continuo: cambian el espacio de acciones, la información de
realización y las predicciones al replanificar.

## Evaluación y contabilidad terminal

Todos los modos de la comparación usan la reserva original de 20 %, sin el
objetivo terminal experimental del 50 %. El costo ajustado de evaluación es
`C_aj=C+v*(E0−ET)`, v fijo 0,1425 SIM/kWh almacenado, con la sensibilidad
predefinida de Fase 4.2. Se informa separado del bruto y del objetivo J.
No es una liquidación real, una recarga forzada ni un costo de búsqueda.

El SOC medio se aproxima por la media de `(SOC_inicio+SOC_fin)/2` de cada hora,
sin contar dos veces fronteras diarias. Se reportan además mínimo y final.
Reducir importación no prueba capacidad de respaldo durante cortes; ENS cero
con red siempre disponible no constituye una evaluación de resiliencia.

## Comandos y preservación histórica

La configuración del experimento está en `configs/energy.modes.json`. Requiere
artefactos congelados de Fase 4.2 y el archivo local de fuentes previos indicado
por `source_archive`. Los resultados nuevos siempre usan carpetas nuevas.

```powershell
.\.venv\Scripts\python.exe -u -m bekantor_energy energy-modes --config configs/energy.modes.json --run-dir experiments/phase4_3/repeat-01
.\.venv\Scripts\python.exe scripts/analyze_energy_modes.py --run-dir experiments/phase4_3/repeat-01
.\.venv\Scripts\python.exe scripts/verify_energy_modes.py --run-dir experiments/phase4_3/repeat-01
```

No hay entrenamiento ni adquisición nueva: se reutilizan las predicciones y
perfiles cronológicos de Córdoba. Los fuentes históricos se archivaron antes
del refactor, sin modificar manifiestos ni resultados anteriores. El nuevo
verificador distingue fuentes archivados de fuentes activos y comprueba la
regresión realizada de Económico contra Fase 4.2. Los verificadores históricos
que exigen el hash del planner anterior requieren esos fuentes históricos;
no se alteran sus manifiestos para aceptar silenciosamente otro código.

Los archivos ignorados por Git —modelos, datos, resultados y archivos de
procedencia/fuentes— deben conservarse junto al experimento para reproducirlo.
Las decisiones y métricas son deterministas con los mismos inputs/software;
tiempos, fechas de creación y cabeceras gzip pueden variar entre corridas.

# Fase 4.2 — Operación continua y energía terminal

**El ahorro simulado de A* + MSE persiste con tarifa variable:** 10,57 % en
costo bruto y **10,49 % en costo ajustado**, frente al reactivo, con **6,36 %
más importación de red**. Con tarifa plana, el reactivo sigue siendo más
favorable: A* original cuesta 3,36 % más en bruto y **3,28 % más ajustado**.
Ninguna de las cinco valoraciones predefinidas invierte esos resultados.

La condición SOC terminal previsto ≥ 50 % no mejora el costo ajustado frente
al A* original con la valoración principal. No garantiza un SOC realizado del
50 % al terminar la semana. Su costo de búsqueda aumenta y sus efectos sobre
las acciones se concentran en dos casos de tarifa plana. No se recomienda
adoptarla como solución validada al problema terminal.

Se completaron **dos bloques disjuntos de siete días, ocho casos pareados y
24 ejecuciones**, con 4.032 transiciones realizadas, sin fallos, violaciones
físicas ni energía no suministrada. Estos resultados son simulados, no ahorros
reales ni una evaluación anual o sobre nuevas viviendas.

## Diseño y continuidad temporal

El [diseño previo](phase4_2-design.md) y la configuración
[`energy.continuous.json`](../configs/energy.continuous.json) se fijaron antes
de evaluar costos. Cada corrida conserva `design.json` antes de simular.

| Bloque UCI | Ejecución, fin exclusivo | Horas | Demanda realizada |
|---|---|---:|---:|
| Temprano | 21/09/2009 00:00 → 28/09/2009 00:00 | 168 | 164,250233 kWh |
| Tardío | 24/01/2010 00:00 → 31/01/2010 00:00 | 168 | 267,905800 kWh |

Se eligió el primer inicio elegible a medianoche en cada mitad cronológica de
validación. Se exige disponibilidad de las ventanas originales y de objetivos
observados, incluida la cola de 23 horas del último pronóstico. Es un filtro
retrospectivo de completitud; no se eligen bloques por costos, demanda alta o
precisión. `candidate_blocks.csv` conserva los candidatos y su elegibilidad.

En cada origen se utiliza la MLP MSE con calendario 64–32, semilla 2026,
congelada, últimas 168 horas y salida de 24 horas. Se reutilizan sus emisiones
causales de validación, cotejadas con los hashes, ventanas y una inferencia CPU.
El contexto pasado puede contener horas de train anteriores a validación;
**todas las horas puntuadas y las colas de pronóstico están en validación**.
No se abre test, reentrena ni introduce demanda realizada futura en A*.

El SOC inicial es 50 % = 2,5 kWh. Persiste durante las 168 horas: el estado
realizado de la hora anterior inicia la siguiente, también a medianoche.
Sólo se reinicia entre bloques independientes y entre estrategias pareadas.
Cada primera acción se ejecuta con demanda realizada mediante el controlador
físico existente, que limita solicitudes al desbalance y límites reales.
Las solicitudes ajustadas quedan registradas; no son violaciones físicas.
El reactivo conserva su hipótesis anterior de realimentación ideal dentro de
la hora, representada por promedios horarios.

Cada bloque se cruza con dos perfiles solares y las tarifas plana/variable.
Las tres estrategias comparten exactamente demanda, generación, precios,
batería, SOC inicial, límites y controlador de realización en cada caso.
No hay oracle ni selector automático de estrategias.

## Solar cronológica de Córdoba

Se descargaron dos trayectorias nuevas de **192 horas consecutivas** mediante
el adaptador PVGIS v6 ya existente, con recibos, URL, fecha y SHA-256:

| Perfil independiente | Período local, Córdoba | Energía de las primeras 168 h |
|---|---|---:|
| Enero | 15–22/01/2023 | 108,698355 kWh |
| Julio | 15–22/07/2023 | 94,408924 kWh |

Ubicación −31,4201 / −64,1888; 3 kWp, ERA5, inclinación 30°, orientación norte,
eficiencia del sistema 0,86. Se usan las primeras 191 muestras: siete días
ejecutados más la cola de 23 horas del último pronóstico. No se concatena ni
repite un día solar. Los períodos se eligieron por estación, no buscando
resultados económicos favorables; las energías anteriores se midieron después.

Se conservan timestamps UTC, `America/Argentina/Cordoba`, índice ordinal y
timestamp UCI asignado en cada CSV de escenario. La conversión es W / 1.000 ×
1 h; el rendimiento PVGIS no se aplica de nuevo. Ambos perfiles se cruzan con
ambos bloques como factores controlados: se reutiliza una trayectoria semanal
entre casos, no se inventa variabilidad meteorológica repitiendo días.

**UCI y PVGIS no son mediciones sincronizadas ni de la misma vivienda,
ubicación o año.** La alineación a medianoche es una combinación simulada de
relojes independientes, sin afirmar equivalencia estacional francesa/argentina.
La trayectoria solar futura del escenario se conoce idealmente en cada origen.
No se ha evaluado un pronóstico meteorológico ni su incertidumbre.

Se verificó la API oficial el 22/10/2026: [documentación v6](https://photovoltaic-geographic-information-system.ec.europa.eu/api/v6/redoc)
y [notas de versión](https://photovoltaic-geographic-information-system.ec.europa.eu/api/release-notes).
El adaptador usa GET, que conserva sus parámetros de consulta.

## Tres estrategias y dos tratamientos distintos

1. **Reactivo:** sin planificación de demanda futura.
2. **A* MSE original, tratamiento A:** operación continua con la reserva
   original de 20 % como única meta terminal mínima.
3. **A* MSE terminal, tratamiento B:** mismo modelo, entradas, tarifas,
   discretización, objetivo de costo de importación y heurística; usa la
   opción existente `terminal_min_soc=0.5` al final de cada horizonte móvil.

La batería sigue en 5 kWh, potencias máximas 2,5 kW AC y eficiencias 0,95.
La grilla original es 0,25 kWh. El nuevo objetivo terminal, 2,5 kWh, pertenece
a esa grilla. No se modifican `planner.py`, `simulator.py`, sus contratos ni
las configuraciones originales. La reserva física de 20 % nunca cambia.

**La condición B se verifica sobre el plan previsto, no sobre toda la
trayectoria realizada ni como una igualdad al final del bloque.** Al
replanificar, el extremo del horizonte avanza. Puede programarse una reposición
futura que todavía no se ejecute en el paso actual; además cambia la demanda
prevista respecto de la realizada. Los resultados no aíslan causalmente estos
dos mecanismos. No debe confundirse esta variante con un cierre semanal de
SOC fijo, ni concluirse que cualquier tratamiento terminal es inútil.

La condición agrega restricciones de meta, sin penalizaciones monetarias ni
aristas negativas. La heurística relajada original ignora la reposición
terminal: su relajación física sigue siendo una cota inferior. La búsqueda
con reapertura conserva optimalidad **en su grafo discreto y horizonte
previsto**, no en toda la operación realizada bajo incertidumbre.
Se comprobó contra UCS en 24 problemas pequeños con tarifas cero/variables,
solar y distintos objetivos terminales, y desde estados intermedios.

## Valoración terminal: contabilidad, no factura

Se informan dos costos separados:

`C_bruto = Σ importación_t × tarifa_t`

`C_ajustado = C_bruto + v × (E_inicial − E_final)`

El valor principal se fija en **v = 0,1425 SIM/kWh almacenado**:
0,15 SIM/kWh AC de referencia × eficiencia de descarga 0,95. Valora la energía
potencialmente entregable al precio de referencia. Es el mismo valor para
todas las estrategias, tarifas, bloques y saldos finales. No se aplica una
segunda eficiencia a las energías almacenadas. La reserva común se cancela
al comparar sus variaciones.

El ajuste carga el agotamiento del inventario inicial y acredita energía
retenida; puede ser positivo o negativo. No modifica A*, no simula una recarga
real de cierre, no es un cobro/reembolso ni garantiza el valor económico
futuro de la batería. Es una comparación de recursos bajo un supuesto fijo.
No elimina toda sensibilidad al horizonte, a precios futuros o al valor de
disponer de energía durante un corte.

Sensibilidad predefinida: 0; 0,076 (=0,08×0,95); 0,1425; 0,1578947368
(=0,15/0,95, referencia de reposición desde red); 0,38 (=0,40×0,95).
El último valor no es una predicción de la tarifa al cierre. No se eligió una
valoración tras observar qué estrategia ganaba.

Los ajustes diarios se calculan usando el SOC que efectivamente inicia y
termina cada día. Sus sumas coinciden con el ajuste semanal por telescopado;
no se acredita de nuevo la batería inicial cada medianoche.

## Resultados: factura simulada y costo ajustado

Medias por bloque de siete días; cada fila de tarifa contiene cuatro cruces
(dos bloques × dos perfiles solares). Todas las estrategias inician con 2,5 kWh.

| Tarifa | Estrategia | Bruto SIM | Ajustado SIM | Red kWh | Energía final kWh |
|---|---|---:|---:|---:|---:|
| Plana | Reactivo | 19,007120 | 19,220870 | 126,714131 | 1,0000 |
| Plana | A* original | 19,646386 | 19,851230 | 130,975909 | 1,0625 |
| Plana | A* terminal | 19,664199 | 19,851230 | 131,094659 | 1,1875 |
| Variable | Reactivo | 26,951947 | 27,165697 | 126,714131 | 1,0000 |
| Variable | A* original | 24,102618 | 24,316368 | 134,778488 | 1,0000 |
| Variable | A* terminal | 24,102618 | 24,316368 | 134,778488 | 1,0000 |

La igualdad del costo ajustado entre variantes se refiere al valor principal,
no a cualquier valoración ni a igualdad de todas las acciones.

| Comparación A* original frente a reactivo | Tarifa plana | Tarifa variable |
|---|---:|---:|
| Reducción del costo bruto agregado | −3,3633 % | +10,5719 % |
| Reducción del costo ajustado agregado | −3,2796 % | +10,4887 % |
| Cambio en red importada | +3,3633 % | +6,3642 % |
| Gana / pierde en costo ajustado | 0 / 4 | 4 / 0 |
| Media de mejora por caso ± DE | −3,7860 ± 1,6720 pp | +9,7779 ± 2,8216 pp |
| Rango de mejora por caso | −5,7260 a −2,2461 % | +5,6288 a +11,9489 % |

Los porcentajes agregados son cocientes de sumas, no medias de porcentajes.
Al aplicar el ajuste, la mejora agregada variable cambia **−0,0832 puntos
porcentuales**. En estos cuatro casos ambas políticas terminan en 1 kWh:
el ahorro absoluto es el mismo, 2,849329 SIM por semana en promedio; el
denominador aumenta por el agotamiento común del inventario inicial.

En tarifa plana, el ajuste reduce el sobrecosto original en 0,0837 puntos
porcentuales. No invierte ningún caso. El peor caso relativo ajustado es el
bloque temprano con PV enero: +5,7260 %. Todos los casos desfavorables y
porcentajes se conservan en `unfavorable_cases.csv` y `paired_results.csv`.

Desglose del costo ajustado, SIM por semana, con v principal:

| Bloque | PV | Tarifa | Reactivo | A* original | A* terminal |
|---|---|---|---:|---:|---:|
| Septiembre | Enero | Plana | 11,485366 | 12,143018 | 12,143018 |
| Septiembre | Enero | Variable | 14,381746 | 13,572222 | 13,572222 |
| Septiembre | Julio | Plana | 13,209906 | 13,821281 | 13,821281 |
| Septiembre | Julio | Variable | 17,270821 | 15,407875 | 15,407875 |
| Enero | Enero | Plana | 25,247951 | 25,815057 | 25,815057 |
| Enero | Enero | Variable | 37,316696 | 32,857758 | 32,857758 |
| Enero | Julio | Plana | 26,940255 | 27,625564 | 27,625564 |
| Enero | Julio | Variable | 39,693525 | 35,427618 | 35,427618 |

Sobre la mezcla factorial completa, el bruto original baja 4,8088 % y el
ajustado 4,7836 %. Medias ajustadas ± DE muestral: reactivo 23,193283 ±
10,950878 SIM; original y terminal 22,083799 ± 9,433015 SIM. La DE mezcla
condiciones y bloques; no es un intervalo de confianza ni variabilidad entre
semillas. El peso igual de tarifas artificiales no representa uso anual real.

![Cambios de costo bruto y ajustado](../experiments/phase4_2/continuous-20261022/plots/paired_costs.png)

## ¿Qué hizo la condición terminal?

Los 1.344 planes de la variante cumplen su condición prevista de 2,5 kWh.
En el original, 1.337 de 1.344 planes terminan por debajo de ese objetivo.
Pese a cambiar planes completos, las primeras acciones son idénticas en seis
de los ocho casos. En los dos restantes, del bloque tardío con tarifa plana,
cambian 27 solicitudes horarias y 25 flujos realizados de batería.

En cada uno de esos dos casos la variante termina con **1,25 kWh frente a
1,00 kWh** del original. Importa 0,2375 kWh adicionales y factura 0,035625 SIM
más. Es exactamente el valor principal de los 0,25 kWh retenidos:
`0,25 × 0,1425 = 0,035625`. Por ello el costo ajustado empata; no es una mejora
de eficiencia ni una ganancia económica independiente del precio contable.
La energía solar capturada es la misma entre ambas variantes.

El original también retiene 1,25 kWh en el caso temprano/enero/plana. El resto
de sus saldos es 1 kWh. La variante retiene 1,25 kWh en tres casos planos;
en los otros cinco termina en 1 kWh. **Ninguna termina la semana en 50 %**.
Las horas por debajo del objetivo previsto no violan la reserva física de 20 %.
`terminal_diagnostics.csv` conserva estas diferencias y el costo de búsqueda.

La condición consume más búsqueda: 305.180 → 445.399 nodos expandidos y
183,06 → 256,06 s acumulados entre llamadas concurrentes. No se justifica
adoptarla como política predeterminada con esta evidencia.

## Sensibilidad y persistencia de la conclusión

Reducción agregada del costo ajustado original frente al reactivo:

| v (SIM/kWh almacenado) | Plana | Variable |
|---:|---:|---:|
| 0 | −3,3633 % | +10,5719 % |
| 0,076 | −3,3184 % | +10,5274 % |
| 0,1425 | −3,2796 % | +10,4887 % |
| 0,1578947368 | −3,2706 % | +10,4798 % |
| 0,38 | −3,1441 % | +10,3529 % |

Los ocho signos pareados frente a reactivo se mantienen para ambas variantes
en las cinco valoraciones. En los dos casos donde la variante retiene más
energía que el original, cambia su preferencia relativa: debajo de 0,1425
cuesta más ajustado; encima cuesta menos ajustado. La magnitud máxima del
beneficio contable por caso dentro del rango es 0,059375 SIM. Es sensibilidad
al valor residual, no demostración de una mejor decisión operativa general.

Las curvas que coinciden se superponen en el gráfico; no son series ausentes.

![Sensibilidad a la valoración terminal](../experiments/phase4_2/continuous-20261022/plots/terminal_sensitivity.png)

## Costo y dependencia de red

Las cuatro victorias de A* con tarifa variable también aumentan importación.
La captación solar media pasa de 88,8483 % del reactivo a 83,4416 % en A*;
con tarifa plana es 84,7374 %. Se define como PV directa a demanda más PV
cargada a batería, dividido por PV disponible. Incluye energía retenida y
pérdidas posteriores; no equivale a autonomía útil ni a cobertura de cortes.

Medias de carga/descarga AC por caso: reactivo 23,0614 / 22,2379 kWh;
original 36,5514 / 34,3829; terminal 36,5514 / 34,3235. La función económica
acepta importación barata y pérdidas para evitar energía cara. Valorar el saldo
final no transforma ese objetivo en minimizar red. No se calculan horas de
autonomía durante cortes porque no se simularon cortes aquí.

![Importación y captación solar](../experiments/phase4_2/continuous-20261022/plots/grid_solar.png)

## Implementación y verificación

Se agregaron `energy/continuous.py`, `continuous_experiment.py`,
`continuous_plots.py`, la CLI `continuous-energy`, dos scripts de análisis y
verificación, configuración, pruebas y documentación. Se reutilizan proveedor
congelado, métricas existentes, PVGIS, simulador y A*. Los contratos públicos
anteriores y los resultados de fases previas se conservan.

Cada caso guarda condiciones, ledger continuo, pronósticos, búsquedas y **cada
transición de cada plan previsto completo** en CSV comprimido. La primera
acción sólo se entrega al simulador tras validar el plan entero. Los fallos
de estrategia o proceso se registran, sin fallback. Los agregados principales
exigen tres ejecuciones completas por caso y publican cobertura: 8/8 en esta
corrida, 0/24 ejecuciones fallidas.

**104 pruebas aprobadas sin entrenamiento:** las 94 anteriores sin ajustes
TensorFlow y diez nuevas. Cubren continuidad a medianoche, causalidad,
unidades/signo y telescopado contable, selección de bloques, cola solar,
contratos inválidos, fallos explícitos, sensibilidad y A* contra UCS. No se
ejecutaron los tres módulos de integración TensorFlow (ocho pruebas), que
incluyen ajustes sobre fixtures sintéticos. No se reentrenó ninguna red.

La verificación independiente reproduce 4.032 transiciones realizadas y
64.512 transiciones previstas de 2.688 planes; comprueba 144 cruces de
medianoche y 1.344 emisiones congeladas. Revisa relojes solares, igualdad de
condiciones, balances telescópicos, tarifas, costos, saldos, agregados y
sensibilidad. **962 archivos protegidos permanecieron intactos**, incluyendo
modelos, resultados anteriores, simulador y planificador originales.

El tiempo de corrida fue aproximadamente 249 s con dos procesos CPU. La
búsqueda más lenta tardó 0,607 s; el máximo de expansiones por llamada fue
417, frente a límites de 10 s y 100.000. Los tiempos sumados de llamadas
concurrentes no equivalen al tiempo total ni a tiempo CPU. No se ocultaron
búsquedas lentas ni casos fallidos.

Se generaron once gráficos: tres comparativos y ocho trayectorias, una por
caso, con demanda, generación, SOC, costo acumulado y decisiones de batería.
Los archivos y el resultado detallado de verificación se enlazan abajo.

## Límites y recomendación para los modos futuros

El resultado conserva la conclusión cualitativa de Fase 4.1, pero su 15,21 %
variable anterior no es un control comparable con el 10,49 % actual: cambian
días, continuidad, SOC inicial, perfiles solares y factorial. No se atribuye
esa diferencia numérica exclusivamente al SOC terminal. La comparación
controlada de esta fase es bruto frente a ajustado sobre las mismas ejecuciones.

Sólo hay dos semanas UCI, de una vivienda, una semilla y dos trayectorias
solares estacionales. Los cruces no son ocho semanas independientes. La
selección por completitud puede excluir períodos difíciles. La validación ya
fue usada en desarrollo; no constituye evidencia final fuera de muestra.
No hay incertidumbre solar, degradación, autodescarga, exportación remunerada,
límite de conexión, cortes ni hardware. La grilla discreta A* y el control
reactivo continuo siguen teniendo distinto espacio de acciones. La valoración
lineal es un supuesto, no una estimación del valor real de la energía retenida.

Para diseñar los modos, sin implementarlos todavía:

- **Económico:** conservar el costo bruto como factura simulada y reportar
  siempre saldo y ajuste. A* MSE es candidato para tarifas variables; el
  reactivo sigue siendo una referencia fuerte con tarifa plana. Revisar un
  cierre con fecha fija u operación más larga antes de adoptar una regla terminal.
- **Autonomía:** acordar explícitamente qué se prioriza —menos importación,
  disponibilidad de batería o cobertura de cortes— y medirlo. La minimización
  de costo actual no respalda prometer autonomía mayor.
- **Equilibrado:** definir con supervisión humana límites aceptables de
  sobrecosto, importación y energía de respaldo; comparar después alternativas
  sobre una misma frontera de compromisos y valoración terminal. No elegir
  pesos por un único porcentaje favorable de estas semanas.

Conservar la variante B como experimento reproducible, no promoverla por
defecto. No cambiar objetivo ni objetivo terminal tras observar estos resultados.
No se implementaron los tres modos, dashboard, Colab ni Fase 4.3; no se hicieron
commits o push.

## Artefactos y reproducción

- [Resultados por caso y estrategia](../experiments/phase4_2/continuous-20261022/results_by_case.csv).
- [Comparaciones pareadas](../experiments/phase4_2/continuous-20261022/paired_results.csv).
- [Agregados por escenario](../experiments/phase4_2/continuous-20261022/by_scenario.csv).
- [Agregados por bloque](../experiments/phase4_2/continuous-20261022/by_block.csv).
- [Agregados y dispersión](../experiments/phase4_2/continuous-20261022/aggregate.csv).
- [Resumen económico](../experiments/phase4_2/continuous-20261022/economic_summary.csv).
- [Contabilidad diaria continua](../experiments/phase4_2/continuous-20261022/daily_metrics.csv).
- [Diagnóstico de la condición terminal](../experiments/phase4_2/continuous-20261022/terminal_diagnostics.csv).
- [Sensibilidad completa](../experiments/phase4_2/continuous-20261022/sensitivity.csv).
- [Verificación y hashes](../experiments/phase4_2/continuous-20261022/verification.json).
- [Ejemplo de trayectoria semanal](../experiments/phase4_2/continuous-20261022/plots/trajectory_early_january_time_of_use.png).

Desde la raíz del proyecto, con dependencias y artefactos originales instalados:

```powershell
.\.venv\Scripts\python.exe -m bekantor_energy download-pvgis --home-config configs/home.phase3.json --latitude -31.4201 --longitude -64.1888 --start 2023-01-15T03:00:00 --end 2023-01-23T02:00:00 --output data/raw/pvgis/cordoba-20230115-8d.json --processed-output data/processed/pvgis/cordoba-20230115-8d.csv
.\.venv\Scripts\python.exe -m bekantor_energy download-pvgis --home-config configs/home.phase3.json --latitude -31.4201 --longitude -64.1888 --start 2023-07-15T03:00:00 --end 2023-07-23T02:00:00 --output data/raw/pvgis/cordoba-20230715-8d.json --processed-output data/processed/pvgis/cordoba-20230715-8d.csv
.\.venv\Scripts\python.exe -u -m bekantor_energy continuous-energy --config configs/energy.continuous.json --run-dir experiments/phase4_2/repeat-01 --scenario-dir data/simulated/phase4_2/repeat-01
.\.venv\Scripts\python.exe scripts/analyze_continuous_energy.py --run-dir experiments/phase4_2/repeat-01
.\.venv\Scripts\python.exe scripts/verify_continuous_energy.py --run-dir experiments/phase4_2/repeat-01
```

Las descargas pueden reutilizar el caché verificado. Las rutas de salida de
simulación deben ser nuevas. `workers: 1` en una copia de la configuración
permite ejecución secuencial sin cambiar decisiones ni objetivos. Las figuras
se generan automáticamente; pueden repetirse desde los CSV sin simular.

Pruebas sin entrenamiento ni apertura de la partición test:

```powershell
New-Item -ItemType Directory -Force -Path experiments/tmp | Out-Null
$env:TEMP = Join-Path (Get-Location) 'experiments/tmp'
$env:TMP = $env:TEMP
.\.venv\Scripts\python.exe -c "import unittest; loader=unittest.TestLoader(); names=('test_acquisition.py','test_core.py','test_data.py','test_pipeline.py','test_forecasting.py','test_calendar.py','test_weighted.py','test_energy*.py'); suite=unittest.TestSuite(loader.discover('tests',pattern=n) for n in names); result=unittest.TextTestRunner(verbosity=2).run(suite); raise SystemExit(not result.wasSuccessful())"
```

Los modelos, datos y resultados pesados siguen excluidos de Git. La
documentación conserva resultados resumidos sin incorporar los datasets.

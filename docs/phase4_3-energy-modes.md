# Fase 4.3 — Económico, Autonomía y Equilibrado

**Los tres modos funcionan sobre un único núcleo A* y el mismo simulador.**
Con tarifa variable, Económico conserva la reducción de costo ajustado de
**10,49 %** frente al reactivo, con **6,36 % más red**. Equilibrado reduce costo
**4,52 %**, con **4,07 % más red**. Autonomía reduce importación respecto de
los otros A* en promedio, pero **no supera al reactivo en ninguno de los ocho
casos**: importa **2,99 % más**. No se recomienda presentarlo como una mejora
de autonomía ya validada frente a la referencia convencional.

La corrida aceptada es **`experiments/phase4_3/modes-corrected-20261022`**:
dos bloques continuos de siete días, ocho casos pareados, 32 ejecuciones,
sin fallos ni violaciones físicas. El resultado no demuestra un compromiso
universalmente mejor ni ahorro real de una instalación.

## Objetivos y parámetros fijados

El [diseño previo](phase4_3-design.md) y la
[configuración](../configs/energy.modes.json) fijan los parámetros; no se
exploraron pesos sobre los nuevos resultados. La
[documentación de interfaz y matemáticas](energy-management-modes.md) contiene
la API, ecuaciones y condiciones de optimalidad.

| Modo | Objetivo previsto de 24 h | Unidades | Heurística |
|---|---|---|---|
| Económico | `min C`, costo de importación | SIM | Cota relajada original |
| Autonomía | `min lex(G,C)`: menor red y, a igualdad, menor costo | kWh, luego SIM | `(0,0)`, UCS lexicográfico |
| Equilibrado | `min 0,5*C/1 SIM + 0,5*G/1 kWh` | Adimensional | Combinación no negativa de cotas |

Equilibrado equivale en orden a minimizar `C + 1 SIM/kWh × G`. Esa valoración
de red es un parámetro del objetivo, **no una tarifa facturada**. Las escalas
de 1 SIM y 1 kWh son fijas, no se estiman con validación. Un peso de 50 %
no es neutral por sí mismo; con estas escalas se da importancia relativamente
alta a evitar red frente a precios de 0,08–0,40 SIM/kWh.

`ObjectiveConfig` selecciona el modo y valida pesos/escalas. `ModePlanner`
usa el grafo físico compartido y `search.astar_search`. No hay tres motores
independientes. `ModeEnergyPlan.predicted_cost` sigue siendo costo SIM;
`objective_primary`, `objective_secondary`, `objective_units` y
`predicted_grid_import_kwh` separan el objetivo de sus magnitudes físicas.

No se incorpora el objetivo terminal experimental del 50 %. Todos los modos
usan la reserva física/terminal original de **20 %**, horizonte de 24 h y
replanificación horaria. El ajuste terminal es sólo evaluación:
`C_aj = C_bruto + 0,1425 SIM/kWh_almacenado × (E0−ET)`.
Se conserva la sensibilidad predefinida de Fase 4.2. Ni el ajuste ni J son
una factura real.

## Condiciones pareadas y causalidad

Se reutilizan exactamente los casos de Fase 4.2, sin elegir otros por sus
resultados. Cada política enfrenta demanda realizada, generación, tarifas,
batería y SOC inicial iguales dentro de un caso.

- Bloque temprano: 21–27/09/2009, 168 horas, 164,250233 kWh de demanda.
- Bloque tardío: 24–30/01/2010, 168 horas, 267,905800 kWh de demanda.
- PVGIS Córdoba, −31,4201 / −64,1888, 3 kWp, ERA5, 30° hacia norte,
  eficiencia PVGIS 0,86: trayectorias cronológicas de enero y julio de 2023,
  con ocho días fuente, sin repetir artificialmente un día. Se ejecutan siete
  días y se conserva cola solar para todos los horizontes.
- Tarifa plana 0,15 SIM/kWh o variable original: 0,08 de 00–06; 0,18 de
  07–17; 0,40 de 18–21; 0,18 de 22–23. Moneda ilustrativa SIM, sin exportación.
- Batería 5 kWh, potencia máxima 2,5 kW AC, eficiencias 0,95, grilla 0,25 kWh,
  SOC inicial 50 % = 2,5 kWh. El estado se conserva entre días; no se reinicia
  a medianoche. Cada bloque independiente reinicia las condiciones iniciales.

Las tres políticas A* reciben las mismas emisiones causales archivadas de la
MLP calendario + MSE 64–32, semilla 2026, sin entrenamiento ni selección de
redes. El proveedor no recibe objetivos realizados futuros. Se ejecuta sólo
la primera orden con demanda realizada y se vuelve a planificar con el SOC
alcanzado. El controlador físico limita solicitudes al desbalance realizado,
sin ejecutar acciones que violen potencia, capacidad, reserva o balance.

El reactivo conserva su realimentación ideal del desequilibrio presente dentro
de la hora, representada mediante promedios horarios. Sus acciones continuas
no tienen la discretización de A*. No utiliza demanda futura.

La trayectoria solar del escenario es conocida idealmente en cada horizonte.
**UCI y PVGIS no son mediciones sincronizadas ni de la misma vivienda,
ubicación o año.** Los relojes UTC/local y su asignación ordinal a UCI se
conservan. No se evalúa un pronóstico meteorológico ni hardware real.

## Resultados agregados

Medias por episodio continuo de siete días. DE muestral entre ocho cruces de
condiciones, no ocho semanas independientes ni variación entre semillas.

| Estrategia | Costo bruto SIM, media ± DE | Costo ajustado SIM, media ± DE | Red kWh, media ± DE |
|---|---:|---:|---:|
| Reactivo | 22,979533 ± 10,950878 | 23,193283 ± 10,950878 | 126,714131 ± 49,362195 |
| Económico | 21,874502 ± 9,427658 | 22,083799 ± 9,433015 | 132,877198 ± 49,147295 |
| Autonomía | 23,180698 ± 10,515886 | 23,394448 ± 10,515886 | 130,498526 ± 49,769501 |
| Equilibrado | 22,689541 ± 9,596767 | 22,894384 ± 9,607260 | 131,454395 ± 49,753300 |

La mezcla factorial da peso igual a ambas tarifas y perfiles, no a su
probabilidad real. Por ello no se presenta su porcentaje global como ahorro
anual esperado. Los resultados condicionales son más útiles para decidir.

### Tarifa plana: el reactivo sigue siendo más favorable

| Estrategia | Bruto SIM | Ajustado SIM | Red kWh | Cambio de costo ajustado vs reactivo |
|---|---:|---:|---:|---:|
| Reactivo | 19,007120 | 19,220870 | 126,714131 | — |
| Económico | 19,646386 | 19,851230 | 130,975909 | +3,2796 % |
| Autonomía | 19,574779 | 19,788529 | 130,498526 | +2,9533 % |
| Equilibrado | 19,655293 | 19,851230 | 131,035284 | +3,2796 % |

Los tres A* pierden frente a reactivo en costo ajustado e importación en los
cuatro casos planos. Con tarifa constante, `C=p*G`: sus objetivos primarios
son colineales. Las diferencias realizadas entre modos no constituyen una
nueva oportunidad tarifaria: influyen los empates de planes, heurísticas y
precisión numérica, ejecución discreta, errores predictivos y replanificación.
No deben interpretarse como tres óptimos físicos continuos distintos.

### Tarifa variable: existe un compromiso medible

| Estrategia | Bruto SIM | Ajustado SIM | Red kWh | Reducción costo ajustado vs reactivo | Aumento de red vs reactivo |
|---|---:|---:|---:|---:|---:|
| Reactivo | 26,951947 | 27,165697 | 126,714131 | — | — |
| Económico | 24,102618 | 24,316368 | 134,778488 | 10,4887 % | 6,3642 % |
| Autonomía | 26,786616 | 27,000366 | 130,498526 | 0,6086 % | 2,9866 % |
| Equilibrado | 25,723788 | 25,937538 | 131,873505 | 4,5210 % | 4,0717 % |

Los porcentajes son cocientes de sumas sobre cuatro casos, no medias de
porcentajes. Económico reduce costo ajustado en **4/4** casos; Autonomía y
Equilibrado en **2/4**, con deterioro en los otros dos.

Comparado con Económico, Equilibrado evita **2,904983 kWh/semana en promedio**
(−2,1554 % de su importación) a cambio de **1,621170 SIM adicionales**
(+6,6670 % de su costo ajustado). El cociente realizado es 0,558065 SIM por
kWh evitado. No es una tarifa ni el precio sombra configurado: resulta del
replay con errores y sólo describe estos casos.

Autonomía reduce importación respecto de Económico en **3,1756 %** bajo tarifa
variable, con **11,0378 % más costo ajustado**. Sigue importando más que el
reactivo. Todos terminan los casos variables en **1 kWh**, desde los mismos
2,5 kWh iniciales: estas diferencias no se deben a saldos finales distintos.

![Costo frente a importación](../experiments/phase4_3/modes-corrected-20261022/plots/cost_grid_tradeoff.png)

Cada línea conecta las cuatro políticas del mismo caso. Los puntos pueden
superponerse. Son cuatro opciones observadas, no una frontera Pareto completa
ni una garantía de eficiencia fuera de estos escenarios.

## Desglose por escenario y resultados desfavorables

Costo ajustado SIM por semana, con valoración principal:

| Bloque | PV | Tarifa | Reactivo | Económico | Autonomía | Equilibrado |
|---|---|---|---:|---:|---:|---:|
| Septiembre | Enero | Plana | 11,485366 | 12,143018 | 12,073595 | 12,143018 |
| Septiembre | Enero | Variable | 14,381746 | 13,572222 | 15,019440 | 15,102748 |
| Septiembre | Julio | Plana | 13,209906 | 13,821281 | 13,639900 | 13,821281 |
| Septiembre | Julio | Variable | 17,270821 | 15,407875 | 17,324103 | 17,416196 |
| Enero | Enero | Plana | 25,247951 | 25,815057 | 25,815057 | 25,815057 |
| Enero | Enero | Variable | 37,316696 | 32,857758 | 36,542063 | 34,413605 |
| Enero | Julio | Plana | 26,940255 | 27,625564 | 27,625564 | 27,625564 |
| Enero | Julio | Variable | 39,693525 | 35,427618 | 39,115860 | 36,817605 |

En los dos casos tempranos variables, Equilibrado y Autonomía son superados
por el reactivo tanto en costo ajustado como en importación. En los dos casos
tardíos variables aparecen compromisos no dominados entre las cuatro opciones.
Con tarifa plana, el reactivo domina a los tres A* en los cuatro casos.
`observed_dominance.csv` conserva esta comparación, sin generalizarla a todas
las políticas posibles.

No hay casos donde alguno de los tres A* importe menos que el reactivo. Para
Autonomía el deterioro relativo de importación va de **2,2653 % a 5,2187 %**.
Sus ocho derrotas no se ocultan en el promedio. Cambiar el objetivo previsto
no elimina las diferencias entre el grafo discreto/MPC con MLP y la
realimentación continua ideal del reactivo.

![Diferencias pareadas](../experiments/phase4_3/modes-corrected-20261022/plots/paired_changes.png)

## Qué explica físicamente el exceso de red

Con demanda, solar y energía inicial iguales, sin exportación ni cortes,
el balance entre políticas satisface:

`Δ red = Δ recorte solar + Δ pérdidas de conversión + Δ energía final`.

Promedio de los casos variables, diferencias frente al reactivo, kWh/semana:

| Modo | Red adicional | Recorte adicional | Pérdidas adicionales | Diferencia final |
|---|---:|---:|---:|---:|
| Económico | 8,064357 | 5,488128 | 2,576229 | 0 |
| Autonomía | 3,784396 | 3,730089 | 0,054306 | 0 |
| Equilibrado | 5,159374 | 4,557186 | 0,602188 | 0 |

La inferioridad de Autonomía frente a reactivo está asociada principalmente
al mayor recorte solar, no a pérdidas masivas o una batería final más vacía.
Esta identidad explica las energías; **no identifica causalmente** cuánto del
recorte proviene de errores MLP, discretización, horizonte o reglas de ejecución.
No se realizaron nuevas ablaciones para favorecer un modo.

El reactivo es una referencia especialmente fuerte para minimizar red bajo
estos supuestos de realimentación ideal, eficiencia constante y ausencia de
cortes/límite de conexión. La optimalidad de Autonomía en cada grafo previsto
no implica dominancia realizada sobre esa política.

## Solar, SOC y resiliencia

Medias entre los ocho casos; SOC medio calculado con trapecios horarios:

| Estrategia | Captación solar media | Recorte medio kWh | SOC mínimo observado | SOC medio | SOC final medio |
|---|---:|---:|---:|---:|---:|
| Reactivo | 88,8483 % | 11,366264 | 20 % | 34,2158 % | 20,0000 % |
| Económico | 84,0895 % | 16,184376 | 20 % | 44,9369 % | 20,6250 % |
| Autonomía | 85,1640 % | 15,096353 | 20 % | 35,9775 % | 20,0000 % |
| Equilibrado | 84,5333 % | 15,718906 | 20 % | 42,0652 % | 21,2500 % |

Captación es PV directa más PV cargada, dividido por PV disponible. La media
de porcentajes no es el cociente global de energías. Incluye energía retenida
y pérdidas posteriores, no sólo demanda útil servida.

Autonomía tiene menor SOC medio que Económico: **evitar red y mantener SOC
alto son objetivos distintos**. Ninguno garantiza respaldo. ENS fue cero
porque se evaluó red disponible; no se simularon cortes ni se implementó
Resiliencia. No se infieren horas de respaldo o seguridad de suministro.

La sensibilidad contable de cinco valores predefinidos mantiene el signo de
las comparaciones frente al reactivo. En los casos variables todos comparten
saldo final, por lo que las diferencias absolutas de costo son independientes
de v. En algunos planos cambian preferencias entre A* por energía retenida;
no cambian los pesos de Equilibrado ni se altera ninguna política.

## Garantías, corrección numérica y pruebas

El grafo físico y el bucle A* son compartidos. Económico conserva costo,
heurística, grilla y orden de comparación escalar de referencia. Se verificó
contra **todas las trayectorias realizadas de los ocho casos de Fase 4.2**:
acciones, SOC, flujos y costos coinciden dentro de 1e−9, excluyendo tiempos.

Equilibrado combina cotas inferiores con coeficientes no negativos. Autonomía
usa h=(0,0) y orden lexicográfico racional. El tiempo siempre avanza; los costos
son aditivos y no negativos, la frontera reabre estados mejorados. La garantía
es optimalidad en el grafo/horizonte previsto, con la tolerancia escalar de
referencia 1e−12 y la precisión física original, no optimalidad continua o
global de la semana realizada.

**Se corrigió un defecto numérico antes de aceptar resultados.** Sumar como
fracciones los flujos previamente redondeados introducía falsas diferencias
primarias. Una prueba sintética con dos cargas de 1 kWh, eficiencia 0,95,
0,5 kWh utilizables y tarifas 0,40/0,08 elegía costo 0,366 SIM en vez de
0,290 SIM, ambos con 1,525 kWh de importación física. El objetivo se calcula
ahora racionalmente desde demanda, solar, transiciones de energía y eficiencias,
antes de redondear flujos. Se preservan así los empates físicos y el desempate
por costo. No se introdujo epsilon ni se modificaron pesos o condiciones.

La corrida preliminar `modes-20261022`, sus fuentes y diagnóstico se conservan
íntegros como auditoría; **no se usan sus métricas para la conclusión**.
La corrida aceptada `modes-corrected-20261022` repite los mismos escenarios y
objetivos tras la corrección. Se conserva además el diseño previo de esa
revisión, sin reescribir resultados históricos para esconder el defecto.

**118 pruebas aprobadas sin entrenamiento**, incluidas 14 nuevas: selección y
unidades de objetivos, normalización, prioridad lexicográfica ante diferencias
mínimas, empates con eficiencia realista, extremos de pesos, cotas desde estados
intermedios, límites de búsqueda, causalidad, continuidad, métricas SOC y fallos
sin fallback. Se contrastaron **36 problemas modo/escenario** pequeños contra
UCS y enumeración independiente, además de pruebas de estados intermedios.
Se excluyeron los módulos TensorFlow con ajustes de fixtures: no se reentrenó
ninguna red ni se abrió la partición test.

La verificación independiente comprobó:

- **5.376 transiciones realizadas** y 192 cruces de medianoche sin reinicios.
- **4.032 planes completos y 96.768 transiciones previstas** físicamente factibles.
- 1.344 emisiones congeladas, comunes a los tres modos, sin objetivos futuros
  realizados en el proveedor.
- Los tres objetivos en sus unidades, condiciones pareadas, costos, SOC,
  balances de energía, agregados, descomposición de red y ajuste/sensibilidad.
- **1.148 artefactos y archivos previos protegidos**, incluidos modelos, datos,
  resultados históricos, fuentes archivados y corrida preliminar intactos.
- Evolución explícita del fuente activo del planner/evaluador. No se afirma
  falsamente que conserve su hash antiguo tras añadir modos.

No hubo búsquedas o ejecuciones fallidas en los ocho casos: cobertura 8/8 de
cuatro vías y 32/32 ejecuciones completas. Los mecanismos de fallos/límites
se probaron con fixtures; los resultados parciales se registrarían sin fallback
y fuera de agregados completos, con cobertura explícita.

## Costo computacional

Medias ± DE muestral de búsqueda acumulada por semana; puntos de escenarios
repetidos, no mediciones CPU aisladas:

| Estrategia | Tiempo de búsqueda s | Nodos expandidos medios |
|---|---:|---:|
| Reactivo | 0 | 0 |
| Económico | 14,0424 ± 3,5845 | 38.147,50 |
| Autonomía | 40,4351 ± 2,1208 | 55.704,75 |
| Equilibrado | 14,2372 ± 3,4112 | 37.006,63 |

Autonomía paga la búsqueda h=0 y aritmética racional; no se oculta ese costo.
La corrida tardó **315,87 s con dos procesos CPU**. La llamada más lenta
tardó 0,637 s y el máximo de expansiones observado por búsqueda fue 408,
frente a límites originales de 10 s y 100.000. La suma de tiempos de llamadas
concurrentes no equivale al tiempo total ni a tiempo CPU.

## Recomendación para el producto

No fijar un único compromiso universal a partir de estas semanas:

1. **Económico:** conservarlo como candidato para usuarios que priorizan costo
   y aceptan el mayor uso de red. Es el más consistente de los tres A* en
   tarifa variable en esta muestra. Con precio plano, mantener reactivo como
   alternativa de referencia más favorable.
2. **Autonomía:** el objetivo y desempate están implementados y verificados,
   pero no promoverlo como superior al reactivo. Para una prioridad real de
   mínima importación, conservar esa política convencional como referencia
   operativa hasta obtener evidencia de mejora. No se implementó un fallback
   automático ni se cambió silenciosamente el modo A* por reactivo.
3. **Equilibrado:** ofrecerlo como compromiso explícito, no como opción
   automáticamente mejor. Con esta configuración sacrifica 6,67 % de costo
   ajustado frente a Económico para evitar 2,16 % de su importación variable.
   Sólo reduce costo frente a reactivo en 2/4 casos variables. La aceptación
   de ese intercambio debe decidirla el usuario, con más evidencia.

Antes de recomendar pesos para producción, ampliar cobertura de validación,
revisar la pérdida de captación solar y contrastar la discretización/ejecución
con supervisión humana. Son recomendaciones, no experimentos implementados
ni ajustes retrospectivos de esta fase.

## Límites y reproducción

Dos semanas de una vivienda, una semilla y dos perfiles solares no sostienen
una conclusión poblacional o anual. La validación ya se utilizó en desarrollo.
No hay errores de pronóstico solar, degradación, autodescarga, exportación
remunerada, límite de conexión, cortes ni hardware. Los ocho cruces reutilizan
los mismos bloques/perfiles: la dispersión es descriptiva, sin intervalos de
confianza ni inferencia entre semillas. Los objetivos usan predicciones, no
conocimiento perfecto de la demanda. No se generó dashboard ni Colab y no se
avanzó a otra fase, ni se hicieron commits o push.

Artefactos aceptados:

- [Resultados por caso](../experiments/phase4_3/modes-corrected-20261022/results_by_case.csv).
- [Resumen por tarifa y modo](../experiments/phase4_3/modes-corrected-20261022/mode_summary.csv).
- [Agregados por escenario](../experiments/phase4_3/modes-corrected-20261022/by_scenario.csv).
- [Agregados por bloque](../experiments/phase4_3/modes-corrected-20261022/by_block.csv).
- [Diferencias y compromisos](../experiments/phase4_3/modes-corrected-20261022/mode_tradeoffs.csv).
- [Descomposición del exceso de red](../experiments/phase4_3/modes-corrected-20261022/grid_gap_decomposition.csv).
- [Comparación de dominancia observada](../experiments/phase4_3/modes-corrected-20261022/observed_dominance.csv).
- [Sensibilidad terminal](../experiments/phase4_3/modes-corrected-20261022/sensitivity.csv).
- [Regresión de Económico](../experiments/phase4_3/modes-corrected-20261022/economic_regression.json).
- [Verificación y hashes](../experiments/phase4_3/modes-corrected-20261022/verification.json).

Se generaron 13 gráficos: cinco comparativos —costo/red, diferencias pareadas,
SOC, esfuerzo y sensibilidad— y ocho trayectorias semanales. Por ejemplo,
[SOC comparativo](../experiments/phase4_3/modes-corrected-20261022/plots/soc_comparison.png)
y [trayectoria variable](../experiments/phase4_3/modes-corrected-20261022/plots/trajectory_early_january_time_of_use.png).

Desde la raíz, conservando los artefactos congelados y el archivo de fuentes
anteriores indicado en la configuración:

```powershell
.\.venv\Scripts\python.exe -u -m bekantor_energy energy-modes --config configs/energy.modes.json --run-dir experiments/phase4_3/repeat-01
.\.venv\Scripts\python.exe scripts/analyze_energy_modes.py --run-dir experiments/phase4_3/repeat-01
.\.venv\Scripts\python.exe scripts/verify_energy_modes.py --run-dir experiments/phase4_3/repeat-01
```

Las salidas deben ser nuevas. No se descarga ni entrena nada. `workers: 1`
permite ejecución secuencial; no cambia objetivos ni datos. Las figuras pueden
regenerarse con `mode_figures` desde los CSV. Los verificadores históricos
mantienen sus exigencias de fuentes históricos; el nuevo verificador usa el
archivo explícito y la regresión, sin cambiar sus manifiestos.

Pruebas sin entrenamiento ni partición test:

```powershell
New-Item -ItemType Directory -Force -Path experiments/tmp | Out-Null
$env:TEMP = Join-Path (Get-Location) 'experiments/tmp'
$env:TMP = $env:TEMP
.\.venv\Scripts\python.exe -c "import unittest; loader=unittest.TestLoader(); names=('test_acquisition.py','test_core.py','test_data.py','test_pipeline.py','test_forecasting.py','test_calendar.py','test_weighted.py','test_energy*.py'); suite=unittest.TestSuite(loader.discover('tests',pattern=n) for n in names); result=unittest.TextTestRunner(verbosity=2).run(suite); raise SystemExit(not result.wasSuccessful())"
```

Los datos, modelos y resultados pesados siguen excluidos de Git. La API y los
nuevos módulos usan dependencias existentes y biblioteca estándar.

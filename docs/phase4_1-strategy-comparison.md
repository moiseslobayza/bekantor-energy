# Fase 4.1 — Comparación pareada de estrategias energéticas

**Conclusión:** ninguna estrategia operativa domina todos los escenarios.
Con tarifa plana, el autoconsumo reactivo tiene menor costo medio realizado
en los episodios de 24 h. Con tarifa variable, A* + MLP MSE reduce el costo
agregado **15,21 %**, importando **3,92 % más energía**. La MLP ponderada no
presenta una mejora energética suficientemente consistente para reemplazar
la MSE: su ventaja agregada es sólo **0,29 %**, con 24 victorias, 26 derrotas
y 62 empates; muchas victorias coinciden con menor SOC final.

Se completaron **14 días no solapados × 8 escenarios = 112 casos pareados**,
con **448 ejecuciones**, sin fallos ni violaciones físicas. No se modificó el
simulador, el planificador, su objetivo, las tarifas de Fase 3 ni las redes.
La mezcla factorial es un experimento simulado, no ahorro económico real.

## Diseño fijado antes de simular costos

Se identificaron 175 días elegibles de validación. Cada uno debía comenzar a
medianoche, tener 24 emisiones consecutivas de pronóstico y disponer de las
47 horas observadas necesarias para los horizontes rodantes, siempre dentro
de validación. No se abrió test.

La demanda diaria definió dos estratos retrospectivos: tercio bajo ≤
**26,1119 kWh/día** y tercio alto ≥ **32,063367 kWh/día**. De cada estrato se
eligieron siete posiciones distribuidas cronológicamente mediante `linspace`,
sin observar resultados energéticos. El estrato no se entrega al controlador.
Esta selección offline utiliza demanda realizada para diseñar la muestra;
no convierte esos valores futuros en información disponible para una MLP.

| Episodio | Día UCI | Estrato | Demanda realizada (kWh) |
|---|---|---|---:|
| 01 | 2009-09-21 | Bajo | 23,6311 |
| 02 | 2009-10-06 | Bajo | 23,8075 |
| 03 | 2009-10-17 | Alto | 45,7013 |
| 04 | 2009-10-26 | Bajo | 18,5391 |
| 05 | 2009-11-16 | Bajo | 25,2411 |
| 06 | 2009-11-20 | Alto | 32,6836 |
| 07 | 2009-12-13 | Alto | 35,6469 |
| 08 | 2009-12-30 | Alto | 40,8518 |
| 09 | 2010-01-28 | Alto | 35,1706 |
| 10 | 2010-02-15 | Alto | 34,9522 |
| 11 | 2010-02-23 | Bajo | 24,3547 |
| 12 | 2010-03-14 | Bajo | 26,1119 |
| 13 | 2010-03-31 | Alto | 32,0634 |
| 14 | 2010-04-22 | Bajo | 24,3484 |

Hay 11 días laborables y tres de fin de semana, distribuidos entre septiembre
de 2009 y abril de 2010. Son 336 horas únicas ejecutadas; los pronósticos de
24 horas se solapan dentro de cada episodio. El tercio medio de demanda no
se muestrea: el diseño contrasta condiciones, no estima una media poblacional.

Cada día se cruza con:

- Solar alta: 100 % del perfil base; solar baja: 25 % del mismo perfil.
- Tarifa plana: 0,15 SIM/kWh; variable: el perfil ilustrativo de Fase 3,
  0,08 de 00–06, 0,18 de 07–17, 0,40 de 18–21 y 0,18 de 22–23.
- SOC inicial 20 % u 80 %, equivalente a 1 o 4 kWh almacenados.

Todas las estrategias de un caso comparten la demanda realizada, solar,
tarifa, batería de 5 kWh, reserva de 20 %, potencias máximas 2,5 kW AC,
eficiencias 0,95, SOC inicial y restricciones. Cada episodio reinicia el SOC
según su escenario; no representa una operación mensual continua. La energía
inicial es un recurso dado y su adquisición anterior no se factura en el episodio.

Se preserva la grilla A* de **0,25 kWh**, heurística relajada, objetivo exclusivo
de costo de importación y meta terminal de reserva, sin valor monetario del
SOC final. Los hashes de `planner.py`, `simulator.py` y configuraciones originales
se verificaron antes y después. No se ajustó ningún parámetro tras ver costos.

## Las cuatro estrategias y su información

| Estrategia | Información de demanda | Uso |
|---|---|---|
| A. Reactivo | Desequilibrio actual, control ideal dentro de la hora | Referencia sin predicción |
| B. A* + MLP MSE | Predicciones causales congeladas de 24 h | Candidato operativo |
| C. A* + MLP ponderada | Predicciones causales congeladas de 24 h | Candidato experimental |
| D. A* oracle | Demanda futura realizada verdadera, 24 h en cada origen | **Referencia ideal, no producción** |

Se usó la pareja de modelos con **semilla 2026**, predefinida desde Fase 3.
No se eligió una semilla por sus costos. No hubo entrenamiento ni cambios de
pesos. Se reutilizaron sus matrices de pronósticos de validación, protegidas
por hashes y cotejadas con los orígenes/objetivos del pipeline original. Una
comprobación de inferencia CPU contra cada bundle confirmó concordancia
dentro de 2e−6 kWh. Las MLP no reciben el estrato de demanda ni los objetivos
realizados futuros; sólo se reproducen pronósticos emitidos causalmente en
las fases predictivas. El oracle es un proveedor separado y sus filas se
marcan `is_oracle=True`.

En cada origen se busca un plan de 24 h y se ejecuta únicamente su primera
orden. El simulador utiliza demanda realizada y SOC alcanzado, y vuelve a
planificar. El reactivo conserva el control instantáneo ideal representado
por promedios horarios; sus órdenes son continuas y las decisiones A* están
discretizadas, como en Fase 3. Las restricciones físicas son las mismas.

### Solar y consistencia temporal

El perfil base procede del PVGIS de Córdoba ya descargado en Fase 3: latitud
−31,4201, longitud −64,1888, 3 kWp, ERA5, inclinación 30°, orientación norte,
eficiencia del sistema 0,86. Sus 48 horas corresponden al **21 y 22/09/2023**
en `America/Argentina/Cordoba`, conservando también los timestamps UTC.

Se asigna el índice 0 a medianoche de cada episodio UCI y se mantiene la
secuencia horaria para las replanificaciones; la fuente UCI sigue con reloj
ingenuo. **No son mediciones sincronizadas ni de la misma vivienda o año.**
El mismo perfil de dos días se reutiliza en todos los episodios. La variante
25 % es una atenuación sintética uniforme, no otro día meteorológico observado
ni una simulación detallada de nubosidad. No se equiparan estaciones o días
de semana entre Córdoba y la vivienda francesa.

La trayectoria solar modelada y sus factores son conocidos durante cada
horizonte: **previsión solar idealizada**, compartida por las cuatro estrategias.
No se usa generación futura medida de una vivienda. La conversión de W a kWh
y las eficiencias son las de Fase 3; no se aplica dos veces el rendimiento PVGIS.
Los escenarios derivados y su procedencia están separados en `data/simulated/phase4_1/`.

## Resultados agregados

Medias por caso y desviación estándar muestral entre los 112 casos. La
dispersión mezcla días y escenarios; **no es incertidumbre entre semillas ni
intervalo de confianza**. Los escenarios repetidos no son 112 días independientes.

| Estrategia | Costo SIM, media ± DE | Red kWh, media ± DE | Captación solar media | SOC final medio |
|---|---:|---:|---:|---:|
| Reactivo | 3,874687 ± 2,024093 | 19,738347 ± 8,269809 | 90,15 % | 20,00 % |
| A* + MSE | 3,563623 ± 1,659440 | 20,478113 ± 7,958275 | 84,84 % | 24,37 % |
| A* + ponderada | 3,553411 ± 1,672114 | 20,414896 ± 7,973180 | 84,59 % | 21,98 % |
| Oracle ideal | 3,490182 ± 1,684077 | 19,992119 ± 8,166304 | 90,07 % | 22,32 % |

La captación solar es `(PV directa a demanda + PV cargada a batería) / PV
disponible`, como en Fase 3. El agregado mostrado es la media de porcentajes
por caso, no el cociente de todas las energías sumadas. Incluye pérdidas
posteriores y posible energía retenida: no equivale a autonomía útil.

El SOC mínimo observado en todas las estrategias fue **20 %**. Se registraron
**cero violaciones físicas y cero kWh de energía no suministrada**. No se
simularon cortes en este factorial; el planificador mantiene su hipótesis
original de red disponible. Las pruebas de cortes existentes siguen vigentes.

| Métrica por caso | Reactivo | A* MSE | A* ponderada | Oracle |
|---|---:|---:|---:|---:|
| Carga AC de batería, kWh, media ± DE | 1,9535 ± 1,8390 | 3,1665 ± 2,5248 | 3,1360 ± 2,4896 | 3,3435 ± 2,3910 |
| Descarga AC, kWh, media ± DE | 3,1880 ± 2,1917 | 4,0751 ± 2,2995 | 4,1613 ± 2,2298 | 4,3323 ± 2,1730 |
| Búsqueda por episodio, s, media ± DE | 0 | 1,4560 ± 0,7877 | 1,4439 ± 0,8294 | 1,4836 ± 0,7338 |
| Nodos expandidos totales | 0 | 504.717 | 494.049 | 515.180 |

La corrida tardó aproximadamente **259 s**, con dos procesos locales. La suma
de tiempos de búsqueda de llamadas concurrentes no equivale a ese tiempo
total. La llamada más lenta tardó **0,179 s**; el máximo de nodos expandidos
en una búsqueda fue 383, lejos de los límites de 10 s y 100.000 expansiones.
No se ocultaron casos lentos o fallidos: **0 de 448 ejecuciones fallaron**.

## Escenarios: no hay un ganador operativo universal

Costo medio realizado SIM por día en cada condición, sobre los mismos 14 días:

| Solar | Tarifa | SOC inicial | Reactivo | A* MSE | A* ponderada | Oracle |
|---|---|---:|---:|---:|---:|---:|
| Alta | Plana | 20 % | 2,4685 | 2,6391 | 2,5831 | 2,5294 |
| Alta | Plana | 80 % | 2,0410 | 2,2116 | 2,1632 | 2,1019 |
| Baja | Plana | 20 % | 3,8805 | 3,9219 | 3,9180 | 3,8918 |
| Baja | Plana | 80 % | 3,4530 | 3,4944 | 3,4905 | 3,4643 |
| Alta | Variable | 20 % | 3,7014 | 3,3223 | 3,3334 | 3,2234 |
| Alta | Variable | 80 % | 3,4728 | 3,0652 | 3,0840 | 2,9724 |
| Baja | Variable | 20 % | 6,1044 | 5,0535 | 5,0538 | 4,9954 |
| Baja | Variable | 80 % | 5,8758 | 4,8009 | 4,8012 | 4,7428 |

Agrupando por tarifa, con 56 casos por estrategia en cada grupo:

| Tarifa | Costo reactivo | Costo MSE | Costo ponderada | Costo oracle |
|---|---:|---:|---:|---:|
| Plana | 2,960752 | 3,066759 | 3,038715 | 2,996862 |
| Variable | 4,788622 | 4,060487 | 4,068106 | 3,983502 |

- **Plana:** MSE cuesta 3,58 % más que el reactivo; ponderada, 2,63 % más.
  El reactivo es el más favorable en costo diario medio en las cuatro
  combinaciones solares/SOC. No se deduce dominancia en operación continua.
- **Variable:** MSE cuesta 15,21 % menos; ponderada, 15,05 % menos. Ambos
  reducen costo en 52 de 56 casos y lo aumentan en cuatro. MSE es ligeramente
  más favorable que ponderada en las cuatro combinaciones solares/SOC.
- En ambos estratos de demanda, las medias agrupadas conservan el patrón de
  menor costo combinado con más importación de las MLP; el desglose completo
  está en `by_demand.csv`. La agregación por demanda mezcla los dos tipos de
  tarifas, por lo que no sustituye el análisis condicional anterior.

## Dispersión, mejoras y casos desfavorables

Las mejoras porcentuales son `100 × (costo_reactivo − costo_estrategia) /
costo_reactivo`. Si el denominador es cero, se conserva la diferencia absoluta
y el porcentaje queda indefinido, no se inventa un cero o infinito. En esta
corrida los 112 denominadores fueron positivos.

| Estrategia | Reducción del costo total factorial | Media de mejora por caso ± DE (pp) | Peor / mejor mejora por caso | Gana / pierde / empata vs reactivo |
|---|---:|---:|---:|---:|
| MSE | 8,03 % | 4,01 % ± 13,99 | −58,71 % / +28,40 % | 52 / 48 / 12 |
| Ponderada | 8,29 % | 4,60 % ± 13,38 | −64,25 % / +28,40 % | 52 / 46 / 14 |
| Oracle | 9,92 % | 7,16 % ± 11,90 | −41,07 % / +35,48 % | 56 / 38 / 18 |

El cociente de sumas no es la media de porcentajes; ambos se informan para
evitar confundirlos. El total factorial pondera por igual condiciones
artificiales, no probabilidades reales de tarifas, nubosidad o estados de batería.

Ejemplos desfavorables conservados:

- **22/04/2010, solar alta, plana, SOC 80 %:** reactivo 0,642491 SIM;
  MSE 1,019685 y ponderada 1,055310. La ponderada cuesta 0,412819 SIM más
  (+64,25 %); termina con SOC 60 %, frente al 20 % del reactivo.
- **15/02/2010, solar alta, plana, SOC 20 %:** MSE cuesta 0,579823 SIM más
  (+18,33 %) que el reactivo y termina con SOC 100 %. El recurso retenido
  al final explica por qué no deben compararse sólo costos truncados.
- **14/03/2010, solar alta, variable, SOC 20 %:** MSE cuesta 2,263494 SIM
  frente a 2,120322 (+6,75 %). Aquí ambos terminan al 20 %: el resultado
  desfavorable tampoco se explica siempre por distinto SOC final.

Los cuatro casos de deterioro MSE con tarifa variable corresponden a los días
14/03 y 22/04 con solar alta, para ambos SOC iniciales. La lista completa,
con diferencias absolutas y porcentuales, está en `unfavorable_cases.csv`.

## Costo frente a dependencia de red

Los **52 casos** en los que MSE reduce costo también aumentan importación;
todos son de tarifa variable. Ocurre lo mismo con la ponderada en sus 52
victorias contra el reactivo. La oportunidad de trasladar energía desde
horas baratas hace que el objetivo económico acepte pérdidas de conversión,
mayor consumo de red y, en ocasiones, menor captación solar.

Con tarifa variable, las importaciones agregadas aumentan **3,92 %** para MSE
y **4,22 %** para ponderada frente al reactivo. La solar media capturada cae
de 90,15 % del reactivo a 83,68 % y 83,28 %, respectivamente. Es un compromiso
esperable de la función de costos existente, que no penaliza explícitamente
dependencia de red. No se ha cambiado esa función ni impuesto un nuevo objetivo.

![Compromiso entre costo e importación](../experiments/phase4_1/paired-20261022/plots/cost_grid_tradeoff.png)

## ¿La MLP ponderada gestiona mejor pese a peor RMSE?

En los 14 episodios, usando igual número de pares origen/horizonte por día,
el RMSE predictivo es **0,661233 kWh para MSE** y **0,684842 para ponderada**.
La segunda empeora el RMSE en 11 de 14 días. Estos valores describen la muestra
seleccionada y no sustituyen el RMSE de toda validación de fases anteriores.

La ponderada reduce el costo factorial total frente a MSE **0,2866 %**:
diferencia media **−0,010213 ± 0,090624 SIM por caso**. Gana en 24 casos,
pierde en 26 y empata en 62. En nueve días mejora la media de las ocho
condiciones, pero esto no implica mejora uniforme por condición o magnitud.

| Comparación ponderada frente a MSE | Plana, 56 casos | Variable, 56 casos |
|---|---:|---:|
| Cambio en costo agregado | −0,9144 % | +0,1876 % |
| Gana / pierde / empata | 18 / 12 / 26 | 6 / 14 / 36 |
| Diferencia media de costo (SIM) | −0,028044 | +0,007618 |
| Diferencia media de red (kWh) | −0,186957 | +0,060523 |

Además, **14 de las 24 victorias** de ponderada terminan con menos energía
almacenada que MSE. Sólo 10 victorias tienen el mismo SOC terminal. En el
subconjunto descriptivo de 90 casos con SOC final igual, ponderada gana en 10,
pierde en 18 y empata en 62, con costo medio **0,007830 SIM mayor**. Es una
condición posterior a las decisiones, no un experimento nuevo con restricción
terminal controlada ni una prueba causal de superioridad.

Por lo tanto, **sí existen casos de mejor costo pese a peor RMSE**, pero la
evidencia no respalda reemplazar la MLP MSE por ponderada. El efecto económico
es pequeño, dependiente del escenario y sensible a la energía que queda al
cortar la evaluación. No se añadió una valoración terminal ni se alteró el
objetivo para corregir el resultado después de observarlo.

## Interpretación correcta del oracle

Oracle conoce el consumo futuro de validación de cada horizonte y sólo sirve
como referencia ideal. Tiene el menor costo medio de las cuatro estrategias,
pero no es una política disponible en producción.

**No es una cota inferior garantizada del costo realizado del día evaluado.**
Optimiza horizontes móviles de 24 h, parte de los cuales se extiende al día
siguiente; se reportan sólo las primeras 24 h de ejecución. Puede conservar
energía al final, resuelve un grafo discreto y enfrenta la misma comparación
con un reactivo continuo. En 38 casos el reactivo tiene menor costo diario
que este oracle rodante; MSE lo supera en ocho y ponderada en seis. Eso no
contradice la optimalidad de cada búsqueda respecto de su propio horizonte
y estado inicial. No se presenta como un oracle global continuo con igual SOC
terminal ni como una cota matemática para todas las políticas realizadas.

## Implementación, verificación y límites

Se agregaron `energy/comparison.py`, `comparison_plots.py`, la configuración
`configs/energy.comparison.json`, la CLI `compare-energy` y scripts de análisis
y verificación. Se reutilizan el motor físico y A* sin cambios de lógica.
La muestra y el factorial se guardan en `episodes.csv`, `cases.csv` y
`design.json` antes de lanzar las simulaciones.

Cada carpeta `cases/<id>/` conserva condiciones, métricas, ledger realizado,
pronósticos con oracle marcado, búsquedas, auditoría física y `failures.json`.
Se comprueban físicamente **los 24 pasos de cada plan antes de ejecutar la
primera orden**. No se usa una estrategia de reemplazo cuando una búsqueda
falla. Las ejecuciones incompletas quedan registradas con tipo de excepción,
hora y trazas parciales; los agregados principales requieren cuatro ejecuciones
completas por caso, con cobertura explícita. No hubo fallos en esta corrida.

**94 pruebas aprobadas**, incluyendo las 87 anteriores sin entrenamiento y
siete nuevas: diseño no solapado, mismas condiciones, métricas energéticas,
factibilidad de todos los planes, causalidad y separación oracle, fallos
sin fallback, dispersión pareada y denominadores cero. En este entorno
Windows es necesario dirigir `TEMP` y `TMP` a la carpeta del proyecto al
ejecutar pruebas con archivos temporales.

La verificación independiente comprobó:

- **10.752 transiciones realizadas** mediante replay físico.
- Auditoría de **8.064 planes completos y 193.536 transiciones previstas**.
- **8.064 emisiones** contra las predicciones congeladas o, sólo para oracle,
  contra las mediciones futuras de validación.
- Igualdad de condiciones, métricas y agregados; ausencia de fugas oracle
  en las matrices neurales; **143 archivos protegidos intactos**.
- Conservación de pesos, escenarios anteriores, costos y restricciones A*.

La dispersión es descriptiva: 14 días reutilizados no proporcionan 112
réplicas independientes. Se evaluó sólo la semilla 2026, sin estudiar variación
entre semillas. Se usó un perfil solar de dos días con atenuación artificial,
no una distribución meteorológica anual. No hay cortes, degradación,
autodescarga, errores solares, capacidad limitada de conexión ni hardware.
Los datos UCI pertenecen a otra ubicación y ya fueron usados durante desarrollo.
No se extrapolan resultados a costos comerciales, autonomía anual o nuevas viviendas.

## Artefactos, gráficos y reproducción

- [Resultados de las 448 ejecuciones](../experiments/phase4_1/paired-20261022/results_by_case.csv).
- [Diferencias y mejoras por caso](../experiments/phase4_1/paired-20261022/paired_results.csv).
- [Agregados con media, DE, mediana, mínimo y máximo](../experiments/phase4_1/paired-20261022/aggregate.csv).
- [Agregados por solar, tarifa y SOC](../experiments/phase4_1/paired-20261022/by_scenario.csv).
- [Agregados por estrato de demanda](../experiments/phase4_1/paired-20261022/by_demand.csv).
- [Ponderada frente a MSE y SOC final](../experiments/phase4_1/paired-20261022/model_paired_diagnostics.csv).
- [Relación entre error predictivo y costo por día](../experiments/phase4_1/paired-20261022/prediction_vs_energy_by_day.csv).
- [Casos desfavorables](../experiments/phase4_1/paired-20261022/unfavorable_cases.csv).
- [Verificación y hashes](../experiments/phase4_1/paired-20261022/verification.json).

Se inspeccionaron cinco gráficos reproducibles:
[distribuciones](../experiments/phase4_1/paired-20261022/plots/strategy_distributions.png),
[costo por escenario](../experiments/phase4_1/paired-20261022/plots/cost_by_scenario.png),
[costo frente a red](../experiments/phase4_1/paired-20261022/plots/cost_grid_tradeoff.png),
[ponderada frente a MSE](../experiments/phase4_1/paired-20261022/plots/weighted_vs_mse.png) y
[resultados por día](../experiments/phase4_1/paired-20261022/plots/cost_by_day.png).
Las barras con DE describen variación entre casos; no son intervalos de confianza.

Desde la raíz, con los artefactos congelados anteriores y el entorno instalado:

```powershell
.\.venv\Scripts\python.exe -u -m bekantor_energy compare-energy --config configs/energy.comparison.json --run-dir experiments/phase4_1/repeat-01 --scenario-dir data/simulated/phase4_1/repeat-01
.\.venv\Scripts\python.exe scripts/analyze_energy_comparison.py --run-dir experiments/phase4_1/repeat-01
.\.venv\Scripts\python.exe scripts/verify_energy_comparison.py --run-dir experiments/phase4_1/repeat-01
```

Las rutas de salida deben ser nuevas. El comando comprueba las particiones,
los pronósticos guardados y sus modelos antes de ejecutar. TensorFlow sólo se
usa para la comprobación de inferencia; las simulaciones reproducen los
pronósticos congelados y usan dos procesos locales. Para ejecutar secuencialmente
puede fijarse `workers: 1` en una copia de la configuración; no cambia el objetivo.
Los tiempos de planificación dependen de carga y paralelismo, no son deterministas.
Los perfiles grandes y resultados permanecen ignorados por Git.

Pruebas sin entrenar redes ni abrir test:

```powershell
New-Item -ItemType Directory -Force -Path experiments/tmp | Out-Null
$env:TEMP = Join-Path (Get-Location) 'experiments/tmp'
$env:TMP = $env:TEMP
.\.venv\Scripts\python.exe -c "import unittest; loader=unittest.TestLoader(); names=('test_acquisition.py','test_core.py','test_data.py','test_pipeline.py','test_forecasting.py','test_calendar.py','test_weighted.py','test_energy*.py'); suite=unittest.TestSuite(loader.discover('tests',pattern=n) for n in names); result=unittest.TextTestRunner(verbosity=2).run(suite); raise SystemExit(not result.wasSuccessful())"
```

## Recomendación para revisión humana

Conservar el reactivo como referencia fuerte, especialmente con tarifa plana,
y la MLP MSE como referencia predictiva. A* + MSE es prometedor para tarifas
variables en estos escenarios, aceptando explícitamente el aumento de red.
No sustituir MSE por ponderada basándose en una ventaja global de 0,29 % que
no es consistente y coincide con diferencias terminales.

Antes de la siguiente decisión de producto conviene, **sin implementarlo aquí**:

1. Definir cómo comparar energía terminal y evaluar operación continua, para
   distinguir ahorro de corto plazo de energía retenida para el futuro.
2. Repetir con las semillas congeladas 2027/2028 y más perfiles PVGIS
   independientes, sin elegir escenarios por sus resultados favorables.
3. Acordar el compromiso aceptable entre costo e importación/autonomía antes
   de autorizar cualquier cambio de objetivo o regla de selección de estrategia.

No se implementó un selector automático, ni se modificó la función de costos.
No se abrió test, no hubo entrenamiento, dashboard, Colab, commits o push.
**No se avanzó a la Fase 4.2.**

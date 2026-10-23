# Fase 4.4 — Diagnóstico del modo Autonomía

## Pregunta y alcance

¿Puede la planificación predictiva reducir materialmente la importación frente
al autoconsumo reactivo bajo el modelo físico actual? Se conservan los ocho
escenarios de Fase 4.3: dos bloques consecutivos independientes de siete días
de validación, dos perfiles solares cronológicos de Córdoba y dos tarifas.
Cada caso empieza con 2,5 kWh almacenados, batería de 5 kWh y reserva de 1 kWh.
La demanda UCI francesa y los perfiles PVGIS de Córdoba son independientes;
la combinación es simulada y la previsión solar continúa siendo idealizada.

No se abre el test real, no se ajustan ni reentrenan redes y no se modifica el
objetivo lexicográfico de Autonomía `(importación kWh, costo SIM)`. Los ocho
casos no representan ocho semanas independientes: hay dos semanas UCI y cuatro
combinaciones físicas demanda/solar, repetidas bajo dos tarifas.

El [diseño previo](phase4_4-design.md) y la
[configuración](../configs/energy.autonomy-diagnostic.json) fijan antes de las
nuevas corridas las grillas, casos, presupuestos y umbral de mejora material.
La derivación completa figura en [análisis físico](autonomy-physical-bound.md).

## Hallazgo físico independiente

El reactivo alcanza el mínimo físico de importación en **los ocho escenarios**.
Se compara su trayectoria factible con una cota independiente de toda la semana,
obtenida mediante flujo máximo temporal y comprobación flujo/corte mínimo.
La diferencia máxima absoluta es 2,85e-14 kWh, dentro de tolerancia 1e-8 kWh.
La referencia física utiliza demanda perfecta de toda la semana, más información
que el oracle MPC de 24 horas; se utiliza sólo como certificado diagnóstico.

| Bloque UCI | Solar independiente | Importación mínima / reactivo (kWh/semana) |
|---|---|---:|
| 21–27/09/2009 | Enero Córdoba | 75,144107 |
| 21–27/09/2009 | Julio Córdoba | 86,641040 |
| 24–30/01/2010 | Enero Córdoba | 166,894675 |
| 24–30/01/2010 | Julio Córdoba | 178,176701 |

Cada fila se verifica para ambas tarifas. Los precios no cambian la cota de
energía; sí pueden cambiar el desempate económico del planificador.

La media histórica de importación es 126,714131 kWh para el reactivo y
130,498526 kWh para Autonomía MLP: exceso de 3,784396 kWh/semana, un 2,9866 %
calculado como razón de sumas. No existe margen de reducción >1 % y >0,5 kWh
contra ese reactivo en estos escenarios y con estas restricciones, porque ya
coincide con la cota inferior.

Esta conclusión depende de red ilimitada y disponible, eficiencias constantes,
ausencia de exportación, batería sin autodescarga/degradación y demanda fija.
No es una afirmación universal sobre hardware ni otras funciones objetivo.

## Qué implementaciones se comparan

El reactivo usa el desequilibrio realizado de la hora y solicita cargar todo el
excedente o descargar todo el déficit. El mismo simulador limita potencia,
capacidad y reserva. Se interpreta como realimentación instantánea idealizada
de valores medios horarios.

Autonomía usa A* con h=0, aritmética lexicográfica racional, grilla de energía,
horizonte de 24 horas y replanificación horaria. Ejecuta sólo la primera acción
con demanda realizada. La reparación recorta acciones inviables, pero no crea
descargas/cargas que el pronóstico no solicitó. Una subestimación puede dejar
demanda a red o convertir una carga prevista como solar en carga desde red.

El oracle diagnóstico reemplaza sólo la matriz de demanda por las próximas
24 observaciones de validación. No es un controlador aplicable en producción.
La sensibilidad cambia sólo el paso de grilla a 0,5 y 0,125 kWh, manteniendo
0,25 kWh como referencia; se ejecuta en las cuatro condiciones con tarifa
variable, con ambas fuentes predictivas. No se selecciona un nuevo paso.

## Experimentos controlados: resultados

Se completaron **24 corridas nuevas sin fallos**, además de conservar como
referencia las trayectorias de Fase 4.3. Son ocho oracle a 0,25 kWh y dieciséis
sensibilidades (cuatro casos × dos resoluciones × MLP/oracle). Cada corrida
reproduce también su reactivo con las mismas condiciones. Ningún caso mejora
materialmente ni supera en importación al reactivo.

### Oracle a resolución original: ocho escenarios

Medias por semana ± desviación estándar muestral entre escenarios; no son
intervalos de confianza ni ocho réplicas independientes.

| Política | Red kWh, media ± DE | Exceso sobre reactivo kWh, media ± DE | Cambio de red | Costo bruto SIM | Costo ajustado SIM |
|---|---:|---:|---:|---:|---:|
| Reactivo | 126,714131 ± 49,362195 | 0 | 0 % | 22,979533 | 23,193283 |
| Autonomía MLP, 0,25 | 130,498526 ± 49,769501 | 3,784396 ± 0,649460 | +2,9866 % | 23,180698 | 23,394448 |
| Autonomía oracle, 0,25 | 126,874461 ± 49,375078 | 0,160330 ± 0,019235 | +0,1265 % | 22,533491 | 22,747241 |

Reemplazar MLP por oracle reduce la importación 3,624066 ± 0,651055 kWh/semana,
con reducción individual entre 2,696724 y 4,395531 kWh. Elimina **95,7634 % del
exceso** (razón de sumas), sin alcanzar el reactivo. Es evidencia controlada de
que los errores predictivos y su interacción con el control explican la mayor
parte de la desventaja a esa resolución; no demuestra que otro modelo entrenado
vaya a conseguir el resultado ideal del oracle.

El oracle obtiene menor costo agregado con una importación ligeramente mayor;
no domina al reactivo en el objetivo primario. Tampoco es una estrategia
desplegable. Las tablas por tarifa y caso permanecen en los CSV.

### Sensibilidad de grilla: mismos cuatro casos con tarifa variable

Todos los cambios porcentuales son razones de sumas frente a su reactivo pareado.

| Predicción | Paso kWh | Exceso de red kWh/semana, media ± DE | Cambio de red | Tiempo de planificación medio s/semana |
|---|---:|---:|---:|---:|
| MLP | 0,5 | 3,919470 ± 0,708269 | +3,0932 % | 14,34 |
| MLP, referencia histórica | 0,25 | 3,784396 ± 0,701497 | +2,9866 % | 41,48 |
| MLP | 0,125 | 3,846781 ± 0,640427 | +3,0358 % | 184,58 |
| Oracle | 0,5 | 0,579148 ± 0,170361 | +0,4571 % | 13,45 |
| Oracle | 0,25 | 0,160330 ± 0,020776 | +0,1265 % | 49,11 |
| Oracle | 0,125 | 0,080666 ± 0,023923 | +0,0637 % | 177,24 |

Con oracle, afinar de 0,25 a 0,125 recupera 0,079664 ± 0,008555 kWh/semana:
el **49,6873 %** de su brecha residual. Mejora los cuatro casos. Con MLP,
la importación media aumenta 0,062385 kWh/semana; mejora dos casos y empeora dos.
No se elige una resolución nueva a partir de estos resultados. Los tiempos de
MLP a 0,25 son históricos: no constituyen una comparación aislada de rendimiento
con las corridas nuevas. Los oracle nuevos sí muestran el costo computacional
creciente de refinar, con la misma implementación y dos procesos locales.

Exceso de importación por caso, en kWh/semana:

| Bloque / solar | MLP 0,5 | MLP 0,25 | MLP 0,125 | Oracle 0,5 | Oracle 0,25 | Oracle 0,125 |
|---|---:|---:|---:|---:|---:|---:|
| Septiembre / enero | 4,0562 | 3,9215 | 3,4354 | 0,6325 | 0,1293 | 0,0461 |
| Septiembre / julio | 3,1334 | 2,8666 | 3,1795 | 0,7903 | 0,1699 | 0,0947 |
| Enero / enero | 3,6708 | 3,7807 | 4,2553 | 0,3985 | 0,1689 | 0,0984 |
| Enero / julio | 4,8175 | 4,5687 | 4,5169 | 0,4953 | 0,1732 | 0,0834 |

La primera fecha identifica el bloque UCI y la segunda el perfil independiente
de Córdoba; no indican simultaneidad real. A* sigue siendo óptimo dentro del
grafo y para el pronóstico en cada origen. Esa garantía no implica que el replay
con demanda desconocida sea óptimo en el espacio físico continuo.

![Oracle y MLP a grilla original](../experiments/phase4_4/autonomy-diagnostic-20261022/plots/oracle_vs_mlp.png)

![Sensibilidad de grilla](../experiments/phase4_4/autonomy-diagnostic-20261022/plots/grid_sensitivity.png)

## Balance, cargas desde red y SOC terminal

Con igual energía inicial, demanda y solar, sin exportación ni demanda no servida:

```
deltaImportacion = deltaRecorteSolar + deltaPerdidas + deltaEnergiaFinal
```

En Fase 4.3, reactivo y Autonomía terminaron los ocho casos con **1 kWh**.
El SOC terminal no explica la desventaja histórica. El exceso medio se concilia
con 3,730089 kWh de recorte solar adicional y 0,054306 kWh de pérdidas adicionales.
La carga desde red fue 4,287078 kWh/semana en promedio, frente a cero del reactivo.
No debe sumarse esa carga otra vez a la identidad: ya está contenida en sus flujos.

Se conserva la valoración monetaria `C + 0,1425*(E0-ET)` en SIM, separada del
costo bruto. La sensibilidad de energía `G + lambda*(E0-ET)` utiliza lambda
0, eta_d=0,95 y 1/eta_c. Es equivalencia contable, no energía importada ni factura.
Con igual ET no cambia ninguna diferencia entre estrategias.

En **todas las corridas nuevas** ET vuelve a ser 1 kWh (máxima diferencia
2,23e-16 kWh). La sensibilidad terminal no cambia los rankings ni las brechas.
El ajuste monetario añade 0,21375 SIM a cada costo bruto semanal.

Para las medias de los ocho escenarios a 0,25 kWh:

| Predicción | Carga de red prevista kWh | Carga de red realizada kWh | Exceso imprevisto positivo kWh | Menor carga que la prevista kWh |
|---|---:|---:|---:|---:|
| MLP | 1,503251 | 4,287078 | 3,577430 | 0,793603 |
| Oracle | 1,197387 | 1,197387 | 0 | 0 |

El exceso imprevisto se calcula hora a hora con la misma acción: carga de red
real menos la que el plan esperaba necesitar, reteniendo sólo diferencias
positivas. Su conciliación es `real = prevista + exceso - reducción`. No se
puede sumar directamente al exceso neto de importación contra otro controlador.

El oracle todavía carga desde red porque el grafo discreto no siempre permite
captar exactamente un excedente pequeño. Puede elegir un escalón mayor que
también compra red y aprovechar mejor ese solar que esperando. La regresión
de dos horas demuestra este mecanismo con predicción perfecta. En el espacio
continuo esa carga de red no es necesaria para minimizar importación.

Con oracle a 0,25, la brecha energética de 0,160330 kWh se descompone en
0,048293 kWh de recorte adicional y 0,112037 kWh de pérdidas adicionales. En
los cuatro casos a 0,125 baja a 0,016738 + 0,063928 = 0,080666 kWh. Las pérdidas
residuales cobran más peso cuando se elimina la mayor parte del error predictivo.
La MLP fina conserva casi toda la brecha como recorte (3,846997 kWh), con una
pequeña reducción de pérdidas de 0,000217 kWh frente al reactivo.

En la primera hora de cada origen, la MLP tiene MAE 0,329253 kWh en el bloque
temprano y 0,531406 kWh en el tardío. El sesgo medio es +0,057048 y -0,001290
kWh respectivamente: no todo el error es subestimación global. Hay 6–9 horas
por semana con signo de desequilibrio previsto distinto del realizado, según
perfil solar; se solicita carga durante déficit real en 1–5 horas. La secuencia
de errores, el estado acumulado y la reparación importan más que un único RMSE.

![Descomposición física de las brechas](../experiments/phase4_4/autonomy-diagnostic-20261022/plots/physical_gap_decomposition.png)

## Autonomía y resiliencia

Minimizar `sum G` no maximiza SOC. Usar batería para atender demanda presente
puede ser óptimo en importación y dejar menos energía disponible para un corte.
La reserva es una restricción física; no constituye una evaluación de respaldo.

Un futuro modo Resiliencia necesitaría escenarios explícitos de disponibilidad
de red, cargas críticas, energía no suministrada y una política de uso de reserva.
No se implementa ni se prueba ese modo en esta fase. No se pueden deducir horas
de respaldo de resultados sin interrupciones.

## Recomendación y límites de interpretación

Conservar el A* Autonomía actual como referencia experimental congelada y
**modificar posteriormente su papel operativo**, bajo revisión humana. Para
minimizar exclusivamente importación bajo estas restricciones, el autoconsumo
reactivo ya consigue la referencia física; no se justifica promover el A* actual
como una mejora de autonomía ni afinar redes sólo para intentar superar esa cota.
No se sustituye ningún modo ni se incorpora un supervisor nuevo en esta fase.

Una futura revisión podría utilizar realimentación del desequilibrio medido para
la operación de autoconsumo, conservando planificación para objetivos donde sí
exista margen. Estados adicionales alineados con carga solar/déficit podrían
reducir la desventaja de discretización, pero son otro cambio y no se implementan.
La planificación también puede buscar menor costo entre despachos de igual
importación: es una mejora secundaria distinta de reducir energía de red. No
se ha resuelto aquí ese óptimo continuo de costo sujeto a importación mínima.

La predicción puede ser útil si cambian las restricciones: cargas desplazables,
límites de importación, eficiencias variables o cortes y cargas críticas. Estas
son oportunidades de investigación, no prestaciones demostradas ni cambios
realizados. Un modo Resiliencia requiere su propia evaluación de interrupciones.

Los efectos de oracle y grilla no son causas sumables. En MPC, cada intervención
modifica el SOC de horas posteriores. El oracle usa sólo 24 horas y desempate
económico; la cota física conoce toda la semana. Una brecha residual del oracle
no identifica por sí sola toda la contribución de horizonte, empates y grilla.
Las oportunidades instantáneas no utilizadas suman decisiones bajo SOC distintos;
no representan energía adicional recuperable al sumarlas ni otra cota óptima.

## Pruebas y reproducción

Se aprobaron **129 pruebas sin entrenamiento** en 13,872 s: las 118 anteriores y
11 nuevas. Entre ellas hay 120 escenarios sintéticos aleatorios que contrastan
el reactivo con la cota independiente, casos de cronología, potencia, eficiencia,
capacidad, SOC inicial, contabilidad terminal, cola desconocida del oracle y una
regresión de carga desde red causada por la grilla aun con demanda perfecta.
No se ejecutó el bloque de ocho pruebas TensorFlow, que incluye entrenamiento
de fixtures temporales.

La auditoría independiente aprobó 8.064 pasos realizados, 96.768 pasos de planes
previstos completos, 96.768 valores de pronóstico y 288 cruces de medianoche.
Los 1.347 archivos protegidos (resultados, datos/modelos y árbol previamente
versionado) conservaron sus hashes. Hubo cero violaciones físicas, cero energía
no suministrada y cero búsquedas fallidas, sin fallback. La búsqueda individual
más lenta tomó 2,275 s y el máximo fue 774 nodos expandidos, por debajo de los
límites originales de 10 s / 100.000 nodos. La corrida completa tardó 990,50 s
con dos procesos; no implica una latencia de operación real ni incluye una nueva
inferencia neuronal, porque se utilizaron predicciones congeladas.

Se inspeccionaron visualmente los tres gráficos. `verification.json` conserva
210 hashes de artefactos nuevos y hashes del runner, módulo diagnóstico,
análisis, renderizador y pruebas. El verificador contrasta los planes con el
simulador sin reparación, el replay con demanda realizada, las fuentes del
oracle/MLP, las métricas, diferencias controladas y sensibilidad terminal.

Artefactos principales dentro de la corrida:

- `metrics.csv`: valores por caso, fuente de predicción y resolución.
- `summary.csv`: medias, desviaciones, cobertura y casos desfavorables por alcance.
- `physical_bound.csv`: certificados de importación mínima por escenario.
- `controlled_differences.csv` / `controlled_summary.csv`: intervenciones pareadas.
- `grid_charge_sources.csv` / `forecast_dispatch_diagnostics.csv`: mecanismo de carga y errores actuales.
- `terminal_sensitivity.csv`: valoración energética sin cambiar decisiones.
- `cases/`: balances, planes completos, búsquedas, pronósticos y fallos explícitos.
- `plots/`: comparación oracle/MLP, sensibilidad y balance de brechas.
- `design.json`, `preservation.json`, `coverage.json`, `verification.json`: reproducibilidad y auditoría.

Los registros de pruebas, ejecución y auditoría están en
`experiments/verification/phase4_4-{new-tests,all-tests,run,verification}.txt`.

Desde la raíz, en PowerShell:

```powershell
New-Item -ItemType Directory -Force experiments/tmp | Out-Null
$env:TEMP = (Resolve-Path experiments/tmp).Path
$env:TMP = $env:TEMP
.\.venv\Scripts\python.exe -c "import unittest,pathlib; suite=unittest.TestSuite(unittest.defaultTestLoader.discover('tests',pattern=p.name) for p in sorted(pathlib.Path('tests').glob('test_*.py')) if 'tensorflow' not in p.name); r=unittest.TextTestRunner(verbosity=2).run(suite); raise SystemExit(not r.wasSuccessful())"
```

Para repetir los experimentos, usar una carpeta nueva; requiere los artefactos
locales congelados de Fase 4.3 y validation.csv con su hash original. No requiere
SciPy, nuevas descargas ni invocar TensorFlow. El runner conserva los parámetros
originales y registra sus entradas, versiones, tareas, fallos y hashes antes
de observar los nuevos resultados.

```powershell
.\.venv\Scripts\python.exe -u scripts/diagnose_autonomy.py --config configs/energy.autonomy-diagnostic.json --run-dir experiments/phase4_4/repeat-01
.\.venv\Scripts\python.exe scripts/analyze_autonomy_diagnostic.py --run-dir experiments/phase4_4/repeat-01
.\.venv\Scripts\python.exe scripts/plot_autonomy_diagnostic.py --run-dir experiments/phase4_4/repeat-01
.\.venv\Scripts\python.exe scripts/verify_autonomy_diagnostic.py --run-dir experiments/phase4_4/repeat-01
```

Corrida de esta fase: `experiments/phase4_4/autonomy-diagnostic-20261022`.
Las tablas y gráficos son locales e ignorados por Git; código, configuración y
documentación quedan preparados para revisión. No se hacen commits ni push.

# BEKANTOR Energy

Sistema local y modular para investigar predicción de demanda eléctrica y
planificación de energía solar, batería y red para viviendas y pequeños negocios.
**Estado: implementado hasta Fase 4.5.** Incluye adquisición y EDA, predicción
MLP TensorFlow/Keras de 168 a 24 horas en CPU, simulador físico, A* con
replanificación horaria y evaluación continua con SOC persistente. Los modos
Económico, Autonomía y Equilibrado comparten el núcleo de búsqueda y se comparan
con autoconsumo reactivo sobre validación. Incluye diagnóstico físico del modo
Autonomía y un componente experimental de resiliencia ante cortes imprevistos,
con cargas críticas sintéticas, reservas operativas y emergencia común.

Los informes conservan el contexto y los resultados de cada fase. Para la
predicción, ver [protocolo y comandos](docs/phase2-methodology.md) y
[resultados de referencia](docs/phase2-results.md); para el estado energético
actual, ver [comparación de modos](docs/phase4_3-energy-modes.md),
[diagnóstico de Autonomía](docs/phase4_4-autonomy-diagnostic.md) y
[resiliencia](docs/phase4_5-resilience.md). Son resultados de escenarios
simulados UCI + PVGIS independientes, con previsión solar idealizada; no prueban
ahorros reales ni garantizan capacidad de respaldo de una instalación real.

## Instalación local en Windows (PowerShell, CPU)

Usar Python de 64 bits, versión 3.11 o posterior. El entorno completo comprobado
usa Windows, Python 3.13.15 y TensorFlow 2.21.0 en CPU. No hace falta GPU ni CUDA.
Los comandos de datos y las pruebas sin entrenamiento no requieren TensorFlow.
Ejecutar los comandos desde la raíz del proyecto:

```powershell
python --version
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m bekantor_energy --help
.\.venv\Scripts\python.exe -m bekantor_energy check-config
```

No es necesario activar el entorno ni cambiar la política de ejecución de
PowerShell. `requirements.txt` expresa rangos compatibles; `requirements-lock.txt`
congela el entorno probado, incluidas dependencias transitivas. Para otra versión
de Python/plataforma puede ser necesario resolver nuevamente los rangos y validar.
Internet sólo se requiere para instalación y adquisición; los datos cacheados y
las pruebas se usan offline. El comando `bekantor` también queda instalado.
Para predicción y para ejecutar también las pruebas de integración TensorFlow:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-ml-lock.txt
```

## Prueba rápida sin red (datos sintéticos)

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m bekantor_energy process --input tests/fixtures/uci_synthetic.txt --source-kind synthetic_fixture --output data/processed/demo --reports experiments/demo
```

El fixture reproduce columnas UCI, pero **no contiene mediciones reales**. Para
entornos Codex cuyo directorio temporal esté restringido:

```powershell
New-Item -ItemType Directory -Force experiments/tmp | Out-Null
$env:TEMP = (Resolve-Path experiments/tmp).Path
$env:TMP = $env:TEMP
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## Descarga, procesamiento y EDA de UCI

```powershell
.\.venv\Scripts\python.exe -m bekantor_energy download-uci
.\.venv\Scripts\python.exe -m bekantor_energy process --input data/raw/uci/household_power_consumption.txt --source-kind uci_measured
```

El procesamiento valida columnas, fechas, duplicados y dominio numérico;
completa la grilla de minutos; rellena causalmente hasta 5 minutos consecutivos;
integra a horas; genera indicadores de calidad y particiones cronológicas
70/15/15. No inventa energía para horas que siguen incompletas ni extrapola las
horas parciales de inicio/fin. `--max-ffill-minutes 0` desactiva el relleno.

Salidas: `data/processed/uci/{hourly,train,validation,test}.csv`,
`manifest.json` con SHA-256, parámetros, versiones y límites temporales;
`experiments/uci/quality.json`; y `experiments/uci/eda/` con resúmenes CSV/JSON
y cuatro gráficos PNG (serie horaria, distribución, perfil horario y cobertura).
Los datos brutos y procesados y los gráficos quedan fuera de Git.

El EDA usa **sólo train por defecto**. Para repetirlo:

```powershell
.\.venv\Scripts\python.exe -m bekantor_energy eda --input data/processed/uci/train.csv --source-kind uci_measured --scope train --output experiments/uci/eda
```

`process --eda-scope all` habilita explícitamente una descripción global. Si se
usa, ya no se debe considerar el período de prueba completamente ciego para
decisiones de modelado. No se calculan escaladores ni ventanas de entrenamiento
en Fase 1. Las futuras etiquetas de evaluación deberán ser observadas, usando
`is_observed_complete`, para no evaluar contra demanda imputada.

## Generación fotovoltaica modelada con PVGIS vigente

Se consultaron la documentación oficial, OpenAPI v6 y una respuesta real el
22/10/2026. Se usa **GET `/api/v6/power/broadband`** con timestamps UTC,
potencia pico en kWp y ERA5 para el ejemplo Buenos Aires. La API 5.3 tiene otro
contrato y otras convenciones de orientación; no se mezclan parámetros.

```powershell
.\.venv\Scripts\python.exe -m bekantor_energy download-pvgis
.\.venv\Scripts\python.exe -m bekantor_energy download-pvgis --latitude -34.6037 --longitude -58.3816 --start 2023-01-01T00:00:00 --end 2023-01-07T23:00:00 --output data/raw/pvgis/week.json --processed-output data/processed/pvgis/week.csv
```

El comando lee potencia pico/eficiencia de `configs/home.example.json`. Admite
`--home-config`, `--tilt`, `--orientation`, `--irradiance-source`; consulta su
ayuda para más opciones. Coordenadas, tarifas y sistema son **supuestos de
ejemplo**, no una vivienda del usuario. v6 usa norte=0°, este=90°, sur=180°.
Se preservan JSON bruto y recibo con consulta, fecha de adquisición y SHA-256.
Una caché con otra consulta o hash incorrecto se rechaza; usar una ruta nueva o
`--force` para reemplazarla conscientemente. Ambas adquisiciones tienen timeout,
reintentos acotados y límites de tamaño; no se descargan en las pruebas.

PVGIS modela potencia histórica a partir de datos meteorológicos; no es una
medición de paneles ni un pronóstico operativo del día siguiente.
`pv_kwh_estimate` aproxima energía de cada hora manteniendo constante esa
potencia. **No se unen automáticamente UCI y PVGIS:** difieren en vivienda,
ubicación, calendario y tratamiento de zona horaria. Cualquier combinación
en las fases energéticas es un escenario simulado explícito, no evidencia de
una misma vivienda.

## Estructura y responsabilidades

```text
bekantor-energy/
├── src/bekantor_energy/
│   ├── data/                 # UCI, PVGIS, HTTP y particiones cronológicas
│   ├── config.py             # Vivienda, restricciones y tarifas simuladas
│   ├── units.py              # Conversiones físicas
│   ├── contracts.py          # Interfaces predictivas, simulador, A*, evaluación
│   ├── forecasting/          # Ventanas, baselines, normalización, MLP y experimentos
│   ├── energy/               # Simulador, A*, modos, diagnóstico y resiliencia experimental
│   ├── pipeline.py           # Flujo reproducible compartido
│   ├── eda.py                # Estadísticas y gráficos sin interfaz gráfica
│   ├── cli.py                # Comandos locales
│   ├── __init__.py
│   └── __main__.py
├── configs/                  # Vivienda, MLP, A*, modos y protocolos experimentales
├── data/{raw,processed,simulated}/
├── models/                   # Modelos y checkpoints locales, ignorados
├── experiments/              # EDA y ejecuciones, ignorados
├── scripts/                  # Análisis y auditorías físicas reproducibles
├── tests/                    # Suite unittest y fixtures sintéticos pequeños
├── docs/                     # Arquitectura, datos, unidades y validación
├── colab/README.md           # Plan de extracción; sin notebook todavía
├── pyproject.toml
├── requirements.txt
├── requirements-lock.txt
├── requirements-ml.txt
├── requirements-ml-lock.txt
└── .gitignore
```

No existen dos implementaciones. El núcleo es importable por CLI y por un
futuro exportador académico. El adaptador `MLPDemandPredictor` cumple el protocolo
predictivo existente. El simulador y los objetivos energéticos reutilizan los
contratos y el mismo motor A*, sin duplicar tres planificadores.

Documentación: [arquitectura](docs/architecture.md), [fuentes y calidad](docs/data-sources.md),
[unidades](docs/units.md), [validación realizada](docs/phase1-validation.md).

## Fuentes y límites

- [UCI Individual Household Electric Power Consumption](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption):
  Hebrail y Berard (2006), DOI [10.24432/C58K54](https://doi.org/10.24432/C58K54),
  licencia CC BY 4.0. Una vivienda francesa, diciembre 2006–noviembre 2010.
- [PVGIS v6](https://photovoltaic-geographic-information-system.ec.europa.eu/api/),
  Comisión Europea, JRC; [OpenAPI](https://photovoltaic-geographic-information-system.ec.europa.eu/api/v6/openapi.json).
  Generación modelada independiente; disponibilidad, términos y cobertura pueden cambiar.

Las métricas predictivas se documentan en la Fase 2 y las económicas simuladas
en las Fases 3 y 4; no hay ahorros reales demostrados.
TensorFlow 2.21.0 se instaló y validó en Windows/Python 3.13 con CPU,
sin requerir GPU. Es una dependencia opcional para los comandos de datos.

## Predicción de demanda (Fase 2)

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-ml-lock.txt
.\.venv\Scripts\python.exe -u -m bekantor_energy train-mlp --config configs/mlp.reference.json --run-dir experiments/phase2/repeat-01 --model-dir models/phase2/repeat-01
```

Ese comando compara tres arquitecturas sólo con train/validation. Las salidas
deben ser nuevas. La referencia entregada está en
`experiments/phase2/reference-20261022-cpu/` y `models/phase2/reference-20261022-cpu/`.
Consultar sus archivos sin volver a abrir test. El comando final para una
**corrida nueva y congelada**, una única vez, es:

```powershell
.\.venv\Scripts\python.exe -m bekantor_energy evaluate-mlp --run-dir experiments/phase2/repeat-01
```

La generación de ventanas rechaza entradas desconocidas y etiquetas imputadas;
normaliza exclusivamente con train. El modelo seleccionado estima 24 valores
horarios en kWh a partir de 168 horas pasadas; su integración con A* se implementa en
el módulo energético de Fase 3. Se guardan checkpoints,
normalizador, configuración, métricas, predicciones, metadatos y gráficos.
No hay búsqueda automática de hiperparámetros; modificar explícitamente el JSON
para futuros experimentos supervisados. Detalles y límites en
[metodología](docs/phase2-methodology.md).

El [experimento de calendario de Fase 2.2](docs/phase2_2-calendar.md) compara la
MLP 64–32 original con la misma red y variables futuras conocidas (hora, día y
mes, seno/coseno), usando tres semillas y sólo train/validación. Incluye métricas
por semilla, agregados, análisis de picos, gráficos y comandos reproducibles.
Se ejecuta con `compare-calendar`; los resultados y modelos originales se
conservan; la optimización posterior continúa bajo supervisión humana.

El [experimento de MSE ponderada de Fase 2.3](docs/phase2_3-weighted-mse.md)
compara pesos 1/2 por objetivo > P90 de train contra los controles con calendario
del Experimento 1. Incluye tres semillas, métricas de consumos altos y bajos,
análisis del compromiso global y reproducción con `compare-weighted-mse`.
Se recomienda mantener MSE convencional; la pérdida ponderada queda como
experimento conservado. No se probaron otras ponderaciones.

## Simulación energética y A* (Fase 3)

El módulo `energy/` implementa balance AC y de almacenamiento, eficiencias,
límites físicos, reserva, replay de cortes, autoconsumo reactivo y A* sobre un
grafo discreto de SOC. La demostración usa MLP con calendario sin reentrenarla,
24 replanificaciones y demanda **realizada de validación**.

```powershell
.\.venv\Scripts\python.exe -m bekantor_energy energy-demo --run-dir experiments/phase3/repeat-01 --scenario-dir data/simulated/phase3/repeat-01
.\.venv\Scripts\python.exe scripts/verify_energy_run.py --run-dir experiments/phase3/repeat-01
```

Requiere las particiones y modelos anteriores y el perfil PVGIS de Córdoba
descargado para esta fase. La [documentación de Fase 3](docs/phase3-energy.md)
incluye su adquisición, ecuaciones, heurística, pruebas y resultados simulados.
Las rutas de salida deben ser nuevas. No se abrió test ni se modificaron modelos.

## Evaluación comparativa (Fase 4.1)

La [Fase 4.1](docs/phase4_1-strategy-comparison.md) ya compara reactivo, A* + MSE,
A* + ponderada y oracle ideal sobre 14 días de validación y 112 casos factoriales,
sin cambiar A*, redes ni objetivo. Se reproduce con `compare-energy` y conserva
resultados por caso, dispersión, casos desfavorables y auditoría física completa.

## Operación continua y SOC terminal (Fase 4.2)

La [Fase 4.2](docs/phase4_2-continuous-energy.md) implementa operación continua
con SOC persistente sobre dos semanas de validación y perfiles PVGIS
cronológicos. Compara A* original con una condición terminal predefinida y
separa costo bruto de valoración contable del saldo final. El
[diseño previo](docs/phase4_2-design.md) documenta supuestos y sensibilidad.
Se ejecuta con `continuous-energy`; no modifica modelos ni el motor A* original.

## Modos de gestión energética (Fase 4.3)

La [Fase 4.3](docs/phase4_3-energy-modes.md) añade objetivos configurables
Económico, Autonomía y Equilibrado sobre un único núcleo A*. Conserva las
condiciones continuas de Fase 4.2 y compara contra reactivo. La CLI es
`energy-modes`; pesos, escalas y garantías se fijan en el
[diseño previo](docs/phase4_3-design.md) y `configs/energy.modes.json`.

La [interfaz y verificación matemática](docs/energy-management-modes.md)
documenta unidades, heurísticas y desempate lexicográfico. La corrida aceptada
es `experiments/phase4_3/modes-corrected-20261022`; la preliminar se conserva
como auditoría de una corrección numérica, sin modificar los pesos fijados.

La corrección calcula el objetivo lexicográfico de Autonomía con aritmética
racional antes de redondear los flujos, para preservar empates físicos y elegir
el menor costo. Incluye una prueba de regresión y comparaciones con UCS y
enumeración independiente. El modo Económico reproduce las ocho trayectorias
de referencia de Fase 4.2.

Con tarifa variable, Económico reduce el costo ajustado agregado un 10,49 %
frente al reactivo, pero importa un 6,36 % más de red. Equilibrado reduce ese
costo un 4,52 % e importa un 4,07 % más. Autonomía tampoco supera al reactivo
en importación en los casos evaluados. Con tarifa plana, el reactivo resulta
más favorable en costo e importación. La valoración terminal no es una factura;
los límites y la dispersión están en el informe de Fase 4.3.

## Diagnóstico de Autonomía (Fase 4.4)

La [Fase 4.4](docs/phase4_4-autonomy-diagnostic.md) contrasta el reactivo con
una [cota física independiente](docs/autonomy-physical-bound.md), demanda
perfecta diagnóstica, distintas resoluciones y valoración del SOC terminal.
El reactivo alcanza el mínimo físico de importación en los ocho escenarios
sin cortes: 126,714131 kWh/semana de media. Autonomía + MLP importa un 2,9866 %
más; el oracle reduce casi toda esa desventaja, pero tampoco supera al reactivo.
Todos terminan con 1 kWh, por lo que el SOC terminal no explica la diferencia.

Se conserva Autonomía como referencia experimental, sin atribuirle mejoras de
importación no demostradas. El resultado depende de red disponible sin límite,
demanda fija y eficiencias constantes; no es una conclusión universal ni una
evaluación de respaldo. Las 24 corridas nuevas y sus casos desfavorables se
conservan, sin seleccionar retrospectivamente otra grilla. Configuración:
`configs/energy.autonomy-diagnostic.json`.

## Resiliencia ante cortes imprevistos (Fase 4.5)

La [Fase 4.5](docs/phase4_5-resilience.md) implementa seis estrategias sobre
dos bloques independientes de siete días continuos de validación, dos perfiles
PVGIS de Córdoba y tarifas plana/variable. Hay 72 casos con cortes de 2/6/12 h
a las 06:00/14:00/22:00 y ocho controles sin corte: **480 replays principales**.
Con piloto y sensibilidades se completaron 656 replays semanales sin fallos.

La demanda crítica es un **40 % sintético** de UCI. El 60 % no esencial se
desconecta durante cortes y se registra como energía sacrificada. Se conserva
SOC entre días, piso físico 20 %, reserva convencional 50 % y el mismo
controlador de emergencia para todos. Ninguna estrategia conoce los cortes
futuros; el oracle sólo conoce demanda futura de validación.

Medias sobre 72 replays semanales con un corte cada uno; ENS es por evento:

| Estrategia | ENS crítica kWh/evento | Casos sin déficit crítico | Costo ajustado semanal SIM |
|---|---:|---:|---:|
| Reactivo sin reserva adicional | 1,945820 | 16/72 | 21,935228 |
| Reserva fija 50 % | 1,017527 | 28/72 | 23,676292 |
| Reserva adaptativa con MLP | 0,737537 | 44/72 | 22,983586 |
| A* con persistencia de 24 h | 0,623267 | 44/72 | 23,577745 |
| A* con MLP congelada | 0,653646 | 44/72 | 22,711361 |
| A* oracle de demanda | 0,599772 | 50/72 | 21,942328 |

A* + MLP evita 0,363881 kWh críticos por evento frente a reserva fija
(**35,7613 %** de reducción agregada), cumpliendo los umbrales predefinidos.
Mejora 36 casos y empata 36 en ENS. Reduce costo ajustado medio un 4,08 %, con
0,178374 kWh semanales más de red; hay casos individuales con mayor costo.
Con la misma MLP, A* reduce ENS un 11,3745 % frente a reserva adaptativa,
pero importa más. **No se demuestra superioridad neuronal sobre persistencia**:
MLP produce un 4,8741 % más de ENS, aunque con menor costo e importación.

El objetivo experimental es lexicográfico: primero riesgo de déficit crítico
en cortes hipotéticos de cuatro horas, después costo. No utiliza el calendario
real de cortes ni cambia Económico, Autonomía o Equilibrado. Usa h=0, aritmética
racional y comparaciones con UCS/programación dinámica en problemas pequeños.
El riesgo no equivale a ENS realizada de cortes de 2/6/12 h; ni siquiera el
oracle es una cota superior universal de servicio. La grilla fina del panel
no mejora ENS de MLP y casi duplica expansiones; se conserva la resolución
principal de 0,25 kWh sin elegirla retrospectivamente.

Son simulaciones horarias con solar ideal, tarifas ilustrativas SIM y batería
sin degradación. Los dos bloques UCI no son 72 muestras independientes. Se
presupone observabilidad de demanda potencial pasada aun durante cortes;
no se modelan pérdida de telemetría, respuesta del usuario ni hardware de isla.
El costo ajustado valora energía terminal y **no es una factura**. La autonomía
horaria censurada no garantiza horas de respaldo en una instalación real.
Se recomienda mantener resiliencia como componente experimental y conservar
los controles de reserva adaptativa y persistencia para evaluar su beneficio.

Para ejecutar en carpetas nuevas, con los artefactos históricos locales y
fuentes compatibles con sus manifiestos:

```powershell
.\.venv\Scripts\python.exe scripts/diagnose_autonomy.py --config configs/energy.autonomy-diagnostic.json --run-dir experiments/phase4_4/repeat-01
.\.venv\Scripts\python.exe scripts/run_resilience.py --stage pilot --run-dir experiments/phase4_5/repeat-pilot
.\.venv\Scripts\python.exe scripts/analyze_resilience.py --run-dir experiments/phase4_5/repeat-pilot
.\.venv\Scripts\python.exe scripts/verify_resilience.py --run-dir experiments/phase4_5/repeat-pilot
.\.venv\Scripts\python.exe scripts/run_resilience.py --stage full --run-dir experiments/phase4_5/repeat-full --cache-source experiments/phase4_5/repeat-pilot
```

Los informes de ambas fases contienen los comandos de análisis, gráficos,
sensibilidades y auditoría completa. Las previsiones neuronales se reutilizan
congeladas; estos experimentos no reentrenan redes ni abren el test final.

## Verificación y archivos versionados

La suite hasta Fase 4.5 incluye **146 pruebas sin entrenamiento y ocho de
integración TensorFlow** con datos sintéticos temporales. Verifica continuidad
de SOC, causalidad, conservación física, límites de batería, costos terminales,
objetivos frente a UCS, cota de importación, cargas críticas y emergencia común.
El comando de prueba rápida ejecuta ambas partes cuando TensorFlow está
instalado; ninguna abre el test UCI real ni modifica los modelos congelados.
Las integraciones entrenan únicamente redes pequeñas y temporales de prueba.
En esta consolidación del 23/10/2026, las **154 pruebas aprobaron en 177,889 s**,
sin fallos ni omisiones. Registro local:
`experiments/verification/consolidation-4_4-4_5-tests-20261023.txt`.

Git conserva código, configuraciones sin secretos, pruebas, fixtures pequeños
y documentación. `.gitignore` excluye `.venv/`, datos brutos, procesados y
simulados, modelos, checkpoints, resultados experimentales, cachés, temporales
y los nombres habituales de archivos de credenciales. En `data/`, `models/` y
`experiments/` sólo se versionan los marcadores y README permitidos.

Las pruebas sintéticas funcionan desde un clon con las dependencias instaladas.
Para repetir los experimentos históricos hacen falta además los datos adquiridos,
los modelos congelados y sus manifiestos/archivos de auditoría locales, que no
se distribuyen en Git. Las configuraciones conservan esas rutas explícitas; los
informes de cada fase documentan los comandos y prerrequisitos. Usar directorios
de salida nuevos para preservar las corridas anteriores.

Los verificadores históricos también protegen las fuentes y el README vigente
al ejecutar cada corrida. Esta consolidación actualiza ese README, sin cambiar
los manifiestos originales: para una auditoría histórica estricta se necesita
el árbol fuente congelado compatible con sus hashes, en un checkout separado.
La comprobación de consolidación registra el cambio documental autorizado
por separado de modelos, datos y resultados, que permanecen intactos.

## Etapas posteriores (sin implementar)

Quedan pendientes mayor diversidad de perfiles y semillas, errores de previsión
solar, telemetría durante cortes y validación específica de hardware. Dashboard
y extracción académica siguen sin implementar. El notebook Colab todavía no se
ha comenzado; la arquitectura modular permite prepararlo más adelante.

Repositorio: [moiseslobayza/bekantor-energy](https://github.com/moiseslobayza/bekantor-energy).

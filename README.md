# BEKANTOR Energy

Sistema local y modular para investigar predicción de demanda eléctrica y
planificación de energía solar, batería y red para viviendas y pequeños negocios.
**Estado: Fases 1 y 2 implementadas.** Hay procesamiento reproducible y predicción
MLP TensorFlow/Keras de 168 a 24 horas en CPU. Despacho energético, búsqueda A*,
control de hardware, dashboard y notebook académico quedan pendientes.

La instalación siguiente conserva el entorno de datos de Fase 1. Para entrenar
y consultar los resultados de Fase 2, ver [protocolo y comandos](docs/phase2-methodology.md)
y [resultados de referencia](docs/phase2-results.md).

## Instalación local en Windows (PowerShell, CPU)

Usar Python de 64 bits, versión 3.11 o posterior. La versión comprobada para
Fase 1 figura en `docs/phase1-validation.md`. No hace falta GPU, CUDA ni
TensorFlow en esta fase. Ejecutar los comandos desde la raíz del proyecto:

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
futura será un escenario simulado explícito, no evidencia de una misma vivienda.

## Estructura y responsabilidades

```text
bekantor-energy/
├── src/bekantor_energy/
│   ├── data/                 # UCI, PVGIS, HTTP y particiones cronológicas
│   ├── config.py             # Vivienda, restricciones y tarifas simuladas
│   ├── units.py              # Conversiones físicas
│   ├── contracts.py          # Interfaces predictivas, simulador, A*, evaluación
│   ├── forecasting/          # Ventanas, baselines, normalización, MLP y experimentos
│   ├── pipeline.py           # Flujo reproducible compartido
│   ├── eda.py                # Estadísticas y gráficos sin interfaz gráfica
│   ├── cli.py                # Comandos locales
│   ├── __init__.py
│   └── __main__.py
├── configs/                  # Vivienda y comparación MLP de referencia
├── data/{raw,processed,simulated}/
├── models/                   # Artefactos futuros, ignorados
├── experiments/              # EDA y ejecuciones, ignorados
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
predictivo existente. Las interfaces energéticas siguen sin lógica de despacho
o búsqueda.

Documentación: [arquitectura](docs/architecture.md), [fuentes y calidad](docs/data-sources.md),
[unidades](docs/units.md), [validación realizada](docs/phase1-validation.md).

## Fuentes y límites

- [UCI Individual Household Electric Power Consumption](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption):
  Hebrail y Berard (2006), DOI [10.24432/C58K54](https://doi.org/10.24432/C58K54),
  licencia CC BY 4.0. Una vivienda francesa, diciembre 2006–noviembre 2010.
- [PVGIS v6](https://photovoltaic-geographic-information-system.ec.europa.eu/api/),
  Comisión Europea, JRC; [OpenAPI](https://photovoltaic-geographic-information-system.ec.europa.eu/api/v6/openapi.json).
  Generación modelada independiente; disponibilidad, términos y cobertura pueden cambiar.

Las métricas predictivas se documentan en la Fase 2; no hay ahorros energéticos
demostrados. TensorFlow 2.21.0 se instaló y validó en Windows/Python 3.13 con CPU,
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
normaliza exclusivamente con train. El modelo seleccionado estima 24 kWh
horarios a partir de 168 horas pasadas, sin integrar A*. Se guardan checkpoints,
normalizador, configuración, métricas, predicciones, metadatos y gráficos.
No hay búsqueda automática de hiperparámetros; modificar explícitamente el JSON
para futuros experimentos supervisados. Detalles y límites en
[metodología](docs/phase2-methodology.md).

El [experimento de calendario de Fase 2.2](docs/phase2_2-calendar.md) compara la
MLP 64–32 original con la misma red y variables futuras conocidas (hora, día y
mes, seno/coseno), usando tres semillas y sólo train/validación. Incluye métricas
por semilla, agregados, análisis de picos, gráficos y comandos reproducibles.
Se ejecuta con `compare-calendar`; los resultados y modelos originales se
conservan. El siguiente experimento requiere revisión humana.

El [experimento de MSE ponderada de Fase 2.3](docs/phase2_3-weighted-mse.md)
compara pesos 1/2 por objetivo > P90 de train contra los controles con calendario
del Experimento 1. Incluye tres semillas, métricas de consumos altos y bajos,
análisis del compromiso global y reproducción con `compare-weighted-mse`.
Se recomienda mantener MSE convencional; la pérdida ponderada queda como
experimento conservado. No se probaron otras ponderaciones ni se avanzó a A*.

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

## Etapas posteriores (sin implementar)

Quedan pendientes la revisión del horizonte/SOC terminal, más semillas y
perfiles solares, robustez ante incertidumbre y cortes, dashboard y extracción
académica. No se avanzó a Fase 4.2 ni se generó el notebook Colab.

El remoto Git está configurado a `moiseslobayza/bekantor-energy`; este trabajo no
realiza commits, pushes ni modificaciones en GitHub.

# Fase 2: protocolo de predicción y repetición en CPU

Esta fase agrega `src/bekantor_energy/forecasting/` al núcleo existente. Utiliza
los CSV y el manifiesto de Fase 1; no vuelve a procesar el dataset ni cambia sus
límites cronológicos. No incorpora solar, simulador, A*, dashboard o notebook.

## Ventanas y disponibilidad temporal

Un origen `t` representa el límite de emisión después de observar la hora
`[t-1h,t)`. Entrada: energía de las 168 horas `[t-168h,t)`. Salida simultánea:
energía de cada uno de los 24 intervalos `[t,t+24h)`. El modelo es un MLP
directo multisalida, sin realimentar sus propias predicciones.

Las particiones se conservan. Todas las etiquetas de una ventana deben estar
dentro de su partición objetivo. Validation puede usar contexto de train; test
puede usar contexto ya observado de validation y horas anteriores de test.
Esto es evaluación **rolling origin horaria**: cada pronóstico usa únicamente
el pasado disponible en su emisión, no una única predicción emitida al comenzar
todo el bloque test. `stride=1` es común a MLP y baselines. Hay ventanas y
etiquetas solapadas; las métricas pesan pares origen/horizonte y no equivalen a
sumar consumo de horas únicas ni a muestras estadísticamente independientes.

Política de ausentes, fijada antes de ver resultados:

- Una entrada requiere 168 valores finitos. Puede usar las horas rellenadas
  causalmente por Fase 1, que ya estaban disponibles al emitir el pronóstico.
- Una salida requiere 24 horas **totalmente observadas**:
  `is_observed_complete=True`. Incluso una energía finita con algún minuto
  imputado queda fuera de las etiquetas de entrenamiento, validación y test.
- No se rellenan nuevos objetivos ni entradas desconocidas. Se excluye la ventana
  y se informa el motivo. Los timestamps ausentes se reindexan, nunca se eliminan
  para unir artificialmente horas separadas. Los huecos reducen cobertura.
- La normalización usa una sola vez los valores horarios únicos y observados
  de train, sin reponderarlos por la cantidad de ventanas que los contienen.

## Normalización y arquitectura común

Se guardan media y desviación estándar poblacional de train. Entradas:
`x_scaled=(x-mean_train)/std_train`. Objetivos: `y_scaled=y/std_train`, sin
centrar, de modo que la salida `softplus` pueda garantizar energía no negativa.
Si la serie de entrenamiento fuera constante se usa escala 1. Las predicciones
se convierten a kWh multiplicando por esa escala; ninguna métrica final queda
expresada como un error normalizado sin unidad física.

Tres arquitecturas de referencia, fijadas en `configs/mlp.reference.json`:

| Nombre | Capas ocultas | Activación oculta | Salida |
|---|---|---|---|
| small | 64, 32 | ReLU | 24 neuronas softplus |
| medium | 128, 64 | ReLU | 24 neuronas softplus |
| large | 256, 128, 64 | ReLU | 24 neuronas softplus |

Las tres usan Adam con learning rate 0.001, MSE, métricas Keras MAE/RMSE,
batch 128, máximo 40 épocas, semilla 2026 y paciencia 6. Sin dropout ni L2 en
esta primera comparación. El JSON permite modificar capas, activación, dropout,
L2, learning rate y otros parámetros para experimentos **supervisados** futuros.
No hay búsqueda automática, bucles de optimización ni reentrenamiento al ver test.

Dentro de cada arquitectura, EarlyStopping y ModelCheckpoint monitorean el
mismo `val_rmse`, con modo mínimo y `min_delta=0`. Se recarga el checkpoint
de mejor validación; las métricas train/validation reportadas corresponden a él,
no necesariamente a la última época. Los logs Keras están en unidades escaladas;
los gráficos convierten la pérdida a MSE equivalente kWh². Si se activa L2, la
curva de pérdida incluye su penalización, y se indica en el eje.

Se selecciona entre las tres por menor RMSE de validation en kWh. Un empate se
resuelve con menos parámetros y luego nombre. Los baselines se comparan, pero
no intervienen en elegir cuál MLP es la mejor. El modelo seleccionado conserva
su entrenamiento original: no se reajusta con train+validation.

## Baselines y métricas

Para cada horizonte `h=0..23`:

```text
persistencia diaria:  y_hat(t+h) = y(t+h-24h)
persistencia semanal: y_hat(t+h) = y(t+h-168h)
```

Ambas usan exactamente las mismas ventanas elegibles que cada MLP, los mismos
objetivos y las mismas unidades kWh. No se introduce el valor real del horizonte
futuro para prolongar recursivamente una baseline. Se reportan MAE, RMSE, MSE
(kWh²), sesgo (kWh), cantidad de orígenes y error por horizonte. No se usa MAPE
como criterio por su sensibilidad a consumos cercanos a cero.

## Separación de selección y test

`train-mlp` lee solamente train/validation. Verifica sus hashes contra Fase 1,
guarda configuración, reportes de ventanas, normalizador, metadatos, checkpoints
y comparación por validación. Escribe `selection.json` al terminar, con hashes
del bundle elegido y de la configuración. Las rutas de salida deben ser nuevas;
no sobrescribe experimentos previos.

`evaluate-mlp` verifica esa selección congelada y crea exclusivamente
`test_evaluation.json` **antes de abrir test**. Evalúa el modelo elegido y las
dos baselines sobre las mismas ventanas, preserva predicciones y métricas, y
marca completada la evaluación. Una segunda llamada en la misma corrida se
rechaza, sin volver a leer test. Si falla después de reservar la evaluación, el
marcador queda conservado: no se elimina ni reintenta automáticamente.

El marcador protege una corrida, no impide que una persona cree otra o lo borre.
Después de esta evaluación test ya no puede considerarse ciego para iteraciones
humanas que se hayan informado de sus resultados. Las próximas optimizaciones
deben guiarse por validation o validación temporal interna; reservar otro bloque
final si se requiere una nueva estimación totalmente independiente.

## Instalación y comandos en Windows

Desde la raíz, reutilizando `.venv` de Fase 1:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-ml-lock.txt
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

La dependencia TensorFlow es opcional respecto de adquisición/EDA. Para resolver
otro entorno usar `requirements-ml.txt` y volver a validar: el lock reproduce el
entorno Windows x64/Python 3.13 de esta corrida, no todas las plataformas.
La primera importación de TensorFlow puede ser lenta por carga de DLL.
No hace falta GPU, CUDA, WSL ni cambiar la política de PowerShell.

Repetir sólo entrenamiento/selección en una carpeta nueva:

```powershell
.\.venv\Scripts\python.exe -u -m bekantor_energy train-mlp --config configs/mlp.reference.json --run-dir experiments/phase2/repeat-01 --model-dir models/phase2/repeat-01
```

Tras revisar validation y congelar esa selección, el comando final autorizado
para una corrida nueva sería:

```powershell
.\.venv\Scripts\python.exe -m bekantor_energy evaluate-mlp --run-dir experiments/phase2/repeat-01
```

No ejecutar reiteradamente esa evaluación para orientar ajustes. La corrida
entregada se llama `reference-20261022-cpu`; sus resultados se consultan en archivos,
sin volver a evaluarla. Para personalizar, copiar el JSON a otro archivo en
`configs/`, modificar explícitamente parámetros y utilizar otras rutas de salida.

Regenerar sólo figuras desde las predicciones y métricas ya guardadas, sin
abrir el dataset de test ni volver a ejecutar el modelo:

```powershell
.\.venv\Scripts\python.exe -m bekantor_energy plot-mlp --run-dir experiments/phase2/reference-20261022-cpu
```

En el sandbox Codex de esta máquina, apuntar los temporales al proyecto antes de
pruebas, entrenamiento o evaluación (en una terminal normal puede no ser necesario):

```powershell
New-Item -ItemType Directory -Force experiments/tmp | Out-Null
$env:TEMP = (Resolve-Path experiments/tmp).Path
$env:TMP = $env:TEMP
```

## Artefactos y compatibilidad

`models/phase2/<corrida>/<arquitectura>/` conserva `best.keras`, `training.json`
y `epochs.csv`. `selected/` guarda el bundle de inferencia: `model.keras`,
`normalization.json`, `predictor.json`. Todos los modelos están ignorados por Git.

`experiments/phase2/<corrida>/` conserva configuración, manifiesto de datos,
normalización, contabilidad de ventanas, selección, comparación con baselines,
predicciones de validation/test, métricas y PNG. `reproducibility.json` registra
versiones, sistema, CPU, hilos, semillas, dispositivos y hashes del código.

`MLPDemandPredictor` cumple los métodos `fit`, `predict` y `save` del puerto
`DemandPredictor` existente; `load` permite restaurarlo. `predict` comprueba
168 horas conocidas contiguas y exactamente los próximos 24 inicios horarios,
y devuelve `LoadForecast` con unidades kWh, fecha de emisión e identificador.
No se conecta al planificador energético en esta fase.

Semillas de Python/NumPy/TensorFlow, operaciones deterministas, CPU exclusiva,
oneDNN desactivado e hilos fijados reducen variación. No se promete identidad
binaria entre versiones, hardware o sistemas diferentes; se conserva el lock.

Referencias técnicas oficiales consultadas:
[EarlyStopping](https://keras.io/api/callbacks/early_stopping/),
[ModelCheckpoint](https://keras.io/api/callbacks/model_checkpoint/),
[determinismo TensorFlow](https://www.tensorflow.org/api_docs/python/tf/config/experimental/enable_op_determinism),
[paquete TensorFlow 2.21.0](https://pypi.org/project/tensorflow/2.21.0/).

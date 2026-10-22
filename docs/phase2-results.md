# Fase 2 — resultados de referencia (22/10/2026)

Corrida: `reference-20261022-cpu`. Se completó una comparación de tres MLP en CPU, selección únicamente con validación y una evaluación final de test. No se implementó Fase 3 ni se realizaron commits/push.

## Arquitecturas y entrenamiento

Entrada: 168 energías horarias pasadas; salida directa: 24 energías horarias futuras. Capas ocultas ReLU, salida softplus no negativa. Adam 0.001, MSE, métricas MAE/RMSE, batch 128, máximo 40 épocas, paciencia 6, semilla 2026, sin dropout/L2. Se recargó el mejor checkpoint de cada arquitectura.

| MLP | Capas ocultas | Parámetros | Épocas / mejor | MAE train (kWh) | RMSE train (kWh) | MAE validation (kWh) | RMSE validation (kWh) |
|---|---|---:|---:|---:|---:|---:|---:|
| small | 64–32 | 13688 | 14 / 8 | 0.492504 | 0.686105 | 0.507486 | 0.695581 |
| medium | 128–64 | 31448 | 12 / 6 | 0.483136 | 0.672489 | 0.505544 | 0.695754 |
| large | 256–128–64 | 85976 | 8 / 2 | 0.489611 | 0.685863 | 0.504125 | 0.696207 |

**Seleccionada: small (64–32), checkpoint de época 8.** Su RMSE de validación es el menor de esta corrida. Las diferencias entre MLP son muy pequeñas: no prueban superioridad estadística ni justifican una conclusión general sobre el tamaño de la red. La MLP grande reduce su pérdida de train mientras empeora validation tras su mejor época, compatible con sobreajuste. No se amplió la búsqueda ni se repitió la comparación real para buscar mejores resultados.

## Comparación contra baselines en validación

| Estrategia | MAE (kWh) | RMSE (kWh) |
|---|---:|---:|
| selected_mlp | 0.507486 | 0.695581 |
| seasonal_24h | 0.665125 | 0.957958 |
| seasonal_168h | 0.599349 | 0.877874 |

## Test final: una única evaluación

| Estrategia | MAE (kWh) | RMSE (kWh) | MSE (kWh²) | Sesgo (kWh) |
|---|---:|---:|---:|---:|
| selected_mlp | 0.431285 | 0.593534 | 0.352282 | -0.013123 |
| seasonal_24h | 0.505590 | 0.756014 | 0.571558 | 0.004559 |
| seasonal_168h | 0.552348 | 0.797331 | 0.635737 | 0.024965 |

Reducción descriptiva de RMSE frente a `seasonal_24h`: **21.49%** sobre las mismas ventanas.

Reducción descriptiva de RMSE frente a `seasonal_168h`: **25.56%** sobre las mismas ventanas.

No se evaluaron las MLP descartadas en test. La MLP seleccionada no se reajustó con train+validation ni después de abrir test. Un menor error predictivo no demuestra ahorro energético: todavía no existe simulador ni planificador integrado.

## Ventanas y normalización

| Partición | Candidatas | Historia insuficiente | Entradas desconocidas | Objetivos no observados | Aceptadas |
|---|---:|---:|---:|---:|---:|
| train | 24189 | 168 | 1986 | 1111 | 20924 |
| validation | 5165 | 0 | 423 | 264 | 4478 |
| test | 5166 | 0 | 543 | 170 | 4453 |

Los motivos se contabilizan en orden: primero historia insuficiente, luego entradas desconocidas, luego objetivos no observados. Una ventana con más de un problema se cuenta en el primer motivo aplicable.

Normalizador ajustado sobre **24022 horas únicas observadas de train**: media 1.085161359 kWh y escala 0.927800535 kWh. Ningún dato de validation/test participó en ese ajuste.

Test conserva **4453 orígenes**, cada uno con 24 objetivos: **106872 pares origen/horizonte**. Son ventanas solapadas, no 106872 horas únicas. Toda etiqueta es totalmente observada; las entradas pueden incluir el relleno causal limitado de Fase 1. Se preservan fechas y particiones originales.

| Horizonte | MAE MLP (kWh) | RMSE MLP (kWh) |
|---|---:|---:|
| +1 h | 0.361210 | 0.512680 |
| +6 h | 0.435110 | 0.597871 |
| +12 h | 0.433441 | 0.597884 |
| +18 h | 0.436653 | 0.598697 |
| +24 h | 0.435371 | 0.600574 |

El JSON de métricas conserva las 24 filas de cada estrategia, sin seleccionar únicamente horizontes favorables.

## Gráficos generados y revisados

Los PNG se encuentran en `experiments/phase2/reference-20261022-cpu/plots/`:

- `training_loss.png`: pérdidas train/validation de las tres MLP, escala vertical común y época elegida.
- `architecture_comparison.png`: RMSE train/validation de cada checkpoint.
- `validation_actual_vs_predicted.png` y `test_actual_vs_predicted.png`: consumo observado frente a MLP y ambas persistencias; siempre el primer origen elegible, fijado sin elegir un día favorable.
- `validation_baseline_comparison.png` y `test_baseline_comparison.png`: MAE/RMSE comparables.
- `validation_horizon_error.png` y `test_horizon_error.png`: RMSE según horizonte.

Las figuras se pueden reconstruir desde predicciones/JSON guardados mediante `plot-mlp`, sin abrir de nuevo el dataset de test ni inferir otra vez.

## Pruebas y entorno

**47 pruebas aprobadas, 0 fallos y 0 errores**: 29 de Fase 1 y 18 nuevas. Se verifican alineación exacta 168→24, límites de partición, huecos sin compresión del tiempo, rechazo de etiquetas imputadas, causalidad, normalización exclusiva de train, lags 24/168, métricas físicas, semillas, salida no negativa, CPU, checkpoint, persistencia/restauración, contrato LoadForecast y bloqueo de segunda evaluación.

La integración sintética repite dos entrenamientos diminutos con la misma semilla y obtiene RMSE coincidente a seis decimales; también comprueba que replotear no abre las particiones ni ejecuta inferencia. Las pruebas nunca leen el test real UCI. La suite completa pasó y, después del ajuste de presentación de figuras, se repitieron además las dos pruebas de integración TensorFlow con resultado aprobado.

Entorno: Windows 11 x64, Python 3.13.15, TensorFlow 2.21.0, Keras 3.15.1, NumPy 2.5.4, pandas 3.0.6, Matplotlib 3.11.2. Sólo `/device:CPU:0`, dos hilos intraop, uno interop, oneDNN desactivado y operaciones deterministas. `pip check` no encontró dependencias incompatibles. Lock completo: `requirements-ml-lock.txt`.

El primer intento real (`reference-20261022`) falló al escribir un temporal de checkpoint por permisos del sandbox; se preservó. La corrida final usa TEMP/TMP en `experiments/tmp`, con los mismos datos, configuración y semillas, sin haber evaluado test en el intento fallido. No fue una búsqueda adicional.

Se conservó una nota de procedencia sobre metadatos editables antiguos: el código entrenado es 0.2.0, pero el registro original de distribuciones incluía un egg-info 0.1.0 residual de Fase 1. Se regeneró ese metadato ignorado; los pesos y la selección no cambiaron. El reporte de reproducibilidad preserva la explicación y los hashes originales del código de entrenamiento.

## Archivos entregados y repetición

- Núcleo: `forecasting/{windows,normalization,baselines,metrics,config,mlp,experiment,plots}.py`.
- Configuración editable: `configs/mlp.reference.json`.
- Modelo elegido: `models/phase2/reference-20261022-cpu/selected/model.keras`.
- Bundle: `normalization.json` y `predictor.json` en esa misma carpeta.
- Checkpoints e historias por arquitectura: carpetas `small/`, `medium/`, `large/` del directorio de modelos.
- Configuración, manifiesto, normalización, ventanas, selección, predicciones, métricas y metadatos: `experiments/phase2/reference-20261022-cpu/`.
- Logs: `experiments/verification/phase2-tests.txt`, `phase2-tensorflow-final.txt`, `phase2-training-cpu.txt`, `phase2-test-evaluation.txt`.
- Métodos, instalación Windows y comandos: [phase2-methodology.md](phase2-methodology.md).

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-ml-lock.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -u -m bekantor_energy train-mlp --config configs/mlp.reference.json --run-dir experiments/phase2/repeat-01 --model-dir models/phase2/repeat-01
.\.venv\Scripts\python.exe -m bekantor_energy plot-mlp --run-dir experiments/phase2/reference-20261022-cpu
```

El test de referencia ya está completado: consultar `test_metrics.json`; no repetir `evaluate-mlp` sobre esa corrida. Para una corrida nueva, el procedimiento de evaluación final está separado y documentado. Los modelos, datos y resultados grandes permanecen fuera de Git.

## Limitaciones y mejoras supervisadas posibles

- Una sola vivienda histórica: no demuestra generalización a otros hogares, negocios o países.
- Un único bloque validation y una semilla para la comparación real. La casi igualdad entre MLP no establece una clasificación estable. No se calcularon intervalos de confianza sobre ventanas correlacionadas.
- Excluir huecos puede sesgar la cobertura hacia períodos con medición completa. No se mide rendimiento cuando faltan entradas en operación.
- Modelo univariado, sin calendario explícito, clima, ocupación ni incertidumbre predictiva. Los picos pueden quedar suavizados.
- Test tiene menor error que validation, pero los períodos y niveles de demanda difieren; eso no implica mejora del modelo durante test.
- Después de observar este test, no usarlo para elegir learning rate, regularización o arquitectura. Para una nueva estimación ciega tras optimizar, reservar otro bloque final o una validación temporal anidada.
- Los determinismos y el lock no garantizan identidad binaria entre hardware, sistemas o versiones diferentes.
- Queda a supervisión humana variar arquitectura, learning rate, dropout/L2 y paciencia, siempre mirando validation. Otra mejora metodológica es validación temporal con varios cortes dentro del período de desarrollo, sin reutilizar este test como selector.
- No se implementaron A*, despacho energético, dashboard o Colab. No se avanzó a Fase 3.

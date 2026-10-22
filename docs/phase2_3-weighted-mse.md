# Fase 2.3 — Experimento 2: MSE ponderada para consumo elevado

**Recomendación: descartar la ponderación 1/2 como reemplazo de la pérdida
principal y mantener la MLP con calendario y MSE convencional.** La ponderación
mejora los picos en las tres semillas, pero aumenta el MAE global **5,87 %**,
el RMSE global **4,26 %** y el RMSE de consumos bajos **15,19 %**. El compromiso
no satisface el objetivo de mejorar picos sin perjudicar sensiblemente la
precisión general. El código y todos los resultados quedan conservados como
experimento; no se elimina ni sustituye ningún modelo anterior.

Corrida: `weighted-mse-20261022`. Se entrenaron **tres modelos nuevos en CPU**,
uno por semilla 2026, 2027 y 2028. Los controles son los tres modelos MSE con
calendario congelados del Experimento 1, cuyas predicciones de validación se
reutilizan sin reentrenarlos. No se abrió ni utilizó test.

## Protocolo controlado

Las dos variantes tienen 312 entradas (168 consumos pasados y 144 variables
cíclicas del horizonte futuro), capas ocultas ReLU 64–32, salida softplus de 24
horas y **22.904 parámetros**. Se mantienen Adam, learning rate 0,001, batch 128,
máximo 40 épocas, paciencia 6, ausencia de dropout/L2, ventanas horarias,
particiones, normalización y semillas del Experimento 1.

EarlyStopping y ModelCheckpoint siguen minimizando **`val_rmse` sin ponderar**.
No se cambió la regla de selección al introducir la pérdida. Cada comparación
utiliza el checkpoint mínimo de su propia corrida y la misma semilla; no se
selecciona una semilla ni se promedian predicciones como ensemble.

La única modificación al entrenamiento es:

```text
z = energía_kWh / escala_kWh
t = P90_train / escala_kWh
w(z) = 2 si z > t; 1 en caso contrario
L = media sobre batch y las 24 horas de [w(z_real) × (z_predicho − z_real)²]
```

Los pesos se aplican **por valor objetivo**, no a toda una ventana cuando
contiene un pico. El umbral depende del objetivo real, no de la predicción.
Los pesos son 1 y 2 sin dividir por su suma ni normalizarlos por batch; por eso
las pérdidas de entrenamiento de las dos variantes no se comparan directamente
como si fueran la misma magnitud. Las figuras de aprendizaje comparan RMSE
sin ponderar, en kWh.

Los objetivos se escalan sin centrar: **no se resta la media de las entradas al
umbral**. La escala es 0,92780053496 kWh y el umbral normalizado P90 es
2,57749007092 (float32: 2,57749009132). Se comprobó igualdad exacta de las máscaras
de picos en unidades físicas y normalizadas para todos los objetivos de train
y validación. La pérdida física ponderada es `escala² × pérdida normalizada`.

## Particiones, umbrales y evaluación

Se verifican los hashes train/validation de Fase 1 y su coincidencia con los
metadatos del control, así como normalización, informes de ventanas, orígenes y
objetivos guardados. El lector de datos sólo recibe `train` y `validation`.

- Train: 2006-12-16 17:00 a 2009-09-20 12:00; **20.924 ventanas** de 168→24 horas.
- Validación: 2009-09-20 13:00 a 2010-04-24 16:00; **4.478 ventanas**, **107.472
  pares origen/horizonte** por modelo, idénticos en los seis modelos comparados.
- Todos los cuantiles usan exclusivamente **24.022 horas únicas originalmente
  observadas de train**, interpolación lineal; no objetivos repetidos de ventanas.
- P90 = **2,39139666667 kWh**; consumo elevado `real > P90`: **12.142 pares**.
- P95 = **2,94186333333 kWh**; extremo `real > P95`: **4.898 pares**.
- Consumo bajo, definición fijada antes de entrenar: `real <= P25_train`, con
  P25 = **0,32286666667 kWh**: **10.642 pares**.
- Consumo normal para el diagnóstico: `real <= P90`, **95.330 pares**. Incluye
  el estrato bajo; ambos no deben sumarse como grupos disjuntos.
- Los objetivos desconocidos/imputados siguen excluidos. Las entradas sólo
  contienen consumo anterior al origen y calendario futuro conocido.
- Sesgo = predicho − real. Subestimación: predicho < real; sobreestimación:
  predicho > real. Los estratos por consumo real son diagnósticos retrospectivos.

Para interpretar «sin perjudicar significativamente» se fijó antes de entrenar
un margen **operativo de +1 %** en MAE y RMSE globales. Es una decisión de
evaluación de esta implementación, no un umbral solicitado por el usuario ni
una prueba de significancia estadística. Se informan también todos los cambios
continuos para permitir revisar ese criterio.

## Media y desviación estándar

Media ± desviación estándar **muestral** entre tres semillas (`ddof=1`). MAE,
RMSE y sesgo en kWh por intervalo horario. Los porcentajes tienen unidades de
porcentaje y su dispersión, de puntos porcentuales. No son intervalos de
confianza ni incertidumbre de las predicciones.

| Métrica global | Calendario + MSE | Calendario + MSE ponderada |
|---|---:|---:|
| MAE | 0,503995 ± 0,003010 | 0,533570 ± 0,002257 |
| RMSE | 0,686419 ± 0,001027 | 0,715655 ± 0,003879 |
| Sesgo | +0,014694 ± 0,014274 | +0,131828 ± 0,012732 |
| Subestimación (%) | 43,5248 ± 1,0271 | 37,9367 ± 1,0213 |

| Métrica en consumo elevado | MSE | MSE ponderada |
|---|---:|---:|
| MAE > P90 | 1,097048 ± 0,020481 | 0,941054 ± 0,017499 |
| RMSE > P90 | 1,296872 ± 0,019039 | 1,159403 ± 0,019491 |
| Sesgo > P90 | −1,071890 ± 0,025489 | −0,873060 ± 0,029089 |
| Subestimación > P90 (%) | 95,1902 ± 0,5737 | 89,1973 ± 1,2676 |
| MAE > P95 | 1,533482 ± 0,028880 | 1,331273 ± 0,034383 |
| RMSE > P95 | 1,707726 ± 0,025299 | 1,538432 ± 0,030805 |
| Sesgo > P95 | −1,529673 ± 0,029127 | −1,315380 ± 0,038609 |
| Subestimación > P95 (%) | 99,0676 ± 0,1359 | 96,3250 ± 0,7370 |

| Métrica de consumo bajo (≤ P25) | MSE | MSE ponderada |
|---|---:|---:|
| MAE | 0,351131 ± 0,012714 | 0,401782 ± 0,017788 |
| RMSE | 0,487391 ± 0,008897 | 0,561430 ± 0,010942 |
| Sesgo | +0,342905 ± 0,014320 | +0,390996 ± 0,022531 |
| Sobreestimación (%) | 91,4803 ± 2,0036 | 90,5907 ± 3,2630 |

La frecuencia media de sobreestimación baja ligeramente en el estrato bajo,
pero **aumentan la magnitud del error y el sesgo positivo**. No corresponde
afirmar que allí se sobreestima con mayor frecuencia. En todo el consumo
normal ≤ P90, el MAE aumenta de 0,428459 a 0,481670 kWh (+12,42 %), el RMSE de
0,562947 a 0,637270 kWh (+13,20 %) y la sobreestimación de 63,06 % a 68,59 %.

Las diferencias pareadas (ponderada − MSE) son:

| Métrica | Diferencia media ± DE entre semillas |
|---|---:|
| MAE global | +0,029575 ± 0,005258 kWh |
| RMSE global | +0,029236 ± 0,004281 kWh |
| MAE > P90 | −0,155994 ± 0,024025 kWh |
| RMSE > P90 | −0,137469 ± 0,024754 kWh |
| RMSE > P95 | −0,169294 ± 0,032933 kWh |
| RMSE bajo | +0,074039 ± 0,018114 kWh |

## Resultados por semilla

| Semilla | Pérdida | MAE global | RMSE global | MAE > P90 | RMSE > P90 | Sesgo > P90 | Subestimación > P90 (%) |
|---|---|---:|---:|---:|---:|---:|---:|
| 2026 | MSE | 0,507243 | 0,687596 | 1,085770 | 1,285836 | −1,058043 | 95,0091 |
| 2026 | Ponderada | 0,531260 | 0,714928 | 0,956433 | 1,176595 | −0,896812 | 90,1911 |
| 2027 | MSE | 0,503443 | 0,685954 | 1,120689 | 1,318856 | −1,101305 | 95,8326 |
| 2027 | Ponderada | 0,533681 | 0,712191 | 0,944714 | 1,163384 | −0,881750 | 89,6310 |
| 2028 | MSE | 0,501299 | 0,685707 | 1,084684 | 1,285923 | −1,056321 | 94,7290 |
| 2028 | Ponderada | 0,535769 | 0,719846 | 0,922014 | 1,138229 | −0,840616 | 87,7697 |

| Semilla | Pérdida | RMSE > P95 | Sesgo > P95 | Subestimación > P95 (%) | MAE bajo | RMSE bajo |
|---|---|---:|---:|---:|---:|---:|
| 2026 | MSE | 1,693789 | −1,513942 | 98,9792 | 0,362235 | 0,496968 |
| 2026 | Ponderada | 1,562520 | −1,345413 | 96,6109 | 0,387414 | 0,550191 |
| 2027 | MSE | 1,736929 | −1,563283 | 99,2242 | 0,353896 | 0,485824 |
| 2027 | Ponderada | 1,549056 | −1,328896 | 96,8763 | 0,421677 | 0,572049 |
| 2028 | MSE | 1,692460 | −1,511794 | 98,9996 | 0,337262 | 0,479381 |
| 2028 | Ponderada | 1,503721 | −1,271830 | 95,4880 | 0,396254 | 0,562049 |

En **3/3 semillas** mejoran MAE/RMSE y sesgo de P90/P95 y baja su frecuencia de
subestimación. También en **3/3** empeoran MAE/RMSE globales y bajos. El aumento
de RMSE global va de +3,82 % a +4,98 %: ninguna corrida satisface el margen del
1 %. La ganancia en picos y el costo general son consistentes en esta muestra.

Los mejores checkpoints ponderados se alcanzaron en épocas **8, 3 y 5**;
se ejecutaron 14, 9 y 11 épocas respectivamente. Los controles MSE habían
seleccionado 5, 3 y 5. Los fits nuevos tardaron 7,21, 5,15 y 6,22 segundos en
este equipo, sin contar importación, datos, inferencia, guardado ni figuras.

## Interpretación del compromiso

El RMSE > P90 disminuye **10,60 %**, el MAE **14,22 %**, y la subestimación
**5,9930 puntos porcentuales**. Para P95, el RMSE baja **9,91 %** y el MAE
**13,19 %**. Aun así, casi nueve de cada diez pares > P90 y más del 96 % de
los > P95 siguen subestimados. La modificación atenúa el problema, no lo resuelve.

El mayor peso de objetivos altos es compatible con el desplazamiento observado
de las predicciones hacia arriba. El sesgo global pasa de +0,014694 a
+0,131828 kWh por hora. Esto perjudica la precisión de los valores normales,
que representan **88,70 %** de los pares de validación. La descomposición del
cambio de MAE global es:

- Consumos ≤ P90: contribuyen **+0,047199 kWh** al cambio global.
- Consumos > P90: contribuyen **−0,017624 kWh**.
- Total: **+0,029575 kWh** de MAE global.

Esta suma usa diferencias de MAE ponderadas por la proporción de pares de cada
grupo disjunto; no suma RMSE ni porcentajes de mejora de estratos diferentes.
El RMSE bajo aumenta 15,19 % y el MAE bajo 14,42 %.

El RMSE medio global por horizonte empeora en **24 de 24 horizontes**. H1:
0,598941 → 0,615374 kWh (+2,74 %). H24: 0,696472 → 0,723641 kWh (+3,90 %).
Los deterioros relativos por horizonte abarcan aproximadamente +2,74 % a
+5,68 %. Las bandas de dispersión de las figuras no son una prueba formal de
significancia. La validación ya fue utilizada para desarrollo y selección en
fases anteriores; tres semillas no equivalen a tres períodos independientes.
Las ventanas también se solapan y una misma hora real aparece en varios
horizontes. No se extrapolan estos resultados a otras viviendas o estaciones.

Para BEKANTOR Energy, reducir subestimación podría ser útil si su costo energético
fuera suficientemente asimétrico, pero aquí aumenta el error y el sesgo general.
No se evaluaron costos, baterías ni A*: no hay evidencia para afirmar que el
compromiso reduzca la factura o mejore la autonomía. Con la evidencia disponible,
**mantener MSE convencional es la decisión técnica recomendada**. Descartar este
reemplazo no demuestra que toda función ponderada sea inútil; sólo se probó
esta configuración 1/2 y no se propone ni ejecuta otra búsqueda en esta fase.

## Archivos, figuras y reproducción

Código: `forecasting/weighted_loss.py`, `weighted_experiment.py`,
`weighted_plots.py`; se extendieron el entrenador con una pérdida opcional y
el agregador existente con nombres de variantes parametrizables. El valor
predeterminado del entrenador continúa siendo MSE. Configuración fija en
`configs/mlp.weighted-mse.json`. No se duplicó el procesamiento de datos,
ventanas, calendario, normalización ni el adaptador predictivo.

Cada modelo nuevo se guarda en
`models/phase2_3/weighted-mse-20261022/seed_<semilla>/weighted_mse/`, con checkpoint
serializable, historial, métricas, predicciones de validación comprimidas y
bundle de inferencia. `bundle/training_objective.json` conserva la procedencia
de la pérdida. Los controles se referencian mediante `baseline_sources.json`;
no se duplican ni reemplazan sus archivos grandes.

- [Métricas completas por semilla](../experiments/phase2_3/weighted-mse-20261022/metrics_by_seed.csv).
- [Media, DE, mínimo y máximo](../experiments/phase2_3/weighted-mse-20261022/metrics_aggregate.csv).
- [Diferencias pareadas](../experiments/phase2_3/weighted-mse-20261022/paired_deltas.csv).
- [Cambios globales por semilla](../experiments/phase2_3/weighted-mse-20261022/global_tradeoffs.csv).
- [Comprobación del margen operativo](../experiments/phase2_3/weighted-mse-20261022/global_guardrail.json).
- [Verificación de métricas, checkpoints y hashes](../experiments/phase2_3/weighted-mse-20261022/verification.json).

![Comparación pareada de RMSE global, alto y bajo](../experiments/phase2_3/weighted-mse-20261022/plots/paired_seeds.png)

Se generaron e inspeccionaron visualmente siete figuras reproducibles:
[global](../experiments/phase2_3/weighted-mse-20261022/plots/global_comparison.png),
[P90/P95](../experiments/phase2_3/weighted-mse-20261022/plots/peak_comparison.png),
[consumo bajo](../experiments/phase2_3/weighted-mse-20261022/plots/low_consumption.png),
[horizonte](../experiments/phase2_3/weighted-mse-20261022/plots/error_by_horizon.png),
[hora](../experiments/phase2_3/weighted-mse-20261022/plots/error_by_hour.png),
[semillas](../experiments/phase2_3/weighted-mse-20261022/plots/paired_seeds.png) y
[aprendizaje](../experiments/phase2_3/weighted-mse-20261022/plots/learning_curves.png).

**63 pruebas aprobadas**: 59 de regresión, calendario y métricas (incluyendo tres
nuevas unitarias), y cuatro nuevas de TensorFlow. Estas últimas verifican
pesos por objetivo y gradientes, unidades y borde estricto, mismos pesos
iniciales/Adam, serialización de la pérdida, tres semillas sintéticas repetidas,
guardado/carga y acceso sólo a train/validation. No se ejecutó el evaluador
histórico de test, tampoco con datos sintéticos. Los avisos de deprecación
Keras/NumPy al guardar no impidieron serialización, carga ni determinismo.

La verificación independiente recalculó todas las métricas desde las matrices
guardadas, comprobó los objetivos/orígenes de los seis modelos, la elección del
checkpoint por RMSE no ponderado y la conservación de **76 archivos anteriores
protegidos**. Se conservaron los hashes del código de entrenamiento; la
verificación registra por separado el código final de renderizado de figuras.
Los modelos y resultados grandes continúan ignorados por Git.

Desde la raíz del proyecto y con el entorno local y los artefactos anteriores:

```powershell
$env:TEMP = Join-Path (Get-Location) 'experiments/tmp'
$env:TMP = $env:TEMP
New-Item -ItemType Directory -Force -Path $env:TEMP | Out-Null
.\.venv\Scripts\python.exe -u -m bekantor_energy compare-weighted-mse --config configs/mlp.weighted-mse.json --run-dir experiments/phase2_3/weighted-repeat-01 --model-dir models/phase2_3/weighted-repeat-01
.\.venv\Scripts\python.exe scripts/verify_weighted_run.py --run-dir experiments/phase2_3/weighted-repeat-01 --model-dir models/phase2_3/weighted-repeat-01
```

Las rutas nuevas evitan sobrescribir resultados. No requiere GPU ni nuevas
dependencias; se mantiene `requirements-ml-lock.txt`. Para repetir las 63
pruebas manteniendo excluido el evaluador histórico de test:

```powershell
.\.venv\Scripts\python.exe -c "import unittest; loader=unittest.TestLoader(); names=('test_acquisition.py','test_core.py','test_data.py','test_pipeline.py','test_forecasting.py','test_calendar.py','test_calendar_tensorflow.py','test_weighted.py','test_weighted_tensorflow.py'); suite=unittest.TestSuite(loader.discover('tests',pattern=n) for n in names); result=unittest.TextTestRunner(verbosity=2).run(suite); raise SystemExit(not result.wasSuccessful())"
```

No se probaron otras ponderaciones ni se modificó el modelo de referencia.
No se abrió test, no se implementó A*, no se avanzó a otro experimento y no se
hicieron commits ni push.

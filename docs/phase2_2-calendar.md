# Fase 2.2 — Experimento 1: calendario conocido

**Resultado:** el calendario mejora el RMSE de validación en las tres semillas,
con una reducción media del **1,14 % global** y del **5,63 % en consumo > P90 de
train**. El MAE global mejora sólo un 0,39 %. La subestimación de picos sigue
siendo considerable: 95,19 % para P90 y 99,07 % para P95. La mejora es útil y
consistente en RMSE, pero no resuelve los extremos ni mejora todas las horas.

Corrida: `calendar-20261022`. Seis entrenamientos reales en CPU, sin abrir ni
evaluar test. Los modelos y experimentos originales se conservaron.

## Comparación controlada

| Propiedad | Original | Con calendario |
|---|---|---|
| Historia de consumo | 168 horas | Las mismas 168 horas |
| Variables futuras conocidas | Ninguna | Hora, día de semana y mes de cada una de las 24 horas |
| Codificación de calendario | — | Seno/coseno: períodos 24, 7 y 12 |
| Tamaño de entrada | 168 | 312 = 168 + 24 × 6 |
| Capas ocultas | 64–32, ReLU | 64–32, ReLU |
| Salida | 24 valores no negativos, softplus | Igual |
| Parámetros | 13.688 | 22.904 |
| Entrenamiento | Adam, LR 0,001, MSE, batch 128 | Igual |
| Regularización explícita | Sin dropout ni L2 | Igual |
| Parada y checkpoint | Mínimo `val_rmse`, paciencia 6, máximo 40 épocas | Igual |
| Semillas | 2026, 2027, 2028 | Las mismas |

No se agregaron capas ni se modificó la pérdida. La entrada más amplia sí
aumenta los pesos de la primera capa: la comparación mide el efecto práctico
de incorporar calendario, no una ablación con capacidad total igualada.
Cada variante reinicia la misma semilla y utiliza el mismo orden de barajado
de ventanas; las matrices de pesos tienen tamaños diferentes.

Se usa el reloj ingenuo del dataset UCI, sin asignarle otra zona horaria.
Lunes=0 y enero=0. Para cada coordenada `v` de período `P`, la codificación es
`sin(2πv/P), cos(2πv/P)`. El mes trata los doce meses como posiciones angulares
uniformes. No se incorporan meteorología, festivos ni mediciones futuras.

## Datos y prevención de fugas

- Train: 2006-12-16 17:00 a 2009-09-20 12:00; 20.924 ventanas admisibles.
- Validación: 2009-09-20 13:00 a 2010-04-24 16:00; 4.478 ventanas y **107.472
  pares origen/horizonte por modelo**. Mismos objetivos y orígenes en las seis corridas.
- Ventanas rodantes con paso de una hora: los consumos anteriores al origen
  pueden incluir horas de validación que ya serían conocidas en ese momento.
  Los 24 objetivos permanecen dentro de su partición.
- Se rechazan ventanas con entradas desconocidas u objetivos no observados
  originalmente; no se completan objetivos con información futura.
- Normalizador ajustado a 24.022 horas únicas observadas de train: media
  1,0851613590 kWh y escala 0,9278005350 kWh. Sólo se normaliza consumo; el
  calendario conserva su rango natural. Predicciones y métricas se invierten a kWh.
- Umbrales predefinidos exclusivamente con esas horas de train: **P90 =
  2,3913966667 kWh**, **P95 = 2,9418633333 kWh**, cuantiles lineales.
  Consumo elevado significa `real > umbral`, estrictamente.
- P90 agrupa 12.142 pares de validación; P95, 4.898. Se pondera igual cada
  par origen/horizonte. Una hora real puede aparecer en varios horizontes.
- Sesgo = predicho − real; subestimación = porcentaje de pares con predicho < real.
  Son estratos definidos por el consumo real, disponibles para diagnóstico
  retrospectivo; el modelo no recibe esa clasificación futura.

## Resultados agregados

Media ± desviación estándar muestral entre tres semillas (`ddof=1`). Son medias
de métricas por modelo, no métricas de un ensemble ni intervalos de confianza.
MAE, RMSE y sesgo se expresan en **kWh por intervalo de una hora**.

| Métrica de validación | Original | Calendario |
|---|---:|---:|
| MAE global | 0,505988 ± 0,001614 | 0,503995 ± 0,003010 |
| RMSE global | 0,694327 ± 0,001102 | 0,686419 ± 0,001027 |
| Sesgo global | −0,015025 ± 0,015461 | +0,014694 ± 0,014274 |
| Subestimación global (%) | 44,4175 ± 1,0429 | 43,5248 ± 1,0271 |
| MAE > P90 | 1,178549 ± 0,020841 | 1,097048 ± 0,020481 |
| RMSE > P90 | 1,374218 ± 0,016416 | 1,296872 ± 0,019039 |
| Sesgo > P90 | −1,159704 ± 0,023363 | −1,071890 ± 0,025489 |
| Subestimación > P90 (%) | 96,5080 ± 0,2722 | 95,1902 ± 0,5737 |
| MAE > P95 | 1,628003 ± 0,020240 | 1,533482 ± 0,028880 |
| RMSE > P95 | 1,799693 ± 0,016462 | 1,707726 ± 0,025299 |
| Sesgo > P95 | −1,624402 ± 0,021638 | −1,529673 ± 0,029127 |
| Subestimación > P95 (%) | 99,2174 ± 0,1161 | 99,0676 ± 0,1359 |

La diferencia pareada de RMSE global (calendario − original) es
**−0,007907 ± 0,000318 kWh**. En P90 es **−0,077346 ± 0,019136 kWh**.
Para MAE global, la diferencia es −0,001993 ± 0,002536 kWh: el beneficio es
pequeño y su tamaño varía más entre semillas.

## Resultados por semilla

| Semilla | Variante | MAE global | RMSE global | MAE > P90 | RMSE > P90 | Sesgo > P90 | Subestimación > P90 (%) |
|---|---|---:|---:|---:|---:|---:|---:|
| 2026 | Original | 0,507486 | 0,695581 | 1,193351 | 1,385011 | −1,176407 | 96,7715 |
| 2026 | Calendario | 0,507243 | 0,687596 | 1,085770 | 1,285836 | −1,058043 | 95,0091 |
| 2027 | Original | 0,504279 | 0,693512 | 1,187581 | 1,382316 | −1,169698 | 96,5245 |
| 2027 | Calendario | 0,503443 | 0,685954 | 1,120689 | 1,318856 | −1,101305 | 95,8326 |
| 2028 | Original | 0,506200 | 0,693886 | 1,154715 | 1,355327 | −1,133006 | 96,2280 |
| 2028 | Calendario | 0,501299 | 0,685707 | 1,084684 | 1,285923 | −1,056321 | 94,7290 |

MAE y RMSE globales, y MAE/RMSE/sesgo de P90 y P95, mejoran en las tres semillas.
La frecuencia de subestimación P90 disminuye en las tres; para P95 sólo en dos
de tres: en 2027 aumenta 0,1021 puntos porcentuales. La subestimación global
tampoco disminuye en todas las semillas. No se selecciona la mejor semilla.

| Semilla | Mejor época original / ejecutadas | Mejor época calendario / ejecutadas | Tiempo fit original / calendario (s) |
|---|---:|---:|---:|
| 2026 | 8 / 14 | 5 / 11 | 8,03 / 6,59 |
| 2027 | 8 / 14 | 3 / 9 | 8,38 / 5,71 |
| 2028 | 7 / 13 | 5 / 11 | 7,17 / 6,60 |

Los tiempos corresponden sólo a `fit`, sin importación, preparación,
predicción, guardado ni figuras. No son un benchmark controlado de velocidad.

## Dónde mejora y dónde persiste el problema

El RMSE medio mejora en **23 de 24 horizontes**. H1 empeora de 0,593226 a
0,598941 kWh (+0,96 %); H24 mejora de 0,705356 a 0,696472 kWh (−1,26 %).
Las mejoras horarias más claras están a las 17:00 (−4,48 %), 18:00 (−4,09 %),
15:00 (−3,52 %), 14:00 (−3,32 %) y 07:00 (−3,23 %).
Hay deterioros, como 02:00 (0,312942 → 0,335387 kWh, +7,17 %) y 13:00
(+1,47 %). A las 20:00, una de las horas más difíciles, la mejora es mínima
(0,958577 → 0,957257 kWh). No es una mejora uniforme.

En P90, el MAE cae 6,92 %, el RMSE 5,63 % y el sesgo negativo se reduce
0,087814 kWh. La frecuencia de subestimación baja sólo **1,3177 puntos
porcentuales**. En P95 el RMSE cae 5,11 %, pero casi todos los extremos siguen
subestimados. El sesgo global cambia de signo, hacia una pequeña sobreestimación;
el calendario no justifica una corrección general al alza.

La hipótesis compatible con los resultados es que el calendario aporta contexto
para la forma esperada del perfil diario y semanal. No se aisló el aporte de
hora, día y mes, ni se demostró la causa de los picos. Son plausibles actividades
domésticas no observadas, pero no se dispone de información para atribuirlas.
Condicionar por consumo real elevado tampoco demuestra por sí solo falta de
calibración: este diagnóstico mide el comportamiento en ese estrato.

## Figuras y artefactos

Las seis figuras fueron inspeccionadas visualmente. Las bandas y barras de
error representan dispersión entre semillas, no incertidumbre predictiva.

![Comparación pareada por semilla](../experiments/phase2_2/calendar-20261022/plots/paired_seeds.png)

![Comparación en consumo elevado](../experiments/phase2_2/calendar-20261022/plots/peak_comparison.png)

- [MAE/RMSE globales](../experiments/phase2_2/calendar-20261022/plots/global_comparison.png).
- [MAE/RMSE/sesgo por hora](../experiments/phase2_2/calendar-20261022/plots/error_by_hour.png).
- [MAE/RMSE/sesgo por horizonte](../experiments/phase2_2/calendar-20261022/plots/error_by_horizon.png).
- [Curvas de aprendizaje por semilla](../experiments/phase2_2/calendar-20261022/plots/learning_curves.png).
- [Métricas por semilla, incluidos P95, hora y horizonte](../experiments/phase2_2/calendar-20261022/metrics_by_seed.csv).
- [Métricas agregadas, DE y rango](../experiments/phase2_2/calendar-20261022/metrics_aggregate.csv).
- [Diferencias pareadas](../experiments/phase2_2/calendar-20261022/paired_deltas.csv).
- [Comprobación de artefactos y hashes](../experiments/phase2_2/calendar-20261022/verification.json).

Cada carpeta `models/phase2_2/calendar-20261022/seed_<semilla>/<variante>/`
contiene `best.keras`, `training.json`, `epochs.csv`, predicciones de validación
comprimidas y `bundle/{model.keras,normalization.json,predictor.json}`.
Los seis bundles son cargables por el mismo adaptador; se conservó la carga de
bundles anteriores sin `feature_mode`. Los archivos grandes y resultados
siguen excluidos de Git. No se cambió el bundle seleccionado de la Fase 2.

## Verificación y reproducción en Windows

Se aprobaron **56 pruebas**: 45 regresiones sin evaluación de test y 11 nuevas.
Las nuevas cubren ciclos, alineación, ausencia de fuga, soporte común,
normalización, umbrales train, métricas, agregación, arquitectura fija,
compatibilidad de inicialización, seis fits sintéticos CPU y guardado/carga.
El entorno conserva las dependencias fijadas en `requirements-ml-lock.txt`.
Keras emitió avisos de deprecación de la conversión NumPy al guardar; no
impidieron los checkpoints ni su carga y verificación.

La verificación independiente recalculó las métricas desde las seis matrices
guardadas, comprobó objetivos/orígenes idénticos y los hashes de los 16 archivos
originales protegidos. La corrida de referencia con semilla 2026 reproduce
**exactamente** MAE, RMSE y sesgo de validación originales.

Desde la raíz, con el entorno instalado y las particiones originales presentes:

```powershell
$env:TEMP = Join-Path (Get-Location) 'experiments/tmp'
$env:TMP = $env:TEMP
New-Item -ItemType Directory -Force -Path $env:TEMP | Out-Null
.\.venv\Scripts\python.exe -u -m bekantor_energy compare-calendar --config configs/mlp.calendar-comparison.json --run-dir experiments/phase2_2/calendar-repeat-01 --model-dir models/phase2_2/calendar-repeat-01
.\.venv\Scripts\python.exe scripts/verify_calendar_run.py --run-dir experiments/phase2_2/calendar-repeat-01 --model-dir models/phase2_2/calendar-repeat-01
```

Las rutas de salida deben ser nuevas. Se valida que el procedimiento coincida
con la referencia original y que los archivos train/validation mantengan sus
hashes de Fase 1. No hay comando de evaluación de test en este procedimiento.

Para repetir las 56 pruebas sin ejecutar el evaluador de test heredado:

```powershell
.\.venv\Scripts\python.exe -c "import unittest; loader=unittest.TestLoader(); names=('test_acquisition.py','test_core.py','test_data.py','test_pipeline.py','test_forecasting.py','test_calendar.py','test_calendar_tensorflow.py'); suite=unittest.TestSuite(loader.discover('tests',pattern=n) for n in names); result=unittest.TextTestRunner(verbosity=2).run(suite); raise SystemExit(not result.wasSuccessful())"
```

## Conclusión y propuesta para revisión humana

**Conviene conservar el calendario como candidato de referencia para el
siguiente experimento.** Mejora de forma consistente el RMSE en esta validación,
especialmente en consumos altos; la mejora de MAE global es modesta. Esta
validación ya se utilizó en etapas anteriores para seleccionar y diagnosticar
modelos: es evidencia de desarrollo, no una evaluación independiente nueva.
Tres semillas sobre el mismo período no equivalen a tres períodos temporales
independientes. No se puede concluir que la mejora se mantenga en otras
viviendas o estaciones, ni estimar ahorros o desempeño de baterías.

Para el próximo experimento, propongo comparar **MSE frente a MSE ponderada**
sobre la variante con calendario, manteniendo todo lo demás fijo y las mismas
tres semillas. Una única configuración inicial a revisar sería peso 2 para
objetivos > P90 de train y peso 1 para el resto, con normalización del peso
medio. Fijar antes de ejecutar la regla de checkpoint y el compromiso aceptable
entre error global y error alto; examinar también sobreestimación de consumo
bajo. El objetivo sería reducir la magnitud de los errores extremos, no forzar
arbitrariamente un 50 % de subestimación en un estrato definido por el valor real.

**Propuesta solamente:** no se implementó MSE ponderada, no se ejecutó el
Experimento 2 ni se integró A*. No se hicieron commits ni push.

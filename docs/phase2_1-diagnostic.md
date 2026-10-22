# BEKANTOR Energy — Diagnóstico predictivo de validación

La MLP pequeña mejora el promedio, pero comprime los extremos: sobreestima consumos bajos y subestima picos. En validación, los consumos reales superiores a **2,391 kWh por hora** representan **11,3 % de los pares** y **44,8 % del error cuadrático**. Conviene estudiar calendario y sensibilidad a picos antes de ampliar la red.

Se analizaron exclusivamente predicciones guardadas de validación: **20/09/2009–24/04/2010**, 4.478 orígenes de 24 horas, 107.472 pares y 4.754 horas objetivo únicas. No hubo entrenamiento, inferencia nueva, cambios de arquitectura, A*, commits ni push; no se abrió test.

## Comparación de validación

| Estrategia | MAE (kWh) | RMSE (kWh) | Sesgo (kWh) | MAE en consumo alto |
|---|---:|---:|---:|---:|
| MLP pequeña | **0,5075** | **0,6956** | −0,0203 | 1,1934 |
| Persistencia 24 h | 0,6651 | 0,9580 | +0,0051 | 1,3377 |
| Persistencia 168 h | 0,5993 | 0,8779 | +0,0037 | **1,1663** |

La MLP reduce RMSE **27,4 %** frente a la diaria y **20,8 %** frente a la semanal. Mejora RMSE en los 24 horizontes, las 24 horas y los 7 días, comparando las mismas ventanas. Eso no significa que gane cada pronóstico: en consumo alto la semanal tiene un MAE ligeramente menor, aunque la MLP conserva mejor RMSE (1,3850 frente a 1,4609 kWh).

## Dónde se concentran los errores

- **Horizonte:** RMSE **0,5957 en +1 h → 0,7088 en +24 h**. Desde aproximadamente +4 h oscila cerca de 0,70; no crece de manera monotónica. Como la salida es directa, no corresponde atribuirlo a propagación recursiva.
- **Hora objetivo:** RMSE mínimo alrededor de las 03:00–04:00 (**0,263–0,264**) y máximo a las 20:00 (**0,960**). Las 17:00–21:00 reúnen **36,1 % del SSE**. A las 07:00 hay sesgo **−0,273**; a las 05:00–06:00 cambia a **+0,161/+0,177 kWh**.
- **Día objetivo:** RMSE martes **0,566**, sábado **0,817**, domingo **0,838**. Fines de semana: **0,827**, frente a **0,640** entre semana. El sábado se subestima **−0,123**; el domingo combina error alto con sesgo casi nulo (**+0,018**). En celdas comparables de hora×rango real (99,9 % de pares laborables y 99,0 % de fin de semana), **83,6 % de la brecha de MSE** corresponde contablemente a distinta mezcla. No es una atribución causal al calendario.
- **Consumo alto:** por encima del P90 de las horas observadas de train (**2,391 kWh**), consumo real medio **2,995**, predicción media **1,819**, sesgo **−1,176** y subestimación en **96,8 %** de los pares. Por encima de P95 (**2,942 kWh**), el sesgo llega a **−1,641**.
- **Compensación de sesgos:** en consumos ≤P50 train (**0,763 kWh**) se sobreestima **+0,399 kWh**. Globalmente se sobreestima el 55,3 % de las veces, pero la subestimación media condicional es mayor: **0,590** frente a **0,441 kWh** de sobreestimación. Un sesgo global pequeño no equivale a errores pequeños.

Los grupos de consumo real son diagnósticos **ex post**: no se conocen al emitir el pronóstico. No prueban por sí solos una mala calibración corregible. De hecho, el decil de predicción más alto sobreestima en promedio **+0,159 kWh**; no conviene sumar una corrección uniforme al pronóstico.

## Causas posibles y evidencia disponible

| Factor | Lectura del diagnóstico |
|---|---|
| Variables insuficientes | Sólo hay 168 consumos, sin calendario explícito, clima ni ocupación. Los patrones por hora y fin de semana respaldan probar calendario conocido; todavía no demuestran que mejore. |
| Arquitectura | Las MLP casi empatan en RMSE de validación: 0,69558 / 0,69575 / 0,69621. Aumentar capacidad no aportó mejora en esta corrida. Una sola semilla limita la conclusión. |
| Regularización | No hay L2 ni dropout. La grande empeora después de época 2 y la pequeña después de época 8 mientras baja el error de entrenamiento: compatible con sobreajuste, no prueba de que regularizar resuelva los picos. |
| Ventana y pérdida | 168 h cubren un ciclo semanal; no existe una ablación de lookback. Una estimación puntual con MSE puede suavizar picos inciertos. No debe concluirse que la pérdida “ignora” los picos: ya explican 44,8 % del SSE. |
| Distribución temporal | La media observada cambia de **1,085 train a 1,246 kWh validación (+14,8 %)**. Ponderar medias train por la mezcla de mes/hora de validación produce **1,284**: el calendario es compatible con la diferencia; no demuestra deriva permanente. RMSE mensual: septiembre parcial 0,513, enero 0,798. Clima y ocupación no están medidos. |

## Tres experimentos recomendados — sólo propuestas

Ordenados por utilidad esperada frente a su costo. **No se implementó ninguno.** Cada alternativa se contrastaría independientemente contra la referencia, manteniendo particiones, ventanas elegibles, normalización de train, semilla 2026, Adam 0,001, máximo 40 épocas y paciencia 6.

1. **Calendario conocido del horizonte. Utilidad esperada alta; costo bajo/moderado, un ajuste.** Mantener capas 64–32 y MSE; añadir seno/coseno de hora, día semanal y mes de las horas objetivo. Son datos conocidos de antemano. Revisar RMSE global y sesgos de 07:00, tarde y fines de semana. No usar clima futuro observado ni ocupación posterior.
2. **MSE ponderada por consumo alto. Utilidad potencial media/alta para reservas; costo bajo, un ajuste.** Conservar entradas y capas; peso fijo 2 por cada etiqueta train >P90 y 1 en las demás, normalizando por la media de pesos. Medir MAE/sesgo de picos, RMSE global y sobreestimación. Puede reducir subestimaciones a costa del promedio: acordar ese intercambio antes de ejecutarlo. No seleccionar usando tarifas o beneficios energéticos todavía no simulados.
3. **L2 ligera. Utilidad esperada moderada/baja; costo bajo, un ajuste.** Cambiar únicamente L2 a **0,0001**, sin ampliar capas ni sumar dropout. Comprobar validación y picos: una penalización excesiva podría aumentar el suavizado. No hay evidencia para iniciar una búsqueda extensa.

Los criterios de aceptación y la selección quedan bajo supervisión humana. No combinar automáticamente los cambios ni consultar test para elegirlos. Tras optimizar con esta validación, una futura estimación independiente requerirá otro bloque temporal reservado.

## Consecuencias para un futuro planificador con batería y A*

| Error predictivo | Posible consecuencia energética |
|---|---|
| Subestimar picos | Planificar poca carga o descargar antes del pico; mayor importación imprevista y menor reserva disponible. Los límites de SOC deben seguir imponiéndose con demanda real; el error no autoriza violarlos. |
| Sobreestimar consumos bajos | Cargar de red innecesariamente, reservar batería en exceso o dejar menos capacidad para solar posterior. Depende de tarifas y generación. |
| Confundir horario o día | Trasladar energía a horas distintas de las necesarias. El sesgo diario casi nulo puede ocultar un costo alto con tarifas horarias y límites de potencia. |
| Errores en horizontes lejanos | Mayor incertidumbre sobre la reserva futura; convendría estudiar replanteo periódico y márgenes de incertidumbre en otra fase. |

A* puede encontrar el óptimo del grafo definido por predicciones y aun así producir un plan inadecuado con demanda diferente. Mejorar la búsqueda no corrige automáticamente el modelo predictivo. Aquí no se estimaron ahorros, costos ni probabilidades de agotar batería.

## Gráficos y reproducibilidad

- [Horizonte y sesgo](../experiments/phase2_1/diagnostic/horizon.png).
- [Hora del día](../experiments/phase2_1/diagnostic/hour_of_day.png).
- [Día de la semana](../experiments/phase2_1/diagnostic/weekday.png).
- [Consumo alto y sesgo](../experiments/phase2_1/diagnostic/high_load.png).
- [Calibración y variación mensual](../experiments/phase2_1/diagnostic/calibration_month.png).

Se inspeccionaron también los gráficos previos de pérdidas, horizonte y primer pronóstico de validación. Los valores detallados de los 24 horizontes y cada segmento están en los CSV de `experiments/phase2_1/diagnostic/`; `provenance.json` registra fuentes y hashes.

Reproducir **sólo el diagnóstico**, sin cargar TensorFlow ni ejecutar el modelo:

```powershell
.\.venv\Scripts\python.exe scripts/diagnose_validation.py
```

Se verificaron duplicados, offsets 1–24 h, correspondencia de etiquetas con horas observadas de validación y reconciliación con las métricas guardadas. Las fuentes quedaron sin cambios. Las ventanas se solapan: no son 107.472 observaciones independientes. Reponderar cada hora objetivo por igual apenas cambia RMSE a **0,69425**. Hora/día usan el reloj UCI de zona no especificada.

Las fórmulas y la penalización L2 se contrastaron con la documentación oficial de [pérdidas Keras](https://keras.io/api/losses/regression_losses/) y [regularizadores](https://keras.io/api/layers/regularizers/); sus beneficios aquí son hipótesis, no resultados demostrados.

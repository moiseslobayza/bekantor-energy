# Fase 4.3 — Diseño previo de los modos

Se fija antes de ejecutar resultados nuevos. No hay barrido ni ajuste posterior
de ponderaciones, arquitectura, modelo, discretización o restricciones.

- Un solo núcleo A* y un solo grafo físico; objetivos intercambiables.
- Económico: `min C`, costo bruto de importación, como la referencia anterior.
- Autonomía: `min lex(G, C)`, primero importación en kWh, luego costo SIM.
  No se usa `G + epsilon*C`. Se calculan las ecuaciones de importación desde
  demanda, solar, variación de energía almacenada y eficiencias mediante
  fracciones exactas de sus parámetros binarios, antes de redondear los flujos.
  Así se conservan empates físicos además de la prioridad primaria. Se usa h=0
  vectorial por prudencia. Es UCS
  lexicográfico, caso particular correcto de A*. No se promete menos expansión.
- Equilibrado: `J = w*C/C_ref + (1-w)*G/G_ref`, adimensional.
  Se fijan `w=0,5`, `C_ref=1 SIM` y `G_ref=1 kWh`. La tasa marginal implícita
  es 1 SIM por kWh adicional de red. Las escalas son explícitas, constantes y
  no se ajustan con la nueva validación. 50 % no implica neutralidad universal;
  es un compromiso ilustrativo relativamente fuerte contra importación dadas
  las tarifas 0,08–0,40 SIM/kWh. No es una tarifa facturada.
- Para Económico se conserva la heurística relajada original. Equilibrado
  usa la misma cota física inferior de importación, ponderada por el menor
  precio futuro y los coeficientes positivos del objetivo. Se contrasta contra
  UCS y enumeración en grafos pequeños. Exportación remunerada y cortes
  conocidos siguen fuera del planificador; no se introduce Resiliencia.
- Se conserva `EnergyPlan.predicted_cost` en SIM. El valor del objetivo, sus
  unidades y la importación prevista se exponen por separado.
- Se repiten los ocho casos continuos de Fase 4.2: dos bloques disjuntos de
  siete días (21–27/09/2009 y 24–30/01/2010), perfiles Córdoba PVGIS enero/julio
  de ocho días, tarifa plana/variable, batería 5 kWh, SOC inicial 50 %, reserva
  20 %, grilla 0,25 kWh. No hay reinicio a medianoche.
- Las tres políticas A* usan la misma MLP calendario MSE 64–32, semilla 2026,
  congelada, 168 horas de historia, horizonte 24 h y replanificación horaria.
  El proveedor sólo entrega emisiones causales archivadas; no tiene demanda
  realizada futura. El reactivo usa desbalance presente idealizado dentro de
  la hora. La solar del escenario se conoce idealmente en el horizonte.
- No se añade la condición terminal experimental 50 %: todos los modos
  mantienen la reserva terminal original 20 %. Se informan saldo inicial/final
  y el mismo costo ajustado contable `C + v*(E0-ET)`, v principal 0,1425
  SIM/kWh almacenado y sensibilidad original. Este ajuste no es factura ni
  objetivo de los tres modos. Se mide SOC mínimo, medio y final por separado.
- Cada caso incluye reactivo y los tres modos, con las mismas condiciones,
  simulador, realización física y pronósticos. Se comprueba cada plan completo
  antes de ejecutar su primera acción con demanda realizada. Los fallos y
  límites se registran sin fallback; agregados con cobertura de cuatro vías.
- Se compara la trayectoria Económica contra la corrida congelada 4.2, para
  verificar que el refactor mantiene la referencia. No se modifican sus CSV,
  modelos, gráficos, verificación o manifiestos históricos. Se archivan los
  fuentes anteriores antes del cambio autorizado del núcleo y evaluador;
  sus hashes se distinguen de los fuentes actuales. No se falsea que el
  planificador activo conserve el mismo hash después de añadir los modos.
- Autonomía significa menos importación, no SOC alto ni capacidad de respaldo.
  No se presupone que supere al reactivo, cuyo control continuo difiere de las
  acciones discretas A*. La optimalidad de cada horizonte previsto no asegura
  dominancia realizada bajo errores de demanda y horizonte móvil.
- Se informan resultados por caso y escenario, dispersión descriptiva y
  desfavorables, costo bruto/ajustado, red, captación/recorte solar, SOC,
  energía no suministrada, violaciones, tiempos y nodos. Los ocho cruces no son
  ocho semanas independientes. UCI y PVGIS son escenarios combinados, no
  mediciones sincronizadas ni de la misma vivienda. No se afirman ahorros reales.
- No test, entrenamiento, hardware, Resiliencia, dashboard, Colab, commits,
  push ni fase posterior.

Configuración: `configs/energy.modes.json`. Antes de simular, cada corrida
persistirá este diseño operativo, parámetros y hashes en `design.json`.

## Corrección numérica antes de aceptar la implementación

La corrida preliminar `modes-20261022` se conserva íntegra. Una prueba sintética
adicional detectó que sumar con exactitud importaciones previamente redondeadas
no conservaba empates físicos: con dos demandas de 1 kWh, descarga eficiente
0,95, 0,5 kWh utilizables y tarifas 0,40/0,08, el plan elegía costo 0,366 SIM en
vez de 0,290 SIM, ambos con 1,525 kWh de importación física. Se corrigió la
evaluación racional de las ecuaciones, no los pesos ni condiciones, y se agregó
la prueba de regresión. La versión previa de este diseño y fuentes se conserva
en `experiments/phase4_3/preparation-20261022/preliminary-source/`.
Los resultados aceptados se ejecutarán en `modes-corrected-20261022`.

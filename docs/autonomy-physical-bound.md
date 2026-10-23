# Importación mínima y reserva: alcance del diagnóstico

## Balance y valoración terminal

En cada hora, L es demanda AC, S generación solar AC, G importación, C carga
de batería AC, D descarga AC y K recorte solar, todos en kWh. E es energía
almacenada antes de la hora. Sin exportación ni energía no suministrada:

```
S + G + D = L + C + K
E' = E + eta_c*C - D/eta_d
perdida = (1-eta_c)*C + (1/eta_d-1)*D
```

Al sumar el bloque:

```
G_total = L_total - S_total + K_total + perdida_total + ET - E0
```

Por tanto, para escenarios pareados:

```
deltaG = deltaK + deltaPerdida + deltaET
```

La identidad explica el balance, no separa causas independientes. Un error de
predicción puede aumentar el recorte; una grilla también puede hacerlo.
Su atribución requiere cambiar una sola entrada por comparación.

La valoración monetaria histórica se conserva:
`C_ajustado = C_bruto + 0,1425 SIM/kWh_almacenado * (E0-ET)`.
No es factura, ni es el objetivo del modo Autonomía.

Para sensibilidad de energía se informa `G_equivalente = G + lambda*(E0-ET)`:
lambda=0 no valora el stock; eta_d valora su energía AC entregable;
1/eta_c representa la energía AC necesaria para reponer ese stock desde red.
Estas dos últimas convenciones responden preguntas distintas y no deben
presentarse como importación real ni cambiar las acciones del experimento.
Si dos estrategias terminan con el mismo E, ninguna cambia su diferencia.

## Cota temporal independiente de A*

Transformar el almacenamiento a energía AC entregable:
`X = eta_d*(E-reserva)`, con capacidad `eta_d*(capacidad-reserva)`.
Un kWh AC solar cargado aporta `eta_c*eta_d` kWh entregables.

La red dirigida tiene un nodo por hora, origen y sumidero:

- Suministro inicial al nodo 0: `eta_d*(E0-reserva)`.
- Suministro solar al nodo t:
  `eta_c*eta_d*min(max(S_t-L_t,0), Pc*dt)`.
- Salida del nodo t a demanda:
  `min(max(L_t-S_t,0), Pd*dt)`.
- Arco t a t+1 con capacidad máxima de energía entregable almacenada.

Las capacidades de carga y descarga son distintas. La cronología impide usar
solar posterior para una demanda anterior. La energía solar directa ya se
descontó en el desequilibrio; no se vuelve a contar como carga gratuita.

Con red ilimitada y siempre disponible, cargar q desde red no puede reducir el
objetivo primario: esa energía puede entregar a cargas posteriores como máximo
eta_c*eta_d*q <= q. Sustituir esas entregas por red en sus horas necesita menos
o igual importación total, libera capacidad y respeta límites de potencia. Con
meta terminal limitada a la reserva, un remanente de esa carga tampoco es
obligatorio. Este argumento no minimiza costo monetario ni vale ante cortes,
límites de importación, eficiencias variables o requisitos terminales superiores.

Todo despacho físicamente factible sin carga de red induce un flujo de energía
entregada. Permitir descartar suministro no utilizado relaja el problema: el
flujo máximo F limita superiormente la descarga útil total. En consecuencia:

```
G_min >= suma(max(L_t-S_t,0)) - F
```

El algoritmo de flujo máximo utiliza capacidades reales, arcos residuales y
caminos aumentantes. El corte que separa los nodos alcanzables al terminar se
calcula sobre las capacidades originales; se verifica `flujo = capacidad_corte`
con tolerancia 1e-8 kWh. No usa precios, predicciones, grillas de SOC ni A*.
No es un nuevo controlador: es una referencia diagnóstica retrospectiva de
toda la semana, más informada que el oracle MPC de 24 horas.

Si el reactivo reproducido con el simulador alcanza la cota, su factibilidad
proporciona la otra desigualdad y certifica el mínimo para ese bloque. La cota
optimista no se interpreta como un plan factible por sí misma. Tampoco se
generaliza la igualdad observada a instalaciones fuera de estos supuestos.

## Discretización y control de realización

El reactivo solicita el excedente/déficit actual completo. El simulador limita
esa solicitud con la potencia, capacidad, reserva y eficiencias reales. Puede
terminar una hora en cualquier energía factible.

A* sólo admite destinos de una grilla fija, además de fronteras y estado inicial.
En el espacio físico continuo se puede cargar exactamente un excedente solar
pequeño; en la grilla ese destino puede no existir. Una carga mayor puede captar
ese solar y requerir además red. Esa acción puede ser óptima en el grafo aunque
sea inferior al despacho continuo. La prueba de regresión de dos horas conserva
este caso sin modificar el objetivo ni el planificador.

Los errores de predicción añaden otra diferencia. La reparación de ejecución
recorta solicitudes inviables, pero no inventa cargas/descargas no solicitadas.
Una solicitud insuficiente puede dejar demanda a la red o recortar solar aun
con energía o espacio disponibles. El reactivo utiliza medición instantánea
idealizada de promedios horarios; la MLP decide a partir del pronóstico en el
origen. El oracle separa esta fuente de error de la grilla.

En un mismo estado y horizonte, refinar una grilla anidada amplía los planes
factibles. No garantiza monotonicidad del replay MPC realizado con predicciones:
las acciones alteran los estados iniciales posteriores y los empates secundarios.

## Autonomía no es resiliencia

El modo actual resuelve `min_lex(sum G_t, sum precio_t*G_t)`, con una reserva
física mínima. SOC elevado no forma parte del objetivo. Gastar batería en
demanda actual puede minimizar G y dejar menos energía para una interrupción.

Resiliencia necesitaría evaluar explícitamente disponibilidad de red, cargas
críticas y energía no suministrada U. Por ejemplo, una futura formulación podría
priorizar `sum U_critica` o restricciones de servicio bajo escenarios de cortes,
además de costos o importación. No se implementa ni se evalúa esa formulación.
Tener un SOC alto en simulaciones sin cortes no demuestra horas de respaldo.

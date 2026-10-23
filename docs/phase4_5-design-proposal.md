# Fase 4.5 — Propuesta experimental de resiliencia

Estado: **diseño para revisión; sin implementación ni experimentos ejecutados**.
No se modifican código, parámetros, modelos ni resultados de las fases previas.

## Pregunta y criterio de atribución

Determinar si preparar la batería con A* y demanda prevista reduce energía
crítica no suministrada frente a políticas reactivas, y cuánto cuesta hacerlo.
La Fase 4.4 certificó un mínimo de importación sin cortes; esa conclusión no
resuelve el servicio durante una interrupción. Ahora la prioridad es el servicio
crítico, no minimizar importación ni maximizar SOC de forma aislada.

Una mejora frente al reactivo sin reserva no basta para atribuir valor a A*:
puede proceder simplemente de guardar más batería. Se compara también contra
reserva fija y se incorporan controles de reserva adaptativa y predicción simple.
No se exige que A* gane. Los criterios se fijan antes de observar resultados.

## Hipótesis

| ID | Hipótesis | Contraste que la distingue |
|---|---|---|
| H1 | La reserva operativa mejora servicio frente al autoconsumo sin reserva adicional, a cambio de importación/costo. | Reactivos sin/con reserva, condiciones pareadas y controles sin cortes. |
| H2 | Conocer la demanda futura permite preparar energía donde una reserva fija resulta insuficiente o excesiva. | A* MLP frente a reserva fija, por horario, solar y duración. |
| H3 | Parte del beneficio atribuido a A* puede explicarse por una reserva adaptativa, sin búsqueda. | Reactivo adaptativo con las mismas predicciones y criterio de preparación. |
| H4 | Los errores de demanda limitan el beneficio; la grilla puede introducir otra desventaja. | Oracle sólo de demanda; sensibilidad de grilla predefinida. |
| H5 | Los cortes nocturnos/largos y con poco solar son más exigentes; en cortes cortos puede no haber margen de mejora. | Interacciones, resultados por caso y porcentaje de servicio crítico completo. |
| H6 | Un SOC inicial de corte mayor explica preparación, pero no demuestra mejor respuesta durante el corte. | Registrar SOC al inicio; contrafactual diagnóstico con igual SOC y controlador de emergencia común. |

## Física y separación de reservas

Se conservan batería de 5 kWh, PV de 3 kWp, potencias 2,5 kW, eficiencias 0,95,
paso horario y red disponible sin límite de importación fuera de los cortes.
No hay exportación, degradación ni autodescarga. No se representa hardware de
transferencia/isla ni transitorios: se supone que el sistema puede operar aislado.
Son escenarios energéticos horarios, no una certificación de respaldo real.

El SOC mínimo del 20 % (1 kWh) es un **piso físico común**, incluso en cortes.
La reserva operativa fija propuesta es 50 % (2,5 kWh): se protege mientras hay
red y se libera durante el corte hasta el piso físico. No debe implementarse
como `min_soc=0.5`, pues el simulador actual impediría usarla durante la emergencia.
Sin reserva significa sin reserva operativa adicional; tampoco usa el último 20 %.

Todas las estrategias comparten límites, detección del corte, prioridades de
servicio y controlador local de emergencia. Durante un corte se atiende la
demanda crítica con solar y batería, se carga con excedente solar y no se
importa de red. Se desconecta toda carga no esencial en el ensayo principal.
Así se aísla la preparación de batería de diferencias de respuesta de emergencia.
La política de cargas no se optimiza de forma diferente para favorecer una estrategia.

El balance debe conservar demanda original, carga crítica/no crítica requerida,
energía servida, energía no suministrada y desconexión deliberada. Desconectar
cargas no borra su demanda ni permite presentar menor consumo como eficiencia.
No se simula recuperación posterior de tareas aplazadas: son cargas descartadas.

## Cargas esenciales: supuesto explícito

UCI no identifica cargas críticas. Se propone la descomposición sintética
`L_critica=0.40*L_UCI`, `L_no_critica=0.60*L_UCI`, sin inventar mediciones por
electrodoméstico. Para la MLP congelada se usa la misma fracción conocida sobre
la demanda prevista. El oracle utiliza la fracción de demanda realizada futura.
Ningún modelo se reentrena. Una extensión de sensibilidad, limitada al panel
diagnóstico, usa fracciones 0,20 y 0,60; no se selecciona una fracción ganadora.

## Matriz principal: 80 escenarios continuos

Se reutilizan las ocho condiciones de Fase 4.3: dos semanas de validación,
dos perfiles PVGIS cronológicos de Córdoba y tarifas plana/variable.
Cada replay abarca 168 horas, empieza con SOC 50 % y conserva SOC entre días.
No se reinicia la batería al comienzo del corte. La diferencia de SOC en ese
momento es una consecuencia de la preparación y debe medirse.

| Factor | Valores fijados |
|---|---|
| Bloques UCI | 21–27/09/2009 y 24–30/01/2010, sólo validación |
| Solar | Perfiles independientes enero/julio de Córdoba, ya congelados |
| Tarifas | Plana y variable existentes |
| Día de inicio del corte | Cuarto día del bloque, igual para todas las estrategias |
| Hora de inicio | 06:00, 14:00 y 22:00, reloj del escenario |
| Duración | 2, 6 y 12 horas consecutivas |
| Cortes por replay principal | Uno, imprevisto |
| Control | Un replay sin corte para cada una de las ocho condiciones |

Son `8*3*3=72` escenarios con corte y ocho sin corte. Hay al menos 72 horas
antes de la interrupción y más de dos días posteriores aun en el caso más largo.
Se informa servicio durante el corte, costo de preparación y recuperación, y
totales de la semana; no sólo una ventana favorable. Las fechas no se eligen
por errores del predictor, SOC esperado ni resultado económico.

Se preservan procedencia, timestamps y asociación cronológica del solar. UCI y
PVGIS siguen siendo series independientes combinadas en simulación. La previsión
solar se mantiene idealizada en esta fase para aislar demanda/control; no demuestra
robustez ante errores meteorológicos.

## Información disponible: ninguna estrategia anticipa cortes imprevistos

En el origen t sólo se observa disponibilidad de red actual/pasada. El evaluador
retiene el calendario completo y no lo pasa al planificador, ni como máscara
futura, identificador de escenario, duración restante o aviso implícito.
La restitución se conoce cuando sucede. No hay avisos anticipados en el ensayo
principal y el oracle de demanda tampoco recibe información futura de cortes.

La carga realizada sólo se utiliza para evaluación y realimentación local común;
el planificador MLP recibe sus predicciones causales congeladas, calendario y
tarifas conocidos. El oracle recibe las próximas 24 observaciones de validación,
incluida la cola necesaria, exclusivamente como referencia no desplegable.

Todos conocen el mismo supuesto genérico de preparación: un corte hipotético
de cuatro horas puede comenzar en cualquier momento. No se aprende ni calibra
con los 72 eventos de evaluación. Las duraciones realizadas 2/6/12 horas ponen
a prueba ese supuesto; los algoritmos nunca saben cuál les corresponde.

## Estrategias y objetivo propuesto de A*

| Estrategia principal | Preparación con red | Durante el corte |
|---|---|---|
| R0: reactivo sin reserva adicional | Autoconsumo; carga solar, sin carga intencional de red. | Emergencia común, piso 20 %. |
| R50: reactivo con reserva fija | Protege 50 %; repone con solar y, si falta, con red, respetando potencia. Descarga normalmente sólo por encima de 50 %. | Libera hasta 20 %, mismo servicio crítico. |
| P-MLP: A* predictivo | Horizonte 24 h, replanificación horaria, MLP calendario + MSE congelada. | Misma emergencia; no conoce duración restante. |
| P-Oracle: A* con demanda perfecta | Mismo objetivo, discretización y presupuesto; sólo mejora la demanda futura. | Misma emergencia y desconocimiento del calendario futuro de cortes. |

Se propone un objetivo nuevo y separado para este experimento; **no se cambia
la función de Autonomía ni la de los modos históricos**. Minimizar solamente
importación no expresa servicio crítico ante cortes.

Para un estado de energía E_h, calcular Phi_h(E_h): energía crítica no suministrada
si un corte hipotético de cuatro horas comenzara allí, usando demanda/solar
previstos y el controlador físico de emergencia común. Para h=1..20 caben las
cuatro horas en las 24 predichas; no se requieren observaciones más allá de la MLP.
Se excluye h=0 porque en ese origen ya se observó que la red está disponible.
La trayectoria prevista hasta cada inicio supone red disponible; no incluye el
calendario real del corte.

Objetivo propuesto:

```
min_lex ( (1/20)*sum_h=1..20 Phi_h(E_h), costo_electrico_previsto )
```

El primer componente está en kWh críticos y el segundo en SIM. Es un indicador
de preparación bajo un riesgo artificial uniforme condicionado a un corte de
cuatro horas; **no es una estimación calibrada de interrupciones reales ni la
energía no suministrada del calendario de evaluación**. No combina kWh con
dinero mediante un peso arbitrario. Los parámetros cuatro horas y pesos 1/20
se fijan antes de resultados. Con el mismo riesgo mínimo se prefiere menor costo.
El objetivo terminal físico sigue siendo el piso del 20 %; el saldo final se
reporta y valora por separado, sin ajustes retrospectivos al objetivo.

Se permite carga de red bajo las mismas restricciones físicas para las políticas
que la solicitan. Los límites originales son grilla 0,25 kWh, 10 s y 100.000
expansiones por búsqueda. Usar h=0 y comprobar optimalidad lexicográfica contra
UCS/enumeración pequeña antes del experimento. No tratar el límite de cómputo
como un plan óptimo ni sustituir fallos silenciosamente.

Este diseño concentra la comparación en preparación/reaprovisionamiento cuando
hay red. Durante cortes se ejecuta realimentación común. No pretende demostrar
una ventaja de racionamiento predictivo durante una interrupción.

## Controles para no atribuir a A* una ventaja de reserva

Panel predefinido: los ocho escenarios de corte a las 22:00 de seis horas y sus
ocho controles sin corte, 16 casos. No se seleccionan por resultados.

- **Reactivo con reserva adaptativa MLP:** usa las mismas predicciones; su
  objetivo de energía para la próxima frontera horaria es el mínimo stock que
  hace Phi_1=0. Si resulta imposible,
  lo declara y usa capacidad máxima como objetivo, sin negar el déficit de potencia.
  Repone de forma inmediata y descarga sólo por encima de ese objetivo con red.
  Distingue el valor de adaptar la reserva del de buscar su programación con A*.
- **A* con persistencia estacional de 24 horas:** mismo problema y objetivo que
  P-MLP, demanda prevista a partir del pasado. Separa valor específico de la MLP
  de disponer de cualquier previsión. No se entrena ni elige un baseline ganador.
- **Grilla 0,125 kWh:** repetir P-MLP/P-Oracle en ese panel para detectar la
  desventaja de discretización conocida en Fase 4.4. No seleccionar un nuevo paso.
- **Reserva fija 25 % y 75 %:** sensibilidad de R50 en el panel, sin sustituir
  retrospectivamente la reserva principal ni escoger el mejor porcentaje.
- **Igual SOC al inicio del corte:** contrafactual aislado, con estado común y
  el mismo controlador de emergencia. Debe producir igual servicio crítico;
  no se mezclan estos reinicios artificiales con el replay continuo principal.

El núcleo tiene 320 replays semanales (80 casos por cuatro estrategias). Los
controles y sensibilidades son adicionales y se informan separadamente; no se
mezclan poblaciones distintas en una única media ni se ocultan casos costosos.
Primero se valida físicamente un piloto de ocho casos del núcleo (los dos bloques
por dos perfiles solares, tarifa variable, sin corte o corte 22:00/6 h). El piloto
dimensiona cómputo y prueba contratos, no ajusta parámetros por beneficios. Si
el costo no es viable, se informa antes de cambiar la matriz, conservando todo
resultado y distinguiendo claramente cobertura pendiente.

## Métricas y comparaciones pareadas

| Métrica | Definición y tratamiento |
|---|---|
| ENS crítica, principal | Sumatoria de carga crítica requerida menos servida durante el corte, en kWh. También fracción ENS/energía crítica requerida. |
| Servicio crítico completo | Caso con ENS crítica <=1e-6 kWh y sin violaciones. Reportar proporción de casos completos, no probabilidad anual de respaldo. |
| Horas de autonomía crítica | Horas completas desde inicio hasta primera hora con déficit crítico >1e-6 kWh. Si no falla: informar >=duración, observación censurada. |
| Cobertura horaria | Total de horas críticas completamente atendidas, aunque la alimentación se recupere con solar después de un fallo. Separar de autonomía inicial continua. |
| Carga no esencial desconectada | kWh requeridos y deliberadamente no atendidos; no contar como ahorro de eficiencia. |
| Red y costo | Importación kWh y costo bruto SIM en semana completa, antes/durante/después del corte. Costo ajustado terminal separado. |
| Preparación sin cortes | Diferencia frente a R0/R50 en controles sin corte, con igual tarifa y valoración terminal. Mide el costo de estar preparado aunque no ocurra el evento. |
| SOC | Inicial, al comienzo del corte, mínimo durante el corte, final del corte y final semanal; energía almacenada en kWh además del porcentaje. |
| Solar y pérdidas | Captación, recorte, carga solar/red, descarga y pérdidas de conversión. |
| Cómputo/fiabilidad | Nodos, tiempos, búsquedas fallidas, cobertura, violaciones y residuales físicos. |

La autonomía horaria no se interpola como si existieran mediciones subhorarias.
Los límites de potencia pueden causar déficit aun con SOC alto. En casos sin
corte no se asignan artificialmente 168 horas de respaldo: esa métrica no aplica.
Si no hay demanda crítica requerida en todo el evento, su fracción ENS no aplica;
las horas transcurridas no se interpretan como respaldo de una carga positiva.

Conservar costo ajustado `C+0,1425*(E0-ET)` y sensibilidad 0, 0,076,
0,1425, 0,1578947368 y 0,38 SIM/kWh almacenado, como en Fase 4.2. No es una
factura real. Menor costo puede proceder de no atender demanda: siempre se
presenta junto con servicio crítico y cargas sacrificadas.

Por par, informar ENS evitada absoluta, reducción porcentual sólo si la ENS de
referencia es positiva, cambio de horas, importación, costos y stock. También
medias, desviación, medianas, peor caso y cantidad de casos que mejoran/empeoran.
El costo incremental por kWh crítico recuperado sólo se calcula cuando existe
recuperación positiva; no se asigna un precio de valor de la carga perdida.
Separar solar, duración, hora, tarifa y bloque; las 72 combinaciones no son
72 muestras independientes. No inferir significancia poblacional ni confiabilidad anual.

## Criterio de utilidad y límites

Umbral operativo propuesto para discutir antes de ejecutar: mejora material si
P-MLP evita al menos 0,10 kWh críticos de media por evento sobre los 72 casos
con corte (incluidos los de referencia con ENS cero) y reduce al menos 10 %
la ENS agregada frente a R50, cuando la suma de referencia es positiva. Informar todos
los deterioros; ningún promedio convierte en aceptable un caso desfavorable.
La aceptabilidad económica queda a revisión humana, sin seleccionar pesos
retrospectivamente para declarar un ganador.

Si sólo supera a R0, el resultado demuestra valor de reserva, no de A*.
Si no supera al reactivo adaptativo, no queda demostrado valor adicional de la
búsqueda en ese panel. Si no supera a persistencia, no se atribuye la mejora a
la MLP. Si sólo mejora el oracle, el potencial bajo demanda perfecta no es una
ventaja desplegable. Si todas las políticas atienden todo el evento, no hay
ventaja de servicio identificable para ese escenario; se comparan costos.

## Verificaciones previstas antes de evaluar

- Hashes de todos los artefactos históricos, incluida Fase 4.4, sin reescrituras.
- Cambiar el calendario futuro de cortes no cambia acciones anteriores a su
  detección; cambiar una restauración futura no debe filtrarla a la política.
- Sólo oracle recibe demanda futura; MLP/persistencia utilizan pasado permitido.
- Con igual SOC de corte y política de emergencia común, servicio realizado igual.
- Prioridad crítica, red importada cero durante corte, piso/capacidad/potencia,
  separación entre descarte deliberado y ENS, sin energía creada ni demanda borrada.
- Comparación A*/UCS y enumeración en problemas pequeños, costos no negativos,
  pruebas extremas sin solar, batería en piso/llena y demanda crítica sobre potencia.
- Sin corte y sin reserva adicional, el replay reactivo debe reproducir la referencia
  física anterior. Las estrategias con reserva no tienen por qué reproducir su SOC.

Esta propuesta no habilita implementación automática: primero se revisan estos
supuestos, especialmente fracción crítica, reserva fija y riesgo hipotético de
cuatro horas. No se abre test final, no se reentrena, no se ejecutan experimentos,
no se alteran fases anteriores ni se realizan commits/push.

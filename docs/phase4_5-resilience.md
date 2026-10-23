# Fase 4.5 — Resiliencia ante cortes imprevistos

## Resultado principal

En los 72 escenarios con cortes, A* + MLP congelada reduce la ENS crítica de
1,017527 a 0,653646 kWh por evento frente a reserva fija del 50 %: **35,7613 %**
de reducción agregada y 0,363881 kWh evitados de media. Mejora 36 casos y empata
36, sin deterioros de ENS frente a esa referencia. Cumple los dos umbrales
predefinidos (0,10 kWh/evento y 10 %), sin ajustar parámetros por resultados.

Esto **no demuestra una ventaja incremental de la red neuronal** sobre un
pronóstico simple: A* + persistencia obtiene 0,623267 kWh/evento, mejor que la
MLP en el agregado. La MLP reduce costo e importación frente a esa persistencia,
pero produce un 4,8741 % más de ENS crítica. Se deben separar los resultados de
reserva, planificación y modelo neuronal.

La corrida principal y los paneles están completos y auditados. La sensibilidad
de grilla se informa por separado; no modifica los resultados principales.

## Alcance y ejecución

Se conservaron el [diseño previo](phase4_5-design-proposal.md) y el
[protocolo de ejecución](phase4_5-execution-protocol.md). Las precisiones del
pedido ampliaron a las seis estrategias todos los casos de la matriz principal.

- Dos semanas continuas UCI de validación: 21–27/09/2009 y 24–30/01/2010.
- Dos perfiles PVGIS cronológicos independientes de Córdoba: enero/julio.
- Tarifas plana y variable ilustrativas, en SIM, conservadas de fases anteriores.
- Corte en el cuarto día a las 06:00, 14:00 o 22:00, de 2, 6 o 12 horas.
- 72 escenarios con un corte y ocho controles sin corte: 80 casos, seis políticas.
- SOC semanal inicial común 50 %, sin reinicio a medianoche ni al iniciar el corte.
- Batería 5 kWh, PV 3 kWp, potencias de carga/descarga 2,5 kW, eficiencias 0,95.
- Piso físico 20 % para todos; reserva operativa convencional 50 %, liberable
  durante la interrupción hasta el piso físico.

Los perfiles solares y el consumo UCI no son mediciones sincronizadas de una
instalación. El solar futuro se conoce idealmente en el escenario. La carga
crítica es **40 % sintético** del consumo UCI, sin medición por circuito ni
electrodoméstico. El 60 % no esencial se desconecta durante cada corte para
todas las estrategias. No se aplaza ni recupera después esa demanda.

Procedencia solar: Córdoba, -31,4201 / -64,1888; ERA5, 3 kWp, inclinación 30°
hacia norte, rendimiento PVGIS 0,86 ya aplicado. Se conservan los perfiles
15–22/01/2023 y 15–22/07/2023, sus timestamps UTC/locales y el mapeo ordinal
horario a UCI documentado en [Fase 4.2](phase4_2-continuous-energy.md).
Se usan 168 horas realizadas y 23 de cola para previsiones; no se repite un día.
Enero aporta 108,698355 kWh solares semanales y julio 94,408924 kWh en estos
perfiles particulares; no son valores climatológicos generales.

Secuencia ejecutada:

1. 17 pruebas nuevas y suite completa sin entrenamiento: **146 aprobadas en 10,112 s**.
2. Piloto: ocho casos, 48 replays, cero fallos, **214,80 s**.
3. Matriz completa: 80 casos, **480 replays, cero fallos, 299,17 s**, reutilizando
   planes causalmente idénticos del piloto.
4. Panel de grilla 0,125 kWh: **96/96 replays, cero fallos, 1.558,41 s**, dentro
   del presupuesto predefinido de 1.800 s; auditoría física aprobada.
5. Reservas 25/75 %: 32/32 replays y contrafactual de igual SOC. En total se
   ejecutaron **656 replays semanales**, además de 48 trazas aisladas de emergencia.
   No se cambiaron parámetros principales ni se borraron casos desfavorables.

Las 72 combinaciones no son 72 semanas independientes: son dos bloques UCI,
dos perfiles solares y condiciones repetidas. La dispersión es descriptiva,
no un intervalo de confianza ni una estimación de confiabilidad anual.

## Arquitectura, objetivo y causalidad

`energy/resilience.py` añade riesgo, planificación experimental, reserva
operativa, emergencia común y métricas. `energy/resilience_experiment.py`
implementa escenarios, predicciones congeladas, replay continuo y caché causal.
Se reutilizan PhysicalSimulator, el grafo físico y el mismo motor de búsqueda.
No se modifica la lógica de Económico, Autonomía ni Equilibrado, sus archivos,
configuraciones o resultados. La CLI experimental es `scripts/run_resilience.py`.

Para una previsión total de demanda de 24 horas se define demanda crítica
`Lc_h=0,4*Lprevista_h`. Phi_h(E_h) es la ENS crítica de un corte hipotético de
cuatro horas en h..h+3, ejecutado con emergencia común y solar previsto.
El objetivo es:

```
min_lex( (1/20)*sum_{h=1..20} Phi_h(E_h), costo_electrico_previsto )
```

El primer componente tiene unidades kWh críticos; el segundo, SIM. Se excluye
h=0 porque la red actual ya fue observada. Las cuatro horas siempre caben en
las 24 previstas. La trayectoria normal hasta cada inicio hipotético supone red
disponible; no recibe el calendario de cortes efectivos.

Este riesgo usa una distribución artificial uniforme de inicios y una duración
hipotética fija. No es una predicción de cortes, una probabilidad calibrada ni
la ENS realizada para los eventos de 2/6/12 horas. Mejorar ese objetivo tampoco
garantiza mejorar ENS real en cada escenario.

Los costos lexicográficos se calculan con racionales antes de redondear los
flujos. Las aristas son no negativas, h=0 y el grafo temporal finito. A* equivale
a UCS y es óptimo dentro del grafo representado si termina en presupuesto.
Se contrastó con UCS independiente y programación dinámica hacia atrás en
12 problemas pequeños; además, Phi se contrastó contra el simulador de emergencia
en 30 escenarios sintéticos. No se afirma optimalidad física continua ni ante
demanda desconocida o una duración de corte distinta de la hipótesis.

Ninguna política, incluido el oracle, conoce anticipadamente inicio, duración
ni restauración. El evaluador sólo revela disponibilidad actual. La demanda
realizada llega al simulador/control local, no al planificador MLP. El oracle
recibe únicamente demanda futura de validación. La persistencia usa las últimas
24 horas conocidas, con contexto pasado de train sólo al inicio, sin entrenar
ni evaluar train. No se abre test real ni se invoca entrenamiento TensorFlow.

Durante cortes todas las políticas usan la misma emergencia: servicio crítico
con solar y batería, carga de excedentes, importación cero y piso 20 %.
La comparación principal mide preparación y recuperación con red disponible,
no superioridad de racionamiento predictivo durante el corte.

Las políticas quedan fijadas así:

| Política | Preparación mientras hay red |
|---|---|
| Reactivo | Autoconsumo instantáneo, sin carga intencional desde red; sólo protege el piso físico. |
| Reserva fija | Objetivo 2,5 kWh; repone con solar/red y descarga sólo el excedente sobre ese objetivo. |
| Reserva adaptativa MLP | Cada hora calcula el menor stock en la siguiente frontera que anula Phi_1; repone inmediatamente y protege ese objetivo. Si capacidad/potencia no permiten anularlo, declara imposibilidad y apunta a capacidad máxima. |
| A* persistencia | Demanda prevista igual a las últimas 24 horas observadas; programa carga/descarga con el objetivo lexicográfico. |
| A* MLP | Mismo A*, con previsiones congeladas de MLP calendario 64-32 + MSE, identificador `calendar-20261022:2026:calendar`. |
| A* oracle | Mismo A*, con demanda potencial futura perfecta; desconocimiento idéntico de cortes. |

Las reposiciones respetan potencia y capacidad; un objetivo de reserva no implica
que pueda alcanzarse en una hora. Con el SOC convencional 50 %, la energía AC
disponible hasta el piso es `(2,5-1)*0,95=1,425 kWh`; con batería llena son
3,8 kWh. Estas cantidades no equivalen a un número fijo de horas de respaldo.
Los planes previstos se verifican sin reparación. En ejecución, el controlador
físico común limita comandos frente al consumo realizado; no permite acciones
entregadas inviables ni crea una descarga no solicitada para ocultar errores.

## Comparación de las seis estrategias

Medias sobre 72 replays semanales con un corte cada uno. ENS es por evento;
red/costos son de la semana completa. DE es desviación muestral entre escenarios.
ENS relativa es la razón entre sumas de ENS y energía crítica requerida durante
los cortes, no la media no ponderada de porcentajes individuales.

| Estrategia | ENS crítica kWh, media ± DE | ENS relativa | Casos con servicio crítico completo | Red semanal kWh | Costo bruto SIM | Costo ajustado SIM |
|---|---:|---:|---:|---:|---:|---:|
| Reactivo, piso 20 % | 1,945820 ± 1,732860 | 52,3181 % | 16/72 | 119,764151 | 21,721478 | 21,935228 |
| Reserva fija 50 % | 1,017527 ± 1,481831 | 27,3587 % | 28/72 | 128,299775 | 23,676292 | 23,676292 |
| Reserva adaptativa MLP | 0,737537 ± 1,404776 | 19,8305 % | 44/72 | 124,239264 | 23,002932 | 22,983586 |
| A* persistencia 24 h | 0,623267 ± 1,286679 | 16,7580 % | 44/72 | 132,520634 | 23,613370 | 23,577745 |
| A* MLP | 0,653646 ± 1,304861 | 17,5748 % | 44/72 | 128,478149 | 22,782611 | 22,711361 |
| A* oracle de demanda | 0,599772 ± 1,242868 | 16,1263 % | 50/72 | 123,685776 | 21,902250 | 21,942328 |

Se desconectan **5,578820 kWh no esenciales por evento en promedio**, iguales
para todas las estrategias en cada par. No son ahorro de eficiencia ni servicio
abastecido. Las horas de autonomía se refieren exclusivamente a demanda crítica.

| Estrategia | Horas completas de servicio crítico | Prefijo continuo hasta primer déficit, h | SOC medio semanal | Energía final kWh | Solar aprovechada kWh/semana | Pérdidas kWh/semana |
|---|---:|---:|---:|---:|---:|---:|
| Reactivo | 2,806 | 1,694 | 34,646 % | 1,000 | 89,653 | 2,364 |
| Fija 50 % | 4,278 | 3,167 | 59,338 % | 2,500 | 82,897 | 1,715 |
| Adaptativa MLP | 4,972 | 4,417 | 53,268 % | 2,636 | 89,623 | 3,965 |
| A* persistencia | 5,222 | 4,667 | 63,310 % | 2,750 | 80,822 | 3,216 |
| A* MLP | 5,083 | 4,528 | 61,674 % | 3,000 | 85,357 | 3,490 |
| A* oracle | 5,236 | 5,236 | 60,404 % | 2,219 | 89,336 | 3,404 |

Son medias sobre cortes de duraciones distintas; no representan una promesa de
horas de respaldo. Los casos sin déficit tienen autonomía censurada al final del
corte, y no se calcula autonomía para controles sin cortes. Todas las políticas
parten de 2,5 kWh y alcanzan el piso 20 % en al menos un escenario. SOC al inicio
y al final de cada corte, mínimos y dispersión se conservan por caso en las tablas.

![ENS por duración](../experiments/phase4_5/paired-20261023/plots/critical_ens.png)

## Beneficio incremental y casos desfavorables

ENS evitada positiva indica mejora del candidato respecto a su referencia.
Cambios de costo son ajustados y semanales; cambios de red son kWh semanales.

| Contraste | ENS evitada media kWh | Reducción agregada de ENS | Mejora / empeora / empate | Delta costo SIM | Delta red kWh |
|---|---:|---:|---:|---:|---:|
| A. Fija frente a ausencia de reserva | +0,928294 | 47,7071 % | 50 / 0 / 22 | +1,741064 | +8,535623 |
| B. Adaptativa frente a fija | +0,279989 | 27,5167 % | 36 / 2 / 34 | -0,692706 | -4,060510 |
| C. A* persistencia frente a adaptativa MLP | +0,114270 | 15,4934 % | 20 / 10 / 42 | +0,594159 | +8,281370 |
| D. A* MLP frente a A* persistencia | -0,030379 | -4,8741 % | 11 / 16 / 45 | -0,866384 | -4,042486 |
| E. A* MLP frente a fija | +0,363881 | 35,7613 % | 36 / 0 / 36 | -0,964931 | +0,178374 |
| F. Oracle frente a A* MLP | +0,053874 | 8,2420 % | 16 / 6 / 50 | -0,769033 | -4,792372 |

El contraste C cambia búsqueda y fuente predictiva. Para aislar mejor el efecto
de planificación se calculó también **A* MLP frente a reactivo adaptativo con
la misma MLP**: evita 0,083891 kWh/evento (11,3745 %), mejora 22 casos y empata
50, sin deterioros de ENS; baja costo 0,272225 SIM pero importa 4,238884 kWh más.
Este contraste todavía incluye diferencias de despacho continuo/discreto.

La red neuronal no mejora la ENS frente al baseline estacional en el agregado.
Su peor deterioro individual es 0,475 kWh críticos; el mejor beneficio también
0,475 kWh. La adaptación reactiva empeora dos casos hasta 0,511063 kWh y el contraste
C empeora diez casos hasta 0,371590 kWh. Los casos están conservados en
`adverse_cases.csv` y todos los valores en `paired_differences.csv`.

Aunque E no tiene deterioros de ENS, **14 casos cuestan más**, con máximo
0,566163 SIM adicionales; en 36 casos aumenta la importación, hasta 6,770050 kWh.
El promedio favorable no constituye dominancia económica por escenario.

![Beneficio incremental](../experiments/phase4_5/paired-20261023/plots/incremental_benefit.png)

## Cuándo aparece el beneficio

| Duración | ENS reactivo | ENS fija 50 % | ENS adaptativa | ENS A* persistencia | ENS A* MLP | ENS oracle |
|---|---:|---:|---:|---:|---:|---:|
| 2 h | 0,934553 | 0,080885 | 0 | 0 | 0 | 0 |
| 6 h | 1,633471 | 0,694847 | 0,344769 | 0,196308 | 0,248844 | 0,120658 |
| 12 h | 3,269436 | 2,276847 | 1,867843 | 1,673495 | 1,712094 | 1,678659 |

Cada duración tiene 24 escenarios. A* MLP atiende completamente 24/24 cortes de
dos horas, 16/24 de seis y sólo 4/24 de doce. El oracle tampoco resuelve todos
los cortes largos: completa 8/24 de doce horas. No se declara respaldo ilimitado.

Frente a fija, la ENS evitada media por A* MLP es 0,552346 kWh para inicios a
las 06:00, cero a las 14:00 y 0,539296 kWh a las 22:00. La reducción es mayor
en el perfil julio (0,426663 kWh) que enero (0,301099), y en el bloque tardío
(0,529845) que temprano (0,197917). No se extrapolan estas diferencias a estaciones
reales sincronizadas; son combinaciones simuladas de perfiles independientes.

![Servicio completo](../experiments/phase4_5/paired-20261023/plots/critical_service_coverage.png)

## Por qué el oracle puede perder

Conocer demanda perfecta no implica conocer el corte ni optimizar su ENS real.
Por ejemplo, en el primer bloque con tarifa variable y corte nocturno de seis horas, el oracle puede
llegar con 2,25 kWh almacenados: suficiente para su criterio de cuatro horas,
pero pierde 0,169247 kWh críticos durante las seis realizadas. La reserva fija
con 2,5 kWh y la MLP más conservadora abastecen ese evento completo.

El oracle empeora frente a MLP en seis casos y frente a fija en cuatro, aunque
mejora sus medias. Es una referencia de demanda perfecta para el mismo problema,
**no una cota superior global de resiliencia**. No se amplió retrospectivamente
el corte hipotético de cuatro horas para forzar un resultado más favorable.

![SOC y servicio](../experiments/phase4_5/paired-20261023/plots/soc_and_critical_service.png)

## Costo, energía servida y SOC terminal

Se conserva `C_ajustado=C_bruto+v*(E0-ET)`, con v=0,1425 SIM/kWh almacenado.
No es factura. A* MLP termina con 3 kWh de media, fija con 2,5 y reactivo con 1;
los saldos se muestran, sin imponer igualdad artificial al final de cada semana.
La ventaja económica media E persiste sin valoración terminal: costo bruto
menor en 0,893681 SIM, frente a 0,964931 SIM ajustados. Con la sensibilidad
predefinida v entre 0 y 0,38, la reducción media va de 0,893681 a 1,083681 SIM.

Con cortes no vale usar sin cambios la identidad de Fase 4.4. Para iguales
demanda requerida, solar y energía inicial, sin exportación:

```
deltaRed = ENScritica_recuperada - deltaNoEsencialDesconectada
           + deltaRecorte + deltaPerdidas + deltaEnergiaFinal
```

Se conciliaron 800 pares con residual máximo 6,09e-14 kWh. En E, los 0,178374
kWh adicionales de red acompañan 0,363881 kWh críticos recuperados, menor
recorte, más pérdidas y más stock final. No se interpretan automáticamente como
energía desperdiciada: parte de la comparación cambia el servicio entregado.

En controles sin corte, fija añade 1,592839 SIM ajustados y 7,551792 kWh de red
frente al reactivo. A* MLP añade 0,526915 SIM frente al reactivo y cuesta
1,065924 SIM menos que fija, con 0,095600 kWh más de red. La preparación tiene
costo aun cuando el evento no ocurre; la disponibilidad de la red no se revela
anticipadamente para esos controles.

Con tarifa variable, E reduce costo ajustado medio 1,689190 SIM e importa
1,248731 kWh más. Con tarifa plana reduce costo 0,240672 SIM e importa
0,891983 kWh menos. No hay una conclusión única de costo/autonomía para todo caso.

![Servicio frente a costo](../experiments/phase4_5/paired-20261023/plots/service_cost_tradeoff.png)

## Reserva fija y contrafactual de igual SOC

Panel predefinido de ocho cortes 22:00/6 h, más ocho controles sin corte:

| Política en el panel con corte | ENS crítica media kWh | Costo ajustado semanal SIM | Red semanal kWh |
|---|---:|---:|---:|
| Fija 25 % | 2,304287 | 22,545722 | 121,750947 |
| Fija 50 % | 1,150913 | 24,061982 | 129,452590 |
| Fija 75 % | 0,557163 | 25,957589 | 138,317516 |
| A* MLP, referencia 0,25 | 0,616538 | 23,121650 | 130,052365 |
| A* persistencia, referencia 0,25 | 0,527476 | 23,967311 | 134,152823 |

La reserva 75 % supera a MLP en ENS de ese panel, pero cuesta e importa más.
No se selecciona el mejor porcentaje después de observar resultados. Los
32 replays de sensibilidad y sus 5.376 pasos físicos fueron verificados.

El contrafactual aislado inicia el corte con 2,5 kWh para todos: las seis
trayectorias de emergencia coinciden exactamente en los ocho eventos. No es
un replay continuo y no se mezcla con las métricas principales. Confirma que
las diferencias de esta fase proceden de preparación/SOC al corte, no de un
controlador de emergencia distinto. SOC alto no garantiza servicio completo si
la energía, potencia o duración son insuficientes.

## Sensibilidad de grilla: no seleccionar una resolución ganadora

Se repitieron las seis políticas en los 16 casos del panel predefinido (ocho
cortes 22:00/6 h y ocho controles), con 0,125 kWh. Sólo cambia la resolución;
se mantienen fracción crítica, riesgo, límites, previsiones y presupuesto por
búsqueda. Son 96 replays adicionales completos, sin fallos ni violaciones.

Medias sobre los mismos ocho cortes del panel:

| Política | ENS kWh, paso 0,25 | ENS kWh, paso 0,125 | Costo ajustado 0,25 SIM | Costo ajustado 0,125 SIM | Nodos semanales 0,25 / 0,125 |
|---|---:|---:|---:|---:|---:|
| A* persistencia | 0,527476 | 0,527476 | 23,967311 | 23,972210 | 37.770 / 72.909 |
| A* MLP | 0,616538 | 0,661070 | 23,121650 | 23,075084 | 38.256 / 73.951 |
| A* oracle | 0,361975 | 0,361975 | 22,286847 | 22,179644 | 37.776 / 73.127 |

Los tres reactivos reproducen sus resultados exactamente. La grilla fina
aumenta ENS de MLP en 0,044531 kWh/evento: empeora tres casos en 0,118750 kWh y
empata cinco, sin mejoras. En esos tres llega al corte con 0,125 kWh menos;
la diferencia de energía crítica es `0,125*0,95=0,11875 kWh`. El costo medio
baja 0,046566 SIM, mientras la red sube 0,160411 kWh. Las expansiones casi se
duplican. Oracle y persistencia mantienen ENS, aunque cambian costos y stock.

Esto no identifica una ventaja de refinar la grilla para el servicio realizado:
prepararse según riesgo hipotético de cuatro horas no equivale a minimizar ENS
del corte de seis horas. Una grilla más gruesa puede dejar un margen adicional
de energía. No se ajusta el riesgo ni se escoge retrospectivamente la resolución
favorable; la matriz principal conserva 0,25 kWh. Estas ocho interrupciones no
permiten extrapolar el efecto de resolución a toda la matriz de 72 cortes.

El panel verifica 16.128 pasos realizados y 97.656 previstos de 4.069 planes
únicos, además de preservar 6.511 artefactos de su referencia principal.
`grid-comparison-20261023` conserva emparejamiento, diferencias y auditoría de
tablas; el panel y la matriz completa nunca se mezclan en una única media.

## Verificación, cómputo y conservación

La auditoría principal reconstruyó **80.640 pasos realizados**, 2.880 cruces de
medianoche, 37.152 pasos de prefijos previos a cortes y 38.592 pasos de prefijos
con restauración futura desconocida. Se comprobaron 4.552 planes únicos: 2.032
ya auditados en el piloto y 2.520 nuevos, con replay físico y objetivo previsto
independiente. Cero violaciones físicas, importación cero durante cortes y
separación explícita de ENS y desconexión no esencial.

Los planes se almacenan por una firma que excluye cortes futuros y depende de
origen, previsión, solar, precios, configuración, energía exacta y código.
Se separan tiempo/nodos lógicos y trabajo único. Por replay semanal con corte,
los A* referencian aproximadamente 31–33 s de búsqueda y 38.000 nodos; los
aciertos de caché no significan latencia cero desplegada. La reserva adaptativa
no tiene búsqueda, pero calcular su objetivo e inferir pronósticos tampoco es
trabajo gratuito en hardware. No se incluye nueva inferencia neuronal aquí.

La suite incluye casos extremos, presupuesto agotado sin fallback, causalidad
de cortes/restauración, conservación de SOC, emergencia idéntica, liberación de
reserva y ausencia de conocimiento futuro de cortes en el oracle. Las pruebas
anteriores de los tres modos siguen aprobando. No se entrenó ni evaluó test real.
Se excluyeron los tres módulos de integración TensorFlow (ocho pruebas, varias
con entrenamiento sintético). Las 146 ejecutadas incluyen toda la suite restante,
sin omisiones dentro de ella, y las 17 pruebas nuevas de esta fase.

Los manifiestos conservan hashes de datos, modelos, fuentes y resultados previos,
incluida Fase 4.4; el piloto y la matriz principal se preservan al ejecutar los
paneles. Los registros fallidos/incompletos tendrían estado explícito y se excluirían
sólo de agregados pareados completos; en piloto y matriz principal hubo cero.
El cierre comprobó **14.825 archivos por hash**, incluidos entradas conservadas,
piloto, matriz principal y panel fino. El recibo final, con fuentes y registro
de pruebas, es `experiments/verification/phase4_5-final-audit.json`. Los 656
replays semanales completaron sin fallos. Los archivos versionados y el índice
Git coinciden con HEAD; los archivos nuevos de esta fase quedan sin commit.

## Recomendación y límites

Hay una ventaja verificable de preparación predictiva frente a reserva fija
50 % para estas condiciones simuladas. Se recomienda conservar el componente
como **resiliencia experimental**, no afirmar respaldo real ni integrar una
selección automática de estrategia en los modos existentes.

La reserva adaptativa proporciona una alternativa simple con buena parte del
beneficio y menor importación. A* añade una mejora moderada de ENS con la misma
MLP, a costa de más red. Para priorizar estrictamente ENS, la MLP no supera a
persistencia; no se justifica atribuirle un beneficio neuronal incremental de
resiliencia. Sí muestra un compromiso de menor costo/red con un pequeño deterioro
de servicio frente a ese baseline. La elección depende de prioridades humanas.

Persisten límites importantes: dos bloques y un modelo/semilla congelados;
40 % crítico artificial; solar perfecto; corte único; riesgo de cuatro horas
distinto de duraciones realizadas; grilla discretizada; tarifas SIM; batería
sin degradación/autodescarga; y simulación horaria con detección inmediata.
Se supone operación aislada técnicamente posible, sin evaluar inversor,
transferencia, transitorios ni potencia subhoraria. La desconexión no esencial
es completa, aun si podría reconectarse con excedentes. No se recuperan cargas.
Las previsiones congeladas suponen disponible la demanda potencial pasada aun
durante interrupciones: no se simula pérdida de telemetría ni se reconstruye
demanda de circuitos desenergizados. Tampoco se modela respuesta de usuarios a
desconexiones. Es una hipótesis de observabilidad para este ensayo; antes de uso
real habría que resolver esas entradas sin asumir medición de cargas no servidas.
La sensibilidad de fracción crítica 20/60 % queda fuera de la evaluación ejecutada:
se mantuvo el 40 % indicado en las precisiones, sin inferir robustez a otro reparto.

No se ajustó duración hipotética, reserva, pesos ni modelo para mejorar resultados;
no se hizo commit/push ni se avanzó a otra fase. Revisar estos resultados antes
de cualquier optimización o integración operativa posterior.

## Archivos y reproducción

Los resultados pesados quedan ignorados por Git:

- `experiments/phase4_5/pilot-20261023`: piloto y auditoría.
- `experiments/phase4_5/paired-20261023`: matriz principal, tablas y cinco gráficos.
- `experiments/phase4_5/reserve-panel-20261023`: reservas 25/75 y contrafactual.
- `experiments/phase4_5/accounting-20261023`: conservación con servicio variable.
- `experiments/phase4_5/fine-grid-20261023`: panel de grilla fina y auditoría.
- `experiments/phase4_5/grid-comparison-20261023`: comparación entre resoluciones.
- `experiments/verification/phase4_5-*.txt`: pruebas, ejecución y verificación.

En cada corrida principal: `metrics.csv`, `paired_metrics.csv`, `summary.csv`,
`paired_differences.csv`, `comparison_summary.csv`, `adverse_cases.csv`,
`terminal_sensitivity.csv`, `period_metrics.csv`, `same_forecast_search*.csv`,
`bases/*/replays/*/ledger.csv`, planes únicos con sus entradas y manifiestos.

Código nuevo: `energy/resilience.py`, `energy/resilience_experiment.py`,
`tests/test_resilience.py`, `configs/energy.resilience.json` y los scripts
`run_resilience.py`, `analyze_resilience.py`, `analyze_resilience_incremental.py`,
`analyze_resilience_energy_balance.py`, `verify_resilience.py`,
`plot_resilience.py`, `run_resilience_panel.py`, `run_resilience_reserve_panel.py`,
`analyze_resilience_grid.py`.
Sólo se usan dependencias ya instaladas y CPU.

Requiere artefactos históricos locales congelados; un clon no contiene modelos
ni datasets. Usar carpetas nuevas. Desde la raíz, en PowerShell:

```powershell
New-Item -ItemType Directory -Force experiments/tmp | Out-Null
$env:TEMP = (Resolve-Path experiments/tmp).Path
$env:TMP = $env:TEMP
.\.venv\Scripts\python.exe -c "import unittest,pathlib; s=unittest.TestSuite(unittest.defaultTestLoader.discover('tests',pattern=p.name) for p in sorted(pathlib.Path('tests').glob('test_*.py')) if 'tensorflow' not in p.name); r=unittest.TextTestRunner(verbosity=2).run(s); raise SystemExit(not r.wasSuccessful())"
.\.venv\Scripts\python.exe -u scripts/run_resilience.py --stage pilot --run-dir experiments/phase4_5/repeat-pilot
.\.venv\Scripts\python.exe scripts/analyze_resilience.py --run-dir experiments/phase4_5/repeat-pilot
.\.venv\Scripts\python.exe scripts/verify_resilience.py --run-dir experiments/phase4_5/repeat-pilot
.\.venv\Scripts\python.exe -u scripts/run_resilience.py --stage full --run-dir experiments/phase4_5/repeat-full --cache-source experiments/phase4_5/repeat-pilot
.\.venv\Scripts\python.exe scripts/analyze_resilience.py --run-dir experiments/phase4_5/repeat-full
.\.venv\Scripts\python.exe scripts/analyze_resilience_incremental.py --run-dir experiments/phase4_5/repeat-full
.\.venv\Scripts\python.exe scripts/plot_resilience.py --run-dir experiments/phase4_5/repeat-full
.\.venv\Scripts\python.exe scripts/verify_resilience.py --run-dir experiments/phase4_5/repeat-full --verified-cache experiments/phase4_5/repeat-pilot
.\.venv\Scripts\python.exe scripts/run_resilience_reserve_panel.py --reference experiments/phase4_5/repeat-full --run-dir experiments/phase4_5/repeat-reserves
.\.venv\Scripts\python.exe scripts/analyze_resilience_energy_balance.py --reference experiments/phase4_5/repeat-full --output experiments/phase4_5/repeat-accounting
.\.venv\Scripts\python.exe -u scripts/run_resilience_panel.py --reference experiments/phase4_5/repeat-full --run-dir experiments/phase4_5/repeat-fine --variant fine_grid
.\.venv\Scripts\python.exe scripts/analyze_resilience_grid.py --reference experiments/phase4_5/repeat-full --supplemental experiments/phase4_5/repeat-fine --output experiments/phase4_5/repeat-grid-comparison
```

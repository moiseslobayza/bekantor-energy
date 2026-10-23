# Fase 4.5 — Protocolo de implementación, previo al piloto

Se acepta la propuesta `phase4_5-design-proposal.md` con las precisiones del
pedido de implementación. Las seis estrategias se ejecutarán en todos los
casos del núcleo, ampliando los dos controles inicialmente limitados al panel.
Son 80 casos × seis políticas = 480 replays semanales; el piloto usa ocho
casos y las seis políticas (48 replays). Los resultados del piloto se conservan.

No se modifica ningún archivo previo de código/configuración/modelos/resultados.
La implementación nueva reutiliza PhysicalSimulator, el grafo físico y el único
motor A*, y conserva exactamente los objetivos anteriores. ResilienceProblem
prioriza el par racional `(riesgo_kWh_criticos, costo_SIM)` con h=0. Sus aristas
son no negativas; en un grafo temporal finito, UCS/A* con reapertura devuelve
el mínimo lexicográfico si termina dentro del presupuesto. Las pruebas incluyen
un UCS independiente y programación dinámica hacia atrás, además de comparar
Phi con el replay físico de emergencia. No se cambia h por resultados económicos.

Para H=24 y K=4, Phi_h(E) es ENS crítica simulada en un corte hipotético en
h..h+3. Usa exclusivamente 0,4 veces la previsión de demanda y el perfil solar
idealizado del escenario. El riesgo de un plan es `(1/20)*sum(Phi_h(E_h))`
para h=1..20, excluyendo disponibilidad actual ya observada. Cada Phi se calcula
con batería continua y emergencia común. El grafo sigue discretizado en 0,25 kWh;
las magnitudes físicas y los objetivos se mantienen separados. La suma se forma
con racionales a partir de las entradas binarias representadas, antes del redondeo
de los flujos. El costo secundario también usa las ecuaciones físicas racionales.

Ese riesgo corresponde a inicios futuros equiponderados y duración hipotética
fija, no al calendario efectivo ni a una probabilidad de cortes calibrada. ENS
real se obtiene ejecutando con demanda realizada, cortes 2/6/12 y SOC acumulado.
No debe confundirse riesgo previsto bajo red operativa con ENS realizada.

La persistencia estacional de 24 horas reutiliza sólo las últimas 24 horas
conocidas en cada origen, como perfil para las 24 futuras. El contexto inicial
requiere las últimas horas anteriores a validación, conservadas en train: se
usan sólo como pasado de entrada, nunca para entrenar ni puntuar. Todos los
objetivos evaluados y toda la demanda futura oracle pertenecen a validación.
La MLP se consume a través de sus predicciones congeladas de Fase 4.3; no hay fit
ni nuevas inferencias que alteren modelos o requieran TensorFlow.

Presupuestos fijados antes del piloto: dos procesos; 10 s y 100.000 expansiones
por búsqueda; 1.200 s de reloj para piloto y 3.600 s para matriz completa.
No se eligen pesos, fracciones, reservas ni escenarios a partir de beneficios.
Si un límite interrumpe una ejecución, se guardan trazas parciales y estado
fallido; los casos no se convierten en éxito con ENS cero.

Para no recalcular prefijos causalmente idénticos, se reutilizan planes mediante
una clave de política, origen, demanda prevista, solar, tarifas, configuración,
SOC exacto y hashes del código. No incluye corte futuro ni duración. Se audita
físicamente cada plan único cargado/calculado y se conserva su entrada completa.
El cache del piloto puede leerse desde la evaluación completa sin modificarlo.
Los reportes separan nodos/tiempo lógico referenciado de trabajo único realizado;
un acierto de caché no prueba latencia cero en un controlador desplegado.

Las sensibilidades de la propuesta son suplementarias, separadas del núcleo;
no sustituyen escenarios ni estrategias. Se priorizan pruebas, piloto y matriz
completa. Cualquier sensibilidad no ejecutada se declarará pendiente por recursos,
sin cambiar los parámetros principales ni presentar la omisión como un resultado.

Los controles sin cortes y la valoración terminal histórica se informan aparte
del costo bruto. El objetivo operativo principal se juzga frente a reserva fija:
al menos 0,10 kWh críticos evitados de media en 72 eventos y reducción agregada
de al menos 10 %, únicamente como criterio de evaluación y mostrando deterioros.

# Fase 4.4 — Diseño diagnóstico previo

Se conserva el objetivo lexicográfico (importación kWh, costo SIM), el motor A*,
la reserva, las eficiencias, el control de realización, los presupuestos de
búsqueda y el horizonte de 24 horas. No se cambia ninguna implementación previa.
Los modelos y resultados históricos quedan congelados; sólo se usa validación.

## Comparaciones fijadas antes de los nuevos resultados

- Reproducir métricas de reactivo y Autonomía MLP a grilla 0,25 kWh de los ocho
  casos de Fase 4.3, sin ejecutar ni reescribir la corrida histórica.
- Oracle diagnóstico: demanda futura realizada de las próximas 24 horas,
  disponible sólo para esta referencia explícita, en los ocho casos a 0,25 kWh.
- Sensibilidad: 0,5 y 0,125 kWh, MLP congelada y oracle, sobre los cuatro casos
  con tarifa variable (dos bloques por dos perfiles solares). Las grillas son
  anidadas. No se eligen casos por resultados y no se busca una resolución ganadora.
- Cada caso conserva 168 horas consecutivas, E0=2,5 kWh, SOC persistente y el
  perfil PVGIS cronológico con cola de 23 horas. El oracle debe obtener su cola
  exclusivamente de validation.csv y cotejar las 168 horas con la referencia.
- La MLP reutiliza las predicciones históricas emitidas causalmente. El solar
  futuro sigue siendo idealizado para todas las estrategias. UCI y PVGIS son
  series independientes combinadas en un escenario simulado.
- Los presupuestos originales (10 segundos y 100.000 expansiones por búsqueda)
  siguen vigentes. Las búsquedas fallidas quedan registradas, sin sustitución.
  Dos procesos locales; los tiempos no son una comparación de implementaciones.

## Diagnóstico físico independiente

Construir una red temporal de flujo en unidades AC entregables:
capacidad de almacenamiento eta_d*(C-reserva), suministro inicial
eta_d*(E0-reserva), suministro solar por hora
eta_c*eta_d*min(excedente,Pc*dt), y demanda de descarga
min(déficit,Pd*dt). Los arcos temporales limitan el almacenamiento. Un flujo
máximo, verificado con el corte mínimo residual, limita superiormente lo que
la batería puede servir. La cota de importación es suma(déficits)-flujo máximo.
Permitir descartar suministro no usado hace la cota optimista, nunca exigente.
La eliminación de carga de red no empeora el mínimo primario cuando la red es
ilimitada, está siempre disponible y eta_c*eta_d<=1: desplazar ese suministro
directamente a la carga futura requiere menos o igual importación.

Si el replay reactivo es factible y alcanza esta cota, certifica el mínimo de
importación para esa semana, aun frente a previsión perfecta de todo el bloque,
dentro del modelo físico horario. No se afirma optimalidad universal en otras
instalaciones, interrupciones o modelos de batería.

## SOC y métricas

Separar G real, E0, ET, costo bruto y C+0,1425*(E0-ET). Informar también
G+lambda*(E0-ET) para lambda=0, eta_d=0,95 y 1/eta_c; es una equivalencia
contable, no importación medida. Verificar directamente los pares con igual ET.
No añadir condición terminal ni recargar físicamente al final de forma encubierta.

Descomponer cada brecha mediante conservación:
deltaG = deltaRecorte + deltaPérdidas + deltaET, con igual demanda, solar, E0,
sin exportación ni energía no suministrada. Medir carga de red, carga solar,
descarga, oportunidades de carga/descarga no usadas, reparaciones y errores de
signo del desequilibrio actual. Esta identidad contable no atribuye causalidad
por sí sola: oracle y resolución son intervenciones diagnósticas independientes,
con interacción posible y sin sumar sus efectos como causas aditivas.

La mejora material se fija como reducción simultáneamente >1% y >0,5 kWh por
semana frente al reactivo. Informar resultados desfavorables y cobertura; no
promediar fallos como ceros. La decisión final puede ser conservar sólo la
referencia experimental. No se implementa Resiliencia ni se ajustan objetivos.

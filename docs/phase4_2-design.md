# Fase 4.2 — Diseño fijado antes de evaluar

El experimento se especifica antes de ejecutar resultados energéticos. No se
modifican simulador, A*, redes ni resultados de las fases 1–4.1.

- Dos bloques de siete días: primer comienzo completo a medianoche dentro de
  cada mitad cronológica de validación. Disponibilidad de ventanas y objetivos
  observados es un filtro retrospectivo de elegibilidad, no entrada del control.
  Resultan 21–27/09/2009 y 24–30/01/2010. Los 23 pasos finales de cada pronóstico
  también quedan dentro de validación. El SOC no se reinicia entre días.
- Tres estrategias: reactivo, MLP MSE + A* original, y la misma MLP/A* con
  condición **SOC previsto ≥ 50 % al final de cada horizonte móvil de 24 h**.
  El SOC inicial de cada bloque es 50 %. La condición es sobre el pronóstico,
  no una reserva física del 50 % durante la ejecución ni garantía del saldo
  realizado al terminar el bloque. La reserva física original sigue en 20 %.
- La variante usa `PlannerConfig.terminal_min_soc`, ya implementado. No agrega
  penalizaciones monetarias ni aristas negativas. El objetivo sigue siendo el
  costo de importación. Restringir metas preserva la cota inferior original,
  que ignora la reposición terminal. Se comprobará contra UCS en casos pequeños.
- Factorial: dos perfiles solares × dos tarifas × dos bloques = ocho casos,
  cada uno con tres estrategias bajo exactamente las mismas condiciones.
  Tarifa plana 0,15 SIM/kWh; variable original de Fase 3.
- Perfiles PVGIS Córdoba, −31,4201 / −64,1888, 3 kWp, ERA5, tilt 30°, norte,
  eficiencia 0,86. Trayectorias cronológicas de ocho días, 15–22/01/2023 y
  15–22/07/2023, con timestamps UTC y locales conservados. No se eligen por
  resultados de costos ni se repite un día. Se ejecutan siete días y se conserva
  cola para el pronóstico. Enero/julio son condiciones estacionales declaradas,
  no clases meteorológicas elegidas tras observar la generación.
- Solar futura conocida idealmente en los tres controladores; combinación
  ordinal simulada de relojes independientes. No son mediciones de la misma
  vivienda, ubicación, año ni períodos sincronizados de UCI y PVGIS.
- Costo bruto: suma de importación × precio. Costo ajustado contable:
  `C_aj = C_bruto + v × (E_inicial − E_final)`. No es factura, liquidación real,
  objetivo de A*, garantía de reemplazo físico ni simulación de recarga final.
- Valor fijo principal **v = 0,1425 SIM/kWh almacenado**, dado por precio de
  referencia 0,15 SIM/kWh AC × eficiencia de descarga 0,95. Representa energía
  potencialmente entregable valorada al precio de referencia; no depende de
  qué estrategia, tarifa, bloque ni resultado se observe. La reserva común
  se cancela al comparar variaciones de energía almacenada.
- Sensibilidad predefinida: 0; 0,076 (=0,08×0,95); 0,1425; 0,1578947368
  (=0,15/0,95, referencia de reposición desde red); 0,38 (=0,40×0,95).
  Es una sensibilidad contable, no otra búsqueda de hiperparámetros.
- Modelo MSE con calendario 64–32, semilla 2026, pesos congelados; predicciones
  causales ya emitidas y verificadas por hashes/inferencia CPU. No oracle,
  test, entrenamiento ni exposición de objetivos realizados futuros al planner.
- Cada plan previsto completo se reproduce físicamente antes de aplicar su
  primera orden a demanda realizada. Las limitaciones de búsqueda y fallos se
  conservan explícitamente, sin reemplazar estrategias. Los agregados principales
  exigen las tres estrategias completas y declaran cobertura.
- Se informan resultados por bloque y escenario, dispersión descriptiva,
  costo bruto y ajustado, energía inicial/final, red, solar, restricciones,
  energía no suministrada, carga/descarga, nodos y tiempo. Los cruces no son
  ocho semanas independientes. No se afirma ahorro real ni superioridad general.

API verificada el 22/10/2026: [documentación oficial v6](https://photovoltaic-geographic-information-system.ec.europa.eu/api/v6/redoc)
y [notas de versión](https://photovoltaic-geographic-information-system.ec.europa.eu/api/release-notes).
Se reutiliza el adaptador GET vigente; los cambios de POST no lo afectan.

La configuración reproducible está en `configs/energy.continuous.json`. Este
diseño también se guarda con hashes antes de simular en `design.json` de cada
corrida. No se implementan modos, dashboard, Colab ni Fase 4.3.

# Validación de Fase 1 — 22/10/2026

## Repositorio y entrega

Directorio inicial vacío salvo `.git`. Rama `main`; remoto fetch/push:
`https://github.com/moiseslobayza/bekantor-energy.git`.
`git ls-remote origin` finalizó con código 0 y sin referencias: acceso de lectura
comprobado, remoto sin ramas/tags publicados al consultar. La consulta inicial
dentro del sandbox falló por DNS; fuera de esa restricción funcionó.
No se creó commit, rama nueva, PR ni push. No se cambió el remoto.

La estructura completa y el inventario de módulos están en `README.md`.
Se implementaron 13 archivos Python del paquete, 4 módulos de pruebas, 2
fixtures sintéticos, configuración JSON, metadatos de instalación, dependencias
congeladas y documentación. Los directorios de datos/modelos/resultados sólo
dejan documentación y marcadores visibles a Git. La prueba `git ls-files --others
--exclude-standard` excluye las descargas, el entorno y los resultados.

## Entorno e instalación

- Windows x64, Python **3.13.15**.
- Entorno virtual local `.venv`, instalación editable con backend setuptools.
- NumPy **2.5.4**, pandas **3.0.6**, Matplotlib **3.11.2**, tzdata **2026.5**.
- Todas las versiones transitivas están en `requirements-lock.txt`.
- `python -m pip check`: **No broken requirements found**.
- Ayuda CLI, validación de configuración, demo offline, procesamiento real y
  EDA ejecutados desde el entorno virtual. No se instaló TensorFlow ni GPU.

La primera suite se ejecutó también sobre el Python del sistema, con pandas
3.0.5, NumPy 2.5.2 y Matplotlib 3.11.1. El entorno aislado congelado es el
referente reproducible de esta entrega; no se promete compatibilidad probada
con todas las combinaciones admitidas por los rangos.

## Pruebas automáticas

Comando: `.\.venv\Scripts\python.exe -m unittest discover -s tests -v`.
**29 tests aprobados, 0 fallos, 0 errores** en la ejecución aislada.
Salida local completa: `experiments/verification/tests.txt` (ignorada por Git).

Cobertura funcional comprobada:

- Conversiones W/kW, Wh/kWh, potencia por duración, SOC y valores físicos inválidos.
- Configuración ejemplo, tarifas fijas y por paso, límites de potencia/capacidad,
  eficiencia, reserva, horizonte y tarifas imposibles.
- Columnas ausentes, vacío, `?`, campos vacíos, fechas imposibles, timestamp
  fuera de minuto, duplicados, texto numérico incorrecto, negativos e infinitos.
- Integración de potencia y submedidores con unidades distintas, horas parciales,
  minutos ausentes en la grilla, huecos largos, ausencia inicial y relleno desactivado.
- Invariancia del pasado al multiplicar por 1000 valores futuros.
- Particiones disjuntas, ordenadas y exhaustivas; fracciones y entradas inválidas.
- Descarga acotada con limpieza de parcial, retry 529 y ausencia de retry HTTP400.
- Extracción ZIP restringida, recibos SHA-256 y detección de caché alterada.
- Esquema PVGIS v6, parámetros UTC/kWp, conversión W a kW, rechazos de fechas,
  longitud, unidades de potencia pico y valores inválidos, caché por consulta.
- Pipeline completo con fixture, repetibilidad de manifiesto/hashes, EDA limitado
  a train, archivos CSV/JSON y cuatro PNG válidos, error de CLI ante configuración inválida.

Las pruebas son offline y no dependen de disponibilidad de las APIs. Inicialmente
7 tests de archivos fallaron por permisos del directorio temporal del sandbox,
no por sus aserciones de datos. Se reejecutaron con TEMP/TMP en `experiments/tmp`
y pasó la suite completa. El README incluye esa alternativa para Codex.

## Comprobación con UCI completo

Adquisición y pipeline ejecutados contra el archivo original descargado, no
sólo contra fixtures. Archivo bruto: 132 960 755 bytes, excluido de Git.

SHA-256 TXT:
`4259c9d7ece5dbee9ab8d53682baac68d791c864f0f64a52b4043cb3b90894b7`

| Comprobación | Resultado |
|---|---:|
| Filas por minuto | 2 075 259 |
| Minutos esperados entre primera y última observación | 2 075 259 |
| Timestamps faltantes en esa grilla | 0 |
| Valores faltantes originales por columna eléctrica | 25 979 |
| Minutos rellenados por columna, límite causal 5 | 156 |
| Valores aún faltantes por columna | 25 823 |
| Filas horarias conservadas | 34 589 |
| Horas completas después del relleno | 34 141 |
| Horas con 60 minutos originales de potencia activa | 34 085 |
| Horas incompletas, energía global mantenida como NaN | 448 |

| Partición | Horas | Inicio (reloj UCI) | Último inicio horario |
|---|---:|---|---|
| Train | 24 212 | 2006-12-16 17:00 | 2009-09-20 12:00 |
| Validation | 5 188 | 2009-09-20 13:00 | 2010-04-24 16:00 |
| Test | 5 189 | 2010-04-24 17:00 | 2010-11-26 21:00 |

La primera hora es parcial y permanece incompleta; no se extrapola. El EDA
principal describe sólo train, con 149 horas sin energía global completa. Las
cuatro visualizaciones se abrieron e inspeccionaron: ejes/unidades legibles,
títulos con fuente/partición, huecos sin consumo inventado y sin texto recortado.
No se eliminaron picos ni se ajustaron hiperparámetros a partir de esas figuras.

Los artefactos completos están en `data/processed/uci/` y `experiments/uci/`.
El manifiesto conserva límites y hashes de cada partición para reproducirlos.

## Comprobación real de PVGIS v6

Se leyó el OpenAPI actual y el endpoint oficial de convenciones; luego se
ejecutó la adquisición del adaptador, con respuesta HTTP 200:

- Buenos Aires (-34.6037, -58.3816), ERA5, 3 kWp, eficiencia de sistema 0.86,
  inclinación 30°, orientación norte 0°.
- 48 timestamps horarios UTC: 2023-01-01 00:00 a 2023-01-02 23:00, inclusive.
- Se verificaron todos los timestamps y valores finitos/no negativos.
- JSON original y recibo en `data/raw/pvgis/`; CSV independiente en
  `data/processed/pvgis/hourly.csv`.
- SHA-256 de la respuesta del adaptador:
  `48f487e1aa094bc09cd40dba900ac9f90c91524e2d76bb200e7b063417f5dd92`.

Es una prueba de adquisición y conversión de generación modelada, no de
precisión solar, medición de paneles o coincidencia con UCI. El `unit` de la
respuesta v6 es angular. La aproximación rectangular a energía se identifica
con `pv_kwh_estimate`, sin afirmar energía medida.

## Limitaciones reales y Fase 2

- El cargador procesa UCI en memoria; no se midieron garantías de latencia o
  memoria para máquinas de pocos recursos. Para volúmenes mayores se necesitará
  lectura por chunks/streaming con estado causal entre bloques.
- No se resuelve la zona horaria ni DST de UCI por suposición. Tampoco se
  sincronizan fuentes cuyo calendario, hogar y ubicación son distintos.
- Los huecos largos permanecen desconocidos. Deben excluirse ventanas o
  definirse políticas explícitas antes del entrenamiento; no usar NaN como cero.
- Tarifas, parámetros de batería y ubicación solar son ejemplos simulados.
- PVGIS depende de una API externa cambiante. El adaptador sólo cubre el
  contrato v6 revisado y valida la respuesta; no tiene fallback silencioso.
- No se entrenó ninguna red ni se midieron MAE/RMSE, ahorro o autonomía. Los
  contratos de MLP, simulador, A* y evaluador son exclusivamente interfaces.
- Quedan pendientes para Fase 2 las ventanas causales, transformaciones
  ajustadas sólo a train, baseline y MLP TensorFlow/Keras CPU, optimización
  con validation y evaluación con test. Validar compatibilidad TensorFlow/Python
  al fijar esa dependencia. Implementar posteriormente el simulador, A*, las
  comparaciones y la extracción del único notebook académico según el alcance
  que se acuerde para cada fase. No se avanzó a ellas en esta entrega.

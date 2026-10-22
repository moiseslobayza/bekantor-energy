# Fuentes, calidad y procedencia

Revisión de fuentes oficiales: **22 de octubre de 2026**.

## UCI: consumo medido histórico

- Ficha: https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption
- Descarga: https://archive.ics.uci.edu/static/public/235/individual+household+electric+power+consumption.zip
- Autores: Georges Hebrail y Alice Berard (2006).
- DOI: https://doi.org/10.24432/C58K54 ; licencia **CC BY 4.0**, conservar atribución.
- 2 075 259 observaciones por minuto, una vivienda en Sceaux, Francia;
  diciembre de 2006 a noviembre de 2010. Aproximadamente 1.25% de observaciones
  ausentes según la descripción oficial; se calculan los valores efectivos en
  `quality.json` para cada ejecución, sin confiar en flags contradictorios de
  ausencia de la tabla de variables del sitio.
- TXT delimitado por `;`, fechas `dd/mm/yyyy`, horas `hh:mm:ss`; números decimales
  con punto, ausentes `?` o campos vacíos. Se exige el esquema de 9 columnas.
  Columnas adicionales se ignoran; las requeridas nunca se renombran por posición.
- Zona horaria no especificada explícitamente en el archivo. Se conservan
  etiquetas de reloj sin asignarles UTC ni resolver DST por suposición.

Validaciones: vacío, columnas ausentes, fechas, segundos distintos de cero,
timestamps duplicados, cadenas numéricas inválidas, infinitos y negativos.
Se ordenan registros desordenados. Se identifican minutos ausentes en la grilla,
se contabilizan nulos por columna antes/después del relleno y se conserva
cobertura observada. No se recortan outliers ni se fuerza consumo cero en huecos.
El relleno toma hasta cinco minutos consecutivos a partir del último valor,
incluso el prefijo de un hueco largo; el resto continúa ausente.

Cada hora requiere 60 valores por magnitud para integrar su energía. Una hora
parcial al borde se considera incompleta incluso si todos sus minutos presentes
son válidos. Submedidores y potencia global tienen cobertura independiente:
puede haber un submedidor nulo con consumo global completo. `is_complete`
se refiere al objetivo global; `is_observed_complete` exige 60 valores globales
originales, no imputados. Los submedidores no son energía adicional al total.

## PVGIS: generación calculada, fuente independiente

- Portal vigente: https://photovoltaic-geographic-information-system.ec.europa.eu/api/
- OpenAPI consultado: https://photovoltaic-geographic-information-system.ec.europa.eu/api/v6/openapi.json
- Documentación interactiva: https://photovoltaic-geographic-information-system.ec.europa.eu/api/v6/redoc
- Endpoint implementado: `GET https://photovoltaic-geographic-information-system.ec.europa.eu/api/v6/power/broadband`.
- PVGIS pertenece al Joint Research Centre de la Comisión Europea. Ver términos
  y licencia de sus datos en el servicio; la licencia de un esquema/software no
  sustituye la de los datos meteorológicos subyacentes. No se atribuye CC BY 4.0
  de UCI a PVGIS. No se redistribuyen respuestas completas por Git.

La especificación v6 consultada documenta series de 2014–2025, irradiancia SARAH-3
y ERA5, y variables meteorológicas ERA5/ERA5-Land. La cobertura y disponibilidad
se validan finalmente en el servidor; no todas las fuentes cubren cualquier
coordenada. Buenos Aires usa ERA5 como ejemplo explícito. Los límites temporales
del adaptador se fijan a la versión revisada; ampliarlos requiere volver a
verificar el contrato. PVGIS v6 no se presume equivalente a v5.3.

Parámetros: `latitude`, `longitude`, `start_time`, `end_time`, `frequency=Hourly`,
`timezone=UTC`, `irradiance_source=ERA5`, `peak-power` (con guion, en kWp),
`system_efficiency` (fracción), `surface_tilt`, `surface_orientation`,
`photovoltaic_module` y `verbose=1`. Se proporcionan exactamente tres de los
cuatro parámetros de tiempo: inicio, fin y frecuencia; no `periods`.
Los límites son horas completas UTC e inclusivos en la lista de timestamps.

La respuesta real incluye `timestamps` y `power`. El adaptador exige exactamente
la grilla UTC horaria solicitada y valores no negativos finitos; rechaza una
respuesta parcial o diferente sin reconstruir fechas por conjetura. Comprueba
potencia pico declarada. `unit=radians` es angular: no usarlo como unidad de
generación. El vector de potencia W se divide por 1000; la aproximación horaria
a kWh y sus limitaciones están documentadas en `units.md`.

v6: orientación desde norte (0°), este (90°), sur (180°), oeste (270°).
La [API 5.3 oficial](https://joint-research-centre.ec.europa.eu/photovoltaic-geographical-information-system-pvgis/using-pvgis-5/api-non-interactive-service_en)
usa `seriescalc`, `lat/lon`, `peakpower`, `loss` porcentual y azimut desde sur;
es un contrato diferente y no se usa como sustitución silenciosa.

## Separación real/simulado

- `data/raw/uci/`: mediciones reales históricas.
- `data/raw/pvgis/`: respuestas originales con **generación modelada**, no medida.
- `data/processed/uci/` y `data/processed/pvgis/`: fuentes procesadas por separado.
- `data/simulated/`: reservado para futuros escenarios con mapeo explícito,
  supuestos de estacionalidad, ubicación, zona y escalamiento; hoy no se generan.
- `tests/fixtures/`: datos artificiales pequeños y rotulados, no extractos reales.

No se afirma coincidencia de vivienda ni simultaneidad. UCI 2006–2010 y la
cobertura v6 documentada ni siquiera se superponen. Un futuro alineamiento por
hora del día/estación o fecha artificial debe identificarse como simulación;
no es una validación de autoconsumo histórico en la vivienda UCI. PVGIS histórico
también actúa como información solar oráculo si se usa para planificar el mismo
período: explicitarlo hasta desarrollar un pronóstico solar operativo.

## Reproducibilidad y límites

Cada adquisición conserva URL exacta, timestamp UTC y SHA-256 en un recibo
adyacente. El hash identifica bytes descargados, **no certifica autenticidad** ni
igualdad con una versión UCI oficial fijada externamente. Se mantiene verificación
TLS; no se desactiva el certificado. Para reproducir una ejecución, preservar
datos brutos/recibos, manifest y entorno congelado fuera del repositorio.

El pipeline registra hashes de entrada/CSV, parámetros y versiones y no tiene
pasos aleatorios. Reejecutar sobre los mismos bytes y entorno produce iguales
particiones y valores; no se garantiza identidad binaria de PNG entre versiones
de Matplotlib/fuentes/sistemas operativos. No se guarda todo el ambiente del usuario.

Las consultas HTTP tienen timeout de 45 s por operación, hasta 3 intentos y
esperas de 1/2 s entre reintentos transitorios; el timeout no es un límite total
de duración de una descarga larga. El tamaño se limita y la escritura descargada
es atómica. No hay reintentos infinitos, ejecución oculta ni descargas en tests.
Para un archivo solar de muchos años, fraccionar consultas en períodos menores.

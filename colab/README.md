# BEKANTOR Energy — versión académica

Trabajo Final de Desarrollo de Sistemas de Inteligencia Artificial,
Instituto Superior Santo Domingo. El único notebook principal es
[BEKANTOR_Energy_Trabajo_Final.ipynb](BEKANTOR_Energy_Trabajo_Final.ipynb).
Incluye resultados, cinco gráficos y código ejecutable; no traslada todo el MVP.

## Abrir y ejecutar en Google Colab

1. [Abrir el notebook en Google Colab](https://colab.research.google.com/github/moiseslobayza/bekantor-energy/blob/main/colab/BEKANTOR_Energy_Trabajo_Final.ipynb).
   También se puede usar **Archivo → Subir notebook** para abrir el `.ipynb`.
   Sólo se carga el notebook, ningún dato.
2. Elegir CPU/sin acelerador. Usar **Entorno de ejecución → Reiniciar sesión y
   ejecutar todo** para reproducir desde un estado limpio.
3. Esperar la descarga automática de UCI (~20 MB) y, si falta alguna biblioteca,
   su instalación. Las tablas, gráficos y conclusiones se producen en orden.

No requiere Drive, GPU, claves, rutas Windows, instalar BEKANTOR ni descargar
modelos. El perfil solar pequeño está incluido en el propio notebook. Funciona
sin publicar previamente los assets en GitHub. Requiere acceso público a UCI y,
cuando corresponda, PyPI. Se utiliza TensorFlow/Keras, NumPy, pandas y Matplotlib.

**Validación:** la ejecución local se realizó en un kernel nuevo de CPU, con un
entorno virtual independiente y directorio de trabajo inicialmente vacío.
Además, el usuario confirmó una ejecución exitosa dentro de Google Colab antes
de esta consolidación. Esa comprobación en el servicio fue realizada por el
usuario; las tablas y los registros locales de referencia se conservan sin
cambios. Las versiones de Colab y sus tiempos pueden diferir del entorno local.

## Datos y protocolo fijado

- Consumo: UCI, Hebrail y Berard, DOI `10.24432/C58K54`, **CC BY 4.0**.
  Selección calendario 01/01–30/06/2007: 4.344 horas, todas dentro del train
  original del MVP (que termina el 20/09/2009). Se descarga el ZIP público y se
  procesa su prefijo necesario, sin abrir el `test.csv` histórico del proyecto.
- Potencia activa en kW integrada por minuto a kWh horarios. Relleno causal
  máximo de cinco minutos; las horas todavía desconocidas se conservan como NaN.
  No se evalúan objetivos imputados. La agregación se contrastó con el train
  procesado del núcleo; el CSV académico se redondea a nueve decimales y verifica
  por SHA-256. No se borran huecos para unir horas no consecutivas.
- Particiones académicas: train 126 días (01/01–06/05), validación 27 días
  (07/05–02/06) y test 28 días (03/06–30/06). Este **test académico es diferente
  del test histórico** del MVP. EDA y normalización usan sólo train; EarlyStopping
  usa validación; no se seleccionan parámetros con test.
- MLP 168→64→32→24, ReLU/softplus, Adam 0,001 y MSE; semilla 2026, máximo
  25 épocas, paciencia 4. Sin calendario, pérdidas ponderadas ni búsqueda de
  hiperparámetros. Persistencia de 24 h evaluada sobre los mismos objetivos.
- Solar: primeras 48 horas cronológicas de PVGIS 6 Córdoba, 15–16/01/2023,
  ERA5, 3 kWp, inclinación 30° norte, eficiencia 0,86 ya aplicada. Timestamps UTC
  y locales, consulta original y atribución JRC se conservan en la procedencia.
  Se alinean horas ordinales; **UCI y PVGIS no son mediciones sincronizadas**.
- Episodio energético: primer origen de test académico aceptado a medianoche,
  posterior a toda la selección por EarlyStopping. Se fija por fecha/calidad,
  nunca por costos favorables. Horizonte 24 h, replanificación cada hora, sólo
  la primera acción se ejecuta con consumo realizado. No se actualizan pesos.
- Batería 5 kWh, SOC inicial 50 %, piso 20 %, potencia 2,5 kW, eficiencias 95 %,
  grilla 0,5 kWh, solar futuro ideal y tarifas horarias ilustrativas en SIM.
  Se comparan reactivo, A* con persistencia y A* con MLP en condiciones pareadas.

Los resultados son una demostración: una vivienda, muestra/semilla únicas y
un día simulado. La cobertura se reduce por ventanas con datos desconocidos.
El test académico procede del train del MVP y no constituye un nuevo test
independiente del producto completo. No hay degradación, hardware, cortes,
resiliencia, dashboard ni garantía de ahorro real. Mayor precisión predictiva
no garantiza menor costo energético. El costo ajustado por SOC es contabilidad
ilustrativa, no una factura.

## Resultados locales de referencia

TensorFlow 2.21.0, Python 3.13.15, CPU; una configuración y semilla fijas.
EarlyStopping restaura el mejor modelo tras 16 épocas ejecutadas (~2,2 s de
entrenamiento en este equipo). Hay 841 ventanas de train, 25 de validación y
17 de test; las 408 etiquetas de test son observadas y no solapadas.

| Partición | Modelo | MAE kWh | RMSE kWh |
|---|---|---:|---:|
| Validación académica | Persistencia 24 h | 0,6458 | 0,9535 |
| Validación académica | MLP | 0,5784 | 0,7583 |
| Test académico | Persistencia 24 h | 0,6205 | 0,9700 |
| Test académico | MLP | 0,5965 | 0,7687 |

Episodio fijado: 03/06/2007, posterior al fin de selección de pesos
(02/06/2007 23:00). SOC inicial común 50 %; cero violaciones físicas.

| Estrategia | Costo bruto SIM | Costo ajustado SIM | Red kWh | SOC final |
|---|---:|---:|---:|---:|
| Reactivo | 4,8161 | 5,0299 | 18,6521 | 20 % |
| A* persistencia | 4,7397 | 4,7397 | 21,3515 | 50 % |
| A* MLP | 4,2329 | 4,4466 | 19,9119 | 20 % |

MLP mejora la predicción y A* + MLP reduce costo en este ejemplo, pero importa
más red que el reactivo. No se extrapola a otros días ni se selecciona una
estrategia del producto a partir de este test académico. Los saldos finales
difieren: por eso se muestran ambos costos y el SOC.

La ejecución aceptada y su auditoría están en
`experiments/phase5/validation-05-delivery/`. Se conservan los intentos de
instalación y ejecución anteriores. Los primeros replays de desarrollo usaban
un episodio de validación: se corrigió a test académico para que EarlyStopping
no consultara observaciones posteriores al origen. No se cambiaron arquitectura,
semilla, entrenamiento ni tarifas para favorecer resultados. También se probó
la instalación automática de Matplotlib al faltar en el entorno descartable.

## Archivos y extracción del núcleo

- `academic_data.py`: descarga pública y extracción horaria reproducible,
  compartidas con las celdas; no depende del paquete del proyecto.
- `assets/consumption_hourly.csv` (~178 KiB): copia reducida para procedencia,
  verificación y reconstrucción; el notebook la reproduce desde UCI.
- `assets/solar_cordoba_48h.csv` (~3 KiB) y `assets/provenance.json`: perfil
  independiente, fuentes, licencia UCI, períodos, transformación y hashes.
- `build_notebook.py`: genera las celdas desde siete archivos del núcleo.
- `validate_notebook.py`: ejecución secuencial aislada y auditoría independiente.
- `tests/test_academic.py`: nueve pruebas pequeñas sin entrenamiento.
- `.gitattributes`: conserva LF en los CSV y en `academic_data.py` para que
  sus hashes no cambien por la conversión automática de finales de línea.

El generador extrae ventanas, persistencia, normalización, contratos físicos,
simulador, cola A* y métodos del grafo. No se mantienen a mano otras versiones
de esos algoritmos. Sólo adapta contenedores y construcción del grafo a
parámetros fijos; no conserva toda la infraestructura del MVP. Los hashes y el
commit fuente están en `metadata.bekantor` del notebook. Ante cambios del núcleo,
regenerar, revisar simplificaciones y volver a ejecutar; no editar sus copias
manualmente. El notebook no importa esos archivos al ejecutarse en Colab.

## Mantenimiento y validación local opcionales

Esto no es necesario para el profesor. Desde la raíz, usar un entorno separado
del MVP. En Windows conviene una ruta corta para las cabeceras de TensorFlow:

```powershell
python -m venv experiments/e
.\experiments\e\Scripts\python.exe -m pip install tensorflow==2.21.0 numpy==2.5.4 pandas==3.0.6 matplotlib==3.11.2 nbformat nbclient ipykernel
.\experiments\e\Scripts\python.exe colab/build_notebook.py
.\experiments\e\Scripts\python.exe -m unittest discover -s colab/tests -v
.\experiments\e\Scripts\python.exe colab/validate_notebook.py --run-dir experiments/phase5/repeat-01
```

Regenerar elimina las salidas: ejecutar después el validador para guardarlas.
Usar un directorio nuevo por corrida. El validador no necesita BEKANTOR instalado.
Compara métricas con los arreglos originales, verifica 24 orígenes causales,
72 pasos físicos, normalización sólo de train y que la selección de pesos sea
anterior al episodio. Contrasta A*/UCS/enumeración pequeña en las celdas y
exporta los cinco gráficos para inspección. El notebook también guarda tablas
y arreglos de auditoría en `bekantor_academic/` dentro del runtime, sin modelos.

Para reconstruir los assets desde los caches originales del proyecto:

```powershell
.\experiments\e\Scripts\python.exe colab/build_notebook.py --prepare-assets --archive data/raw/uci/uci_household.zip
```

Requiere train procesado y el perfil/recibos PVGIS originales sólo para esa
extracción de mantenimiento. Omitir `--archive` descarga UCI automáticamente.
Los recibos y gráficos de validación quedan localmente en `experiments/phase5/`,
ignorados por Git. El MVP, sus modelos y resultados históricos se conservan.
Los registros originales de validación corresponden al estado previo a la
consolidación académica en GitHub.

# Extracción académica futura (sin notebook en Fase 1)

Se mantendrá un solo núcleo en `src/bekantor_energy`. El futuro notebook Colab
será una extracción acotada y versionada, autocontenida en sus celdas y ejecutable
sin un checkout local. Un exportador futuro copiará desde funciones del núcleo
(sin mantener a mano una segunda implementación), con versión/hash de origen.
También podrá descargar las fuentes públicas. No dependerá de rutas Windows,
credenciales, modelos preentrenados ni hardware. Se verificará con ejecución
completa en CPU desde un runtime limpio.

Secuencia prevista: fuentes y unidades → EDA de train → preprocesamiento causal
y ventanas con límites cronológicos → MLP Keras pequeño → evaluación → simulador
reducido → A* con demanda predicha → comparación convencional y discusión de
búsqueda/optimización. PVGIS se presentará como escenario independiente modelado.

Fase 2 ya implementa ventanas, baselines, MLP y entrenamiento en el núcleo Python.
El simulador, A* y el notebook siguen pendientes; no se duplica esa lógica aquí.

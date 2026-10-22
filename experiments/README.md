# Resultados reproducibles

Salidas de procesamiento, EDA, entornos y consultas de verificación. Todo el
contenido generado se excluye de Git. Registrar cada experimento futuro con
configuración, semilla, versiones, fuentes y métricas. Fase 1 no entrena modelos;
Fase 2 agrega la primera comparación MLP controlada.

Fase 2 guarda en `phase2/<corrida>/` la configuración, selección por validación,
métricas, predicciones, gráficos y metadatos reproducibles. El marcador de test
impide reevaluarlo en la misma corrida. No eliminarlo para repetir una medición.

# IE1 — Documentación de fuente de datos externa

## Fuente: megaGymDataset (Kaggle)

- **Origen:** Kaggle — dataset "Gym Exercise Data" de niharika41298
  (https://www.kaggle.com/datasets/niharika41298/gym-exercise-data)
- **Formato original:** CSV
- **Tamaño:** ~2500+ ejercicios, columnas incluyen Title, Type, BodyPart, Equipment, Level, Rating, Desc
- **Licencia:** [Revisar y confirmar en la página del dataset en Kaggle antes de publicar/distribuir]
- **Fecha de descarga:** [fecha exacta en que se descargó el archivo megaGymDataset.csv]
- **Ruta local:** data/01_raw/megaGymDataset.csv

## Propósito de la integración

Enriquecer el catálogo base de ejercicios (`exercises_raw` / `exercises.json`) con
metadatos adicionales no presentes en la fuente original: `Type`, `Level` y `Rating`.

## Método de integración

- **Técnica:** Fuzzy matching sobre nombres de ejercicios, usando la librería `rapidfuzz`
  (scorer: `WRatio`, umbral mínimo de aceptación: 80/100)
- **Campos cruzados:** `name` (catálogo base) ↔ `Title` (megaGymDataset)
- **Nodo Kedro:** `enrich_with_external_metadata` (pipeline `data_processing`)

## Cobertura real del match (evidencia, no asumida)

- **Cobertura obtenida:** 99.62%
- **Método de cálculo:** porcentaje de registros del catálogo base que encontraron
  una coincidencia con score ≥ 80 en el dataset externo
- **Registros sin match:** se rellenan con el valor `"No especificado"` en las
  columnas `Type`, `Level` y `Rating` (no se descartan filas)

## Salida

- **Archivo generado:** `data/03_primary/exercise_features_enriched.parquet`
- **Formato:** Parquet (columnar, tipado estricto — por eso las columnas mixtas
  Type/Level/Rating se normalizaron a tipo string antes de guardar)

## Limitaciones conocidas

- El fuzzy matching puede generar falsos positivos en nombres de ejercicios muy
  genéricos o cortos (no verificado manualmente registro por registro)
- La licencia de Kaggle debe confirmarse antes de cualquier uso comercial o
  redistribución del dataset combinado
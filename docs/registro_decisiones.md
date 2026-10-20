# Registro de decisiones

Qué se probó, qué se descartó y por qué. El README y los notebooks muestran el camino
final; este registro guarda también los descartes, para que cualquier integrante pueda
explicar por qué el proyecto no hizo otra cosa.

Cada fila indica dónde está la evidencia.

## EV1 — Comprensión y preparación de los datos

| Decisión | Alternativas consideradas | Por qué | Evidencia |
|---|---|---|---|
| Músculo primario = `target` | `muscle_group`, usado en la versión preliminar | `muscle_group` está dentro de `secondary_muscles` en el 100 % de los registros: es un sinergista. Usarlo invertía todas las aristas agonista/sinergista | README §3.3 |
| `id` declarado como texto en el catálogo de Kedro | Corregir el tipo después de cargar | `pandas.read_json` convierte `"0001"` en `1` al cargar; con la llave dañada solo el 46,8 % reconstruía su ruta de medios | README §3.3, §4.4 |
| Parquet en la capa intermedia | CSV | Conserva los tipos (la llave `id`) y reduce el tamaño ~94 % | README §3.3 |
| Emparejamiento estricto con megaGymDataset (`token_sort_ratio` ≥ 85 + coherencia de región y equipamiento) | `WRatio` ≥ 80 (rama `benja`) | `WRatio` daba 99,6 % de cobertura con 56 % de precisión; la regla estricta, 31,9 % con 96 %. Se prefiere un nulo honesto a una etiqueta de otro ejercicio | README §3.5, notebook 01 §14 |
| Regla de contención para recuperar cobertura | — | Queda como mejora: requiere una muestra nueva revisada a mano | README §6 punto 6 |
| Detectar atípicos con IQR y no eliminarlos | *z-score*; eliminar filas | Las variables no son normales (el IQR es robusto). Los atípicos son extremos legítimos: eliminarlos sesgaría el catálogo hacia lo simple | README §4.1 |
| No imputar mecánica, fuerza ni dificultad | Imputar desde `n_pasos` o `n_musculos_total` | La información no existe en la fuente y la dificultad no se asocia con esas variables (Mann-Whitney p > 0,1) | README §3.4, §4.5 |
| Usar solo `nivel_curado` de megaGymDataset | Usar `Level` tal cual | `Intermediate` es el valor de relleno de las fichas sin valoración (99,7 %) | README §3.5 |
| No entrenar con los GIF | Modelo temporal sobre pose extraída de los GIF | 180×180 por contrato, 4 FPS, una repetición, ilustraciones sin etiquetas: el problema es la señal, no el volumen | README §4.4 |
| Databricks por gobierno del dato | Justificarlo por volumen | El dato pesa 260 KB; el argumento de volumen no se sostiene | README §2.3 |

## EV2 — Modelado y evaluación

| Decisión | Alternativas consideradas | Por qué | Evidencia |
|---|---|---|---|
| Regiones compuestas como nodo propio, con articulación solo por consenso | Excluir la arista; *fan-out* a los músculos componentes | No pierde ejercicios y no afirma una articulación que la fuente no afirma | README §11.2 |
| *Fan-out* de regiones en el grafo | — | Se probó en la fase 3: empeoró la cobertura de sustitución sin mejorar la mezcla de fuentes. Descartado | README §13.4 |
| Vetar pares con modificadores opuestos | Aceptar todo par sobre el umbral | Se aceptaba `hip abduction` ↔ `hip adduction` | README §11.3 |
| Perfil de exigencia en vez de dificultad | Predecir la dificultad combinada de dos fuentes | Las fuentes coinciden igual que al azar (kappa −0,01) | README §11.5, §12.1 |
| Etiquetas solo de free-exercise-db | Mezclar con megaGymDataset | Un solo curador, criterio coherente | README §12.1 |
| Variables idénticas en las tres fuentes | Usar `category` de free-exercise-db | El modelo aprendería a reconocer la fuente | README §12.2 |
| Grupos de variantes para separar train y test | Separación al azar | Las variantes casi idénticas a ambos lados miden memoria | README §12.2 |
| F1 macro | *Accuracy* | `static` es el 12 %: la *accuracy* esconde el fallo en la clase chica | README §12.2 |
| Regresión logística y Random Forest | XGBoost | El Random Forest ya cubre las interacciones no lineales; XGBoost no se agregó | `docs/plan_ev2.md` |
| `C` con la regla de un error estándar | El máximo exacto de la CV | El F1 es casi plano entre `C` = 10 y 300: el máximo exacto es ruido | README §12.2, notebook 02 §6 |
| Umbral de abstención con 95 % de acierto | 90 % | La validación externa muestra una caída de 4–5 puntos en el catálogo | README §12.2 |
| Un ejercicio carga una articulación si la cruza un agonista **o** un sinergista | Solo agonistas | En la sentadilla, el cuádriceps (sinergista) carga la rodilla | README §13.2 |
| Node2Vec + K-Means como definición de familia | Louvain | Louvain gana en modularidad, pero 8 de sus 24 comunidades son de una sola fuente y su cobertura es 55,8 % frente a 91,0 % (57,4 % frente a 90,7 % con la carga por patrón) | README §13.3 |
| K-Means sin HDBSCAN | HDBSCAN | K-Means cumplió el KPI con familias estables; HDBSCAN no hizo falta | `docs/plan_ev2.md` |
| k = 12 | k = 4 a 28 | Último k antes de que caiga la *silhouette*, con estabilidad 0,91 | README §13.3 |
| "Fuerza por fuerza, estiramiento por estiramiento" | Sin esa regla | La demostración proponía estiramientos como alternativa a una sentadilla | README §1.4, §13.5 |

## Revisión del 2026-10-20 — diagnóstico de los modelos supervisados

| Qué se probó | Resultado | Decisión |
|---|---|---|
| CV repetida 5 × 5 con contraste pareado (Nadeau-Bengio) entre logística y Random Forest | Ninguna diferencia supera el ruido. En fuerza, el Random Forest gana en el 80 % de los pliegues por 0,016 (t = 1,20, p = 0,24) | Se informa en README §12.6. Lectura: en fuerza los dos algoritmos empatan dentro del ruido |
| Reentrenar con validación estratificada por grupos y elegir el algoritmo con el contraste pareado (más estricto) | Gana la logística en los tres objetivos; en mecánica cambia `C` a 0,3. Cobertura del catálogo: mecánica 85,5 % → 70,3 %, fuerza 95,5 % → 93,3 %. F1 externo de fuerza 0,89 → 0,80. Cobertura de sustitución 91,0 % → 88,4 % (rodilla 80,9 % → 67,5 %) | **Descartado.** No mejora ninguna métrica fuera de muestra y empeora las del producto. Se conserva la regla original, fijada antes de entrenar |
| Recalibrar las probabilidades (sigmoide e isotónica) | Con los modelos actuales, el Brier mejora solo en fuerza (0,164 → 0,130). En la variante estricta, la isotónica de mecánica bajó el F1 externo de 0,81 a 0,73: deshizo el balanceo de clases y la calibración ajustada en free-exercise-db no se trasladó al catálogo | **No se recalibra.** El umbral de abstención ya se eligió por acierto real. Si la aplicación muestra la confianza, recalibrar con etiquetas del catálogo |
| Carga articular por patrón de movimiento: sumar a la carga deducida de los músculos las articulaciones que el nombre delata (`data/01_raw/carga_por_patron.csv`) | La revisión manual encontró *Kettlebell deadlifts* como alternativa a un peso muerto rumano con la lumbar vetada: su ficha no anota el músculo lumbar. Al 40 % de los ejercicios que cargan la lumbar por su nombre les faltaba la articulación. La regla toca 145 ejercicios; la cobertura baja de 91,0 % a 90,7 % (lumbar 92,6 % → 88,8 %), siempre sobre la meta | **Adoptado.** Corrige un defecto de seguridad sin cambiar las familias. La cifra anterior contaba como seguras alternativas que no lo eran |
| Objetivo barajado, validación adversarial, curva de aprendizaje, importancia en validación, revisión de errores y concordancia de etiquetas | Sin fuga; fuerte cambio de dominio entre fuentes (AUC 0,999); fuerza y complejidad mejorarían con más datos; 23,5 % de los grupos de variantes con complejidad discordante | Se informa en README §12.6 y notebook 02 §13. Recomendación: corregir las etiquetas discordantes antes de reentrenar |

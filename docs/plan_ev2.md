# Plan de trabajo — EV2 (MLY1101)

Hoja de ruta de las fases que faltan para la Evaluación Parcial N°2. Se actualiza a
medida que se cierra cada tarea (`[x]`).

## Qué evalúa la rúbrica

| Indicador | Peso | Qué tiene que existir |
|---|---|---|
| IE5 — Metodología (CRISP-DM) | 20 % | Las seis fases, coherentes entre sí, documentadas en el README |
| IE6 — Modelos supervisados | 30 % | **Al menos 2 modelos comparados**, con justificación del algoritmo y su configuración |
| IE7 — Modelo no supervisado | 30 % | **Al menos 1 técnica** que descubra patrones o grupos, interpretada para el negocio |
| IE8 — Interpretación | 20 % | Métricas explicadas y traducidas a recomendaciones |

**Entregables:** README.md como informe técnico, notebook `.ipynb` ejecutable de
principio a fin, datos incluidos y carpetas `data/`, `notebooks/`, `models/`, `images/`.
**Presentación:** 10 minutos, con preguntas cruzadas: cada integrante debe poder defender
cualquier parte.

El profesor autorizó seguir con el proyecto del gimnasio en lugar de los casos A/B/C.

---

## Fase 1 — Integración de fuentes ✅ (cerrada el 2026-09-22)

Detalle en README §11. Resultado: 2.592 ejercicios únicos, ontología común de músculos,
articulaciones y equipamiento, y etiquetas `mecanica` / `fuerza` / `nivel` donde las
fuentes las traen.

**Pendiente del equipo:**

- [ ] Revisar los veredictos de `data/01_raw/fuentes_match_validacion_manual.csv`,
      sobre todo los marcados *variante* y *distinto*.
- [ ] Revisar los veredictos pendientes de la EV1 en
      `data/01_raw/megagym_match_validacion_manual.csv` (el archivo tiene cambios sin
      commitear).
- [ ] Revisar las decisiones de ontología: `regiones_compuestas.csv` (en especial
      `spine` → erector espinal) y `vocabulario_muscular_externo.csv` (`middle back` →
      `upper back`, `Anterior deltoid` → `deltoids`).

---

## Fase 2 — Modelos supervisados: completar las etiquetas (IE6 · 30 %)

**Pregunta de negocio:** ¿se puede asignar mecánica (compuesto/aislado) y tipo de fuerza
(empuje/tracción/estático) a todo el catálogo, para que el recomendador pueda sustituir
un ejercicio por otro del mismo patrón de movimiento?

**Qué hacer con el nivel de dificultad (decisión del 2026-09-22).** El nivel *combinado*
no sirve: megaGymDataset y free-exercise-db coinciden igual que al azar (kappa = −0,01,
README §11.5). El nivel de free-exercise-db **por sí solo** sí sirve, porque tiene un
solo curador y sigue una lógica coherente: los levantamientos técnicos son 91 %
intermedio o experto, los ejercicios aislados 78 % principiante y la banda elástica 100 %
principiante. Lo que mide es **complejidad técnica**, no riesgo: marca como principiante
el 93 % de los saltos.

Por eso la dificultad se reemplaza por un **perfil de exigencia** de varias dimensiones:

| Dimensión | Cómo se obtiene | Para quién importa |
|---|---|---|
| Complejidad técnica | Modelo supervisado entrenado **solo** con el nivel de free-exercise-db (principiante vs. intermedio/experto) | Principiante |
| Mecánica (compuesto/aislado) | Modelo supervisado | Principiante, sustitución |
| Fuerza (empuje/tracción/estático) | Modelo supervisado | Sustitución por patrón de movimiento |
| Impacto | Regla transparente: saltos o pliometría | Lesión, adulto mayor |
| Estabilidad | Regla: apoyado (máquina, sentado, acostado) o inestable (una pierna, balón) | Ambos |
| Carga graduable | Regla: banda, máquina o cable | Ambos |
| Articulaciones que carga | Grafo (Fase 3) | **Lesión** |

- **Principiante:** baja complejidad técnica, apoyado, sin impacto y con carga graduable.
- **Lesión:** el grafo descarta los ejercicios que cargan la articulación afectada; entre
  los que quedan se prefieren los apoyados y sin impacto. Si quedan menos de 3
  alternativas, el sistema se abstiene (README §5.3).
- El nivel de megaGymDataset queda solo como dato descriptivo.
- **Límite declarado:** qué combinación del perfil es "apta para principiante" o "apta
  para lesión" es un criterio de diseño del equipo, no una validación clínica.

**Tareas**

- [ ] Agregar `scikit-learn` (y opcionalmente `xgboost`) a `requirements.txt`.
- [ ] Pipeline Kedro `modelado_supervisado` o `notebooks/02_modelado_supervisado.ipynb`.
- [ ] **Datos de entrenamiento:** los 876 ejercicios de free-exercise-db
      (`data/02_intermediate/free_exercise_db.parquet`: 789 con `mecanica`, 846 con
      `fuerza`). Se usa esa tabla y no `ejercicios.parquet`, que solo conserva como
      nodo propio a los 549 sin par en el catálogo.
      **Datos a predecir:** los ejercicios sin etiqueta (catálogo y wger).
- [ ] **Variables:**
  - [ ] TF-IDF del nombre y de `instrucciones_en`.
  - [ ] Músculos agonistas y sinergistas en *multi-hot* (vocabulario canónico).
  - [ ] Articulaciones derivadas (`musculo_articulacion.csv`) y número de articulaciones
        implicadas. Es el indicador biomecánico directo de "compuesto".
  - [ ] Familia de equipamiento (*one-hot*).
- [ ] **Evitar fuga de información:** agrupar por `nombre_norm` o `grupo_variacion` al
      separar train/test (`GroupKFold`), para que variantes casi idénticas no queden a
      ambos lados.
- [ ] **Tres objetivos:** `mecanica` (binario), `fuerza` (3 clases) y complejidad técnica
      (binario: `beginner` vs. `intermediate`/`expert`; *expert* solo tiene 57 casos).
- [ ] **Reglas del perfil de exigencia** (impacto, estabilidad, carga graduable) con una
      muestra revisada a mano, porque se basan en palabras clave.
- [ ] **Modelos a comparar (mínimo 2):**
  - [ ] Regresión logística (base interpretable; coeficientes por variable).
  - [ ] Random Forest o XGBoost (interacciones no lineales entre músculos y equipo).
  - [ ] Ajuste de hiperparámetros con validación cruzada (`GridSearchCV` o
        `RandomizedSearchCV`), documentando el espacio de búsqueda.
- [ ] **Métricas:** F1 macro (hay desbalance: `static` es 12 %), matriz de confusión,
      *precision/recall* por clase, importancia de variables (coeficientes y
      *permutation importance*).
- [ ] **Validación externa:** comparar la predicción con las etiquetas transferidas a los
      ejercicios del catálogo con par (`mecanica_origen == "free_exercise_db"`). Mide si
      el modelo generaliza al vocabulario y la redacción del catálogo.
- [ ] **Regla de abstención:** etiquetar solo si la probabilidad supera un umbral
      elegido en validación; si no, dejar nulo. Es el criterio de README §5.3.
- [ ] **KPI nuevo:** % del catálogo con mecánica y fuerza asignadas con confianza.
- [ ] Guardar el mejor modelo en `models/` (joblib) y las predicciones en
      `data/07_model_output/`.
- [ ] Commit por parte: dependencias · código del modelo · modelo y predicciones ·
      notebook · README.

---

## Fase 3 — Grafo biomecánico no supervisado (IE7 · 30 %)

**Pregunta de negocio:** ¿qué familias de movimiento existen en el catálogo, y cuántas
alternativas reales tiene un ejercicio cuando se veta una articulación lesionada?

**Tareas**

- [ ] Agregar `networkx` (y `node2vec` o `gensim` para los *embeddings*) a
      `requirements.txt`.
- [ ] **Construir el grafo** a partir de:
  - [ ] Nodos Ejercicio (`ejercicios.parquet`, con mecánica y fuerza de la Fase 2).
  - [ ] Aristas Ejercicio → Músculo (`ejercicio_musculo.parquet`), con peso según el rol
        (agonista > sinergista).
  - [ ] Aristas Músculo → Articulación (`musculo_articulacion.csv`).
  - [ ] Aristas Región → Músculo *parte de* (`regiones_compuestas.csv`).
  - [ ] Aristas Ejercicio → Familia de equipamiento.
- [ ] **Técnicas no supervisadas (mínimo 1; se proponen 2 para comparar):**
  - [ ] Detección de comunidades con Louvain sobre la proyección ejercicio–ejercicio.
        Métrica: modularidad.
  - [ ] *Embeddings* con Node2Vec + K-Means / HDBSCAN. Métricas: *silhouette* y
        Davies-Bouldin; elegir k con el método del codo y *silhouette*.
  - [ ] Visualizar con UMAP o t-SNE, coloreando por *cluster* y por contexto.
- [ ] **Interpretar los clusters** como familias de movimiento (p. ej. "empuje
      horizontal de hombro", "extensión de cadera"). Revisar si el catálogo y las
      fuentes nuevas se mezclan o quedan separados: si quedan separados, el modelo
      aprendió la fuente y no la biomecánica.
- [ ] **Demostración de negocio: sustitución con articulación vetada.**
  - [ ] Para un ejercicio y una articulación lesionada (p. ej. rodilla), buscar vecinos
        del mismo cluster que no carguen esa articulación.
  - [ ] Medir el KPI "Cobertura de sustitución": % de ejercicios con ≥ 3 alternativas
        (meta 85 %, README §1.3), por articulación y por contexto.
  - [ ] Demostración para la presentación: "principiante con rodilla lesionada busca
        alternativa a la sentadilla", aplicando el perfil de exigencia sobre el grafo.
  - [ ] Implementar la mitigación de sesgo de §5.1: cuota de alternativas sin
        equipamiento y abstención cuando el músculo tenga < 10 ejercicios
        (`levator scapulae` sigue en 2).
- [ ] Guardar el grafo y los *embeddings* en `data/06_models/` o `models/`.
- [ ] Commit por parte: dependencias · construcción del grafo · clustering ·
      sustitución · notebook · README.

---

## Fase 4 — Evaluación, informe y defensa (IE5 + IE8 · 40 %)

- [ ] **README como informe EV2:**
  - [ ] Reescribir el aviso de "Alcance": las fases 4 y 5 de CRISP-DM ahora están
        cerradas.
  - [ ] Actualizar la tabla de §1.4 (estado de cada fase CRISP-DM).
  - [ ] KPIs: agregar los nuevos (cobertura de etiquetas con confianza, cobertura de
        sustitución medida) y actualizar la tabla de §1.3 con valores reales.
  - [ ] Secciones de modelado supervisado, no supervisado y evaluación, con métricas
        interpretadas en lenguaje de negocio.
  - [ ] Recomendaciones de negocio concretas a partir de los resultados.
  - [ ] Actualizar §6 (limitaciones) y §7 (arquitectura: qué quedó implementado).
- [ ] **Sesgo después del modelado:** ¿mejoró la cobertura de rehabilitación?
      ¿El modelo se equivoca más en algún contexto o fuente?
- [ ] **Estructura exigida:** crear `models/` e `images/` (o documentar que las figuras
      viven en `data/08_reporting/figures/`).
- [ ] **Reproducibilidad:** ejecutar `kedro run` y los notebooks desde cero en un entorno
      limpio (`pip install -r requirements.txt`) antes de entregar.
- [ ] **Presentación (10 min):** guion con un bloque por indicador de la rúbrica.
      Ensayar preguntas cruzadas: cada integrante explica el pipeline, el modelo
      supervisado, el grafo y el hallazgo del kappa.

---

## Opcional (solo si sobra tiempo)

- **KIMORE** (78 sujetos, 5 ejercicios de rehabilitación, puntaje clínico 0–50): tercer
  modelo supervisado, regresión de la calidad de ejecución. Requiere acuerdo del equipo,
  porque en la EV1 se decidió no usar datos de pose.
- **Regla de contención** para megaGymDataset (README §10 punto 3).
- **Validación clínica** del mapeo articular con un profesional (README §10 punto 1).
  Sigue siendo el prerrequisito para producción, no para la evaluación.

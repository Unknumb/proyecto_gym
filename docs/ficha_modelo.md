# Ficha del modelo — Recomendador biomecánico

Resumen de qué es cada modelo, para qué sirve, para qué no y con qué limitaciones. El
detalle está en el README (§12–§14) y en los notebooks 02 y 03.

## Modelos

| Componente | Tipo | Archivo | Entrenado con |
|---|---|---|---|
| Mecánica (compuesto / aislado) | Regresión logística, `C` = 10, `class_weight="balanced"` | `models/modelos_supervisados.pkl` | free-exercise-db, 789 ejercicios |
| Fuerza (empuje / tracción / estático) | Random Forest, 400 árboles | `models/modelos_supervisados.pkl` | free-exercise-db, 846 ejercicios |
| Complejidad técnica (baja / alta) | Regresión logística, `C` = 1 | `models/modelos_supervisados.pkl` | free-exercise-db, 876 ejercicios |
| Familias de movimiento | Node2Vec (16 dimensiones) + K-Means (k = 12) | `models/embeddings_node2vec.parquet`, `data/07_model_output/familias_movimiento.parquet` | Grafo de 2.529 nodos y 13.907 aristas |
| Sustitución con articulación vetada | Reglas de seguridad + similitud del *embedding* | `Sustituidor` en `grafo_biomecanico/nodes.py` | — |

Reentrenamiento: `kedro run` (30 nodos, ~3 minutos), con semillas fijas y versiones
exactas en `requirements-lock.txt`.

## Uso previsto

- Completar las etiquetas de mecánica, fuerza y complejidad técnica de un catálogo de
  ejercicios, **absteniéndose** cuando la confianza no alcanza el umbral.
- Proponer alternativas a un ejercicio que trabajen el mismo músculo sin cargar una
  articulación declarada como lesionada, o abstenerse cuando no existen.
- Como apoyo informativo para el usuario y su kinesiólogo o entrenador.

## Usos no previstos

- **Diagnóstico, prescripción o tratamiento médico.** El sistema no conoce el historial
  clínico ni las contraindicaciones del usuario.
- Graduar la progresión de un usuario por "dificultad": la dificultad no es una etiqueta
  objetiva (kappa −0,01 entre fuentes).
- Aplicarlo a catálogos con otra redacción u otro vocabulario sin una validación propia:
  el cambio de dominio entre fuentes es fuerte (AUC adversarial 0,999).
- Mostrar la confianza del modelo de fuerza como una probabilidad: no está calibrada.

## Datos

- **Entrenamiento:** free-exercise-db (*The Unlicense*), un solo curador.
- **Predicción:** catálogo `exercises.json` (MIT) y wger (CC-BY-SA / CC0 por ficha).
- **Evaluación:** test por grupos de variantes (20 % de los grupos) y validación externa
  con los gemelos del catálogo (~60 ejercicios por objetivo).
- Detalle de fuentes, licencias y preparación: README §2 y §11.

## Métricas

| Objetivo | F1 macro test (IC 95 %) | F1 macro catálogo (IC 95 %) | Azar | Catálogo con etiqueta |
|---|---|---|---|---|
| Mecánica | 0,88 (0,83–0,93) | 0,81 (0,68–0,91) | 0,50 | 85,5 % |
| Fuerza | 0,88 (0,82–0,93) | 0,89 (0,78–0,96) | 0,34 | 95,5 % |
| Complejidad técnica | 0,74 (0,68–0,81) | 0,67 (0,53–0,80) | 0,49 | 32,9 % |

Sustitución: 91,0 % de los ejercicios sustituibles con ≥ 3 alternativas de su familia
(meta 85 %); hombro 78,9 %, rodilla 80,9 % y cadera 82,4 % bajo la meta. El 40,5 % de los
casos no tiene sustituto anatómico y el sistema se abstiene.

### Por subgrupo

| Subgrupo | Resultado |
|---|---|
| Contexto rehabilitación / funcional | Acierto en mecánica ≈ 78 % frente a ≈ 91 % en gimnasio; sustitución 89,5 % |
| Contexto peso corporal | Acierto en complejidad técnica ≈ 61 % |
| Contexto cardio / otros | Sustitución 75,7 %, bajo la meta |
| Fuente wger | Peor etiquetada (fuerza 69 %, complejidad 11,5 %): el modelo se abstiene más con sus instrucciones cortas |

Muestras de 18 a 100 casos: diferencias de menos de ~10 puntos no son concluyentes
(README §14.3).

## Consideraciones éticas

- Sesgo de muestreo hacia el gimnasio comercial (Gini de equipamiento 0,737). Mitigado
  con una cuota de alternativas sin equipamiento (51 % → 99 % de las listas).
- La abstención protege la calidad de las etiquetas pero redistribuye la cobertura: los
  ejercicios de wger casi no llegan al perfil de principiante.
- No hay datos personales en el entrenamiento. El diseño de destino procesa el vídeo del
  usuario solo en el dispositivo (README §5.4).

## Limitaciones

- Las etiquetas reproducen el criterio de free-exercise-db, sin validación clínica. En
  complejidad técnica, el 23,5 % de los grupos de variantes tiene etiquetas discordantes.
- El mapeo músculo → articulación, del que depende la seguridad de la sustitución, no fue
  revisado por un profesional de la salud.
- Las familias reflejan cómo están anotados los ejercicios, no una medición del
  movimiento (*silhouette* 0,23).
- El KPI de sustitución mide que las alternativas sean seguras y de la misma familia, no
  que un entrenador las considere buenas.

## Responsables

Equipo del proyecto (MLY1101, Duoc UC). Registro de decisiones en
`docs/registro_decisiones.md`.

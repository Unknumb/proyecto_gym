# Informe ejecutivo — Sustituto Seguro

**MLY1101 Machine Learning · Duoc UC · Evaluación Final Transversal**
Recomendación biomecánica de ejercicios para personas con una articulación lesionada.
Informe técnico completo: [`README.md`](../README.md) (las referencias § apuntan a él).

---

## El problema

Las aplicaciones de entrenamiento recomiendan ejercicios por similitud de texto o por
popularidad, no por mecánica corporal. A una persona con la rodilla lesionada le proponen
como "alternativa" a la sentadilla otro ejercicio que carga la misma rodilla, porque ambos
llevan la etiqueta *pierna*. El error es frecuente y puede agravar la lesión.

## Qué construimos

Un sistema que modela cada ejercicio por lo que trabaja (músculos), lo que carga
(articulaciones) y lo que exige (mecánica, tipo de fuerza, complejidad técnica). Dado un
ejercicio y una articulación lesionada, propone hasta cinco alternativas que trabajan el
mismo músculo sin cargarla, o **se abstiene** y explica por qué cuando no existe una
alternativa segura.

| Pieza | Técnica | Para qué |
|---|---|---|
| Integración de datos | Pipeline Kedro, emparejamiento por nombre validado a mano | 2.592 ejercicios de tres fuentes abiertas con un vocabulario muscular común |
| Perfil de exigencia | Regresión logística y Random Forest, con umbral de abstención | Etiquetar mecánica, fuerza y complejidad donde el catálogo no las trae |
| Familias de movimiento | Grafo biomecánico, Node2Vec + K-Means | Agrupar ejercicios por patrón para que la alternativa sea del mismo tipo |
| Sustitución | Veto duro de la articulación + ranking por similitud | Recomendar sin cargar la zona lesionada, o abstenerse |
| Despliegue | Página web, app de iPhone, reentrenamiento con los usuarios | Llevar el modelo a las personas y aprender de sus correcciones |

## Resultados contra los KPIs

| KPI | Meta | Resultado |
|---|---|---|
| Ejercicios con ≥ 3 alternativas seguras de su familia | ≥ 85 % | ✅ **89,4 %** (hombro 79,6 % y rodilla 69,3 %, bajo la meta) |
| Alternativas que cargan la articulación vetada | 0 | ✅ **0** en 13.304 consultas |
| Calidad de los modelos (F1 macro en test, mecánica / fuerza) | ≥ 0,85 | ✅ **0,88 / 0,88** |
| Catálogo con mecánica / fuerza etiquetadas | — | De 0 % a **85,5 % / 95,5 %** |
| Acierto de lo etiquetado en el catálogo | ≥ 90 % | ⚠️ 89,8 % / 88,5 %, con ±8 puntos: lo certifica la revisión manual de 105 etiquetas, preparada |
| Predicción en el teléfono, sin servidor | ≤ 33 ms | ✅ **0,2 ms** (medido en Mac) |
| Datos personales enviados a servidores | 0 | ✅ 0: edad, peso, sesiones y dolor no salen del dispositivo |

## Cómo se eligió la solución

- **Supervisado (EV2):** se compararon la regresión logística y el Random Forest con
  validación cruzada por grupos de variantes, para que un mismo ejercicio no quedara a
  ambos lados de la partición. Se eligió la logística para mecánica y complejidad técnica,
  y el Random Forest para fuerza.
- **Optimización y ensambles (EV3):** se ajustó un boosting y se probaron siete
  candidatos (*bagging*, *boosting*, *voting* y *stacking*) con una regla fijada antes de
  ver los resultados: mejora de al menos 0,02 de F1, en el 80 % de los pliegues y con
  una prueba estadística corregida. El mejor, el *stacking*, mejora solo +0,008 a +0,010
  (p ≥ 0,24). **Se mantienen los modelos de la EV2**: un modelo más caro y menos
  interpretable no se justifica por una mejora del tamaño del ruido.
- **No supervisado:** el grafo con Node2Vec + K-Means encontró 12 familias de movimiento.
  Ordenan las alternativas para que primero aparezcan las del mismo patrón, y permiten
  avisar a la persona cuando la alternativa segura es "otro movimiento".

## Lo que aprendimos

1. **Abstenerse es parte de la solución.** En el 40 % de los casos con una articulación
   vetada no hay sustituto anatómico posible: decirlo es más útil que inventar uno.
2. **La seguridad es una regla verificable, no una probabilidad.** El veto se aplica antes
   del modelo y lo comprueban tests en cada cambio.
3. **El techo está en los datos.** Ningún algoritmo mejoró a los modelos simples. Por eso
   el despliegue recoge correcciones de las personas: en simulación, 200 correcciones
   suben mecánica de 0,811 a 0,887 en el catálogo externo, y los aportes maliciosos se
   rechazan.

## Riesgos y ética

- **Seguridad:** toda la protección depende del mapeo músculo → articulación, armado
  desde Wikipedia y ExRx.net y aún sin validación clínica. La planilla para que la revise
  un profesional de la salud está lista.
- **Sesgo:** sin corrección, el ranking favorecía el gimnasio. Con una cuota sin
  equipamiento, el 98 % de las listas incluye una opción sin equipamiento. La abstención deja peor
  cubiertos a wger y al cardio, y está declarado.
- **Privacidad:** todo corre en el dispositivo y no se procesa video. Solo se exportan
  correcciones de ejercicios, cuando la persona lo decide.
- **No es un dispositivo médico:** es una herramienta informativa, y ante dolor recomienda
  consultar a un profesional.

## Recomendaciones

1. **Antes de abrirlo al público:** validar el mapeo articular con un profesional de la
   salud y completar la revisión de las 105 etiquetas, que certifica el KPI de acierto y
   reentrena el modelo.
2. **En el producto:** mantener la abstención visible, la marca "mismo / otro movimiento"
   y la cuota sin equipamiento.
3. **Con usuarios reales:** exigir tres votos coincidentes antes de que una corrección
   cambie una etiqueta, y volver a medir el aprendizaje.
4. **Siguiente versión:** análisis de postura con la cámara, en el teléfono, una vez
   validado el mapeo.

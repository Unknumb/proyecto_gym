"""
Nodos del pipeline Modelado Supervisado — completar las etiquetas del perfil de
exigencia (EV2, fase 2; decisión en ``docs/plan_ev2.md``).

Tres objetivos, todos aprendidos de free-exercise-db, la única fuente que los trae
con un solo curador:

- ``mecanica``: compuesto / aislado.
- ``fuerza``: empuje / tracción / estático.
- ``complejidad_tecnica``: ``baja`` (``beginner``) / ``alta`` (``intermediate`` o
  ``expert``). Es el nivel de free-exercise-db **solo**, nunca mezclado con
  megaGymDataset: las dos fuentes coinciden igual que al azar (README §11.5).

**Variables idénticas en las tres fuentes.** Si una variable existe solo en
free-exercise-db (su ``category``), el modelo aprendería a reconocer la fuente y
no el movimiento, y fallaría en silencio al predecir sobre el catálogo. Por eso
todo se calcula del nombre normalizado, las instrucciones en inglés, la ontología
muscular común y la familia de equipamiento.

**Validación sin fuga.** Los nombres casi idénticos (``"Dumbbell Bench Press"`` /
``"Dumbbell Bench Press with Neutral Grip"``) se agrupan, y un grupo nunca queda
repartido entre entrenamiento y prueba. La validación externa usa ejercicios del
catálogo cuyo gemelo de free-exercise-db quedó en prueba: mide si el modelo
generaliza a otra redacción y otro vocabulario sin haber visto la respuesta.
"""

from __future__ import annotations

import logging
import re
from typing import Any

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from scipy.sparse.csgraph import connected_components
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, GroupKFold, GroupShuffleSplit, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler, OneHotEncoder

logger = logging.getLogger(__name__)

OBJETIVOS = ["mecanica", "fuerza", "complejidad_tecnica"]


# ─────────────────────────────────────────────────────────────────────────────
# 1. Reglas del perfil de exigencia
# ─────────────────────────────────────────────────────────────────────────────
#: Saltos, rebotes, carrera y desplazamientos con aterrizaje: cargan rodilla y
#: tobillo con varias veces el peso corporal. No incluye los lanzamientos de balón
#: medicinal que free-exercise-db clasifica como ``plyometrics``: son balísticos,
#: pero de tren superior y sin aterrizaje.
_IMPACTO = re.compile(
    r"\b(?:jump\w*|hop\w*|bound\w*|plyo\w*|burpee\w*|sprint\w*|skip\w*"
    r"|(?:jumping|plank|step|star|split) jacks?|jack jump|leap\w*|skater|clap\w*|depth"
    r"|butt kick|shuffle|carioca|mountain climber|run|running|jog\w*|high knee|push off)\b")
#: Máquinas de cardio sin aterrizaje, aunque el nombre diga "run".
_SIN_IMPACTO = re.compile(r"\b(?:bike|cycle|cycling|ergometer|elliptical|rower|rowing)\b")
#: Base de apoyo reducida o superficie móvil.
_INESTABLE = re.compile(
    r"\b(?:single leg|one leg|single arm leg|pistol|bosu|exercise ball|stability ball"
    r"|swiss ball|ball|unilateral|balance|on one foot)\b")
#: El cuerpo está apoyado o la trayectoria la guía una máquina.
_APOYADO = re.compile(
    r"\b(?:seated|sitting|lying|supine|prone|bench|machine|lever|smith|supported"
    r"|kneeling|incline|decline|preacher|sled|assisted)\b")
#: Familias en que la carga puede bajarse por debajo del peso corporal o en pasos
#: pequeños sin exigir control de la trayectoria (banda, polea, máquina).
FAMILIAS_CARGA_GRADUABLE = {"maquina", "cable", "banda elastica", "asistido"}


def derivar_perfil_reglas(nombre_norm: pd.Series, familia: pd.Series) -> pd.DataFrame:
    """Dimensiones del perfil de exigencia que no necesitan un modelo.

    Son reglas transparentes sobre el nombre normalizado y la familia de
    equipamiento. ``estabilidad`` vale ``inestable`` si hay base reducida o
    superficie móvil (aunque también haya apoyo: un *crunch* sobre balón es
    inestable), ``apoyado`` si el cuerpo o la trayectoria están guiados, y
    ``libre`` en otro caso.
    """
    inestable = nombre_norm.str.contains(_INESTABLE) | familia.eq("balon o rodillo")
    apoyado = nombre_norm.str.contains(_APOYADO) | familia.isin({"maquina", "asistido"})
    return pd.DataFrame({
        "impacto": nombre_norm.str.contains(_IMPACTO) & ~nombre_norm.str.contains(_SIN_IMPACTO),
        "estabilidad": np.select([inestable, apoyado], ["inestable", "apoyado"], "libre"),
        "carga_graduable": familia.isin(FAMILIAS_CARGA_GRADUABLE),
        "es_estiramiento_nombre": nombre_norm.str.contains(r"\bstretch"),
    }, index=nombre_norm.index)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Tabla de modelado
# ─────────────────────────────────────────────────────────────────────────────
def _agrupar_variantes(nombres: pd.Series, umbral: float) -> np.ndarray:
    """Componentes conexas del grafo "nombre casi idéntico" (``token_sort_ratio``
    ≥ ``umbral``). Cada componente es un grupo para la validación cruzada."""
    similitud = process.cdist(nombres.tolist(), nombres.tolist(),
                              scorer=fuzz.token_sort_ratio, workers=-1)
    _, grupos = connected_components(similitud >= umbral, directed=False)
    return grupos


def construir_tabla_modelado(
    fedb: pd.DataFrame,
    ejercicios: pd.DataFrame,
    match_fedb: pd.DataFrame,
    musculo_articulacion: pd.DataFrame,
    parametros: dict[str, Any],
) -> pd.DataFrame:
    """Une entrenamiento y predicción en una tabla con las mismas variables.

    - ``rol = entrenamiento``: los 876 ejercicios de free-exercise-db, con sus
      etiquetas, un ``grupo`` de variantes y una ``particion`` train/test.
    - ``rol = prediccion``: los nodos del catálogo y de wger. Los del catálogo
      con par en free-exercise-db guardan la etiqueta transferida en
      ``<objetivo>_referencia`` y el id del gemelo en ``gemelo_fedb``.

    Returns:
        Una fila por ejercicio, con texto, músculos y articulaciones en
        *multi-hot*, equipamiento, reglas del perfil y objetivos.
    """
    semilla = int(parametros["semilla"])
    entrenamiento = fedb.assign(rol="entrenamiento")
    entrenamiento["complejidad_tecnica"] = entrenamiento["nivel"].map(
        {"beginner": "baja", "intermediate": "alta", "expert": "alta"})
    prediccion = ejercicios.loc[ejercicios["fuente"] != "free_exercise_db"].assign(
        rol="prediccion")

    # Etiquetas transferidas a los nodos del catálogo: sirven de referencia en la
    # validación externa, nunca de entrenamiento.
    pares = (match_fedb.loc[match_fedb["aceptado"]]
             .sort_values(["similitud", "ejercicio_id"], ascending=[False, True])
             .drop_duplicates("candidato_id")
             .set_index("candidato_id")["ejercicio_id"])
    prediccion["gemelo_fedb"] = prediccion["ejercicio_id"].map(pares)
    etiquetas_fedb = entrenamiento.set_index("ejercicio_id")
    for objetivo in OBJETIVOS:
        prediccion[f"{objetivo}_referencia"] = prediccion["gemelo_fedb"].map(
            etiquetas_fedb[objetivo])

    columnas = ["ejercicio_id", "fuente", "rol", "nombre", "nombre_norm",
                "instrucciones_en", "familia_equipamiento", "musculos_primarios",
                "musculos_secundarios"]
    tabla = pd.concat([
        entrenamiento[columnas + OBJETIVOS],
        prediccion[columnas + ["gemelo_fedb"] + [f"{o}_referencia" for o in OBJETIVOS]],
    ], ignore_index=True)
    tabla["instrucciones_en"] = tabla["instrucciones_en"].fillna("")
    for objetivo in OBJETIVOS:
        for columna in [objetivo, f"{objetivo}_referencia"]:
            tabla[columna] = tabla[columna].astype(object).where(tabla[columna].notna(), None)

    # Músculos y articulaciones en multi-hot, sobre la ontología común.
    articulaciones = musculo_articulacion.groupby("musculo")["articulacion"].agg(set)
    vocabulario = sorted(set(musculo_articulacion["musculo"]))
    juntas = sorted(set(musculo_articulacion["articulacion"]))

    def _articulaciones(musculos) -> set[str]:
        return set().union(*(articulaciones.get(m, set()) for m in musculos))

    art_agonistas = tabla["musculos_primarios"].map(_articulaciones)
    art_total = [a | _articulaciones(s) for a, s in zip(art_agonistas, tabla["musculos_secundarios"])]
    multi_hot = {}
    for m in vocabulario:
        multi_hot[f"agonista__{m}"] = tabla["musculos_primarios"].map(lambda l, m=m: m in l)
        multi_hot[f"sinergista__{m}"] = tabla["musculos_secundarios"].map(lambda l, m=m: m in l)
    for a in juntas:
        multi_hot[f"articulacion__{a}"] = [a in s for s in art_total]
    tabla = pd.concat([tabla, pd.DataFrame(multi_hot).astype("int8")], axis=1)
    tabla["n_agonistas"] = tabla["musculos_primarios"].map(len)
    tabla["n_sinergistas"] = tabla["musculos_secundarios"].map(len)
    tabla["n_articulaciones_agonistas"] = art_agonistas.map(len)
    tabla["n_articulaciones_total"] = [len(s) for s in art_total]

    reglas = derivar_perfil_reglas(tabla["nombre_norm"], tabla["familia_equipamiento"])
    tabla = pd.concat([tabla, reglas], axis=1)

    # Grupos de variantes y partición train/test (solo entrenamiento).
    es_entrenamiento = tabla["rol"] == "entrenamiento"
    grupos = _agrupar_variantes(tabla.loc[es_entrenamiento, "nombre_norm"],
                                float(parametros["umbral_variantes"]))
    tabla.loc[es_entrenamiento, "grupo"] = grupos
    divisor = GroupShuffleSplit(n_splits=1, test_size=float(parametros["proporcion_test"]),
                                random_state=semilla)
    idx = np.flatnonzero(es_entrenamiento)
    _, test = next(divisor.split(idx, groups=grupos))
    tabla["particion"] = np.where(es_entrenamiento, "train", "prediccion")
    tabla.loc[idx[test], "particion"] = "test"

    logger.info(
        "Tabla de modelado: %d entrenamiento (%d train / %d test, %d grupos de variantes), "
        "%d a predecir.", int(es_entrenamiento.sum()), int((tabla["particion"] == "train").sum()),
        int((tabla["particion"] == "test").sum()), len(set(grupos)),
        int((~es_entrenamiento).sum()),
    )
    return tabla.drop(columns=["musculos_primarios", "musculos_secundarios"])


# ─────────────────────────────────────────────────────────────────────────────
# 3. Entrenamiento y comparación
# ─────────────────────────────────────────────────────────────────────────────
def _columnas_numericas(tabla: pd.DataFrame) -> list[str]:
    prefijos = ("agonista__", "sinergista__", "articulacion__")
    return ([c for c in tabla.columns if c.startswith(prefijos)]
            + ["n_agonistas", "n_sinergistas", "n_articulaciones_agonistas",
               "n_articulaciones_total", "impacto", "carga_graduable",
               "es_estiramiento_nombre"])


def _preprocesador(numericas: list[str]) -> ColumnTransformer:
    """Texto del nombre (palabras y bigramas), texto de las instrucciones,
    equipamiento y estabilidad en *one-hot*, y variables anatómicas."""
    return ColumnTransformer([
        ("nombre", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True), "nombre_norm"),
        ("instrucciones", TfidfVectorizer(stop_words="english", min_df=2, max_features=5000,
                                          sublinear_tf=True), "instrucciones_en"),
        ("categoricas", OneHotEncoder(handle_unknown="ignore"),
         ["familia_equipamiento", "estabilidad"]),
        ("anatomia", MaxAbsScaler(), numericas),
    ])


def _un_error_estandar(cv_results: dict[str, Any]) -> int:
    """Regla de un error estándar (Hastie, Tibshirani y Friedman, *ESL* §7.10).

    Entre los ``C`` cuyo F1 medio queda a menos de un error estándar del mejor,
    elige el **menor** (el modelo más regularizado). Con pocas filas y miles de
    variables de texto, el F1 es casi plano en un rango amplio de ``C``: elegir
    el máximo exacto es elegir ruido, y deja un modelo que memoriza palabras
    raras. Se usa como ``refit`` de ``GridSearchCV``.
    """
    medias = np.asarray(cv_results["mean_test_score"])
    errores = np.asarray(cv_results["std_test_score"]) / np.sqrt(
        sum(k.startswith("split") and k.endswith("_test_score") for k in cv_results))
    mejor = int(np.argmax(medias))
    admisibles = np.flatnonzero(medias >= medias[mejor] - errores[mejor])
    valores_c = np.asarray(cv_results["param_clf__C"], dtype=float)
    return int(admisibles[np.argmin(valores_c[admisibles])])


def _candidatos(parametros: dict[str, Any]) -> dict[str, tuple[Any, dict[str, list], Any]]:
    """Modelos a comparar y su espacio de hiperparámetros.

    - **Regresión logística**: línea base lineal e interpretable; sus
      coeficientes dicen qué palabra o músculo empuja a cada clase.
      ``class_weight="balanced"`` compensa el desbalance (``static`` ~12 %).
      ``C`` se elige con la regla de un error estándar (``_un_error_estandar``).
    - **Random Forest**: captura interacciones que un modelo lineal no ve (un
      mismo músculo es compuesto con barra y aislado con polea). Es robusto con
      muchas variables binarias dispersas y pocas filas.
    """
    semilla = int(parametros["semilla"])
    return {
        "regresion_logistica": (
            LogisticRegression(max_iter=5000, class_weight="balanced"),
            {"clf__C": parametros["grilla_logistica"]["C"]},
            _un_error_estandar,
        ),
        "random_forest": (
            RandomForestClassifier(n_estimators=int(parametros["n_arboles"]),
                                   class_weight="balanced_subsample", random_state=semilla,
                                   n_jobs=-1),
            {f"clf__{k}": v for k, v in parametros["grilla_random_forest"].items()},
            True,
        ),
    }


def _metricas(y_real, y_pred) -> dict[str, float]:
    return {
        "f1_macro": f1_score(y_real, y_pred, average="macro"),
        "accuracy": accuracy_score(y_real, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_real, y_pred),
    }


def _umbral_abstencion(y_real: np.ndarray, probas: np.ndarray, clases: np.ndarray,
                       precision_objetivo: float) -> float:
    """Menor umbral de confianza con el que los ejercicios que sí se etiquetan
    alcanzan ``precision_objetivo`` de acierto (sobre predicciones fuera de
    muestra). Si ninguno la alcanza, se devuelve 1.0: el modelo se abstiene
    siempre."""
    confianza = probas.max(axis=1)
    acierto = clases[probas.argmax(axis=1)] == y_real
    for umbral in np.round(np.arange(0.40, 1.0, 0.01), 2):
        retenidos = confianza >= umbral
        if retenidos.sum() >= 20 and acierto[retenidos].mean() >= precision_objetivo:
            return float(umbral)
    return 1.0


def _elegir_ganador(resultados: dict[str, GridSearchCV], preferido: str) -> str:
    """Mayor F1 macro en validación cruzada; pero si ``preferido`` (el modelo más
    simple e interpretable) queda a menos de un error estándar del mejor, gana
    ``preferido``. Es la misma lógica de la regla de un error estándar, aplicada
    a la elección entre algoritmos: una diferencia menor que el ruido entre
    pliegues no justifica un modelo más complejo."""
    def _media_y_error(busqueda: GridSearchCV) -> tuple[float, float]:
        k = busqueda.best_index_
        n_folds = busqueda.n_splits_
        return (busqueda.cv_results_["mean_test_score"][k],
                busqueda.cv_results_["std_test_score"][k] / np.sqrt(n_folds))

    mejor = max(resultados, key=lambda n: _media_y_error(resultados[n])[0])
    media_mejor, error_mejor = _media_y_error(resultados[mejor])
    if preferido in resultados and _media_y_error(resultados[preferido])[0] >= media_mejor - error_mejor:
        return preferido
    return mejor


def entrenar_modelos(
    tabla: pd.DataFrame, parametros: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Compara los modelos por objetivo y deja el mejor reentrenado.

    Por objetivo y modelo:

    1. ``GridSearchCV`` con ``GroupKFold`` sobre *train*, optimizando F1 macro.
    2. Evaluación en *test* (grupos nunca vistos) y en la **validación externa**
       (ejercicios del catálogo cuyo gemelo quedó en *test*).
    3. Un ``DummyClassifier`` estratificado da la referencia de azar.

    El ganador es el de mayor F1 macro en validación cruzada (no en *test*, para
    no elegir mirando la prueba), salvo empate dentro de un error estándar, en
    cuyo caso gana el modelo más simple (``_elegir_ganador``). Su umbral de abstención se fija con
    predicciones fuera de muestra sobre *train*, y luego se reentrena con todo
    free-exercise-db.

    Returns:
        ``comparacion`` (una fila por objetivo × modelo × conjunto),
        ``predicciones_test`` (para matrices de confusión) y ``modelos``
        (por objetivo: estimador final, umbral, clases y nombre del ganador).
    """
    numericas = _columnas_numericas(tabla)
    cv = GroupKFold(n_splits=int(parametros["n_folds"]))
    precision_objetivo = float(parametros["precision_objetivo_abstencion"])
    filas, predicciones, modelos = [], [], {}

    for objetivo in OBJETIVOS:
        datos = tabla.loc[(tabla["rol"] == "entrenamiento") & tabla[objetivo].notna()]
        train = datos.loc[datos["particion"] == "train"]
        test = datos.loc[datos["particion"] == "test"]
        externa = tabla.loc[(tabla["rol"] == "prediccion")
                            & tabla[f"{objetivo}_referencia"].notna()
                            & tabla["gemelo_fedb"].isin(test["ejercicio_id"])]

        resultados = {}
        for nombre, (clf, grilla, refit) in _candidatos(parametros).items():
            pipe = Pipeline([("prep", _preprocesador(numericas)), ("clf", clf)])
            busqueda = GridSearchCV(pipe, grilla, scoring="f1_macro", cv=cv, n_jobs=-1,
                                    refit=refit)
            busqueda.fit(train, train[objetivo], groups=train["grupo"])
            resultados[nombre] = busqueda
            cv_f1 = busqueda.cv_results_["mean_test_score"][busqueda.best_index_]
            cv_std = busqueda.cv_results_["std_test_score"][busqueda.best_index_]
            for conjunto, df in [("test", test), ("externa_catalogo", externa)]:
                y_real = df[objetivo] if conjunto == "test" else df[f"{objetivo}_referencia"]
                y_pred = busqueda.predict(df)
                filas.append({"objetivo": objetivo, "modelo": nombre, "conjunto": conjunto,
                              "n": len(df), "cv_f1_macro": cv_f1,
                              "cv_f1_std": cv_std,
                              "mejores_parametros": str(busqueda.best_params_),
                              **_metricas(y_real, y_pred)})
                predicciones.append(pd.DataFrame({
                    "objetivo": objetivo, "modelo": nombre, "conjunto": conjunto,
                    "ejercicio_id": df["ejercicio_id"].to_numpy(),
                    "y_real": y_real.to_numpy(), "y_pred": y_pred,
                    "confianza": busqueda.predict_proba(df).max(axis=1),
                }))

        azar = DummyClassifier(strategy="stratified", random_state=int(parametros["semilla"]))
        azar.fit(train, train[objetivo])
        filas.append({"objetivo": objetivo, "modelo": "azar_estratificado", "conjunto": "test",
                      "n": len(test), **_metricas(test[objetivo], azar.predict(test))})

        def _cv_f1(busqueda: GridSearchCV) -> float:
            return busqueda.cv_results_["mean_test_score"][busqueda.best_index_]

        ganador = _elegir_ganador(resultados, parametros["modelo_preferido_en_empate"])
        mejor = resultados[ganador].best_estimator_
        probas_oof = cross_val_predict(clone(mejor), train, train[objetivo],
                                       groups=train["grupo"], cv=cv, method="predict_proba")
        clases = np.sort(train[objetivo].unique())
        umbral = _umbral_abstencion(train[objetivo].to_numpy(), probas_oof, clases,
                                    precision_objetivo)
        final = clone(mejor).fit(datos, datos[objetivo])
        modelos[objetivo] = {"estimador": final, "modelo": ganador, "umbral": umbral,
                             "clases": list(final.classes_),
                             "modelo_train": mejor,
                             "cv_results": {n: pd.DataFrame(b.cv_results_).filter(
                                 regex=r"^(param_|mean_test|std_test)") for n, b in resultados.items()}}
        logger.info("%s: gana %s (CV F1 macro %.3f); umbral de abstención %.2f.",
                    objetivo, ganador, _cv_f1(resultados[ganador]), umbral)

    comparacion = pd.DataFrame(filas)
    for columna in ["cv_f1_macro", "cv_f1_std", "f1_macro", "accuracy", "balanced_accuracy"]:
        comparacion[columna] = comparacion[columna].round(4)
    return comparacion, pd.concat(predicciones, ignore_index=True), modelos


# ─────────────────────────────────────────────────────────────────────────────
# 4. Etiquetas finales del perfil de exigencia
# ─────────────────────────────────────────────────────────────────────────────
def predecir_etiquetas(tabla: pd.DataFrame, ejercicios: pd.DataFrame,
                       modelos: dict[str, Any]) -> pd.DataFrame:
    """Perfil de exigencia completo para cada nodo del grafo.

    Por objetivo, la etiqueta final sigue un orden de confianza:

    1. ``fuente``: la trae la propia ficha (nodos de free-exercise-db).
    2. ``transferida``: viene del gemelo en free-exercise-db (README §11.3).
    3. ``modelo``: predicción con confianza ≥ umbral de abstención.
    4. Nulo, si el modelo no alcanza el umbral. Se prefiere un nulo honesto
       a una etiqueta dudosa (README §5.3).

    ``<objetivo>_prediccion`` y ``<objetivo>_confianza`` se guardan siempre,
    también cuando la etiqueta final viene de la fuente, para poder auditar
    dónde discrepan.
    """
    nodos = ejercicios[["ejercicio_id", "fuente"]].merge(
        tabla.drop(columns=["fuente"]), on="ejercicio_id", how="left", validate="one_to_one")
    perfil = nodos[["ejercicio_id", "fuente", "impacto", "estabilidad", "carga_graduable"]].copy()

    for objetivo, modelo in modelos.items():
        probas = modelo["estimador"].predict_proba(nodos)
        clases = np.asarray(modelo["clases"])
        prediccion = clases[probas.argmax(axis=1)]
        confianza = probas.max(axis=1)
        propia = nodos[objetivo].astype(object)
        transferida = nodos[f"{objetivo}_referencia"].astype(object)
        condiciones = [propia.notna().to_numpy(), transferida.notna().to_numpy(),
                       confianza >= modelo["umbral"]]

        perfil[f"{objetivo}_prediccion"] = prediccion
        perfil[f"{objetivo}_confianza"] = confianza.round(3)
        perfil[objetivo] = np.select(
            condiciones, [propia.to_numpy(), transferida.to_numpy(), prediccion], default=None)
        perfil[f"{objetivo}_origen"] = np.select(
            condiciones, ["fuente", "transferida", "modelo"], default=None)

    resumen = {o: perfil[f"{o}_origen"].value_counts(dropna=False).to_dict() for o in modelos}
    logger.info("Origen de las etiquetas finales: %s", resumen)
    return perfil

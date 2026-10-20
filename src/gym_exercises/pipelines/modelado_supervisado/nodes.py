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
from scipy import stats
from scipy.sparse.csgraph import connected_components
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             f1_score, roc_auc_score)
from sklearn.model_selection import (GridSearchCV, GroupKFold, GroupShuffleSplit,
                                     StratifiedGroupKFold, StratifiedKFold, cross_val_predict,
                                     learning_curve)
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


def _brier(y_real: np.ndarray, probas: np.ndarray, clases: np.ndarray) -> float:
    """Brier multiclase: error cuadrático medio entre la probabilidad y el 0/1 real,
    sumado sobre las clases. 0 es perfecto; mide a la vez acierto y calibración."""
    reales = (np.asarray(y_real)[:, None] == np.asarray(clases)[None, :]).astype(float)
    return float(((probas - reales) ** 2).sum(axis=1).mean())


def _pr_auc_macro(y_real: np.ndarray, probas: np.ndarray, clases: np.ndarray) -> float:
    """PR-AUC (*average precision*) promedio entre clases, una contra el resto. Su
    referencia de azar es la prevalencia de cada clase, no 0,5."""
    return float(np.mean([average_precision_score(np.asarray(y_real) == c, probas[:, k])
                          for k, c in enumerate(clases)]))


def _intervalo_bootstrap(y_real, y_pred, n: int, semilla: int) -> tuple[float, float]:
    """Intervalo de confianza del 95 % del F1 macro por *bootstrap* de los ejemplos.
    Con ~60 casos en la validación externa, el valor puntual solo no basta para
    afirmar una meta."""
    y_real, y_pred = np.asarray(y_real), np.asarray(y_pred)
    rng = np.random.default_rng(semilla)
    muestras = [f1_score(y_real[i], y_pred[i], average="macro", zero_division=0)
                for i in (rng.integers(0, len(y_real), len(y_real)) for _ in range(n))]
    return float(np.percentile(muestras, 2.5)), float(np.percentile(muestras, 97.5))


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
                probas = busqueda.predict_proba(df)
                inferior, superior = _intervalo_bootstrap(
                    y_real, y_pred, int(parametros["diagnosticos"]["n_bootstrap"]),
                    int(parametros["semilla"]))
                filas.append({"objetivo": objetivo, "modelo": nombre, "conjunto": conjunto,
                              "n": len(df), "cv_f1_macro": cv_f1,
                              "cv_f1_std": cv_std,
                              "mejores_parametros": str(busqueda.best_params_),
                              **_metricas(y_real, y_pred),
                              "f1_ic95_inf": inferior, "f1_ic95_sup": superior,
                              "brier": _brier(y_real, probas, busqueda.classes_),
                              "pr_auc_macro": _pr_auc_macro(y_real, probas, busqueda.classes_)})
                predicciones.append(pd.DataFrame({
                    "objetivo": objetivo, "modelo": nombre, "conjunto": conjunto,
                    "ejercicio_id": df["ejercicio_id"].to_numpy(),
                    "y_real": y_real.to_numpy(), "y_pred": y_pred,
                    "confianza": probas.max(axis=1),
                    **{f"proba_{c}": probas[:, k] for k, c in enumerate(busqueda.classes_)},
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
                             "candidatos_train": {n: b.best_estimator_ for n, b in resultados.items()},
                             "cv_results": {n: pd.DataFrame(b.cv_results_).filter(
                                 regex=r"^(param_|mean_test|std_test)") for n, b in resultados.items()}}
        logger.info("%s: gana %s (CV F1 macro %.3f); umbral de abstención %.2f.",
                    objetivo, ganador, _cv_f1(resultados[ganador]), umbral)

    comparacion = pd.DataFrame(filas)
    for columna in ["cv_f1_macro", "cv_f1_std", "f1_macro", "accuracy", "balanced_accuracy",
                    "f1_ic95_inf", "f1_ic95_sup", "brier", "pr_auc_macro"]:
        comparacion[columna] = comparacion[columna].round(4)
    return comparacion, pd.concat(predicciones, ignore_index=True), modelos


# ─────────────────────────────────────────────────────────────────────────────
# 4. Diagnósticos: informe de cómo se analizaron los modelos
# ─────────────────────────────────────────────────────────────────────────────
# Nada de esta sección cambia los modelos ni las etiquetas: reanaliza a los
# candidatos ya entrenados con pruebas más exigentes y deja el resultado como
# informe. Sus parámetros están en ``modelado_supervisado.diagnosticos``.

#: Familias de variables que se permutan juntas: las columnas de una misma familia
#: están correlacionadas y, permutadas por separado, se repartirían la importancia.
FAMILIAS_VARIABLES = {
    "Nombre (texto)": ["nombre_norm"],
    "Instrucciones (texto)": ["instrucciones_en"],
    "Músculos agonistas": ["agonista__", "n_agonistas"],
    "Músculos sinergistas": ["sinergista__", "n_sinergistas"],
    "Articulaciones": ["articulacion__", "n_articulaciones_agonistas", "n_articulaciones_total"],
    "Equipamiento": ["familia_equipamiento", "carga_graduable"],
    "Estabilidad e impacto": ["estabilidad", "impacto", "es_estiramiento_nombre"],
}

METODOS_CALIBRACION = {"sin calibrar": None, "sigmoide": "sigmoid", "isotonica": "isotonic"}


def _validacion_estratificada(parametros: dict[str, Any], semilla: int) -> StratifiedGroupKFold:
    """Pliegues estratificados **y** por grupos: ningún grupo de variantes queda
    repartido, y cada pliegue conserva la proporción de clases."""
    return StratifiedGroupKFold(n_splits=int(parametros["n_folds"]), shuffle=True,
                                random_state=semilla)


def _columnas_familia(tabla: pd.DataFrame, patrones: list[str]) -> list[str]:
    return [c for c in tabla.columns
            if any(c == p or (p.endswith("__") and c.startswith(p)) for p in patrones)]


def _cv_repetida(candidatos: dict[str, Pipeline], train: pd.DataFrame, objetivo: str,
                 parametros: dict[str, Any]) -> pd.DataFrame:
    """F1 macro de cada candidato en una validación cruzada repetida, con **los
    mismos pliegues** para todos, para compararlos pliegue a pliegue."""
    y, grupos = train[objetivo].to_numpy(), train["grupo"].to_numpy()
    filas = []
    for repeticion in range(int(parametros["diagnosticos"]["n_repeticiones_cv"])):
        cv = _validacion_estratificada(parametros, int(parametros["semilla"]) + repeticion)
        for pliegue, (a, b) in enumerate(cv.split(train, y, grupos)):
            for nombre, estimador in candidatos.items():
                modelo = clone(estimador).fit(train.iloc[a], y[a])
                filas.append({"objetivo": objetivo, "repeticion": repeticion, "pliegue": pliegue,
                              "modelo": nombre,
                              "f1_macro": f1_score(y[b], modelo.predict(train.iloc[b]), average="macro")})
    return pd.DataFrame(filas)


def _contraste_pareado(cv_repetida: pd.DataFrame, simple: str,
                       parametros: dict[str, Any]) -> dict[str, Any]:
    """¿La mejora del modelo complejo sobre el ``simple`` es más que ruido?

    Diferencia pliegue a pliegue y prueba t corregida de Nadeau y Bengio (2003): la
    varianza se infla por ``1/n + n_test/n_train`` porque los pliegues comparten
    datos de entrenamiento. Se informa si la mejora cumple tres condiciones: un
    mínimo relevante de F1, ganar en la mayoría de los pliegues y superar dos
    errores estándar corregidos.
    """
    criterio = parametros["diagnosticos"]
    tabla = cv_repetida.pivot_table(index=["repeticion", "pliegue"], columns="modelo",
                                    values="f1_macro")
    complejo = tabla.columns.drop(simple)[0]
    diferencia = tabla[complejo] - tabla[simple]
    n, k = len(diferencia), int(parametros["n_folds"])
    error = np.sqrt((1 / n + 1 / (k - 1)) * diferencia.var(ddof=1))
    t = float(diferencia.mean() / error) if error > 0 else 0.0
    return {
        "simple": simple, "complejo": complejo,
        "f1_simple": float(tabla[simple].mean()), "sd_simple": float(tabla[simple].std()),
        "f1_complejo": float(tabla[complejo].mean()), "sd_complejo": float(tabla[complejo].std()),
        "diferencia_media": float(diferencia.mean()), "diferencia_sd": float(diferencia.std()),
        "pct_pliegues_complejo_mejor": float((diferencia > 0).mean() * 100),
        "t_nadeau_bengio": t, "p_valor": float(2 * stats.t.sf(abs(t), n - 1)),
        "cumple_mejora_minima": bool(diferencia.mean() >= float(criterio["mejora_minima_f1"])),
        "cumple_pliegues": bool((diferencia > 0).mean() >= float(criterio["pliegues_minimos_mejora"])),
        "cumple_significancia": bool(t >= float(criterio["t_minimo_mejora"])),
    }


def _calibracion(estimador: Pipeline, train: pd.DataFrame, objetivo: str,
                 parametros: dict[str, Any]) -> list[dict[str, Any]]:
    """Brier fuera de muestra sin calibrar y recalibrado con ``CalibratedClassifierCV``
    (sigmoide e isotónica). La calibración de cada pliegue se ajusta en pliegues
    internos, sobre datos que ese modelo no usó para entrenar."""
    semilla = int(parametros["semilla"])
    y, grupos = train[objetivo].to_numpy(), train["grupo"].to_numpy()
    clases = np.sort(np.unique(y))
    probas = {m: np.zeros((len(train), len(clases))) for m in METODOS_CALIBRACION}
    for a, b in _validacion_estratificada(parametros, semilla).split(train, y, grupos):
        internos = list(_validacion_estratificada(parametros, semilla + 100)
                        .split(train.iloc[a], y[a], grupos[a]))
        for nombre, metodo in METODOS_CALIBRACION.items():
            modelo = (clone(estimador) if metodo is None else
                      CalibratedClassifierCV(clone(estimador), method=metodo, cv=internos))
            probas[nombre][b] = modelo.fit(train.iloc[a], y[a]).predict_proba(train.iloc[b])
    return [{"objetivo": objetivo, "metodo": nombre, "brier": _brier(y, p, clases),
             "f1_macro": f1_score(y, clases[p.argmax(axis=1)], average="macro"),
             "pr_auc_macro": _pr_auc_macro(y, p, clases)} for nombre, p in probas.items()]


def _importancia_en_validacion(estimador: Pipeline, train: pd.DataFrame, objetivo: str,
                               parametros: dict[str, Any]) -> pd.DataFrame:
    """Importancia por permutación **agrupada** y calculada en **validación**: en
    cada pliegue se entrena con el resto y se permuta cada familia de variables en
    el pliegue retenido. Devuelve la caída del F1 macro con su desviación entre
    pliegues y repeticiones."""
    semilla = int(parametros["semilla"])
    rng = np.random.default_rng(semilla)
    y, grupos = train[objetivo].to_numpy(), train["grupo"].to_numpy()
    caidas = []
    for a, b in _validacion_estratificada(parametros, semilla).split(train, y, grupos):
        modelo = clone(estimador).fit(train.iloc[a], y[a])
        retenido = train.iloc[b]
        base = f1_score(y[b], modelo.predict(retenido), average="macro")
        for familia, patrones in FAMILIAS_VARIABLES.items():
            columnas = _columnas_familia(train, patrones)
            for _ in range(int(parametros["diagnosticos"]["repeticiones_permutacion"])):
                barajado = retenido.copy()
                barajado[columnas] = retenido[columnas].iloc[rng.permutation(len(retenido))].to_numpy()
                caidas.append({"familia": familia, "caida_f1": base - f1_score(
                    y[b], modelo.predict(barajado), average="macro")})
    return (pd.DataFrame(caidas).groupby("familia")["caida_f1"]
            .agg(caida_media="mean", caida_sd="std", n="size").reset_index())


def _preprocesador_familia(tabla: pd.DataFrame, patrones: list[str]) -> ColumnTransformer:
    """Preprocesador restringido a una familia de variables (validación adversarial)."""
    columnas = _columnas_familia(tabla, patrones)
    texto = [c for c in columnas if c in {"nombre_norm", "instrucciones_en"}]
    categoricas = [c for c in columnas if c in {"familia_equipamiento", "estabilidad"}]
    resto = [c for c in columnas if c not in texto + categoricas]
    pasos = [(c, TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2), c) for c in texto]
    if categoricas:
        pasos.append(("categoricas", OneHotEncoder(handle_unknown="ignore"), categoricas))
    if resto:
        pasos.append(("numericas", MaxAbsScaler(), resto))
    return ColumnTransformer(pasos)


def _validacion_adversarial(tabla: pd.DataFrame, semilla: int) -> pd.DataFrame:
    """Un clasificador intenta distinguir free-exercise-db (entrenamiento) del
    catálogo y de wger (predicción), con todas las variables y por familia. Un AUC
    cercano a 0,5 indica que las fuentes son intercambiables; cercano a 1, que hay
    cambio de dominio."""
    filas = []
    entrenamiento = tabla.loc[tabla["rol"] == "entrenamiento"]
    numericas = _columnas_numericas(tabla)
    for fuente in ["catalogo", "wger"]:
        datos = pd.concat([entrenamiento.assign(destino=0),
                           tabla.loc[(tabla["rol"] == "prediccion") & (tabla["fuente"] == fuente)]
                           .assign(destino=1)], ignore_index=True)
        for familia, patrones in {"Todas las variables": None, **FAMILIAS_VARIABLES}.items():
            prep = (_preprocesador(numericas) if patrones is None else
                    _preprocesador_familia(datos, patrones))
            clf = Pipeline([("prep", prep), ("clf", LogisticRegression(max_iter=5000))])
            p = cross_val_predict(clf, datos, datos["destino"], method="predict_proba",
                                  cv=StratifiedKFold(5, shuffle=True, random_state=semilla))[:, 1]
            filas.append({"fuente_prediccion": fuente, "familia": familia,
                          "auc": roc_auc_score(datos["destino"], p)})
    return pd.DataFrame(filas)


def diagnosticar_modelos(
    tabla: pd.DataFrame, modelos: dict[str, Any], parametros: dict[str, Any],
) -> tuple[pd.DataFrame, ...]:
    """Informe de diagnóstico de los modelos ya entrenados. No modifica nada.

    Por objetivo, sobre *train* y con validación estratificada por grupos:

    - **CV repetida y contraste pareado** entre regresión logística y Random
      Forest: ¿la diferencia entre ambos es más que ruido entre pliegues?
    - **Calibración**: Brier sin calibrar, con sigmoide y con isotónica.
    - **Curva de aprendizaje**: ¿ayudarían más datos?
    - **Objetivo barajado**: si el F1 no cae al nivel del azar, hay una fuga.
    - **Importancia por permutación** agrupada, en validación.

    Y entre fuentes, la **validación adversarial**.

    Returns:
        ``cv_repetida``, ``contraste``, ``calibracion``, ``aprendizaje``,
        ``control_fuga``, ``importancia`` y ``adversarial``.
    """
    semilla = int(parametros["semilla"])
    repetidas, contrastes, calibracion, aprendizaje, fuga, importancia = [], [], [], [], [], []

    for objetivo in OBJETIVOS:
        datos = tabla.loc[(tabla["rol"] == "entrenamiento") & tabla[objetivo].notna()]
        train = datos.loc[datos["particion"] == "train"]
        y, grupos = train[objetivo].to_numpy(), train["grupo"].to_numpy()
        ganador = modelos[objetivo]["modelo_train"]
        cv = _validacion_estratificada(parametros, semilla)

        repetida = _cv_repetida(modelos[objetivo]["candidatos_train"], train, objetivo, parametros)
        repetidas.append(repetida)
        contrastes.append({"objetivo": objetivo, "ganador_elegido": modelos[objetivo]["modelo"],
                           **_contraste_pareado(repetida, parametros["modelo_preferido_en_empate"],
                                                parametros)})
        calibracion += _calibracion(ganador, train, objetivo, parametros)

        tamanos, _, puntajes = learning_curve(
            clone(ganador), train, y, groups=grupos, cv=cv, scoring="f1_macro",
            train_sizes=np.linspace(0.2, 1.0, 5), shuffle=True, random_state=semilla)
        aprendizaje += [{"objetivo": objetivo, "n_train": int(n), "f1_media": float(p.mean()),
                         "f1_sd": float(p.std())} for n, p in zip(tamanos, puntajes)]

        rng = np.random.default_rng(semilla)
        barajados = [f1_score(yb, cross_val_predict(clone(ganador), train, yb, groups=grupos, cv=cv),
                              average="macro")
                     for yb in (rng.permutation(y) for _ in range(3))]
        azar = DummyClassifier(strategy="stratified", random_state=semilla)
        fuga.append({
            "objetivo": objetivo,
            "f1_real_cv": f1_score(y, cross_val_predict(clone(ganador), train, y, groups=grupos, cv=cv),
                                   average="macro"),
            "f1_objetivo_barajado": float(np.mean(barajados)),
            "f1_azar": f1_score(y, cross_val_predict(azar, train, y, groups=grupos, cv=cv),
                                average="macro")})

        importancia.append(_importancia_en_validacion(ganador, train, objetivo, parametros)
                           .assign(objetivo=objetivo))

    def _redondear(df: pd.DataFrame) -> pd.DataFrame:
        return df.round({c: 4 for c in df.select_dtypes("float").columns})

    return tuple(_redondear(df) for df in [
        pd.concat(repetidas, ignore_index=True), pd.DataFrame(contrastes),
        pd.DataFrame(calibracion), pd.DataFrame(aprendizaje), pd.DataFrame(fuga),
        pd.concat(importancia, ignore_index=True), _validacion_adversarial(tabla, semilla)])


# ─────────────────────────────────────────────────────────────────────────────
# 5. Etiquetas finales del perfil de exigencia
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

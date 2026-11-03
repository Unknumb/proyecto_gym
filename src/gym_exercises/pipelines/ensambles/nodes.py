"""
Nodos del pipeline Ensambles — ¿un ensamble supera a los modelos del perfil de
exigencia? (EV3, README §16).

Compara, para cada objetivo de ``modelado_supervisado``, las tres familias de
ensamble contra el modelo que hoy etiqueta el catálogo:

- **Bagging**: Random Forest (el candidato de la EV2) y Extra Trees.
- **Boosting**: ``HistGradientBoostingClassifier``, con sus valores por defecto y
  ajustado con ``RandomizedSearchCV``.
- **Voting** suave y **stacking** de regresión logística + Random Forest +
  boosting. El *stacking* aprende la combinación con una regresión logística
  sobre predicciones fuera de muestra, con pliegues por grupo de variantes.

**Solo informa: no cambia ningún modelo ni etiqueta.** Un ensamble reemplazaría
al modelo actual solo si cumple la misma regla que el contraste de la EV2
(``prueba_pareada``): mejora de F1 ≥ ``mejora_minima_f1``, gana en al menos el
80 % de los pliegues y t de Nadeau-Bengio ≥ 2. La columna ``adoptar`` dice si se
cumple; la decisión de integrarlo al pipeline queda documentada en el README.

**Misma vara para todos.** La CV repetida usa los mismos pliegues
estratificados por grupo que ``diagnosticar_modelos``: las filas de la
regresión logística y del Random Forest reproducen ``modelos_cv_repetida``.
Test y validación externa son los de ``entrenar_modelos``.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.ensemble import (ExtraTreesClassifier, HistGradientBoostingClassifier,
                              StackingClassifier, VotingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline

from ..modelado_supervisado.nodes import (OBJETIVOS, _brier, _columnas_numericas,
                                          _intervalo_bootstrap, _preprocesador,
                                          _validacion_estratificada, prueba_pareada)

logger = logging.getLogger(__name__)

#: Familia de cada candidato, para la tabla y las figuras.
FAMILIAS = {
    "regresion_logistica": "individual",
    "random_forest": "bagging",
    "extra_trees": "bagging",
    "boosting_por_defecto": "boosting",
    "boosting": "boosting",
    "voting": "voting",
    "stacking": "stacking",
}


class PliegesPorGrupo:
    """Validación cruzada por grupos que lee el grupo de la columna ``grupo`` de X.

    ``StackingClassifier`` no reenvía ``groups`` a su validación interna: sin esto,
    las variantes de un mismo ejercicio quedarían a ambos lados de un pliegue y el
    meta-modelo aprendería a confiar en los modelos que memorizan nombres.
    """

    def __init__(self, n_splits: int = 5):
        self.n_splits = n_splits

    def split(self, X, y=None, groups=None):  # noqa: ARG002
        return GroupKFold(n_splits=self.n_splits).split(X, y, X["grupo"])

    def get_n_splits(self, X=None, y=None, groups=None):  # noqa: ARG002
        return self.n_splits


class VariablesFrecuentes(BaseEstimator, TransformerMixin):
    """Deja las variables con al menos ``minimo`` valores no nulos en el
    entrenamiento y entrega una matriz densa.

    ``HistGradientBoostingClassifier`` no acepta matrices dispersas, y construye un
    histograma por variable en cada iteración. Con hojas de al menos ``minimo``
    ejercicios, una variable presente en menos ejercicios no admite ninguna
    división válida: quitarla no cambia lo que el árbol puede aprender, y reduce
    las ~9.000 columnas del TF-IDF a las que sí sirven. Se ajusta dentro de cada
    pliegue, sin mirar la validación.
    """

    def __init__(self, minimo: int = 5):
        self.minimo = minimo

    def fit(self, X, y=None):  # noqa: ARG002
        no_nulos = np.asarray((X != 0).sum(axis=0)).ravel()
        self.columnas_ = np.flatnonzero(no_nulos >= self.minimo)
        return self

    def transform(self, X):
        X = X[:, self.columnas_]
        return X.toarray() if hasattr(X, "toarray") else np.asarray(X)


def _boosting(numericas: list[str], semilla: int, **hiperparametros) -> Pipeline:
    """Boosting sobre las mismas variables que los demás candidatos. Sin
    ``early_stopping``: con ~650 filas, reservar un 10 % para detenerlo cuesta más
    de lo que ahorra, y el número de iteraciones se ajusta en la búsqueda. El
    espacio de búsqueda no baja de ``min_samples_leaf`` = 5, el mínimo de
    ``VariablesFrecuentes``."""
    return Pipeline([
        ("prep", _preprocesador(numericas)),
        ("frecuentes", VariablesFrecuentes(minimo=5)),
        ("clf", HistGradientBoostingClassifier(early_stopping=False, class_weight="balanced",
                                               random_state=semilla, **hiperparametros)),
    ])


def _conjuntos(tabla: pd.DataFrame, objetivo: str) -> tuple[pd.DataFrame, ...]:
    """*Train*, *test* y validación externa, igual que en ``entrenar_modelos``."""
    datos = tabla.loc[(tabla["rol"] == "entrenamiento") & tabla[objetivo].notna()]
    train = datos.loc[datos["particion"] == "train"]
    test = datos.loc[datos["particion"] == "test"]
    externa = tabla.loc[(tabla["rol"] == "prediccion")
                        & tabla[f"{objetivo}_referencia"].notna()
                        & tabla["gemelo_fedb"].isin(test["ejercicio_id"])]
    return train, test, externa


def _ajustar_boosting(train: pd.DataFrame, objetivo: str, numericas: list[str],
                      parametros: dict[str, Any]) -> tuple[dict[str, Any], pd.DataFrame]:
    """``RandomizedSearchCV`` del boosting con el mismo ``GroupKFold`` con que
    ``entrenar_modelos`` ajusta la logística y el Random Forest."""
    config = parametros["ensambles"]["busqueda_boosting"]
    semilla = int(parametros["semilla"])
    busqueda = RandomizedSearchCV(
        _boosting(numericas, semilla), {f"clf__{k}": v for k, v in config["espacio"].items()},
        n_iter=int(config["n_iter"]), scoring="f1_macro",
        cv=GroupKFold(n_splits=int(parametros["n_folds"])), n_jobs=-1, random_state=semilla)
    busqueda.fit(train, train[objetivo], groups=train["grupo"])
    resultados = (pd.DataFrame(busqueda.cv_results_)
                  .rename(columns={"mean_test_score": "cv_f1_macro", "std_test_score": "cv_f1_std",
                                   "rank_test_score": "rango"})
                  .filter(regex=r"^(param_clf__|cv_f1|rango)")
                  .rename(columns=lambda c: c.removeprefix("param_clf__"))
                  .sort_values("rango")
                  .assign(objetivo=objetivo))
    mejores = {k.removeprefix("clf__"): v for k, v in busqueda.best_params_.items()}
    return mejores, resultados


def _candidatos(modelos_objetivo: dict[str, Any], hiper_boosting: dict[str, Any],
                numericas: list[str], parametros: dict[str, Any]) -> dict[str, Any]:
    """Los siete candidatos de un objetivo. La logística y el Random Forest son los
    de la EV2 ya ajustados (``candidatos_train``); los ensambles los reutilizan."""
    semilla = int(parametros["semilla"])
    config = parametros["ensambles"]
    logistica = modelos_objetivo["candidatos_train"]["regresion_logistica"]
    bosque = modelos_objetivo["candidatos_train"]["random_forest"]
    hiper_bosque = bosque.named_steps["clf"].get_params()
    boosting = _boosting(numericas, semilla, **hiper_boosting)
    base = [("logistica", logistica), ("random_forest", bosque), ("boosting", boosting)]
    return {
        "regresion_logistica": logistica,
        "random_forest": bosque,
        "extra_trees": Pipeline([
            ("prep", _preprocesador(numericas)),
            ("clf", ExtraTreesClassifier(
                n_estimators=int(config["n_arboles_extra"]), class_weight="balanced_subsample",
                max_depth=hiper_bosque["max_depth"], max_features=hiper_bosque["max_features"],
                min_samples_leaf=hiper_bosque["min_samples_leaf"], random_state=semilla,
                n_jobs=-1)),
        ]),
        "boosting_por_defecto": _boosting(numericas, semilla),
        "boosting": boosting,
        "voting": VotingClassifier(base, voting="soft"),
        "stacking": StackingClassifier(
            base, final_estimator=LogisticRegression(max_iter=5000, class_weight="balanced"),
            cv=PliegesPorGrupo(int(parametros["n_folds"])), stack_method="predict_proba"),
    }


def _f1_pliegue(estimador, train: pd.DataFrame, y: np.ndarray, entrenar, validar) -> float:
    modelo = clone(estimador).fit(train.iloc[entrenar], y[entrenar])
    return f1_score(y[validar], modelo.predict(train.iloc[validar]), average="macro")


def _cv_repetida(candidatos: dict[str, Any], train: pd.DataFrame, objetivo: str,
                 parametros: dict[str, Any]) -> pd.DataFrame:
    """F1 macro por pliegue con los mismos pliegues que ``diagnosticar_modelos``,
    en paralelo sobre repetición × pliegue × candidato."""
    y, grupos = train[objetivo].to_numpy(), train["grupo"].to_numpy()
    tareas = [(repeticion, pliegue, nombre, a, b)
              for repeticion in range(int(parametros["diagnosticos"]["n_repeticiones_cv"]))
              for pliegue, (a, b) in enumerate(
                  _validacion_estratificada(parametros, int(parametros["semilla"]) + repeticion)
                  .split(train, y, grupos))
              for nombre in candidatos]
    f1 = Parallel(n_jobs=-1)(delayed(_f1_pliegue)(candidatos[nombre], train, y, a, b)
                             for _, _, nombre, a, b in tareas)
    return pd.DataFrame([{"objetivo": objetivo, "repeticion": r, "pliegue": p, "modelo": n,
                          "f1_macro": v} for (r, p, n, _, _), v in zip(tareas, f1)])


def _evaluar(estimador, train: pd.DataFrame, test: pd.DataFrame, externa: pd.DataFrame,
             objetivo: str, parametros: dict[str, Any]) -> dict[str, float]:
    """Entrena con *train* y mide en *train* (para la brecha de sobreajuste), en
    *test* y en la validación externa."""
    modelo = clone(estimador).fit(train, train[objetivo])
    n_boot, semilla = int(parametros["diagnosticos"]["n_bootstrap"]), int(parametros["semilla"])
    fila = {"f1_train": f1_score(train[objetivo], modelo.predict(train), average="macro")}
    for conjunto, df, y in [("test", test, test[objetivo]),
                            ("externa", externa, externa[f"{objetivo}_referencia"])]:
        y_pred = modelo.predict(df)
        inferior, superior = _intervalo_bootstrap(y, y_pred, n_boot, semilla)
        fila.update({f"f1_{conjunto}": f1_score(y, y_pred, average="macro"),
                     f"f1_{conjunto}_ic95_inf": inferior, f"f1_{conjunto}_ic95_sup": superior,
                     f"brier_{conjunto}": _brier(y, modelo.predict_proba(df), modelo.classes_)})
    return fila


def comparar_ensambles(
    tabla: pd.DataFrame, modelos: dict[str, Any], parametros: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compara los ensambles con el modelo actual de cada objetivo. No modifica nada.

    Por objetivo:

    1. Ajusta el boosting con ``RandomizedSearchCV`` (``GroupKFold`` sobre *train*).
    2. CV repetida 5 × 5 de los siete candidatos con los mismos pliegues, y
       ``prueba_pareada`` de cada uno contra el modelo actual.
    3. F1 en *train*, *test* y validación externa, con intervalo *bootstrap*.

    Returns:
        ``busqueda_boosting`` (todas las configuraciones probadas),
        ``cv_repetida`` (F1 por pliegue) y ``comparacion`` (una fila por
        objetivo × candidato, con el contraste y la columna ``adoptar``).
    """
    numericas = _columnas_numericas(tabla)
    busquedas, repetidas, filas = [], [], []

    for objetivo in OBJETIVOS:
        train, test, externa = _conjuntos(tabla, objetivo)
        actual = modelos[objetivo]["modelo"]
        hiper_boosting, busqueda = _ajustar_boosting(train, objetivo, numericas, parametros)
        busquedas.append(busqueda)
        candidatos = _candidatos(modelos[objetivo], hiper_boosting, numericas, parametros)

        repetida = _cv_repetida(candidatos, train, objetivo, parametros)
        repetidas.append(repetida)
        por_pliegue = repetida.pivot_table(index=["repeticion", "pliegue"], columns="modelo",
                                           values="f1_macro")
        evaluaciones = Parallel(n_jobs=-1)(
            delayed(_evaluar)(estimador, train, test, externa, objetivo, parametros)
            for estimador in candidatos.values())

        for (nombre, estimador), evaluacion in zip(candidatos.items(), evaluaciones):
            fila = {"objetivo": objetivo, "modelo": nombre, "familia": FAMILIAS[nombre],
                    "es_modelo_actual": nombre == actual,
                    "hiperparametros": str(hiper_boosting) if "boosting" == nombre else "",
                    "cv_f1_media": por_pliegue[nombre].mean(),
                    "cv_f1_sd": por_pliegue[nombre].std(), **evaluacion}
            fila["brecha_train_cv"] = fila["f1_train"] - fila["cv_f1_media"]
            if nombre != actual:
                fila.update(prueba_pareada(por_pliegue[nombre] - por_pliegue[actual], parametros))
                fila["adoptar"] = (fila["cumple_mejora_minima"] and fila["cumple_pliegues"]
                                   and fila["cumple_significancia"])
            filas.append(fila)

        mejor = max((f for f in filas if f["objetivo"] == objetivo and not f["es_modelo_actual"]),
                    key=lambda f: f["cv_f1_media"])
        logger.info("%s: modelo actual %s (CV %.3f); mejor alternativa %s (CV %.3f, "
                    "diferencia %+.3f, p = %.2f, adoptar = %s).", objetivo, actual,
                    por_pliegue[actual].mean(), mejor["modelo"], mejor["cv_f1_media"],
                    mejor["diferencia_media"], mejor["p_valor"], mejor["adoptar"])

    comparacion = pd.DataFrame(filas)
    comparacion = comparacion.rename(columns={"pct_pliegues_complejo_mejor": "pct_pliegues_mejor"})
    comparacion["adoptar"] = comparacion["adoptar"].astype("boolean")

    def _redondear(df: pd.DataFrame) -> pd.DataFrame:
        return df.round({c: 4 for c in df.select_dtypes("float").columns})

    return (_redondear(pd.concat(busquedas, ignore_index=True)),
            _redondear(pd.concat(repetidas, ignore_index=True)), _redondear(comparacion))

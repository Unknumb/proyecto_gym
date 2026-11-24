"""
Nodos del pipeline Aprendizaje de usuarios — ¿cuánto aprende el modelo con las
correcciones de las personas que usan la página y la app? (README §18).

Todavía no hay usuarios reales en cantidad, así que se **simulan**: personas que
etiquetan ejercicios del catálogo. La etiqueta simulada es la de su gemelo en
free-exercise-db (README §11.3), solo para ejercicios cuyo gemelo quedó en *train*;
la medición se hace en la validación externa, cuyos gemelos están en *test*. Así
ninguna respuesta de la evaluación se filtra al entrenamiento.

Tres escenarios, con el mismo número de aportes:

- ``honestos``: todas las etiquetas son correctas.
- ``con_errores``: el 30 % de las etiquetas es otra clase al azar.
- ``maliciosos``: todas las etiquetas son incorrectas (alguien intenta dañar el modelo).

Por cada tamaño y repetición se aplica el mismo control que en producción
(``evaluar_aprendizaje``): ¿se habrían adoptado esos aportes?

Solo produce un informe: no cambia ``modelos_supervisados`` ni las etiquetas.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import f1_score

from gym_exercises.pipelines.modelado_supervisado.nodes import (OBJETIVOS, filas_entrenamiento,
                                                                filas_externa, incorporar_aportes)

logger = logging.getLogger(__name__)


def _aportes_simulados(pool: pd.DataFrame, objetivo: str, n: int, proporcion_error: float,
                       clases: list[str], rng: np.random.Generator) -> pd.DataFrame:
    """``n`` correcciones sobre ejercicios del ``pool``; una fracción
    ``proporcion_error`` con una clase distinta de la correcta, elegida al azar."""
    elegidos = pool.iloc[rng.choice(len(pool), size=n, replace=False)]
    correctas = elegidos[f"{objetivo}_referencia"].astype(str).to_numpy()
    errada = rng.random(n) < proporcion_error
    etiquetas = [rng.choice([c for c in clases if c != e]) if malo else e
                 for e, malo in zip(correctas, errada)]
    return pd.DataFrame({"ejercicio_id": elegidos["ejercicio_id"].to_numpy(),
                         "objetivo": objetivo, "etiqueta_corregida": etiquetas})


def simular_aprendizaje(tabla: pd.DataFrame, modelos: dict[str, Any],
                        parametros: dict[str, Any]) -> pd.DataFrame:
    """Curva de aprendizaje con aportes de usuarios simulados.

    Para cada objetivo, escenario, número de aportes y repetición: incorpora los
    aportes con ``incorporar_aportes`` (mismo consenso y misma asignación de grupos
    que en producción), reentrena el modelo publicado con sus mismos
    hiperparámetros y mide el F1 macro en *test* y en la validación externa.

    Returns:
        Una fila por objetivo × escenario × aportes × repetición, con la decisión del
        control de calidad (``adoptado``).
    """
    config = parametros["aportes_usuarios"]
    simulacion = config["simulacion"]
    tolerancia = float(config["tolerancia_f1"])
    parametros_un_voto = {**parametros, "aportes_usuarios": {**config, "votos_minimos": 1}}
    filas = []

    for objetivo in OBJETIVOS:
        modelo = modelos[objetivo]["modelo_train"]
        datos = filas_entrenamiento(tabla, objetivo)
        train_ids = set(datos.loc[datos["particion"] == "train", "ejercicio_id"])
        clases = sorted(datos[objetivo].astype(str).unique())
        pool = tabla.loc[(tabla["rol"] == "prediccion") & tabla[f"{objetivo}_referencia"].notna()
                         & tabla["gemelo_fedb"].isin(train_ids)]
        base = None
        for escenario, proporcion_error in simulacion["escenarios"].items():
            for n in sorted({0, *map(int, simulacion["tamanos"])}):
                n = min(int(n), len(pool))
                repeticiones = 1 if n in (0, len(pool)) else int(simulacion["repeticiones"])
                for rep in range(repeticiones):
                    if n == 0 and base is not None:
                        filas.append({**base, "escenario": escenario})
                        continue
                    rng = np.random.default_rng([int(parametros["semilla"]), rep, n])
                    correcciones = _aportes_simulados(pool, objetivo, n, float(proporcion_error),
                                                      clases, rng)
                    t, _ = incorporar_aportes(tabla, correcciones, parametros_un_voto)
                    d = filas_entrenamiento(t, objetivo)
                    train, test = d.loc[d["particion"] == "train"], d.loc[d["particion"] == "test"]
                    externa = filas_externa(t, objetivo, test)
                    ajustado = clone(modelo).fit(train, train[objetivo])
                    fila = {"objetivo": objetivo, "escenario": escenario, "n_aportes": n,
                            "repeticion": rep, "n_externa": len(externa),
                            "f1_test": f1_score(test[objetivo], ajustado.predict(test),
                                                average="macro"),
                            "f1_externa": f1_score(externa[f"{objetivo}_referencia"],
                                                   ajustado.predict(externa), average="macro")}
                    if n == 0:
                        base = fila
                    fila["adoptado"] = bool(
                        fila["f1_test"] >= base["f1_test"] - tolerancia
                        and fila["f1_externa"] >= base["f1_externa"] - tolerancia)
                    filas.append(fila)
        logger.info("%s: simulación de aportes lista (%d ejercicios disponibles).", objetivo,
                    len(pool))

    resultado = pd.DataFrame(filas)
    resultado[["f1_test", "f1_externa"]] = resultado[["f1_test", "f1_externa"]].round(4)
    return resultado

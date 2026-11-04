"""Exporta los modelos del perfil de exigencia para la app de iPhone (``ios/``).

La app predice en el teléfono, sin servidor, la mecánica, la fuerza y la
complejidad técnica de un ejercicio nuevo. Para eso necesita, de cada modelo
ganador de ``modelado_supervisado`` (el estimador final, entrenado con todo
free-exercise-db), todo lo que el pipeline de scikit-learn aprendió:

- los dos TF-IDF (vocabulario e IDF), el *one-hot* y el ``MaxAbsScaler``;
- los coeficientes de la regresión logística o los 400 árboles del Random Forest;
- las clases y el umbral de abstención.

Escribe también ``recomendaciones.json``: las recomendaciones precalculadas por
el ``Sustituidor``, las mismas de la web (``scripts/exportar_datos_web.py``).

Además lleva la ontología (músculo → articulaciones), las reglas del perfil
(impacto, estabilidad, carga graduable), la normalización de nombres y el perfil
ya calculado de cada ejercicio del grafo, para mostrarlo y corregirlo.

``--casos`` escribe casos de prueba: ejercicios reales con sus campos crudos y la
probabilidad que da scikit-learn. ``swift test`` (``ios/SustitutoSeguroCore``)
comprueba que Swift da las mismas probabilidades.

Uso, desde la raíz del repositorio::

    python scripts/exportar_modelo_ios.py           # escribe los tres JSON de ios/
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
from gym_exercises.pipelines.data_enrichment import nodes as enriquecimiento  # noqa: E402
from gym_exercises.pipelines.modelado_supervisado import nodes as modelado  # noqa: E402

pd.set_option("compute.use_numexpr", False)

SALIDA_MODELO = RAIZ / "ios/SustitutoSeguro/Recursos/modelo_perfil.json"
#: Copia versionada de las recomendaciones de la web: ``web/datos_web.json`` es un
#: intermedio ignorado por git, y la app debe compilar desde un clon limpio.
SALIDA_RECOMENDACIONES = RAIZ / "ios/SustitutoSeguro/Recursos/recomendaciones.json"
SALIDA_CASOS = RAIZ / "ios/SustitutoSeguroCore/Tests/SustitutoSeguroCoreTests/Recursos/casos_prueba.json"


def _patron(regex) -> str:
    return regex.pattern if hasattr(regex, "pattern") else regex


def _tfidf(vectorizador) -> dict:
    vocabulario = sorted(vectorizador.vocabulary_, key=vectorizador.vocabulary_.get)
    return {"vocabulario": vocabulario, "idf": vectorizador.idf_.tolist(),
            "ngramas": list(vectorizador.ngram_range),
            "quitar_stop_words": vectorizador.stop_words is not None,
            "tf_sublineal": bool(vectorizador.sublinear_tf), "patron": vectorizador.token_pattern}


def _preprocesador(prep) -> dict:
    partes = {nombre: (transformador, columnas) for nombre, transformador, columnas in prep.transformers_}
    categoricas, columnas_cat = partes["categoricas"]
    escalador, numericas = partes["anatomia"]
    return {
        "nombre": _tfidf(partes["nombre"][0]),
        "instrucciones": _tfidf(partes["instrucciones"][0]),
        "categoricas": [{"columna": c, "categorias": list(map(str, cats))}
                        for c, cats in zip(columnas_cat, categoricas.categories_)],
        "numericas": {"columnas": list(numericas), "escala": escalador.scale_.tolist()},
        "n_variables": len(prep.get_feature_names_out()),
    }


def _arbol(estimador) -> dict:
    t = estimador.tree_
    hoja = t.children_left == -1
    valores = t.value[:, 0, :]
    valores = valores / valores.sum(axis=1, keepdims=True)
    return {"izq": t.children_left.tolist(), "der": t.children_right.tolist(),
            "variable": np.where(hoja, -1, t.feature).tolist(),
            "umbral": np.where(hoja, 0.0, t.threshold).tolist(),
            "valor": [valores[i].tolist() if hoja[i] else [] for i in range(t.node_count)]}


def _modelo(info: dict) -> dict:
    estimador = info["estimador"]
    clf = estimador.named_steps["clf"]
    salida = {"algoritmo": info["modelo"], "clases": list(map(str, clf.classes_)),
              "umbral_abstencion": float(info["umbral"]),
              "preprocesador": _preprocesador(estimador.named_steps["prep"])}
    if info["modelo"] == "regresion_logistica":
        salida["logistica"] = {"coeficientes": clf.coef_.tolist(), "intercepto": clf.intercept_.tolist()}
    else:
        salida["bosque"] = [_arbol(arbol) for arbol in clf.estimators_]
    return salida


def nombres_en_espanol() -> dict[str, str]:
    """Los 68 ejercicios con nombre en español de la web (``CURADOS`` en
    ``web/app.html``), por ``ejercicio_id``, para buscar en español en la app."""
    html = (RAIZ / "web/app.html").read_text(encoding="utf-8")
    bloque = html[html.index("const CURADOS = ["):html.index("];", html.index("const CURADOS = ["))]
    return dict(re.findall(r"\['([^']+)', '([^']+)'", bloque))


def exportar_modelo(modelos: dict, ejercicios: pd.DataFrame, perfil: pd.DataFrame,
                    musculo_articulacion: pd.DataFrame) -> dict:
    articulaciones = musculo_articulacion.groupby("musculo")["articulacion"].agg(sorted).to_dict()
    perfil = perfil.set_index("ejercicio_id")
    columnas_perfil = [f"{o}{s}" for o in modelado.OBJETIVOS for s in ("", "_origen", "_confianza")]
    return {
        "ontologia": {"musculos": sorted(articulaciones), "articulaciones_por_musculo": articulaciones,
                      "familias_equipamiento": sorted(ejercicios["familia_equipamiento"].unique())},
        "reglas": {
            "impacto": _patron(modelado._IMPACTO), "sin_impacto": _patron(modelado._SIN_IMPACTO),
            "inestable": _patron(modelado._INESTABLE), "apoyado": _patron(modelado._APOYADO),
            "estiramiento": r"\bstretch",
            "familias_inestable": ["balon o rodillo"], "familias_apoyado": ["asistido", "maquina"],
            "familias_carga_graduable": sorted(modelado.FAMILIAS_CARGA_GRADUABLE),
        },
        "normalizacion": {
            "prefijo_codigo": _patron(enriquecimiento._PREFIJO_CODIGO),
            "prefijo_programa": _patron(enriquecimiento._PREFIJO_PROGRAMA),
            "prefijo_programa_sin_mayusculas": bool(enriquecimiento._PREFIJO_PROGRAMA.flags & 2),
            "ruido": list(enriquecimiento._RUIDO),
            "sinonimos": [list(par) for par in enriquecimiento._SINONIMOS],
        },
        "stop_words": sorted(ENGLISH_STOP_WORDS),
        "modelos": {o: _modelo(modelos[o]) for o in modelado.OBJETIVOS},
        "nombres_es": nombres_en_espanol(),
        "perfil": {"columnas": columnas_perfil,
                   "filas": {eid: [None if pd.isna(v) else (round(float(v), 3) if isinstance(v, float) else str(v))
                                   for v in fila] for eid, fila in perfil[columnas_perfil].iterrows()}},
    }


def exportar_casos(modelos: dict, ejercicios: pd.DataFrame, tabla: pd.DataFrame, n: int, semilla: int) -> dict:
    """Ejercicios del catálogo y de wger (filas de predicción de la tabla de
    modelado) con sus campos crudos y la probabilidad que da scikit-learn."""
    prediccion = tabla.loc[tabla["rol"] == "prediccion"].set_index("ejercicio_id")
    muestra = ejercicios.loc[ejercicios["ejercicio_id"].isin(prediccion.index)].sample(n, random_state=semilla)
    filas = prediccion.loc[muestra["ejercicio_id"]].reset_index()
    probas = {o: modelos[o]["estimador"].predict_proba(filas) for o in modelado.OBJETIVOS}
    casos = []
    for k, (_, e) in enumerate(muestra.iterrows()):
        casos.append({"ejercicio_id": e["ejercicio_id"], "nombre": e["nombre"],
                      "instrucciones_en": e["instrucciones_en"] or "",
                      "familia_equipamiento": e["familia_equipamiento"],
                      "musculos_primarios": list(e["musculos_primarios"]),
                      "musculos_secundarios": list(e["musculos_secundarios"]),
                      "esperado": {o: probas[o][k].tolist() for o in modelado.OBJETIVOS}})
    nombres = [{"nombre": nombre, "nombre_norm": norm}
               for nombre, norm in zip(ejercicios["nombre"], ejercicios["nombre_norm"])]
    return {"casos": casos, "nombres": nombres}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--casos", type=int, default=300, help="ejercicios en los casos de prueba")
    args = ap.parse_args()

    modelos = joblib.load(RAIZ / "models/modelos_supervisados.pkl")
    ejercicios = pd.read_parquet(RAIZ / "data/03_primary/ejercicios.parquet")
    perfil = pd.read_parquet(RAIZ / "data/07_model_output/perfil_exigencia.parquet")
    tabla = pd.read_parquet(RAIZ / "data/05_model_input/tabla_modelado.parquet")
    musculo_articulacion = pd.read_csv(RAIZ / "data/03_primary/musculo_articulacion.csv")

    sys.path.insert(0, str(RAIZ / "scripts"))
    import exportar_datos_web  # noqa: PLC0415

    for ruta, datos in [(SALIDA_MODELO, exportar_modelo(modelos, ejercicios, perfil, musculo_articulacion)),
                        (SALIDA_RECOMENDACIONES, exportar_datos_web.exportar(*exportar_datos_web.cargar(None))),
                        (SALIDA_CASOS, exportar_casos(modelos, ejercicios, tabla, args.casos, 2026))]:
        ruta.parent.mkdir(parents=True, exist_ok=True)
        ruta.write_text(json.dumps(datos, ensure_ascii=False, separators=(",", ":"), allow_nan=False),
                        encoding="utf-8")
        print(f"{ruta.relative_to(RAIZ)} ({ruta.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()

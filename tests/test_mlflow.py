"""Tests del seguimiento con MLflow (``gym_exercises.hooks``, README §8.8): las
métricas registradas son las mismas cifras de los reportes versionados, y el hook
completo deja una corrida terminada con un modelo que predice igual que el ``.pkl``."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import pytest

from gym_exercises.hooks import (MlflowHook, aplanar, curva_barrido, metricas_clustering,
                                 metricas_modelos, metricas_sustitucion)

RAIZ = Path(__file__).resolve().parents[1]
REPORTES = RAIZ / "data/08_reporting"


def test_aplanar_anida_y_limpia_nombres():
    plano = aplanar({"a": {"b": 1, "columna (lumbar)": ["x"]}, "c": None}, "grupo.")
    assert plano == {"grupo.a.b": "1", "grupo.a.columna _lumbar_": "['x']", "grupo.c": "None"}


def test_metricas_modelos_coinciden_con_el_reporte():
    comparacion = pd.read_csv(REPORTES / "modelos_comparacion.csv")
    metricas = metricas_modelos(comparacion)
    fila = comparacion.query("objetivo == 'fuerza' and modelo == 'random_forest' "
                             "and conjunto == 'externa_catalogo'").iloc[0]
    assert metricas["fuerza/random_forest/externa_catalogo/f1_macro"] == fila["f1_macro"]
    assert metricas["fuerza/random_forest/cv_f1_macro"] == fila["cv_f1_macro"]
    # El azar no tiene CV ni Brier: no se inventan métricas vacías.
    assert "mecanica/azar_estratificado/test/f1_macro" in metricas
    assert "mecanica/azar_estratificado/test/brier" not in metricas
    assert "mecanica/azar_estratificado/cv_f1_macro" not in metricas


def test_kpi_de_cobertura_es_el_del_metodo_elegido(parametros):
    cobertura = pd.read_csv(REPORTES / "sustitucion_cobertura.csv")
    metodo = parametros["grafo_biomecanico"]["metodo_familias"]
    esperado = cobertura.query("metodo == @metodo and articulacion == 'TODAS'")
    metricas = metricas_sustitucion(cobertura, metodo)
    assert metricas["kpi/cobertura_sustitucion"] == esperado["cobertura_misma_familia"].iloc[0]
    assert "sustitucion/louvain/cobertura_total" in metricas


def test_metricas_clustering_por_metodo():
    metricas = metricas_clustering(pd.read_csv(REPORTES / "clustering_comparacion.csv"))
    assert {"familias/louvain/n_grupos", "familias/kmeans_node2vec/silhouette_embedding"} <= set(metricas)


def test_curva_barrido_usa_pasos_enteros():
    louvain = pd.read_csv(REPORTES / "louvain_barrido.csv")
    pasos = {paso for _, _, paso in curva_barrido(louvain, "resolucion", 10)}
    assert pasos == {int(round(r * 10)) for r in louvain["resolucion"]}


def test_hook_registra_la_corrida_y_el_modelo(tmp_path, parametros):
    mlflow = pytest.importorskip("mlflow")
    modelos = joblib.load(RAIZ / "models/modelos_supervisados.pkl")
    tabla = pd.read_parquet(RAIZ / "data/05_model_input/tabla_modelado.parquet")
    salidas = {"modelos_comparacion": pd.read_csv(REPORTES / "modelos_comparacion.csv"),
               "modelos_supervisados": {"mecanica": modelos["mecanica"]}}

    hook = MlflowHook()
    hook._config = {"activo": True, "tracking_uri": f"sqlite:///{tmp_path / 'mlflow.db'}",
                    "carpeta_artefactos": str(tmp_path / "mlruns"), "experimento": "prueba",
                    "registrar_modelos": True, "alias_modelo": "candidato"}
    hook._raiz = RAIZ
    hook._parametros = parametros
    pipeline = SimpleNamespace(inputs=lambda: {"params:modelado_supervisado"},
                               all_outputs=lambda: set())
    try:
        hook.before_pipeline_run({"pipeline_names": ["prueba"]}, pipeline)
        hook.after_node_run(SimpleNamespace(name="entrenar_modelos"),
                            {"model_input_tabla_modelado": tabla}, salidas)
        hook.after_pipeline_run(pipeline)
    finally:
        if mlflow.active_run():
            mlflow.end_run()

    corrida = mlflow.search_runs(experiment_names=["prueba"]).iloc[0]
    assert corrida["status"] == "FINISHED"
    assert corrida["params.modelado_supervisado.semilla"] == str(
        parametros["modelado_supervisado"]["semilla"])
    assert corrida["metrics.mecanica/regresion_logistica/test/f1_macro"] == pytest.approx(0.8824)
    cargado = mlflow.sklearn.load_model("models:/gym_perfil_mecanica@candidato")
    np.testing.assert_array_equal(cargado.predict_proba(tabla),
                                  modelos["mecanica"]["estimador"].predict_proba(tabla))

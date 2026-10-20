"""Tests unitarios de las funciones puras de los pipelines."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gym_exercises.pipelines.data_enrichment.nodes import _tokens_equivalentes, normalizar_nombre
from gym_exercises.pipelines.integracion_fuentes.nodes import _modificadores_contradicen
from gym_exercises.pipelines.modelado_supervisado.nodes import (
    _contraste_pareado, _umbral_abstencion, _un_error_estandar, derivar_perfil_reglas)


# ── Emparejamiento entre fuentes ─────────────────────────────────────────────
def test_abduccion_y_aduccion_no_son_el_mismo_token():
    assert not _tokens_equivalentes("abduction", "adduction")


@pytest.mark.parametrize("a, b", [
    ("side lying hip abduction", "side lying hip adduction"),
    ("incline dumbbell press", "decline dumbbell press"),
    ("seated calf raise", "standing calf raise"),
    ("lateral raise", "rear lateral raise"),
])
def test_modificadores_opuestos_vetan_el_par(a, b):
    assert _modificadores_contradicen(a, b)


def test_mismo_ejercicio_sin_modificadores_no_se_veta():
    assert not _modificadores_contradicen("dumbbell bench press", "bench press dumbbell")


def test_normalizar_nombre_quita_marcas_de_presentacion():
    assert normalizar_nombre("Barbell Curl (male)") == normalizar_nombre("barbell curl")


# ── Reglas del perfil de exigencia ───────────────────────────────────────────
def test_reglas_del_perfil():
    nombres = pd.Series(["box jump", "stationary bike run", "single leg squat",
                         "seated cable row", "barbell squat", "hamstring stretch"])
    familia = pd.Series(["peso corporal", "maquina", "peso corporal", "cable", "peso libre",
                         "peso corporal"])
    reglas = derivar_perfil_reglas(nombres, familia)
    assert reglas["impacto"].tolist() == [True, False, False, False, False, False]
    assert reglas["estabilidad"].tolist() == ["libre", "apoyado", "inestable", "apoyado", "libre", "libre"]
    assert reglas["carga_graduable"].tolist() == [False, True, False, True, False, False]
    assert reglas["es_estiramiento_nombre"].tolist() == [False] * 5 + [True]


# ── Selección y abstención ───────────────────────────────────────────────────
def test_un_error_estandar_elige_el_c_mas_simple_dentro_de_la_banda():
    cv_results = {
        "mean_test_score": [0.80, 0.89, 0.90, 0.905],
        "std_test_score": [0.02, 0.02, 0.02, 0.02],
        "param_clf__C": [0.1, 1.0, 10.0, 100.0],
        **{f"split{i}_test_score": [0] * 4 for i in range(4)},   # 4 pliegues → SE = 0,01
    }
    assert cv_results["param_clf__C"][_un_error_estandar(cv_results)] == 10.0


def test_umbral_de_abstencion_alcanza_la_precision_pedida():
    rng = np.random.default_rng(0)
    clases = np.array(["a", "b"])
    confianza = rng.uniform(0.5, 1.0, 400)
    acierta = rng.uniform(size=400) < confianza          # más confianza, más acierto
    y_real = np.where(acierta, "a", "b")
    probas = np.column_stack([confianza, 1 - confianza])
    umbral = _umbral_abstencion(y_real, probas, clases, 0.95)
    retenidos = confianza >= umbral
    assert umbral < 1.0
    assert (y_real[retenidos] == "a").mean() >= 0.95


def test_umbral_de_abstencion_se_abstiene_si_no_hay_precision_posible():
    probas = np.full((100, 2), 0.5) + np.array([0.3, -0.3])
    y_real = np.array(["a", "b"] * 50)
    assert _umbral_abstencion(y_real, probas, np.array(["a", "b"]), 0.95) == 1.0


def _cv_sintetica(mejora: float, ruido: float) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    filas = []
    for r in range(5):
        for k in range(5):
            base = 0.85 + rng.normal(0, 0.02)
            filas += [{"repeticion": r, "pliegue": k, "modelo": "simple", "f1_macro": base},
                      {"repeticion": r, "pliegue": k, "modelo": "complejo",
                       "f1_macro": base + mejora + rng.normal(0, ruido)}]
    return pd.DataFrame(filas)


PARAMETROS_CONTRASTE = {"n_folds": 5, "diagnosticos": {
    "mejora_minima_f1": 0.02, "pliegues_minimos_mejora": 0.8, "t_minimo_mejora": 2.0}}


def test_contraste_pareado_acepta_una_mejora_clara():
    r = _contraste_pareado(_cv_sintetica(0.05, 0.005), "simple", PARAMETROS_CONTRASTE)
    assert r["cumple_mejora_minima"] and r["cumple_pliegues"] and r["cumple_significancia"]


def test_contraste_pareado_no_acepta_ruido():
    r = _contraste_pareado(_cv_sintetica(0.0, 0.02), "simple", PARAMETROS_CONTRASTE)
    assert not r["cumple_significancia"]

"""Tests de la comparación de ensambles (pipeline ``ensambles``, README §16): misma
vara que la EV2, las tres familias de ensamble presentes, la búsqueda del boosting
mejora sus valores por defecto y la decisión de adoptar sigue la regla fijada."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gym_exercises.hooks import metricas_ensambles
from gym_exercises.pipelines.ensambles.nodes import FAMILIAS, PliegesPorGrupo
from gym_exercises.pipelines.modelado_supervisado.nodes import OBJETIVOS, prueba_pareada

REPORTES = Path(__file__).resolve().parents[1] / "data/08_reporting"


@pytest.fixture(scope="module")
def ensambles() -> pd.DataFrame:
    return pd.read_csv(REPORTES / "ensambles_comparacion.csv")


@pytest.fixture(scope="module")
def cv_ensambles() -> pd.DataFrame:
    return pd.read_csv(REPORTES / "ensambles_cv_repetida.csv")


@pytest.fixture(scope="module")
def busqueda() -> pd.DataFrame:
    return pd.read_csv(REPORTES / "ensambles_busqueda_boosting.csv")


# ── Unitarios ────────────────────────────────────────────────────────────────
def test_pliegues_por_grupo_no_reparten_un_grupo():
    X = pd.DataFrame({"grupo": np.repeat(np.arange(20), 3), "x": np.arange(60)})
    pliegues = list(PliegesPorGrupo(5).split(X))
    assert len(pliegues) == PliegesPorGrupo(5).get_n_splits() == 5
    for entrenar, validar in pliegues:
        assert not set(X["grupo"].iloc[entrenar]) & set(X["grupo"].iloc[validar])


def test_prueba_pareada_exige_los_tres_criterios(parametros):
    p = parametros["modelado_supervisado"]
    rng = np.random.default_rng(0)
    clara = pd.Series(0.05 + rng.normal(0, 0.005, 25))
    ruido = pd.Series(rng.normal(0.005, 0.03, 25))
    assert all(prueba_pareada(clara, p)[k] for k in
               ["cumple_mejora_minima", "cumple_pliegues", "cumple_significancia"])
    assert not prueba_pareada(ruido, p)["cumple_significancia"]


# ── Sobre los reportes versionados ───────────────────────────────────────────
def test_estan_las_tres_familias_de_ensamble(ensambles):
    for objetivo in OBJETIVOS:
        familias = set(ensambles.loc[ensambles["objetivo"] == objetivo, "familia"])
        assert {"bagging", "boosting", "voting", "stacking"} <= familias
    assert set(ensambles["modelo"]) == set(FAMILIAS)


def test_misma_vara_que_el_diagnostico_de_la_ev2(cv_ensambles):
    """Mismos pliegues: la logística y el Random Forest reproducen la CV repetida de
    ``diagnosticar_modelos`` pliegue a pliegue."""
    ev2 = pd.read_csv(REPORTES / "modelos_cv_repetida.csv")
    claves = ["objetivo", "repeticion", "pliegue", "modelo"]
    cruce = ev2.merge(cv_ensambles, on=claves, suffixes=("_ev2", "_ev3"), validate="one_to_one")
    assert len(cruce) == len(ev2)
    np.testing.assert_allclose(cruce["f1_macro_ev2"], cruce["f1_macro_ev3"], atol=1e-4)


def test_modelo_actual_es_el_de_modelado_supervisado(ensambles, modelos):
    actuales = ensambles.loc[ensambles["es_modelo_actual"]].set_index("objetivo")["modelo"]
    assert actuales.to_dict() == {o: modelos[o]["modelo"] for o in OBJETIVOS}


def test_la_busqueda_mejora_el_boosting_por_defecto(ensambles, busqueda):
    for objetivo in OBJETIVOS:
        filas = ensambles[ensambles["objetivo"] == objetivo].set_index("modelo")
        assert filas.loc["boosting", "cv_f1_media"] >= filas.loc["boosting_por_defecto", "cv_f1_media"]
        probadas = busqueda[busqueda["objetivo"] == objetivo]
        assert probadas["rango"].min() == 1 and len(probadas) >= 10


def test_adoptar_sigue_la_regla_y_ningun_ensamble_la_cumple(ensambles):
    """La decisión de la EV3 (README §16): un ensamble reemplaza al modelo actual
    solo si cumple los tres criterios. Si este test falla tras un ``kedro run``, la
    evidencia cambió y hay que revisar la decisión, no el test."""
    alternativas = ensambles[~ensambles["es_modelo_actual"]]
    regla = (alternativas["cumple_mejora_minima"] & alternativas["cumple_pliegues"]
             & alternativas["cumple_significancia"])
    assert (alternativas["adoptar"] == regla).all()
    assert not alternativas["adoptar"].any()


def test_metricas_ensambles_coinciden_con_el_reporte(ensambles):
    metricas = metricas_ensambles(ensambles)
    fila = ensambles.query("objetivo == 'fuerza' and modelo == 'stacking'").iloc[0]
    assert metricas["ensambles/fuerza/stacking/cv_f1_media"] == fila["cv_f1_media"]
    assert metricas["ensambles/fuerza/stacking/p_valor"] == fila["p_valor"]
    actual = ensambles.query("objetivo == 'fuerza' and es_modelo_actual").iloc[0]["modelo"]
    assert f"ensambles/fuerza/{actual}/p_valor" not in metricas

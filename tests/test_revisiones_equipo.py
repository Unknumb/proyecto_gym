"""Tests de las revisiones humanas de la EFT (``scripts/revisiones_equipo.py``, README §19.3).

Verifican que las planillas versionadas sean la muestra determinista que da hoy el
pipeline (nadie eligió a mano qué revisar), que el acierto y su intervalo se calculen
bien y que las etiquetas revisadas entren al reentrenamiento por el importador de §18.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

RAIZ = Path(__file__).resolve().parents[1]


def _modulo(nombre: str):
    spec = importlib.util.spec_from_file_location(nombre, RAIZ / f"scripts/{nombre}.py")
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


rev = _modulo("revisiones_equipo")
FIJAS_ETIQUETAS = rev.COLUMNAS_ETIQUETAS[:10]  # lo que no completa el equipo
FIJAS_MAPEO = rev.COLUMNAS_MAPEO[:6]


@pytest.fixture(scope="module")
def datos():
    return rev._cargar_datos()


@pytest.fixture(scope="module")
def planilla() -> pd.DataFrame:
    return pd.read_csv(rev.PLANILLA_ETIQUETAS, dtype=str, keep_default_na=False)


def test_la_planilla_de_etiquetas_es_la_muestra_del_pipeline(datos, planilla):
    esperada = rev.muestra_etiquetas(*datos).astype(str)
    assert list(planilla.columns) == rev.COLUMNAS_ETIQUETAS
    pd.testing.assert_frame_equal(planilla[FIJAS_ETIQUETAS], esperada[FIJAS_ETIQUETAS])


def test_la_muestra_mide_lo_que_pide_el_kpi(datos, planilla):
    perfil, _ = datos
    assert planilla.groupby("objetivo").size().to_dict() == {o: rev.POR_OBJETIVO for o in rev.OBJETIVOS}
    assert planilla["ejercicio_id"].is_unique
    origen = perfil.set_index("ejercicio_id")
    for _, r in planilla.iterrows():
        # Solo etiquetas puestas por el modelo en el catálogo: las que mide el KPI de acierto
        assert origen.at[r["ejercicio_id"], "fuente"] == "catalogo"
        assert origen.at[r["ejercicio_id"], f"{r['objetivo']}_origen"] == "modelo"


def test_la_planilla_del_mapeo_cubre_todo_el_veto():
    planilla = pd.read_csv(rev.PLANILLA_MAPEO, dtype=str, keep_default_na=False)
    mapeo = pd.read_csv(RAIZ / "data/03_primary/musculo_articulacion.csv", dtype=str)
    assert list(planilla.columns) == rev.COLUMNAS_MAPEO
    assert set(zip(planilla["musculo"], planilla["articulacion"])) == set(zip(mapeo["musculo"], mapeo["articulacion"]))


def test_intervalo_de_wilson():
    bajo, alto = rev.wilson(9, 10)
    assert bajo == pytest.approx(0.5958, abs=1e-4) and alto == pytest.approx(0.9821, abs=1e-4)
    assert rev.wilson(10, 10)[1] == pytest.approx(1.0)
    assert 0 <= rev.wilson(0, 5)[0] <= rev.wilson(0, 5)[1] < 0.5


def test_el_acierto_cuenta_solo_lo_revisado(planilla):
    revision = planilla.copy()
    # Se revisan 10 de mecánica: 9 confirmadas y 1 corregida; el resto queda sin revisar
    filas = revision.index[revision["objetivo"] == "mecanica"][:10]
    revision.loc[filas, "etiqueta_correcta"] = revision.loc[filas, "etiqueta_pipeline"]
    otra = {"compound": "isolation", "isolation": "compound"}
    revision.loc[filas[0], "etiqueta_correcta"] = otra[revision.at[filas[0], "etiqueta_pipeline"]]
    resultado = rev.evaluar_etiquetas(revision).set_index("grupo")
    assert resultado.at["mecanica", "revisadas"] == 10 and resultado.at["mecanica", "aciertos"] == 9
    assert resultado.at["mecanica", "acierto"] == pytest.approx(0.9)
    assert not resultado.at["mecanica", "cumple_meta"]  # 9/10: el intervalo baja de 0,90
    assert resultado.at["fuerza", "revisadas"] == 0
    assert resultado.at["total", "de"] == len(planilla)


def test_una_opcion_que_no_existe_se_rechaza(planilla):
    revision = planilla.copy()
    revision.loc[revision.index[0], "etiqueta_correcta"] = "compuesto"
    assert rev._validar_revision(revision)
    assert not rev._validar_revision(planilla)


def test_las_etiquetas_revisadas_entran_por_el_importador(datos, planilla):
    """Lo revisado sale en el formato de ``correcciones_etiquetas`` y el importador de §18 lo acepta."""
    importador = _modulo("importar_aportes")
    perfil, _ = datos
    revision = planilla.copy()
    revision.loc[revision.index[:6], "etiqueta_correcta"] = revision.loc[revision.index[:6], "etiqueta_pipeline"]
    revision.loc[revision.index[0], "revisor"] = "Equipo"
    aportes = rev.aportes_de_la_revision(revision, perfil, "2026-11-27")
    assert list(aportes.columns) == importador.COLUMNAS["correcciones_etiquetas"]
    assert len(aportes) == 6 and aportes["comentario"].str.startswith("revisión del equipo").all()
    assert importador.validar("correcciones_etiquetas", aportes, importador._referencias()) == []


def test_mapeo_sin_revisar_y_revisado():
    planilla = pd.read_csv(rev.PLANILLA_MAPEO, dtype=str, keep_default_na=False)
    vacio = rev.evaluar_mapeo(planilla).iloc[0]
    assert vacio["revisadas"] == 0 and vacio["de"] == len(planilla)
    revisado = planilla.copy()
    revisado.loc[revisado.index[:4], "veredicto"] = ["correcto", "Correcto", "incorrecto", "incompleto"]
    fila = rev.evaluar_mapeo(revisado).iloc[0]
    assert fila["revisadas"] == 4 and fila["aciertos"] == 2

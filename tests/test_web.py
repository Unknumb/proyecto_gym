"""Tests de la página web (``docs/index.html``, README §15).

Verifican que la página publicada esté armada desde ``web/app.html``, que sus datos
coincidan con las salidas actuales del pipeline y que ninguna alternativa
precalculada cargue la articulación vetada.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import pytest
import yaml

RAIZ = Path(__file__).resolve().parents[1]
PAGINA = RAIZ / "docs/index.html"
FUENTE = RAIZ / "web/app.html"


@pytest.fixture(scope="module")
def html() -> str:
    return PAGINA.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def datos(html: str) -> dict:
    m = re.search(r'<script type="application/json" id="datos">(.*?)</script>', html, re.S)
    assert m, "docs/index.html no tiene el bloque de datos"
    return json.loads(m.group(1).replace("<\\/", "</"))


def test_pagina_armada_desde_la_fuente(html: str):
    """docs/index.html = web/app.html + datos (scripts/armar_web.py)."""
    antes, despues = FUENTE.read_text(encoding="utf-8").split("__DATOS__")
    assert antes in html and despues in html, "docs/index.html está desactualizado: corre scripts/armar_web.py"


def test_datos_coinciden_con_el_pipeline(datos: dict):
    familias = pd.read_parquet(RAIZ / "data/07_model_output/familias_movimiento.parquet")
    assert [e[0] for e in datos["ejercicios"]] == familias["nombre"].tolist(), \
        "datos de la web desactualizados: corre scripts/exportar_datos_web.py y scripts/armar_web.py"


def test_ninguna_alternativa_carga_la_articulacion_vetada(datos: dict):
    params = yaml.safe_load((RAIZ / "conf/base/parameters.yml").read_text(encoding="utf-8"))
    incluye = params["grafo_biomecanico"]["sustitucion"].get("veto_incluye") or {}
    arts = datos["articulaciones"]
    carga = [set(e[5]) for e in datos["ejercicios"]]
    for por_art in datos["rec"]:
        for a, par in por_art.items():
            vetadas = {arts.index(x) for x in [arts[int(a)], *incluye.get(arts[int(a)], [])]}
            for _, filas in par:
                assert not any(vetadas & carga[j] for j, _, _ in filas)


def test_ejercicios_curados_existen(html: str, datos: dict):
    """Los ejercicios con nombre en español apuntan a índices válidos y sin duplicar."""
    bloque = html.split("const CURADOS = [")[1].split("];")[0]
    ids = [int(x) for x in re.findall(r"\[(\d+), '", bloque)]
    assert ids and len(ids) == len(set(ids))
    assert all(0 <= i < len(datos["ejercicios"]) for i in ids)
    nombres = {i: datos["ejercicios"][i][0].lower() for i in ids}
    assert nombres[1382] == "bodyweight squat" and nombres[98] == "barbell bench press" and nombres[1651] == "plank"

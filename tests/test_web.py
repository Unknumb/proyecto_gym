"""Tests de la página web (``docs/index.html``, README §15).

Verifican que la página publicada esté armada desde ``web/app.html``, que sus datos
coincidan con las salidas actuales del pipeline y que ninguna alternativa
precalculada cargue la articulación vetada.
"""
from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

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
    """Los datos de la página son exactamente los que da hoy el Sustituidor: ejercicios y recomendaciones."""
    spec = importlib.util.spec_from_file_location("exportar_datos_web", RAIZ / "scripts/exportar_datos_web.py")
    exportar_web = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exportar_web)
    # Ida y vuelta por JSON: las claves numéricas de "rec" quedan como texto, igual que en la página
    actuales = json.loads(json.dumps(exportar_web.exportar(*exportar_web.cargar(None)), ensure_ascii=False))
    desactualizado = "datos de la web desactualizados: corre scripts/exportar_datos_web.py y scripts/armar_web.py"
    assert datos["ids"] == actuales["ids"], desactualizado
    for clave in actuales:
        assert datos[clave] == actuales[clave], f"{desactualizado} (difiere '{clave}')"


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


def _ids_en_bloque(html: str, inicio: str, fin: str) -> list[str]:
    return re.findall(r"'([a-z_]+:[^']+)'", html.split(inicio)[1].split(fin)[0])


def test_ejercicios_curados_existen(html: str, datos: dict):
    """Los ejercicios con nombre en español se identifican por ejercicio_id, existen y no se repiten."""
    bloque = html.split("const CURADOS = [")[1].split("];")[0]
    ids = re.findall(r"\['([a-z_]+:[^']+)', '", bloque)
    assert len(ids) == 68 and len(ids) == len(set(ids))
    posicion = {x: k for k, x in enumerate(datos["ids"])}
    faltan = [x for x in ids if x not in posicion]
    assert not faltan, f"ejercicios curados que ya no están en los datos: {faltan}"
    nombre = {x: datos["ejercicios"][posicion[x]][0].lower() for x in ids}
    assert nombre["free_exercise_db:Bodyweight_Squat"] == "bodyweight squat"
    assert nombre["catalogo:0025"] == "barbell bench press"
    assert nombre["free_exercise_db:Plank"] == "plank"
    # Los complementos de cada deporte también son ejercicios curados
    deportes = _ids_en_bloque(html, "const DEPORTES = {", "\n  };")
    assert deportes and set(deportes) <= set(ids)

"""Tests de la app de iPhone (``ios/``): el modelo que lleva la app es el del
pipeline, sus casos de prueba siguen valiendo, y la app y el importador hablan
el mismo formato. La paridad numérica Swift ↔ scikit-learn la comprueban
``swift run verificar`` y ``swift test`` (job ``ios`` del CI, en macOS)."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "scripts"))
import exportar_modelo_ios as exportador  # noqa: E402
import importar_datos_ios as importador  # noqa: E402

MODELO = RAIZ / "ios/SustitutoSeguro/Recursos/modelo_perfil.json"
RECOMENDACIONES = RAIZ / "ios/SustitutoSeguro/Recursos/recomendaciones.json"
CASOS = RAIZ / "ios/SustitutoSeguroCore/Tests/SustitutoSeguroCoreTests/Recursos/casos_prueba.json"
SWIFT_EXPORTACION = RAIZ / "ios/SustitutoSeguroCore/Sources/SustitutoSeguroCore/Exportacion.swift"


@pytest.fixture(scope="module")
def modelo_app() -> dict:
    return json.loads(MODELO.read_text(encoding="utf-8"))


def test_la_app_lleva_el_modelo_actual_del_pipeline(modelos, ejercicios, perfil, musculo_articulacion, modelo_app):
    """Falla si se reentrena (``kedro run``) sin volver a correr
    ``python scripts/exportar_modelo_ios.py``."""
    actual = json.loads(json.dumps(exportador.exportar_modelo(modelos, ejercicios, perfil, musculo_articulacion),
                                   ensure_ascii=False, allow_nan=False))
    assert actual == modelo_app


def test_la_app_recomienda_lo_mismo_que_la_web():
    """La copia de la app es idéntica a los datos de ``docs/index.html``, que
    ``tests/test_web.py`` compara con el ``Sustituidor`` del pipeline."""
    html = (RAIZ / "docs/index.html").read_text(encoding="utf-8")
    web = re.search(r'<script type="application/json" id="datos">(.*?)</script>', html, re.S).group(1)
    assert json.loads(RECOMENDACIONES.read_text(encoding="utf-8")) == json.loads(web.replace("<\\/", "</")), (
        "recomendaciones de la app desactualizadas: corre scripts/exportar_modelo_ios.py")


def test_casos_de_prueba_con_las_probabilidades_actuales(modelos, tabla_modelado):
    casos = json.loads(CASOS.read_text(encoding="utf-8"))["casos"]
    filas = tabla_modelado.set_index("ejercicio_id").loc[[c["ejercicio_id"] for c in casos]].reset_index()
    for objetivo, info in modelos.items():
        esperado = np.array([c["esperado"][objetivo] for c in casos])
        np.testing.assert_allclose(info["estimador"].predict_proba(filas), esperado, atol=1e-12)


def test_el_modelo_exportado_esta_completo(modelo_app, modelos):
    for objetivo, info in modelos.items():
        exportado = modelo_app["modelos"][objetivo]
        assert exportado["algoritmo"] == info["modelo"]
        assert exportado["umbral_abstencion"] == info["umbral"]
        prep = exportado["preprocesador"]
        n = (len(prep["nombre"]["vocabulario"]) + len(prep["instrucciones"]["vocabulario"])
             + sum(len(c["categorias"]) for c in prep["categoricas"]) + len(prep["numericas"]["columnas"]))
        assert n == prep["n_variables"]
    assert len(modelo_app["modelos"]["fuerza"]["bosque"]) == len(modelos["fuerza"]["estimador"].named_steps["clf"].estimators_)
    assert len(modelo_app["nombres_es"]) == 68


def test_app_e_importador_usan_las_mismas_columnas():
    """``TablaExportable`` (Swift) y ``COLUMNAS`` (Python) deben coincidir: si no,
    el importador rechazaría los CSV de la app."""
    fuente = SWIFT_EXPORTACION.read_text(encoding="utf-8")
    casos = dict(re.findall(r'case (\w+) = "(\w+)"', fuente))
    bloques = re.findall(r"case \.(\w+):\s*\[(.*?)\]", fuente, flags=re.S)
    swift = {casos[caso]: re.findall(r'"(\w+)"', columnas) for caso, columnas in bloques}
    assert swift == importador.COLUMNAS


def test_revisiones_suman_con_la_revision_manual_del_equipo():
    manual = pd.read_csv(RAIZ / "data/01_raw/sustitucion_revision_manual.csv")
    assert importador.COLUMNAS["revisiones_recomendaciones"][:9] == list(manual.columns)
    assert set(manual["veredicto"]) <= importador.VEREDICTOS


def _csv(tmp_path: Path, tabla: str, filas: list[list]) -> Path:
    pd.DataFrame(filas, columns=importador.COLUMNAS[tabla]).to_csv(tmp_path / f"{tabla}.csv", index=False)
    return tmp_path


def test_importa_sin_duplicar(tmp_path):
    carpeta = _csv(tmp_path, "revisiones_recomendaciones", [[
        "rodilla", "barbell full squat", "x", "Hip Thrust", "x", "True", "0.9", "bueno", "",
        "catalogo:0043", "catalogo:0043", "cualquiera", "2026-11-03T12:00:00Z"]])
    destino = tmp_path / "destino"
    assert importador.importar(carpeta, destino) == {"revisiones_recomendaciones": 1}
    assert importador.importar(carpeta, destino) == {"revisiones_recomendaciones": 0}


@pytest.mark.parametrize("tabla, fila, problema", [
    ("revisiones_recomendaciones",
     ["rodilla", "a", "x", "b", "x", "True", "0.9", "excelente", "", "catalogo:0043", "catalogo:0043", "cualquiera",
      "2026-11-03T12:00:00Z"], "veredicto"),
    ("correcciones_etiquetas",
     ["catalogo:0043", "a", "mecanica", "compound", "fuente", "", "complejo", "", "2026-11-03T12:00:00Z"], "etiqueta"),
    ("ejercicios_nuevos",
     ["press", "press", "", "peso libre", "pectoral mayor", "", "compound", "0.9", "compound", "push", "0.9", "push",
      "alta", "0.5", "", "False", "libre", "False", "2026-11-03T12:00:00Z"], "músculo"),
    ("entrenamiento_dolor", ["2026-11-03T12:00:00Z", "s", "rodilla", "11"], "dolor"),
])
def test_rechaza_datos_invalidos(tmp_path, tabla, fila, problema):
    carpeta = _csv(tmp_path, tabla, [fila])
    with pytest.raises(SystemExit):
        importador.importar(carpeta, tmp_path / "destino")
    assert not (tmp_path / "destino").exists()
    df = pd.read_csv(carpeta / f"{tabla}.csv")
    assert any(problema in p for p in importador.validar(tabla, df, importador._referencias()))

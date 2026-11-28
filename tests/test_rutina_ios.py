"""La app de iPhone arma la rutina con el mismo generador que la página.

`RutinasTests` (Swift, job `ios` del CI) compara la app con 60 rutinas que armó la web. Aquí se
revisa, sin Mac, que esas rutinas de referencia estén al día con web/app.html y que las tablas
copiadas a Swift (ejercicios conocidos, deportes, objetivos, plantillas, lesiones) sean las mismas.
"""
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[1]
WEB = (RAIZ / "web/app.html").read_text(encoding="utf-8")
SWIFT = (RAIZ / "ios/SustitutoSeguroCore/Sources/SustitutoSeguroCore/Rutinas.swift").read_text(encoding="utf-8")
CASOS = RAIZ / "ios/SustitutoSeguroCore/Tests/SustitutoSeguroCoreTests/Recursos/casos_rutina.json"


def _tramo(texto, desde, hasta):
    a = texto.index(desde)
    return texto[a:texto.index(hasta, a)]


def _curados_web():
    filas = re.findall(r"\[('[^']+'), ('[^']+'), '(\w+)', (\d), ('\w+'|null), (\d), (\d)\]",
                       _tramo(WEB, "const CURADOS = [", "];"))
    return [(i.strip("'"), es.strip("'"), g, int(n), None if eq == "null" else eq.strip("'"), imp == "1", iso == "1")
            for i, es, g, n, eq, imp, iso in filas]


def _curados_swift():
    filas = re.findall(r'\("([^"]+)", "([^"]+)", "(\w+)", (\d), (nil|"\w+"), (true|false), (true|false)\)',
                       _tramo(SWIFT, "public static let curados", "\n    ]\n"))
    return [(i, es, g, int(n), None if eq == "nil" else eq.strip('"'), imp == "true", iso == "true")
            for i, es, g, n, eq, imp, iso in filas]


def test_mismos_ejercicios_conocidos():
    web = _curados_web()
    assert len(web) == 68
    assert _curados_swift() == web


def test_mismos_textos_de_tablas():
    """Nombres de deportes, objetivos, tipos y lesiones, plantillas y prescripciones."""
    for patron in [r"\['(\w+)', '([^']+)', \[('[^\]]+'(?:, '[^\]]+')*)\]\]",      # LESIONES
                   r"(\w+): \['([^']+)', \{ series: (\d), reps: '([^']+)', descanso: (\d+) \}\]"]:  # OBJETIVOS
        for m in re.finditer(patron, WEB):
            for valor in m.groups()[1:]:
                for parte in re.findall(r"'([^']+)'", valor) or [valor]:
                    assert parte in SWIFT, f"Falta en Swift: {parte}"
    for titulo, grupos in re.findall(r"\['([^']+)', \[((?:'\w+', )+'\w+')\]\]", _tramo(WEB, "const PLANTILLAS", "};")):
        swift = "(\"%s\", [%s])" % (titulo, grupos.replace("'", '"'))
        assert swift in SWIFT, f"Plantilla distinta: {titulo}"
    for clave, nombre, complementos in re.findall(r"(\w+): \['([^']+)', (\[.*?\])\],?\n", _tramo(WEB, "const DEPORTES = {", "};")):
        assert f'("{clave}", "{nombre}"' in SWIFT
        for grupo, ids in re.findall(r"\['(\w+)', \[([^\]]+)\]\]", complementos):
            assert f'("{grupo}", [{ids.replace(chr(39), chr(34))}])' in SWIFT, f"{clave}: {grupo}"


def test_casos_de_referencia_al_dia(tmp_path):
    """Las 60 rutinas de referencia de los tests de Swift son las que arma hoy la página."""
    if shutil.which("node") is None:
        pytest.skip("Sin Node.js")
    salida = tmp_path / "casos.json"
    subprocess.run(["node", str(RAIZ / "scripts/casos_rutina_ios.mjs"), str(salida)], check=True, capture_output=True)
    assert json.loads(salida.read_text()) == json.loads(CASOS.read_text()), \
        "La página cambió: corre «node scripts/casos_rutina_ios.mjs» y ajusta Rutinas.swift"

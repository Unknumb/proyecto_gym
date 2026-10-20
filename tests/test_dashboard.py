"""Test de humo del dashboard: cada vista se dibuja sin errores con las salidas
versionadas, y la demo de sustitución responde como en el notebook 03 §11."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = str(Path(__file__).resolve().parents[1] / "app/dashboard.py")
VISTAS = ["1 · KPIs", "2 · Datos", "3 · Supervisado", "4 · No supervisado",
          "5 · Sustitución en vivo", "6 · Sesgo y cierre"]


@pytest.fixture(scope="module")
def app() -> AppTest:
    return AppTest.from_file(APP, default_timeout=120)


@pytest.mark.parametrize("vista", VISTAS)
def test_cada_vista_se_dibuja_sin_errores(app, vista):
    app.run()
    app.sidebar.radio[0].set_value(vista).run()
    assert not app.exception, app.exception


def test_demo_sentadilla_con_rodilla_propone_hip_thrust(app):
    app.run()
    app.sidebar.radio[0].set_value("5 · Sustitución en vivo").run()
    tabla = app.dataframe[0].value
    assert "hip thrust" in tabla["Alternativa"].iat[0].lower()


def test_demo_extension_de_piernas_se_abstiene(app):
    app.run()
    app.sidebar.radio[0].set_value("5 · Sustitución en vivo").run()
    app.button[2].click().run()   # «Extensión de piernas · rodilla»
    assert any("se abstiene" in w.value for w in app.warning)

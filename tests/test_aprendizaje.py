"""Tests del aprendizaje con datos de usuarios (README §18).

Verifican que, sin aportes, todo quede como en la EV2 y la EV3; que los aportes
pasen por consenso, hereden el grupo de su gemelo (sin fuga) y lleguen al perfil
con origen ``usuarios``; que el control de calidad descarte aportes que empeoran
el modelo; y que la simulación muestre aprendizaje con usuarios honestos.
"""

from __future__ import annotations

import json
import sys

import pandas as pd
import pytest
from sklearn.base import clone

from conftest import RAIZ
from gym_exercises.pipelines.modelado_supervisado.nodes import (
    OBJETIVOS, consolidar_aportes, evaluar_aprendizaje, filas_entrenamiento, incorporar_aportes,
    predecir_etiquetas)

sys.path.insert(0, str(RAIZ / "scripts"))
import importar_aportes as importador  # noqa: E402

COLUMNAS = importador.COLUMNAS["correcciones_etiquetas"]


@pytest.fixture(scope="module")
def tabla_base() -> pd.DataFrame:
    return pd.read_parquet(RAIZ / "data/05_model_input/tabla_base.parquet")


@pytest.fixture(scope="module")
def ms(parametros) -> dict:
    return parametros["modelado_supervisado"]


def _correcciones(*filas) -> pd.DataFrame:
    """Filas (ejercicio_id, objetivo, etiqueta_corregida) con las columnas del importador."""
    return pd.DataFrame([{**dict.fromkeys(COLUMNAS), "ejercicio_id": e, "objetivo": o,
                          "etiqueta_corregida": c, "fecha": "2026-11-24"} for e, o, c in filas],
                        columns=COLUMNAS)


def test_sin_aportes_la_tabla_y_los_modelos_no_cambian(tabla_base, tabla_modelado):
    """Con data/01_raw/aportes_usuarios vacío, la tabla publicada es la base."""
    aportes = pd.read_csv(RAIZ / "data/01_raw/aportes_usuarios/correcciones_etiquetas.csv")
    assert list(aportes.columns) == COLUMNAS
    if aportes.empty:
        marcas = [f"{o}_aporte" for o in OBJETIVOS]
        assert not tabla_modelado[marcas].any().any()
        assert tabla_modelado.drop(columns=marcas).equals(tabla_base)
        informe = pd.read_csv(RAIZ / "data/08_reporting/aprendizaje_usuarios_informe.csv")
        assert set(informe["decision"]) == {"sin aportes"}


def test_consenso(tabla_base, ms):
    config = {**ms["aportes_usuarios"], "votos_minimos": 2, "acuerdo_minimo": 0.6}
    c = consolidar_aportes(_correcciones(
        ("catalogo:0001", "mecanica", "isolation"), ("catalogo:0001", "mecanica", "isolation"),
        ("catalogo:0001", "mecanica", "compound"),           # 2 de 3 → aceptada
        ("catalogo:0003", "mecanica", "isolation"),          # 1 voto → faltan votos
        ("catalogo:0006", "fuerza", "push"), ("catalogo:0006", "fuerza", "pull"),  # empate
        ("catalogo:9999", "mecanica", "compound"), ("catalogo:9999", "mecanica", "compound"),
        ("catalogo:0001", "fuerza", "volar"), ("catalogo:0001", "fuerza", "volar"),
        ("catalogo:0001", "dificultad", "alta"), ("catalogo:0001", "dificultad", "alta"),
    ), tabla_base, config).set_index(["ejercicio_id", "objetivo"])
    assert c.at[("catalogo:0001", "mecanica"), "estado"] == "aceptada"
    assert c.at[("catalogo:0001", "mecanica"), "etiqueta"] == "isolation"
    assert c.at[("catalogo:0003", "mecanica"), "estado"] == "faltan votos"
    assert c.at[("catalogo:0006", "fuerza"), "estado"] == "sin acuerdo"
    assert c.at[("catalogo:9999", "mecanica"), "estado"] == "ejercicio desconocido"
    assert c.at[("catalogo:0001", "fuerza"), "estado"] == "etiqueta desconocida"
    assert c.at[("catalogo:0001", "dificultad"), "estado"] == "objetivo desconocido"


def test_los_aportes_entran_sin_fuga(tabla_base, ms):
    """Un ejercicio del catálogo hereda el grupo y la partición de su gemelo; uno de
    free-exercise-db cambia su etiqueta. Ningún grupo queda en train y en test."""
    fedb = tabla_base[tabla_base["rol"] == "entrenamiento"].set_index("ejercicio_id")
    gemelo = tabla_base.set_index("ejercicio_id").at["catalogo:0006", "gemelo_fedb"]
    t, consolidado = incorporar_aportes(tabla_base, _correcciones(
        ("catalogo:0006", "mecanica", "compound"),
        ("free_exercise_db:Ab_Crunch_Machine", "mecanica", "compound"),
        ("wger:31", "fuerza", "static"),
    ), ms)
    assert (consolidado["estado"] == "aceptada").all()
    fila = t.set_index("ejercicio_id")
    assert fila.at["catalogo:0006", "mecanica"] == "compound" and fila.at["catalogo:0006", "mecanica_aporte"]
    assert fila.at["catalogo:0006", "grupo"] == fedb.at[gemelo, "grupo"]
    assert fila.at["catalogo:0006", "particion"] == fedb.at[gemelo, "particion"]
    assert fila.at["free_exercise_db:Ab_Crunch_Machine", "mecanica"] == "compound"
    assert fila.at["wger:31", "particion"] in {"train", "test"} and pd.notna(fila.at["wger:31", "grupo"])
    # El aporte de mecánica no vuelve a wger:31 un ejemplo de mecánica
    assert "wger:31" not in set(filas_entrenamiento(t, "mecanica")["ejercicio_id"])
    assert "wger:31" in set(filas_entrenamiento(t, "fuerza")["ejercicio_id"])
    for objetivo in OBJETIVOS:
        datos = filas_entrenamiento(t, objetivo)
        assert datos.groupby("grupo")["particion"].nunique().max() == 1


def test_el_perfil_marca_las_etiquetas_de_usuarios(tabla_base, ejercicios, modelos, ms):
    t, _ = incorporar_aportes(tabla_base, _correcciones(("catalogo:0006", "mecanica", "compound")), ms)
    perfil = predecir_etiquetas(t, ejercicios, modelos).set_index("ejercicio_id")
    assert perfil.at["catalogo:0006", "mecanica"] == "compound"
    assert perfil.at["catalogo:0006", "mecanica_origen"] == "usuarios"
    assert (perfil["mecanica_origen"] == "usuarios").sum() == 1


def test_el_control_descarta_aportes_que_empeoran(tabla_base, modelos, ms):
    """Cien etiquetas de mecánica invertidas a propósito: se publica el modelo sin ellas."""
    pool = tabla_base[(tabla_base["rol"] == "prediccion") & tabla_base["mecanica_referencia"].notna()
                      & tabla_base["gemelo_fedb"].isin(
                          tabla_base.loc[tabla_base["particion"] == "train", "ejercicio_id"])].head(100)
    invertir = {"compound": "isolation", "isolation": "compound"}
    correcciones = _correcciones(*[(e, "mecanica", invertir[r]) for e, r in
                                   zip(pool["ejercicio_id"], pool["mecanica_referencia"])])
    t, _ = incorporar_aportes(tabla_base, correcciones, ms)
    datos = filas_entrenamiento(t, "mecanica")
    train = datos[datos["particion"] == "train"]
    con = clone(modelos["mecanica"]["modelo_train"]).fit(train, train["mecanica"])
    candidatos = {**modelos, "mecanica": {**modelos["mecanica"], "modelo_train": con,
                                          "estimador": clone(con).fit(datos, datos["mecanica"])}}
    informe, finales, tabla_final = evaluar_aprendizaje(tabla_base, correcciones, t, candidatos, ms)
    fila = informe.set_index("objetivo").loc["mecanica"]
    assert fila["decision"].startswith("descartar")
    assert fila["f1_externa_con_aportes"] < fila["f1_externa_sin_aportes"] - ms["aportes_usuarios"]["tolerancia_f1"]
    assert not tabla_final["mecanica_aporte"].any()
    assert finales["mecanica"]["modelo_train"] is not con
    assert finales["fuerza"] is modelos["fuerza"]


def test_la_simulacion_muestra_aprendizaje_y_rechaza_maliciosos():
    s = pd.read_csv(RAIZ / "data/08_reporting/aprendizaje_simulacion.csv")
    media = s.groupby(["objetivo", "escenario", "n_aportes"])["f1_externa"].mean()
    # Usuarios honestos: mecánica y fuerza mejoran en ejercicios nuevos del catálogo
    for objetivo, minimo in [("mecanica", 0.05), ("fuerza", 0.02)]:
        assert media[objetivo, "honestos", 200] - media[objetivo, "honestos", 0] >= minimo
    # Usuarios maliciosos: desde 25 aportes el control rechaza al menos el 95 % de
    # las corridas, y las que deja pasar no empeoran más que la tolerancia
    base = s[s["n_aportes"] == 0].drop_duplicates("objetivo").set_index("objetivo")
    maliciosos = s[(s["escenario"] == "maliciosos") & (s["n_aportes"] >= 25)]
    assert maliciosos["adoptado"].mean() <= 0.05
    pasaron = maliciosos[maliciosos["adoptado"]]
    assert (pasaron["f1_externa"] >= pasaron["objetivo"].map(base["f1_externa"]) - 0.02).all()


def test_importa_aportes_de_la_pagina_web(tmp_path, perfil):
    fila = perfil.iloc[0]
    archivo = tmp_path / "sustituto-seguro-aportes-2026-11-24.json"
    archivo.write_text(json.dumps({"formato": importador.FORMATO_WEB, "version": 1, "tablas": {
        "correcciones_etiquetas": [dict(zip(COLUMNAS, [
            fila["ejercicio_id"], "x", "mecanica", fila["mecanica"], fila["mecanica_origen"],
            fila["mecanica_confianza"], "isolation", "", "2026-11-24T10:00:00.000Z"]))],
        "revisiones_recomendaciones": []}}), encoding="utf-8")
    destino = tmp_path / "destino"
    assert importador.importar(archivo, destino) == {"correcciones_etiquetas": 1,
                                                    "revisiones_recomendaciones": 0}
    assert importador.importar(archivo, destino)["correcciones_etiquetas"] == 0
    otro = tmp_path / "otro.json"
    otro.write_text(json.dumps({"formato": "otra-cosa"}), encoding="utf-8")
    with pytest.raises(SystemExit):
        importador.importar(otro, destino)

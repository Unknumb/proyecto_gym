"""Tests de modelo: métrica mínima frente al azar, invariancia, dirección y las
reglas de seguridad del sustituidor."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gym_exercises.pipelines.grafo_biomecanico.nodes import Sustituidor
from gym_exercises.pipelines.modelado_supervisado.nodes import OBJETIVOS


@pytest.mark.parametrize("objetivo", OBJETIVOS)
def test_el_ganador_supera_claramente_al_azar(comparacion, modelos, objetivo):
    filas = comparacion[(comparacion["objetivo"] == objetivo) & (comparacion["conjunto"] == "test")]
    ganador = filas.loc[filas["modelo"] == modelos[objetivo]["modelo"], "f1_macro"].item()
    azar = filas.loc[filas["modelo"] == "azar_estratificado", "f1_macro"].item()
    assert ganador >= azar + 0.2


@pytest.mark.parametrize("objetivo, minimo", [("mecanica", 0.85), ("fuerza", 0.85)])
def test_kpi_f1_en_test(comparacion, modelos, objetivo, minimo):
    fila = comparacion[(comparacion["objetivo"] == objetivo) & (comparacion["conjunto"] == "test")
                       & (comparacion["modelo"] == modelos[objetivo]["modelo"])]
    assert fila["f1_macro"].item() >= minimo


def _test(tabla_modelado, objetivo):
    t = tabla_modelado
    return t[(t["particion"] == "test") & t[objetivo].notna()]


def test_invariancia_al_cambiar_barra_por_mancuernas(tabla_modelado, modelos):
    """Sentadilla con barra o con mancuernas es el mismo movimiento: cambiar solo el
    equipamiento entre pesos libres casi nunca debe cambiar la mecánica."""
    datos = _test(tabla_modelado, "mecanica")
    datos = datos[datos["familia_equipamiento"] == "peso libre"]
    modelo = modelos["mecanica"]["modelo_train"]
    original = modelo.predict(datos)
    cambiado = datos.assign(
        nombre_norm=datos["nombre_norm"].str.replace(r"\bbarbell\b", "dumbbell", regex=True),
        instrucciones_en=datos["instrucciones_en"].str.replace(r"\bbarbell\b", "dumbbell",
                                                               regex=True, case=False))
    assert (original == modelo.predict(cambiado)).mean() >= 0.95


def test_direccion_mas_articulaciones_mas_compuesto(tabla_modelado, modelos):
    """Sumar articulaciones implicadas debe subir, en promedio, la probabilidad de
    compuesto: es la definición biomecánica."""
    datos = _test(tabla_modelado, "mecanica")
    modelo = modelos["mecanica"]["modelo_train"]
    k = list(modelo.classes_).index("compound")
    mas = datos.copy()
    for articulacion in ["cadera", "rodilla", "hombro"]:
        mas[f"articulacion__{articulacion}"] = 1
    mas["n_articulaciones_total"] = mas[[c for c in mas.columns if c.startswith("articulacion__")]].sum(axis=1)
    assert modelo.predict_proba(mas)[:, k].mean() > modelo.predict_proba(datos)[:, k].mean()


# ── Sustituidor: reglas de seguridad ─────────────────────────────────────────
@pytest.fixture(scope="module")
def sustituidor(familias, embeddings, musculo_articulacion, parametros):
    return Sustituidor(familias, embeddings, musculo_articulacion, parametros["grafo_biomecanico"])


def test_ninguna_alternativa_carga_la_articulacion_vetada(sustituidor, familias):
    muestra = familias.sample(150, random_state=0)
    revisados = 0
    for _, fila in muestra.iterrows():
        for articulacion in fila["articulaciones_carga"]:
            alternativas, _ = sustituidor.recomendar(fila["ejercicio_id"], articulacion)
            for carga in alternativas["articulaciones_carga"]:
                assert articulacion not in set(carga)
            revisados += len(alternativas)
    assert revisados > 0


def test_sin_sustituto_anatomico_se_abstiene(sustituidor, familias):
    extension = familias[familias["nombre"].str.lower() == "lever leg extension"]
    assert len(extension) == 1
    alternativas, motivo = sustituidor.recomendar(extension["ejercicio_id"].item(), "rodilla")
    assert alternativas.empty and motivo == "sin_sustituto_anatomico"


def test_demo_sentadilla_con_rodilla_vetada(sustituidor, familias):
    sentadilla = familias[familias["nombre"] == "barbell full squat"]
    alternativas, motivo = sustituidor.recomendar(sentadilla["ejercicio_id"].item(), "rodilla")
    assert motivo is None
    assert alternativas["nombre"].iloc[0] == "Hip Thrust"


def test_carga_por_patron_cubre_las_bisagras_con_carga(familias):
    """Todo peso muerto, clean o remo inclinado carga la columna lumbar, aunque su
    ficha no anote el músculo lumbar (README §13.2)."""
    nombres = familias["nombre"].str.lower()
    bisagras = familias[nombres.str.contains(r"deadlift|\bcleans?\b|bent[- ]?over.*\brows?\b")]
    assert len(bisagras) > 20
    assert bisagras["articulaciones_carga"].map(lambda a: "columna (lumbar)" in set(a)).all()


def test_la_carga_por_patron_solo_agrega(familias):
    agregadas = familias["articulaciones_carga_patron"].map(set)
    carga = familias["articulaciones_carga"].map(set)
    assert all(a <= c for a, c in zip(agregadas, carga))


def test_peso_muerto_rumano_con_lumbar_vetada_no_recibe_bisagras(sustituidor, familias):
    """El caso que encontró la revisión manual: antes se proponía Kettlebell deadlifts."""
    rumano = familias[familias["nombre"] == "Romanian Deadlift from Deficit"]
    alternativas, _ = sustituidor.recomendar(rumano["ejercicio_id"].item(), "columna (lumbar)")
    assert not alternativas["nombre"].str.lower().str.contains(r"deadlift|clean|rdl|swing").any()


# ── Casos que la revisión manual marcó como inseguros (README §13.5) ──────────
def test_clean_con_hombro_vetado_no_propone_otro_clean(sustituidor, familias):
    """En un clean la barra se recibe en rack: carga el hombro."""
    clean = familias[familias["nombre"] == "Clean"]
    alternativas, _ = sustituidor.recomendar(clean["ejercicio_id"].item(), "hombro")
    assert not alternativas["nombre"].str.lower().str.contains(r"\bcleans?\b").any()


def test_vetar_la_lumbar_veta_la_toracico_lumbar(sustituidor, familias):
    """Con la lumbar lesionada tampoco sirve una alternativa que cargue la columna
    torácico-lumbar (Cable Woodchoppers → Upper External Oblique)."""
    leñador = familias[familias["nombre"] == "Cable Woodchoppers"]
    alternativas, motivo = sustituidor.recomendar(leñador["ejercicio_id"].item(), "columna (lumbar)")
    assert alternativas.empty and motivo == "sin_sustituto_anatomico"
    muestra = familias[familias["articulaciones_carga"].map(lambda a: "columna (lumbar)" in set(a))].head(60)
    for ejercicio_id in muestra["ejercicio_id"]:
        alternativas, _ = sustituidor.recomendar(ejercicio_id, "columna (lumbar)")
        for carga in alternativas["articulaciones_carga"]:
            assert not {"columna (lumbar)", "columna (torácico-lumbar)"} & set(carga)

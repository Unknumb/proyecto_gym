"""Tests de datos: esquema, unicidad, rangos y nulos de las tablas versionadas."""

from __future__ import annotations

ETIQUETAS = {"mecanica": {"compound", "isolation"},
             "fuerza": {"push", "pull", "static"},
             "complejidad_tecnica": {"baja", "alta"}}


def test_ejercicios_tiene_una_fila_por_ejercicio(ejercicios):
    assert len(ejercicios) == 2592
    assert ejercicios["ejercicio_id"].is_unique
    assert ejercicios["ejercicio_id"].notna().all()


def test_id_del_catalogo_conserva_los_ceros_a_la_izquierda(ejercicios):
    """El bug de la EV1: pandas convertía "0001" en 1 al cargar (README §3.3)."""
    catalogo = ejercicios[ejercicios["fuente"] == "catalogo"]
    assert len(catalogo) == 1324
    assert catalogo["id_fuente"].str.fullmatch(r"\d{4}").all()
    assert "0001" in set(catalogo["id_fuente"])


def test_fuentes_y_licencias(ejercicios):
    assert set(ejercicios["fuente"]) == {"catalogo", "free_exercise_db", "wger"}
    assert ejercicios["licencia"].notna().all()


def test_aristas_ejercicio_musculo(ejercicios, ejercicio_musculo):
    assert set(ejercicio_musculo["rol"]) == {"agonista", "sinergista"}
    assert ejercicio_musculo["ejercicio_id"].isin(ejercicios["ejercicio_id"]).all()
    assert not ejercicio_musculo.duplicated(["ejercicio_id", "musculo", "rol"]).any()


def test_mapeo_musculo_articulacion_sin_nulos(musculo_articulacion):
    assert musculo_articulacion[["musculo", "articulacion"]].notna().all().all()
    assert not musculo_articulacion.duplicated(["musculo", "articulacion"]).any()


def test_perfil_de_exigencia(perfil, ejercicios):
    assert perfil["ejercicio_id"].is_unique
    assert set(perfil["ejercicio_id"]) == set(ejercicios["ejercicio_id"])
    for objetivo, clases in ETIQUETAS.items():
        assert set(perfil[objetivo].dropna()) <= clases
        assert perfil[f"{objetivo}_confianza"].between(0, 1).all()
        assert set(perfil[f"{objetivo}_origen"].dropna()) <= {"fuente", "transferida", "modelo"}
    assert set(perfil["estabilidad"]) <= {"apoyado", "libre", "inestable"}


def test_particion_sin_grupos_compartidos(tabla_modelado):
    """Ningún grupo de variantes queda repartido entre train y test (sin fuga)."""
    entrenamiento = tabla_modelado[tabla_modelado["rol"] == "entrenamiento"]
    particiones_por_grupo = entrenamiento.groupby("grupo")["particion"].nunique()
    assert (particiones_por_grupo == 1).all()


def test_datos_crudos_sin_cambios():
    """Los datos crudos son inmutables: deben coincidir con su huella registrada."""
    import subprocess
    import sys

    from conftest import RAIZ

    resultado = subprocess.run([sys.executable, str(RAIZ / "scripts/registrar_huellas.py"), "--verificar"],
                               capture_output=True, text=True)
    assert resultado.returncode == 0, resultado.stdout

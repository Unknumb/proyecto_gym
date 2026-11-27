"""Tests de la API de cuentas (``api/``, README §20).

Corren sobre SQLite. Si la variable ``DATABASE_URL_TEST`` apunta a un PostgreSQL (el CI
levanta uno), los mismos tests corren también contra él: el SQL es el mismo.
"""
from __future__ import annotations

import json
import os

import pytest

from api.servidor import Catalogo, crear_app, ids_de_la_pagina

IDS = ids_de_la_pagina()
EJ, OTRO = IDS[0], IDS[1]
MOTORES = ["sqlite"] + (["postgresql"] if os.environ.get("DATABASE_URL_TEST") else [])


@pytest.fixture(params=MOTORES)
def cliente(request, tmp_path):
    if request.param == "sqlite":
        url = f"sqlite:///{tmp_path / 'cuentas.db'}"
    else:
        url = os.environ["DATABASE_URL_TEST"]
        from api.basedatos import BaseDatos
        with BaseDatos(url).conectar() as cx:  # base limpia para cada test
            for tabla in ["revisiones", "correcciones", "pesos", "sesiones", "rutinas", "perfiles", "usuarios"]:
                cx.ejecutar(f"DROP TABLE IF EXISTS {tabla}")
    app = crear_app(url, secreto="secreto-de-prueba", catalogo=Catalogo())
    return app.test_client()


def registrar(c, email="ana@correo.cl", nombre="Ana", contrasena="clave-segura-1"):
    r = c.post("/api/registro", json={"email": email, "nombre": nombre, "contrasena": contrasena})
    assert r.status_code == 201, r.json
    return {"Authorization": "Bearer " + r.json["token"]}


PERFIL = {"edad": 25, "peso": 72.5, "altura": 175, "deporte": "futbol", "nivel": "intermedio", "objetivo": "musculo",
          "dias": 4, "minutos": 60, "tipo": "auto", "equipo": ["pesas", "gym"], "lesiones": ["rodilla"]}
RUTINA = {"creada": "2026-11-20", "objetivo": "musculo", "tipo": "tp", "porqueTipo": "Con 4 días…", "perfil": PERFIL,
          "avisos": [["red", "Lesiones excluidas:", "ningún ejercicio carga la rodilla."]], "huecos": [],
          "dias": [{"titulo": "Torso A", "ejercicios": [{"id": EJ, "grupo": "pecho", "series": 3, "reps": "8–12",
                                                         "descanso": 90, "alternativas": [OTRO]}]}]}


def sesion(uid="s1", fecha="2026-11-21", **extra):
    return {"uid": uid, "fecha": fecha, "dia": 0, "titulo": "Torso A", "minutos": 50, "esfuerzo": 7, "notas": "bien",
            "dolor": {"rodilla": 2}, "ejercicios": [{"id": EJ, "series": [{"kg": 40, "reps": 10}, {"kg": None, "reps": 8}]}],
            **extra}


def test_salud_y_pagina(cliente):
    r = cliente.get("/api/salud")
    assert r.status_code == 200 and r.json["ok"] and r.json["cuentas"]
    assert cliente.get("/api/no-existe").status_code == 404


def test_registro_e_inicio_de_sesion(cliente):
    h = registrar(cliente, email="  Ana@Correo.CL ")
    assert cliente.get("/api/cuenta", headers=h).json["usuario"]["email"] == "ana@correo.cl"
    # El mismo correo no se registra dos veces, sin importar mayúsculas
    r = cliente.post("/api/registro", json={"email": "ANA@correo.cl", "nombre": "Otra", "contrasena": "otra-clave-1"})
    assert r.status_code == 409
    assert cliente.post("/api/sesion", json={"email": "ana@correo.cl", "contrasena": "mala"}).status_code == 401
    r = cliente.post("/api/sesion", json={"email": "ANA@correo.cl", "contrasena": "clave-segura-1"})
    assert r.status_code == 200 and r.json["usuario"]["nombre"] == "Ana"


@pytest.mark.parametrize("cuerpo, mensaje", [
    ({"email": "no-es-correo", "nombre": "A", "contrasena": "12345678"}, "correo"),
    ({"email": "a@b.cl", "nombre": "", "contrasena": "12345678"}, "nombre"),
    ({"email": "a@b.cl", "nombre": "A", "contrasena": "corta"}, "8 caracteres"),
])
def test_registro_rechaza_datos_invalidos(cliente, cuerpo, mensaje):
    r = cliente.post("/api/registro", json=cuerpo)
    assert r.status_code == 400 and mensaje in r.json["error"]


def test_la_contrasena_no_se_guarda_en_claro(cliente):
    registrar(cliente)
    with cliente.application.extensions["sustituto_bd"].conectar() as cx:
        fila = cx.una("SELECT hash_contrasena FROM usuarios")
    assert "clave-segura-1" not in fila["hash_contrasena"] and fila["hash_contrasena"].startswith(("scrypt:", "pbkdf2:"))


def test_sin_sesion_o_con_token_falso_no_hay_datos(cliente):
    assert cliente.get("/api/datos").status_code == 401
    assert cliente.get("/api/datos", headers={"Authorization": "Bearer falso"}).status_code == 401


def test_demasiados_intentos_fallidos(cliente):
    registrar(cliente)
    for _ in range(10):
        cliente.post("/api/sesion", json={"email": "ana@correo.cl", "contrasena": "mala"})
    r = cliente.post("/api/sesion", json={"email": "ana@correo.cl", "contrasena": "clave-segura-1"})
    assert r.status_code == 429


def test_perfil_rutina_sesiones_y_pesos(cliente):
    h = registrar(cliente)
    assert cliente.put("/api/datos/perfil", json=PERFIL, headers=h).status_code == 200
    assert cliente.put("/api/datos/rutina", json=RUTINA, headers=h).status_code == 200
    assert cliente.post("/api/datos/sesiones", json=sesion(), headers=h).status_code == 201
    # La misma sesión (mismo uid) se actualiza, no se duplica
    assert cliente.post("/api/datos/sesiones", json=sesion(notas="editada"), headers=h).status_code == 201
    assert cliente.put("/api/datos/pesos", json={"fecha": "2026-11-21", "kg": 72.5}, headers=h).status_code == 200
    d = cliente.get("/api/datos", headers=h).json
    assert d["perfil"] == PERFIL
    assert d["rutina"]["dias"][0]["ejercicios"][0]["id"] == EJ
    assert len(d["sesiones"]) == 1 and d["sesiones"][0]["notas"] == "editada" and d["sesiones"][0]["dolor"] == {"rodilla": 2}
    assert d["pesos"] == [{"fecha": "2026-11-21", "kg": 72.5}]
    assert cliente.delete("/api/datos/sesiones/s1", headers=h).status_code == 200
    assert cliente.get("/api/datos", headers=h).json["sesiones"] == []


@pytest.mark.parametrize("ruta, cuerpo", [
    ("/api/datos/perfil", {**PERFIL, "edad": 7}),
    ("/api/datos/perfil", {**PERFIL, "lesiones": ["oreja"]}),
    ("/api/datos/perfil", {**PERFIL, "minutos": 50}),
    ("/api/datos/rutina", {**RUTINA, "dias": [{"titulo": "x", "ejercicios": [{"id": "inventado:1"}]}]}),
    ("/api/datos/rutina", {**RUTINA, "dias": []}),
    ("/api/datos/sesiones", sesion(esfuerzo=11)),
    ("/api/datos/sesiones", sesion(fecha="21/11/2026")),
    ("/api/datos/sesiones", sesion(dolor={"rodilla": 15})),
])
def test_datos_invalidos_se_rechazan(cliente, ruta, cuerpo):
    h = registrar(cliente)
    metodo = cliente.post if ruta.endswith("sesiones") else cliente.put
    assert metodo(ruta, json=cuerpo, headers=h).status_code == 400


def test_cada_cuenta_ve_solo_sus_datos(cliente):
    ana = registrar(cliente)
    luis = registrar(cliente, email="luis@correo.cl", nombre="Luis")
    cliente.post("/api/datos/sesiones", json=sesion(), headers=ana)
    assert cliente.get("/api/datos", headers=luis).json["sesiones"] == []
    # Luis no puede borrar la sesión de Ana aunque conozca su uid
    cliente.delete("/api/datos/sesiones/s1", headers=luis)
    assert len(cliente.get("/api/datos", headers=ana).json["sesiones"]) == 1


def test_sincronizar_sube_lo_local_sin_pisar_la_cuenta(cliente):
    h = registrar(cliente)
    cliente.put("/api/datos/perfil", json=PERFIL, headers=h)
    cliente.post("/api/datos/sesiones", json=sesion("s1", notas="de la cuenta"), headers=h)
    local = {"perfil": {**PERFIL, "edad": 40}, "rutina": RUTINA,
             "sesiones": [sesion("s1", notas="local"), sesion("s2", "2026-11-22")],
             "pesos": [{"fecha": "2026-11-22", "kg": 71}],
             "correcciones": [{"ejercicio_id": EJ, "objetivo": "mecanica", "etiqueta": "compound"}]}
    d = cliente.post("/api/sincronizar", json=local, headers=h).json
    assert d["perfil"]["edad"] == 25                      # la cuenta ya tenía perfil: gana la cuenta
    assert d["rutina"] is not None                         # no tenía rutina: se toma la local
    assert [s["uid"] for s in d["sesiones"]] == ["s1", "s2"]
    assert d["sesiones"][0]["notas"] == "de la cuenta"     # una sesión que ya existe no se pisa
    assert d["pesos"] == [{"fecha": "2026-11-22", "kg": 71.0}] and len(d["correcciones"]) == 1
    # Sincronizar dos veces no duplica nada
    d2 = cliente.post("/api/sincronizar", json=local, headers=h).json
    assert len(d2["sesiones"]) == 2 and len(d2["correcciones"]) == 1


def test_aportes_un_voto_por_persona(cliente):
    h = registrar(cliente)
    voto = {"ejercicio_id": EJ, "objetivo": "fuerza", "etiqueta": "push", "etiqueta_previa": "pull"}
    cliente.post("/api/datos/aportes", json={"correcciones": [voto]}, headers=h)
    cliente.post("/api/datos/aportes", json={"correcciones": [{**voto, "etiqueta": "static"}],
                                              "revisiones": [{"original_id": EJ, "alternativa_id": OTRO, "articulacion": "rodilla",
                                                              "perfil": "cualquiera", "veredicto": "bueno", "similitud": 0.8}]},
                 headers=h)
    d = cliente.get("/api/datos", headers=h).json
    assert [c["etiqueta"] for c in d["correcciones"]] == ["static"] and d["revisiones"][0]["veredicto"] == "bueno"


def test_borrar_la_cuenta_borra_todos_sus_datos(cliente):
    h = registrar(cliente)
    cliente.put("/api/datos/perfil", json=PERFIL, headers=h)
    cliente.post("/api/datos/sesiones", json=sesion(), headers=h)
    assert cliente.delete("/api/cuenta", json={"contrasena": "mala"}, headers=h).status_code == 403
    assert cliente.delete("/api/cuenta", json={"contrasena": "clave-segura-1"}, headers=h).status_code == 200
    assert cliente.get("/api/datos", headers=h).status_code == 401
    with cliente.application.extensions["sustituto_bd"].conectar() as cx:
        for tabla in ["usuarios", "perfiles", "sesiones"]:
            assert cx.una(f"SELECT COUNT(*) AS n FROM {tabla}")["n"] == 0
    # El correo queda libre para registrarse otra vez
    registrar(cliente)


def test_cuerpo_demasiado_grande(cliente):
    h = registrar(cliente)
    r = cliente.post("/api/sincronizar", data="x" * (3 * 1024 * 1024), headers={**h, "Content-Type": "application/json"})
    assert r.status_code == 413


def test_los_aportes_de_la_base_salen_en_el_formato_del_importador(cliente):
    """Lo que las personas aportan con su cuenta vuelve al reentrenamiento (§18) sin sus datos."""
    import importlib.util
    from pathlib import Path
    raiz = Path(__file__).resolve().parents[1]

    def modulo(nombre):
        spec = importlib.util.spec_from_file_location(nombre, raiz / f"scripts/{nombre}.py")
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        return m

    exportador, importador = modulo("exportar_aportes_bd"), modulo("importar_aportes")
    h = registrar(cliente)
    cliente.post("/api/datos/aportes", headers=h, json={
        "correcciones": [{"ejercicio_id": EJ, "objetivo": "mecanica", "etiqueta": "compound", "etiqueta_previa": "isolation",
                          "origen_previo": "modelo", "confianza_previa": 0.7, "fecha": "2026-11-27T10:00:00Z"}],
        "revisiones": [{"original_id": EJ, "alternativa_id": OTRO, "articulacion": "rodilla", "perfil": "cualquiera",
                        "veredicto": "bueno", "similitud": 0.81, "misma_familia": True, "fecha": "2026-11-27T10:00:00Z"}]})
    bd = cliente.application.extensions["sustituto_bd"]
    url = bd.url if bd.motor == "postgresql" else f"sqlite:///{bd.ruta}"
    archivo = exportador.exportar(url, exportador.datos_pagina())
    texto = json.dumps(archivo)
    assert "ana@correo.cl" not in texto and "Ana" not in texto
    import pandas as pd
    for tabla, filas in archivo["tablas"].items():
        df = pd.DataFrame(filas)
        assert list(df.columns) == importador.COLUMNAS[tabla]
        assert importador.validar(tabla, df, importador._referencias()) == []

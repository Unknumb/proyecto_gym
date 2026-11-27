"""API de cuentas de Sustituto Seguro (README §20).

Las personas se registran con correo y contraseña y guardan en la base de datos su perfil,
su rutina, sus sesiones de entrenamiento, su peso y sus aportes al modelo. La misma cuenta
sirve en la página web y en la app de iPhone. El servidor también entrega la página
(``docs/index.html``), así la web y la API comparten origen y no hace falta CORS.

Desarrollo, desde la raíz del repositorio (SQLite, sin instalar nada más que Flask)::

    python -m api.servidor                  # http://localhost:8000

Producción (AWS, README §20.4)::

    DATABASE_URL=postgresql://… SECRETO=… gunicorn -w 2 -b 127.0.0.1:8000 "api.servidor:crear_app()"

Formato de los datos: los ejercicios se identifican por su ``ejercicio_id``
(``catalogo:0025``), no por su posición en la página, para que la web y la app hablen
del mismo ejercicio y los datos sigan valiendo si se regeneran las recomendaciones.
"""
from __future__ import annotations

import json
import logging
import os
import re
import secrets
import time
from collections import defaultdict, deque
from functools import wraps
from pathlib import Path

from flask import Flask, g, jsonify, request, send_file
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer
from werkzeug.security import check_password_hash, generate_password_hash

from api.basedatos import BaseDatos, a_json, ahora, de_json

RAIZ = Path(__file__).resolve().parents[1]
PAGINA = RAIZ / "docs/index.html"
VERSION = "1.0"

#: Un inicio de sesión dura 30 días; después hay que volver a entrar.
DURACION_SESION = 30 * 24 * 3600
#: Intentos fallidos de inicio de sesión permitidos por correo e IP cada 15 minutos.
INTENTOS_MAXIMOS, VENTANA_INTENTOS = 10, 15 * 60
#: Límites de tamaño: protegen la base de un cliente que envíe datos sin fin.
MAX_CUERPO = 2 * 1024 * 1024
MAX_SESIONES, MAX_PESOS, MAX_APORTES = 5000, 3000, 5000

# Vocabularios: los mismos de la página (web/app.html) y de la app de iPhone.
DEPORTES = {"ninguno", "futbol", "running", "ciclismo", "natacion", "basquet", "tenis", "combate", "escalada", "otro"}
NIVELES = {"principiante", "intermedio", "avanzado"}
OBJETIVOS = {"salud", "fuerza", "musculo", "resistencia"}
TIPOS = {"auto", "full", "tp", "epp", "arnold", "weider"}
EQUIPOS = {"pesas", "gym", "banda", "balon"}
LESIONES = {"hombro", "codo", "muneca", "cuello", "lumbar", "cadera", "rodilla", "tobillo"}
OBJETIVOS_MODELO = {"mecanica", "fuerza", "complejidad_tecnica"}
VEREDICTOS = {"bueno", "aceptable", "malo", "inseguro"}
PERFILES_RECOMENDACION = {"cualquiera", "principiante"}
EMAIL = re.compile(r"^[^@\s]{1,64}@[^@\s]{1,190}\.[^@\s]{2,}$")
FECHA = re.compile(r"^\d{4}-\d{2}-\d{2}$")

log = logging.getLogger("sustituto_seguro.api")


class ErrorDatos(ValueError):
    """Datos inválidos enviados por el cliente: se responde 400 con el mensaje."""


# ═══════════════ Validación ═══════════════

def _entero(v, minimo, maximo, campo, nulo=False):
    if v is None or v == "":
        if nulo:
            return None
        raise ErrorDatos(f"Falta {campo}.")
    if isinstance(v, bool):
        raise ErrorDatos(f"{campo} no es un número.")
    try:
        n = float(v)
    except (TypeError, ValueError):
        raise ErrorDatos(f"{campo} no es un número.") from None
    if not n.is_integer() or not minimo <= n <= maximo:
        raise ErrorDatos(f"{campo} debe ser un entero entre {minimo} y {maximo}.")
    return int(n)


def _decimal(v, minimo, maximo, campo, nulo=False):
    if v is None or v == "":
        if nulo:
            return None
        raise ErrorDatos(f"Falta {campo}.")
    if isinstance(v, bool):
        raise ErrorDatos(f"{campo} no es un número.")
    try:
        n = float(v)
    except (TypeError, ValueError):
        raise ErrorDatos(f"{campo} no es un número.") from None
    if not minimo <= n <= maximo:
        raise ErrorDatos(f"{campo} debe estar entre {minimo} y {maximo}.")
    return round(n, 2)


def _texto(v, maximo, campo, vacio=True):
    if v is None:
        v = ""
    if not isinstance(v, str):
        raise ErrorDatos(f"{campo} debe ser texto.")
    v = v.strip()
    if not vacio and not v:
        raise ErrorDatos(f"Falta {campo}.")
    return v[:maximo]


def _opcion(v, opciones, campo):
    if v not in opciones:
        raise ErrorDatos(f"{campo} no válido.")
    return v


def _fecha(v, campo="fecha"):
    if not isinstance(v, str) or not FECHA.match(v):
        raise ErrorDatos(f"{campo} debe tener el formato AAAA-MM-DD.")
    return v


def _lista(v, campo):
    if v is None:
        return []
    if not isinstance(v, list):
        raise ErrorDatos(f"{campo} debe ser una lista.")
    return v


def validar_perfil(p) -> dict:
    if not isinstance(p, dict):
        raise ErrorDatos("El perfil debe ser un objeto.")
    return {
        "edad": _entero(p.get("edad"), 10, 100, "la edad"),
        "peso": _decimal(p.get("peso"), 30, 250, "el peso", nulo=True),
        "altura": _decimal(p.get("altura"), 120, 230, "la altura", nulo=True),
        "deporte": _opcion(p.get("deporte", "ninguno"), DEPORTES, "El deporte"),
        "nivel": _opcion(p.get("nivel", "principiante"), NIVELES, "El nivel"),
        "objetivo": _opcion(p.get("objetivo", "salud"), OBJETIVOS, "El objetivo"),
        "dias": _entero(p.get("dias", 3), 2, 6, "los días por semana"),
        "minutos": _opcion(p.get("minutos", 45), {30, 45, 60, 75}, "Los minutos por sesión"),
        "tipo": _opcion(p.get("tipo", "auto"), TIPOS, "El tipo de rutina"),
        "equipo": list(dict.fromkeys(_opcion(e, EQUIPOS, "El equipamiento") for e in _lista(p.get("equipo"), "equipo"))),
        "lesiones": list(dict.fromkeys(_opcion(k, LESIONES, "La lesión") for k in _lista(p.get("lesiones"), "lesiones"))),
    }


def ids_de_la_pagina(ruta: Path = PAGINA) -> list[str]:
    """Los ``ejercicio_id`` del bloque de datos de ``docs/index.html``, en su orden."""
    if not ruta.is_file():
        return []
    html = ruta.read_text(encoding="utf-8")
    m = re.search(r'<script type="application/json" id="datos">(.*?)</script>', html, re.S)
    return json.loads(m.group(1).replace("<\\/", "</")).get("ids", []) if m else []


class Catalogo:
    """Los ``ejercicio_id`` que existen, leídos de los datos de la página."""

    def __init__(self, ruta: Path = PAGINA):
        self.ids: set[str] = set(ids_de_la_pagina(ruta))

    def ejercicio(self, v, campo="ejercicio") -> str:
        if not isinstance(v, str) or (self.ids and v not in self.ids):
            raise ErrorDatos(f"{campo} desconocido: {str(v)[:60]}")
        return v


def validar_rutina(r, catalogo: Catalogo) -> dict:
    """La rutina se guarda tal como la arma el cliente, después de revisar su forma y que
    cada ejercicio exista. La seguridad (que ningún ejercicio cargue una lesión) la
    garantiza el generador, en la web y en la app, con los mismos datos."""
    if not isinstance(r, dict):
        raise ErrorDatos("La rutina debe ser un objeto.")
    dias = _lista(r.get("dias"), "dias")
    if not 1 <= len(dias) <= 6:
        raise ErrorDatos("La rutina debe tener entre 1 y 6 días.")
    limpios = []
    for d in dias:
        if not isinstance(d, dict):
            raise ErrorDatos("Cada día debe ser un objeto.")
        ejercicios = []
        for e in _lista(d.get("ejercicios"), "ejercicios")[:12]:
            if not isinstance(e, dict):
                raise ErrorDatos("Cada ejercicio debe ser un objeto.")
            ejercicios.append({
                "id": catalogo.ejercicio(e.get("id")),
                "grupo": _texto(e.get("grupo"), 20, "grupo"),
                "series": _entero(e.get("series", 3), 1, 6, "las series"),
                "reps": _texto(str(e.get("reps", "10")), 12, "repeticiones") or "10",
                "descanso": _entero(e.get("descanso", 60), 15, 300, "el descanso"),
                "iso": e.get("iso") is True, "extra": e.get("extra") is True,
                "alternativas": [catalogo.ejercicio(a, "alternativa") for a in _lista(e.get("alternativas"), "alternativas")[:8]],
            })
        limpios.append({"titulo": _texto(d.get("titulo"), 60, "título del día"),
                        "equilibrio": d.get("equilibrio") is True, "ejercicios": ejercicios})
    avisos = []
    for a in _lista(r.get("avisos"), "avisos")[:8]:
        if isinstance(a, list) and len(a) == 3 and a[0] in {"amber", "red", "teal"}:
            avisos.append([a[0], _texto(a[1], 80, "aviso"), _texto(a[2], 300, "aviso")])
    creada = r.get("creada")
    return {"creada": _fecha(creada, "creada") if creada else ahora()[:10],
            "objetivo": _opcion(r.get("objetivo", "salud"), OBJETIVOS, "El objetivo"),
            "tipo": r.get("tipo") if r.get("tipo") in TIPOS else None,
            "porqueTipo": _texto(r.get("porqueTipo"), 200, "porqueTipo"),
            "perfil": validar_perfil(r["perfil"]) if isinstance(r.get("perfil"), dict) else None,
            "avisos": avisos,
            "huecos": [_texto(h, 20, "hueco") for h in _lista(r.get("huecos"), "huecos")[:12]],
            "dias": limpios}


def validar_sesion(s, catalogo: Catalogo) -> dict:
    if not isinstance(s, dict):
        raise ErrorDatos("La sesión debe ser un objeto.")
    uid = _texto(s.get("uid"), 64, "el identificador de la sesión", vacio=False)
    dolor = {}
    for zona, valor in (s.get("dolor") or {}).items() if isinstance(s.get("dolor"), dict) else []:
        dolor[_opcion(zona, LESIONES, "La zona de dolor")] = _entero(valor, 0, 10, "el dolor")
    ejercicios = []
    for e in _lista(s.get("ejercicios"), "ejercicios")[:20]:
        if not isinstance(e, dict):
            raise ErrorDatos("Cada ejercicio debe ser un objeto.")
        series = [{"kg": _decimal(z.get("kg") if isinstance(z, dict) else None, 0, 1000, "el peso de la serie", nulo=True),
                   "reps": _entero(z.get("reps", 0) if isinstance(z, dict) else 0, 0, 1000, "las repeticiones")}
                  for z in _lista(e.get("series"), "series")[:10]]
        ejercicios.append({"id": catalogo.ejercicio(e.get("id")), "series": series})
    return {"uid": uid, "fecha": _fecha(s.get("fecha")),
            "dia": _entero(s.get("dia", 0), 0, 5, "el día"),
            "titulo": _texto(s.get("titulo"), 60, "el título"),
            "minutos": _entero(s.get("minutos", 1), 1, 300, "los minutos"),
            "esfuerzo": _entero(s.get("esfuerzo", 6), 1, 10, "el esfuerzo"),
            "notas": _texto(s.get("notas"), 500, "las notas"),
            "dolor": dolor, "ejercicios": ejercicios,
            "origen": s.get("origen") if s.get("origen") in {"web", "ios"} else "web"}


def validar_peso(p) -> dict:
    if not isinstance(p, dict):
        raise ErrorDatos("El peso debe ser un objeto.")
    return {"fecha": _fecha(p.get("fecha")), "kg": _decimal(p.get("kg"), 30, 250, "el peso")}


def validar_correccion(c, catalogo: Catalogo) -> dict:
    if not isinstance(c, dict):
        raise ErrorDatos("La corrección debe ser un objeto.")
    return {"ejercicio_id": catalogo.ejercicio(c.get("ejercicio_id")),
            "objetivo": _opcion(c.get("objetivo"), OBJETIVOS_MODELO, "El objetivo"),
            "etiqueta": _texto(c.get("etiqueta"), 20, "la etiqueta", vacio=False),
            "etiqueta_previa": _texto(c.get("etiqueta_previa"), 20, "la etiqueta previa"),
            "origen_previo": _texto(c.get("origen_previo"), 20, "el origen previo"),
            "confianza_previa": _decimal(c.get("confianza_previa"), 0, 1, "la confianza", nulo=True),
            "fecha": _texto(c.get("fecha"), 40, "la fecha") or ahora()}


def validar_revision(r, catalogo: Catalogo) -> dict:
    if not isinstance(r, dict):
        raise ErrorDatos("La calificación debe ser un objeto.")
    return {"original_id": catalogo.ejercicio(r.get("original_id")),
            "alternativa_id": catalogo.ejercicio(r.get("alternativa_id")),
            "articulacion": _texto(r.get("articulacion"), 40, "la articulación", vacio=False),
            "perfil": _opcion(r.get("perfil", "cualquiera"), PERFILES_RECOMENDACION, "El perfil"),
            "veredicto": _opcion(r.get("veredicto"), VEREDICTOS, "El veredicto"),
            "similitud": _decimal(r.get("similitud"), 0, 1, "la similitud", nulo=True),
            "misma_familia": 1 if r.get("misma_familia") else 0,
            "fecha": _texto(r.get("fecha"), 40, "la fecha") or ahora()}


# ═══════════════ Lectura y escritura ═══════════════

def leer_datos(cx, uid: int) -> dict:
    p = cx.una("SELECT * FROM perfiles WHERE usuario_id = ?", (uid,))
    perfil = None
    if p:
        perfil = {k: p[k] for k in ["edad", "peso", "altura", "deporte", "nivel", "objetivo", "dias", "minutos"]}
        perfil.update(tipo=p.get("tipo") or "auto", equipo=de_json(p["equipo"], []), lesiones=de_json(p["lesiones"], []))
    r = cx.una("SELECT datos FROM rutinas WHERE usuario_id = ?", (uid,))
    sesiones = [{"uid": s["uid"], "fecha": s["fecha"], "dia": s["dia"], "titulo": s["titulo"], "minutos": s["minutos"],
                 "esfuerzo": s["esfuerzo"], "notas": s["notas"], "dolor": de_json(s["dolor"], {}),
                 "ejercicios": de_json(s["ejercicios"], []), "origen": s["origen"]}
                for s in cx.todas("SELECT * FROM sesiones WHERE usuario_id = ? ORDER BY fecha, id", (uid,))]
    pesos = [{"fecha": x["fecha"], "kg": x["kg"]}
             for x in cx.todas("SELECT fecha, kg FROM pesos WHERE usuario_id = ? ORDER BY fecha", (uid,))]
    correcciones = cx.todas("SELECT ejercicio_id, objetivo, etiqueta, etiqueta_previa, origen_previo, confianza_previa, fecha "
                            "FROM correcciones WHERE usuario_id = ? ORDER BY id", (uid,))
    revisiones = cx.todas("SELECT original_id, alternativa_id, articulacion, perfil, veredicto, similitud, misma_familia, fecha "
                          "FROM revisiones WHERE usuario_id = ? ORDER BY id", (uid,))
    for x in revisiones:
        x["misma_familia"] = bool(x["misma_familia"])
    return {"perfil": perfil, "rutina": de_json(r["datos"]) if r else None, "sesiones": sesiones, "pesos": pesos,
            "correcciones": correcciones, "revisiones": revisiones}


def guardar_perfil(cx, uid: int, p: dict) -> None:
    cx.ejecutar(
        "INSERT INTO perfiles (usuario_id, edad, peso, altura, deporte, nivel, objetivo, dias, minutos, tipo, equipo, lesiones, actualizado) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT (usuario_id) DO UPDATE SET edad = excluded.edad, "
        "peso = excluded.peso, altura = excluded.altura, deporte = excluded.deporte, nivel = excluded.nivel, "
        "objetivo = excluded.objetivo, dias = excluded.dias, minutos = excluded.minutos, tipo = excluded.tipo, "
        "equipo = excluded.equipo, lesiones = excluded.lesiones, actualizado = excluded.actualizado",
        (uid, p["edad"], p["peso"], p["altura"], p["deporte"], p["nivel"], p["objetivo"], p["dias"], p["minutos"],
         p["tipo"], a_json(p["equipo"]), a_json(p["lesiones"]), ahora()))


def guardar_rutina(cx, uid: int, r: dict) -> None:
    cx.ejecutar("INSERT INTO rutinas (usuario_id, datos, actualizado) VALUES (?, ?, ?) ON CONFLICT (usuario_id) "
                "DO UPDATE SET datos = excluded.datos, actualizado = excluded.actualizado", (uid, a_json(r), ahora()))


def guardar_sesion(cx, uid: int, s: dict, reemplazar: bool = True) -> None:
    accion = ("DO UPDATE SET fecha = excluded.fecha, dia = excluded.dia, titulo = excluded.titulo, minutos = excluded.minutos, "
              "esfuerzo = excluded.esfuerzo, notas = excluded.notas, dolor = excluded.dolor, ejercicios = excluded.ejercicios"
              if reemplazar else "DO NOTHING")
    cx.ejecutar("INSERT INTO sesiones (usuario_id, uid, fecha, dia, titulo, minutos, esfuerzo, notas, dolor, ejercicios, origen) "
                f"VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT (usuario_id, uid) {accion}",
                (uid, s["uid"], s["fecha"], s["dia"], s["titulo"], s["minutos"], s["esfuerzo"], s["notas"],
                 a_json(s["dolor"]), a_json(s["ejercicios"]), s["origen"]))


def guardar_peso(cx, uid: int, p: dict, reemplazar: bool = True) -> None:
    accion = "DO UPDATE SET kg = excluded.kg" if reemplazar else "DO NOTHING"
    cx.ejecutar(f"INSERT INTO pesos (usuario_id, fecha, kg) VALUES (?, ?, ?) ON CONFLICT (usuario_id, fecha) {accion}",
                (uid, p["fecha"], p["kg"]))


def guardar_correccion(cx, uid: int, c: dict) -> None:
    # Un voto por persona, ejercicio y objetivo: el último reemplaza al anterior (como en la página)
    cx.ejecutar("INSERT INTO correcciones (usuario_id, ejercicio_id, objetivo, etiqueta, etiqueta_previa, origen_previo, "
                "confianza_previa, fecha) VALUES (?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT (usuario_id, ejercicio_id, objetivo) "
                "DO UPDATE SET etiqueta = excluded.etiqueta, etiqueta_previa = excluded.etiqueta_previa, "
                "origen_previo = excluded.origen_previo, confianza_previa = excluded.confianza_previa, fecha = excluded.fecha",
                (uid, c["ejercicio_id"], c["objetivo"], c["etiqueta"], c["etiqueta_previa"], c["origen_previo"],
                 c["confianza_previa"], c["fecha"]))


def guardar_revision(cx, uid: int, r: dict) -> None:
    cx.ejecutar("INSERT INTO revisiones (usuario_id, original_id, alternativa_id, articulacion, perfil, veredicto, similitud, "
                "misma_familia, fecha) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT (usuario_id, original_id, alternativa_id, "
                "articulacion, perfil) DO UPDATE SET veredicto = excluded.veredicto, similitud = excluded.similitud, "
                "misma_familia = excluded.misma_familia, fecha = excluded.fecha",
                (uid, r["original_id"], r["alternativa_id"], r["articulacion"], r["perfil"], r["veredicto"], r["similitud"],
                 r["misma_familia"], r["fecha"]))


def _contar(cx, tabla: str, uid: int) -> int:
    return cx.una(f"SELECT COUNT(*) AS n FROM {tabla} WHERE usuario_id = ?", (uid,))["n"]


# ═══════════════ Aplicación ═══════════════

def crear_app(url_base: str | None = None, secreto: str | None = None, catalogo: Catalogo | None = None) -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = MAX_CUERPO
    url_base = url_base or os.environ.get("DATABASE_URL") or f"sqlite:///{RAIZ / 'api/instancia/cuentas.db'}"
    secreto = secreto or os.environ.get("SECRETO")
    if not secreto:
        secreto = secrets.token_urlsafe(32)
        log.warning("Falta la variable SECRETO: las sesiones se cierran cada vez que se reinicia el servidor.")
    bd = BaseDatos(url_base)
    bd.crear_esquema()
    firma = URLSafeTimedSerializer(secreto, salt="sustituto-seguro-sesion")
    catalogo = catalogo or Catalogo()
    intentos: dict[str, deque] = defaultdict(deque)
    app.extensions["sustituto_bd"] = bd

    def cx():
        if "cx" not in g:
            g.cx = bd.conectar().__enter__()
        return g.cx

    @app.teardown_request
    def _cerrar(error=None):
        conexion = g.pop("cx", None)
        if conexion is not None:
            conexion.__exit__(type(error) if error else None, error, None)

    def respuesta_error(mensaje: str, codigo: int):
        return jsonify({"error": mensaje}), codigo

    @app.errorhandler(ErrorDatos)
    def _datos(e):
        return respuesta_error(str(e), 400)

    @app.errorhandler(413)
    def _grande(_):
        return respuesta_error("Los datos enviados son demasiado grandes.", 413)

    @app.errorhandler(404)
    def _no_existe(_):
        return respuesta_error("No existe esa dirección.", 404)

    @app.errorhandler(405)
    def _metodo(_):
        return respuesta_error("Método no permitido.", 405)

    @app.errorhandler(500)
    def _interno(_):
        return respuesta_error("Error interno del servidor. Vuelve a intentarlo.", 500)

    @app.after_request
    def _cabeceras(r):
        r.headers.setdefault("X-Content-Type-Options", "nosniff")
        r.headers.setdefault("Referrer-Policy", "same-origin")
        if request.path.startswith("/api/"):
            r.headers["Cache-Control"] = "no-store"
        return r

    def cuerpo() -> dict:
        datos = request.get_json(silent=True)
        if not isinstance(datos, dict):
            raise ErrorDatos("El cuerpo debe ser un objeto JSON.")
        return datos

    def token_de(usuario_id: int) -> str:
        return firma.dumps({"u": usuario_id})

    def publico(u: dict) -> dict:
        return {"id": u["id"], "email": u["email"], "nombre": u["nombre"], "creado": u["creado"]}

    def requiere_cuenta(f):
        @wraps(f)
        def envoltura(*args, **kwargs):
            cabecera = request.headers.get("Authorization", "")
            if not cabecera.startswith("Bearer "):
                return respuesta_error("Inicia sesión para continuar.", 401)
            try:
                datos = firma.loads(cabecera[7:], max_age=DURACION_SESION)
            except SignatureExpired:
                return respuesta_error("Tu sesión venció. Vuelve a iniciar sesión.", 401)
            except BadSignature:
                return respuesta_error("Sesión no válida. Vuelve a iniciar sesión.", 401)
            usuario = cx().una("SELECT * FROM usuarios WHERE id = ?", (datos.get("u"),))
            if usuario is None:
                return respuesta_error("La cuenta ya no existe.", 401)
            g.usuario = usuario
            return f(*args, **kwargs)
        return envoltura

    # ── Página ──────────────────────────────────────────────────────────────
    @app.get("/")
    def pagina():
        if not PAGINA.is_file():
            return respuesta_error("Falta docs/index.html: corre scripts/armar_web.py.", 404)
        return send_file(PAGINA, mimetype="text/html", max_age=300)

    @app.get("/api/salud")
    def salud():
        with bd.conectar() as c:
            c.una("SELECT 1 AS uno")
        return jsonify({"ok": True, "version": VERSION, "base": bd.motor, "cuentas": True})

    # ── Cuenta ──────────────────────────────────────────────────────────────
    @app.post("/api/registro")
    def registro():
        d = cuerpo()
        email = _texto(d.get("email"), 254, "el correo", vacio=False).lower()
        if not EMAIL.match(email):
            raise ErrorDatos("Ese correo no parece válido.")
        nombre = _texto(d.get("nombre"), 80, "tu nombre", vacio=False)
        contrasena = d.get("contrasena")
        if not isinstance(contrasena, str) or not 8 <= len(contrasena) <= 200:
            raise ErrorDatos("La contraseña debe tener al menos 8 caracteres.")
        if cx().una("SELECT id FROM usuarios WHERE email = ?", (email,)):
            return respuesta_error("Ya existe una cuenta con ese correo. Inicia sesión.", 409)
        fila = cx().una("INSERT INTO usuarios (email, nombre, hash_contrasena, creado) VALUES (?, ?, ?, ?) RETURNING *",
                        (email, nombre, generate_password_hash(contrasena), ahora()))
        return jsonify({"token": token_de(fila["id"]), "usuario": publico(fila)}), 201

    @app.post("/api/sesion")
    def iniciar_sesion():
        d = cuerpo()
        email = _texto(d.get("email"), 254, "el correo", vacio=False).lower()
        contrasena = d.get("contrasena") if isinstance(d.get("contrasena"), str) else ""
        clave = f"{email}|{request.remote_addr}"
        ventana = intentos[clave]
        while ventana and ventana[0] < time.time() - VENTANA_INTENTOS:
            ventana.popleft()
        if len(ventana) >= INTENTOS_MAXIMOS:
            return respuesta_error("Demasiados intentos. Espera unos minutos y vuelve a intentarlo.", 429)
        usuario = cx().una("SELECT * FROM usuarios WHERE email = ?", (email,))
        if usuario is None or not check_password_hash(usuario["hash_contrasena"], contrasena):
            ventana.append(time.time())
            return respuesta_error("Correo o contraseña incorrectos.", 401)
        intentos.pop(clave, None)
        return jsonify({"token": token_de(usuario["id"]), "usuario": publico(usuario)})

    @app.get("/api/cuenta")
    @requiere_cuenta
    def cuenta():
        u = g.usuario
        resumen = {t: _contar(cx(), t, u["id"]) for t in ["sesiones", "pesos", "correcciones", "revisiones"]}
        return jsonify({"usuario": publico(u), "resumen": resumen})

    @app.delete("/api/cuenta")
    @requiere_cuenta
    def borrar_cuenta():
        d = cuerpo()
        if not check_password_hash(g.usuario["hash_contrasena"], d.get("contrasena") or ""):
            return respuesta_error("La contraseña no es correcta.", 403)
        # ON DELETE CASCADE borra perfil, rutina, sesiones, pesos y aportes
        cx().ejecutar("DELETE FROM usuarios WHERE id = ?", (g.usuario["id"],))
        return jsonify({"ok": True, "borrado": True})

    # ── Datos ───────────────────────────────────────────────────────────────
    @app.get("/api/datos")
    @requiere_cuenta
    def datos():
        return jsonify(leer_datos(cx(), g.usuario["id"]))

    @app.put("/api/datos/perfil")
    @requiere_cuenta
    def poner_perfil():
        p = validar_perfil(cuerpo())
        guardar_perfil(cx(), g.usuario["id"], p)
        return jsonify({"perfil": p})

    @app.put("/api/datos/rutina")
    @requiere_cuenta
    def poner_rutina():
        r = validar_rutina(cuerpo(), catalogo)
        guardar_rutina(cx(), g.usuario["id"], r)
        return jsonify({"rutina": r})

    @app.delete("/api/datos/rutina")
    @requiere_cuenta
    def quitar_rutina():
        cx().ejecutar("DELETE FROM rutinas WHERE usuario_id = ?", (g.usuario["id"],))
        return jsonify({"ok": True})

    @app.post("/api/datos/sesiones")
    @requiere_cuenta
    def poner_sesion():
        s = validar_sesion(cuerpo(), catalogo)
        if _contar(cx(), "sesiones", g.usuario["id"]) >= MAX_SESIONES:
            return respuesta_error("Llegaste al máximo de sesiones guardadas.", 409)
        guardar_sesion(cx(), g.usuario["id"], s)
        return jsonify({"sesion": s}), 201

    @app.delete("/api/datos/sesiones/<uid>")
    @requiere_cuenta
    def quitar_sesion(uid):
        cx().ejecutar("DELETE FROM sesiones WHERE usuario_id = ? AND uid = ?", (g.usuario["id"], uid))
        return jsonify({"ok": True})

    @app.put("/api/datos/pesos")
    @requiere_cuenta
    def poner_peso():
        p = validar_peso(cuerpo())
        guardar_peso(cx(), g.usuario["id"], p)
        return jsonify({"peso": p})

    @app.delete("/api/datos/pesos/<fecha>")
    @requiere_cuenta
    def quitar_peso(fecha):
        cx().ejecutar("DELETE FROM pesos WHERE usuario_id = ? AND fecha = ?", (g.usuario["id"], _fecha(fecha)))
        return jsonify({"ok": True})

    @app.post("/api/datos/aportes")
    @requiere_cuenta
    def poner_aportes():
        d = cuerpo()
        correcciones = [validar_correccion(c, catalogo) for c in _lista(d.get("correcciones"), "correcciones")[:MAX_APORTES]]
        revisiones = [validar_revision(r, catalogo) for r in _lista(d.get("revisiones"), "revisiones")[:MAX_APORTES]]
        for c in correcciones:
            guardar_correccion(cx(), g.usuario["id"], c)
        for r in revisiones:
            guardar_revision(cx(), g.usuario["id"], r)
        return jsonify({"correcciones": len(correcciones), "revisiones": len(revisiones)})

    @app.post("/api/sincronizar")
    @requiere_cuenta
    def sincronizar():
        """Sube lo que el dispositivo guardó sin cuenta. Lo que ya está en la cuenta gana:
        el perfil y la rutina locales solo se usan si la cuenta no tiene; las sesiones y
        los pesos se agregan si no existen; los aportes se suman (el último voto vale)."""
        d = cuerpo()
        uid = g.usuario["id"]
        actuales = leer_datos(cx(), uid)
        if d.get("perfil") and actuales["perfil"] is None:
            guardar_perfil(cx(), uid, validar_perfil(d["perfil"]))
        if d.get("rutina") and actuales["rutina"] is None:
            guardar_rutina(cx(), uid, validar_rutina(d["rutina"], catalogo))
        sesiones = [validar_sesion(s, catalogo) for s in _lista(d.get("sesiones"), "sesiones")[:MAX_SESIONES]]
        pesos = [validar_peso(p) for p in _lista(d.get("pesos"), "pesos")[:MAX_PESOS]]
        if len(actuales["sesiones"]) + len(sesiones) > MAX_SESIONES:
            return respuesta_error("Llegaste al máximo de sesiones guardadas.", 409)
        for s in sesiones:
            guardar_sesion(cx(), uid, s, reemplazar=False)
        for p in pesos:
            guardar_peso(cx(), uid, p, reemplazar=False)
        for c in _lista(d.get("correcciones"), "correcciones")[:MAX_APORTES]:
            guardar_correccion(cx(), uid, validar_correccion(c, catalogo))
        for r in _lista(d.get("revisiones"), "revisiones")[:MAX_APORTES]:
            guardar_revision(cx(), uid, validar_revision(r, catalogo))
        return jsonify(leer_datos(cx(), uid))

    return app


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    crear_app().run(host="127.0.0.1", port=int(os.environ.get("PUERTO", "8000")), debug=False)

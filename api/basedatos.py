"""Base de datos de las cuentas de Sustituto Seguro (README §20).

Funciona con **PostgreSQL** en producción (RDS en AWS) y con **SQLite** para desarrollo
y tests, con el mismo SQL: los dos aceptan ``RETURNING`` y ``ON CONFLICT … DO UPDATE``.
La conexión se elige con la variable ``DATABASE_URL``:

- ``postgresql://usuario:clave@host:5432/base`` → PostgreSQL (requiere ``psycopg``).
- ``sqlite:///ruta/al/archivo.db`` → SQLite (incluido en Python).

Las consultas se escriben con ``?`` y se traducen a ``%s`` para PostgreSQL.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

#: Esquema de la base. ``{ID}`` cambia según el motor; el resto es SQL estándar.
ESQUEMA = """
CREATE TABLE IF NOT EXISTS usuarios (
    id {ID},
    email TEXT NOT NULL UNIQUE,
    nombre TEXT NOT NULL,
    hash_contrasena TEXT NOT NULL,
    creado TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS perfiles (
    usuario_id INTEGER PRIMARY KEY REFERENCES usuarios(id) ON DELETE CASCADE,
    edad INTEGER NOT NULL,
    peso DOUBLE PRECISION,
    altura DOUBLE PRECISION,
    deporte TEXT NOT NULL,
    nivel TEXT NOT NULL,
    objetivo TEXT NOT NULL,
    dias INTEGER NOT NULL,
    minutos INTEGER NOT NULL,
    tipo TEXT NOT NULL DEFAULT 'auto',
    equipo TEXT NOT NULL,
    lesiones TEXT NOT NULL,
    actualizado TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS rutinas (
    usuario_id INTEGER PRIMARY KEY REFERENCES usuarios(id) ON DELETE CASCADE,
    datos TEXT NOT NULL,
    actualizado TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS sesiones (
    id {ID},
    usuario_id INTEGER NOT NULL REFERENCES usuarios(id) ON DELETE CASCADE,
    uid TEXT NOT NULL,
    fecha TEXT NOT NULL,
    dia INTEGER,
    titulo TEXT NOT NULL DEFAULT '',
    minutos INTEGER,
    esfuerzo INTEGER,
    notas TEXT NOT NULL DEFAULT '',
    dolor TEXT NOT NULL DEFAULT '{{}}',
    ejercicios TEXT NOT NULL DEFAULT '[]',
    origen TEXT NOT NULL DEFAULT 'web',
    UNIQUE (usuario_id, uid)
);
CREATE TABLE IF NOT EXISTS pesos (
    usuario_id INTEGER NOT NULL REFERENCES usuarios(id) ON DELETE CASCADE,
    fecha TEXT NOT NULL,
    kg DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (usuario_id, fecha)
);
CREATE TABLE IF NOT EXISTS correcciones (
    id {ID},
    usuario_id INTEGER NOT NULL REFERENCES usuarios(id) ON DELETE CASCADE,
    ejercicio_id TEXT NOT NULL,
    objetivo TEXT NOT NULL,
    etiqueta TEXT NOT NULL,
    etiqueta_previa TEXT NOT NULL DEFAULT '',
    origen_previo TEXT NOT NULL DEFAULT '',
    confianza_previa DOUBLE PRECISION,
    fecha TEXT NOT NULL,
    UNIQUE (usuario_id, ejercicio_id, objetivo)
);
CREATE TABLE IF NOT EXISTS revisiones (
    id {ID},
    usuario_id INTEGER NOT NULL REFERENCES usuarios(id) ON DELETE CASCADE,
    original_id TEXT NOT NULL,
    alternativa_id TEXT NOT NULL,
    articulacion TEXT NOT NULL,
    perfil TEXT NOT NULL,
    veredicto TEXT NOT NULL,
    similitud DOUBLE PRECISION,
    misma_familia INTEGER NOT NULL DEFAULT 0,
    fecha TEXT NOT NULL,
    UNIQUE (usuario_id, original_id, alternativa_id, articulacion, perfil)
);
CREATE INDEX IF NOT EXISTS sesiones_por_usuario ON sesiones (usuario_id, fecha);
"""

MOTORES = {
    "sqlite": {"ID": "INTEGER PRIMARY KEY AUTOINCREMENT"},
    "postgresql": {"ID": "BIGSERIAL PRIMARY KEY"},
}


def ahora() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class BaseDatos:
    """Una conexión por petición (``conectar``); el esquema se crea con ``crear_esquema``."""

    def __init__(self, url: str):
        partes = urlparse(url)
        if partes.scheme in ("postgres", "postgresql"):
            self.motor = "postgresql"
            self.url = url.replace("postgres://", "postgresql://", 1)
        elif partes.scheme == "sqlite":
            self.motor = "sqlite"
            # sqlite:///relativa.db o sqlite:////absoluta.db; ":memory:" no sirve con varias conexiones
            self.ruta = url[len("sqlite:///"):]
            if self.ruta and self.ruta != ":memory:":
                Path(self.ruta).parent.mkdir(parents=True, exist_ok=True)
        else:
            raise ValueError(f"DATABASE_URL no reconocida: {url!r} (usa postgresql://… o sqlite:///…)")

    def conectar(self) -> "Conexion":
        if self.motor == "sqlite":
            cx = sqlite3.connect(self.ruta, timeout=10)
            cx.row_factory = sqlite3.Row
            cx.execute("PRAGMA foreign_keys = ON")
            return Conexion(cx, "sqlite")
        import psycopg  # solo en producción: pip install "psycopg[binary]"
        from psycopg.rows import dict_row
        return Conexion(psycopg.connect(self.url, row_factory=dict_row, connect_timeout=10), "postgresql")

    def crear_esquema(self) -> None:
        sql = ESQUEMA.format(**MOTORES[self.motor])
        with self.conectar() as cx:
            for sentencia in filter(str.strip, sql.split(";")):
                cx.ejecutar(sentencia)


class Conexion:
    """Envoltorio mínimo: mismas consultas con ``?`` en los dos motores y filas como dict."""

    def __init__(self, cx, motor: str):
        self.cx, self.motor = cx, motor

    def __enter__(self) -> "Conexion":
        return self

    def __exit__(self, tipo, *_):
        if tipo is None:
            self.cx.commit()
        else:
            self.cx.rollback()
        self.cx.close()

    def _sql(self, sql: str) -> str:
        return sql.replace("?", "%s") if self.motor == "postgresql" else sql

    def ejecutar(self, sql: str, parametros: tuple | list = ()):
        cursor = self.cx.cursor()
        cursor.execute(self._sql(sql), tuple(parametros))
        return cursor

    def una(self, sql: str, parametros: tuple | list = ()) -> dict | None:
        fila = self.ejecutar(sql, parametros).fetchone()
        return dict(fila) if fila is not None else None

    def todas(self, sql: str, parametros: tuple | list = ()) -> list[dict]:
        return [dict(f) for f in self.ejecutar(sql, parametros).fetchall()]


def a_json(valor) -> str:
    return json.dumps(valor, ensure_ascii=False, separators=(",", ":"))


def de_json(texto, defecto=None):
    if texto is None or texto == "":
        return defecto
    return json.loads(texto)

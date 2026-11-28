"""Prueba de punta a punta de las cuentas en la página (README §20): servidor real, base de
datos real y navegador real.

Levanta ``api.servidor`` con una base SQLite temporal (o la de ``DATABASE_URL``), y en
Chrome: arma una rutina sin cuenta, entrena, crea la cuenta (lo local sube), comprueba en
la base que quedó guardado, entra desde otro navegador y ve lo mismo, cierra sesión y
borra la cuenta.

    pip install playwright && playwright install chromium
    python web/probar_cuentas.py [ruta a Chrome]
"""
import os
import socket
import sys
import tempfile
import threading
import time
from pathlib import Path

from playwright.sync_api import sync_playwright
from werkzeug.serving import make_server
import logging

logging.getLogger("werkzeug").setLevel(logging.ERROR)

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
from api.servidor import crear_app  # noqa: E402

CHROME = sys.argv[1] if len(sys.argv) > 1 else None
fallos, errores = [], []


def check(cond, msg):
    print("OK   " if cond else "FALLA", msg)
    if not cond:
        fallos.append(msg)


def puerto_libre():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


carpeta = Path(tempfile.mkdtemp())
url_bd = os.environ.get("DATABASE_URL") or f"sqlite:///{carpeta / 'cuentas.db'}"
app = crear_app(url_bd, secreto="prueba")
bd = app.extensions["sustituto_bd"]
puerto = puerto_libre()
servidor = make_server("127.0.0.1", puerto, app, threaded=True)
threading.Thread(target=servidor.serve_forever, daemon=True).start()
URL = f"http://127.0.0.1:{puerto}/"


def contar(tabla):
    with bd.conectar() as cx:
        return cx.una(f"SELECT COUNT(*) AS n FROM {tabla}")["n"]


with sync_playwright() as p:
    nav = p.chromium.launch(**({"executable_path": CHROME} if CHROME else {}), args=["--no-sandbox"])

    def pagina(**kw):
        ctx = nav.new_context(viewport={"width": 1200, "height": 900}, **kw)
        pg = ctx.new_page()
        pg.on("pageerror", lambda e: errores.append(str(e)))
        # Las respuestas 401/403 son parte de la prueba; las fuentes y el favicon no cargan sin internet
        pg.on("console", lambda m: m.type == "error" and "Failed to load resource" not in m.text and errores.append(m.text))
        pg.goto(URL)
        pg.wait_for_timeout(700)
        return pg

    pg = pagina()
    check(pg.locator(".cuenta-chip", has_text="Iniciar sesión").is_visible(), "con servidor aparece «Iniciar sesión»")

    # 1. Sin cuenta: rutina y una sesión, guardadas solo en el navegador
    pg.goto(URL + "#rutina"); pg.wait_for_timeout(300)
    pg.fill("#f-edad", "31"); pg.fill("#f-peso", "78")
    pg.click("[data-multi=lesiones] [data-v=rodilla]"); pg.click("[data-multi=equipo] [data-v=pesas]")
    pg.click("button[type=submit]"); pg.wait_for_timeout(300)
    pg.goto(URL + "#entrenar"); pg.wait_for_timeout(300)
    pg.locator("[data-dia='0']").click(); pg.wait_for_timeout(200)
    pg.locator("[data-hecho]").first.click(); pg.wait_for_timeout(150)
    pg.click("#terminar"); pg.wait_for_timeout(150)
    pg.click("#guardar-ses"); pg.wait_for_timeout(300)
    check(contar("usuarios") == 0, "sin cuenta no se escribe nada en la base")

    # 2. Crear la cuenta: lo local sube a la base
    pg.goto(URL + "#cuenta"); pg.wait_for_timeout(300)
    pg.click("[data-modo=registro]"); pg.wait_for_timeout(100)
    pg.fill("#c-nombre", "Benja"); pg.fill("#c-email", "benja@correo.cl"); pg.fill("#c-clave", "clave-segura-1")
    pg.click("#c-enviar"); pg.wait_for_timeout(300)
    check("aceptar" in pg.inner_text("#c-error"), "sin aceptar el uso de datos no se crea la cuenta")
    pg.check("#c-acepto"); pg.click("#c-enviar"); pg.wait_for_timeout(1200)
    check("Hola, Benja" in pg.inner_text("#app"), "cuenta creada y sesión iniciada")
    check(contar("usuarios") == 1 and contar("perfiles") == 1 and contar("rutinas") == 1 and contar("sesiones") == 1
          and contar("pesos") == 1, "perfil, rutina, sesión y peso subieron a la base")
    with bd.conectar() as cx:
        fila = cx.una("SELECT ejercicios FROM sesiones")
    check('"id":"' in fila["ejercicios"] and '"i":' not in fila["ejercicios"], "la base guarda los ejercicios por ejercicio_id")

    # 3. Un cambio con sesión iniciada se sube solo
    pg.goto(URL + "#progreso"); pg.wait_for_timeout(300)
    pg.fill("#p-kg", "77"); pg.click("#f-pesaje button"); pg.wait_for_timeout(1500)
    with bd.conectar() as cx:
        kg = cx.una("SELECT kg FROM pesos ORDER BY fecha DESC")["kg"]
    check(kg == 77, f"el peso nuevo se guardó en la cuenta ({kg})")

    # 4. Otro navegador, misma cuenta: ve los mismos datos
    otro = pagina()
    otro.goto(URL + "#cuenta"); otro.wait_for_timeout(300)
    check(otro.locator("#c-email").is_visible(), "otro navegador empieza sin sesión")
    otro.fill("#c-email", "benja@correo.cl"); otro.fill("#c-clave", "mala-clave-1"); otro.click("#c-enviar"); otro.wait_for_timeout(500)
    check("incorrectos" in otro.inner_text("#c-error"), "contraseña incorrecta se rechaza")
    otro.fill("#c-clave", "clave-segura-1"); otro.click("#c-enviar"); otro.wait_for_timeout(1200)
    check("Hola, Benja" in otro.inner_text("#app"), "inicia sesión en el otro navegador")
    otro.goto(URL + "#rutina"); otro.wait_for_timeout(400)
    check("1 día" in otro.inner_text("#app") or "días por semana" in otro.inner_text("#app"), "el otro navegador ve la rutina de la cuenta")
    otro.goto(URL + "#progreso"); otro.wait_for_timeout(400)
    check("Historial" in otro.inner_text("#app"), "el otro navegador ve la sesión de la cuenta")

    # 5. Cerrar sesión borra los datos de este navegador, no de la cuenta
    otro.goto(URL + "#cuenta"); otro.wait_for_timeout(300)
    otro.click("#c-salir"); otro.wait_for_timeout(500)
    check(otro.evaluate("JSON.parse(localStorage.getItem('sustituto-seguro-v1')).sesiones.length") == 0,
          "al cerrar sesión el navegador queda sin los datos")
    check(contar("sesiones") == 1, "la sesión sigue en la cuenta")

    # 6. Borrar la cuenta
    pg.goto(URL + "#cuenta"); pg.wait_for_timeout(300)
    pg.click("#c-borrar"); pg.wait_for_timeout(100)
    pg.fill("#c-clave-borrar", "mala"); pg.click("#c-borrar-si"); pg.wait_for_timeout(400)
    check("no es correcta" in pg.inner_text("#c-error-borrar"), "borrar exige la contraseña correcta")
    pg.fill("#c-clave-borrar", "clave-segura-1"); pg.click("#c-borrar-si"); pg.wait_for_timeout(600)
    check(all(contar(t) == 0 for t in ["usuarios", "perfiles", "rutinas", "sesiones", "pesos"]), "la cuenta y todos sus datos se borraron")
    pg.set_viewport_size({"width": 390, "height": 844}); pg.goto(URL + "#cuenta"); pg.wait_for_timeout(300)
    check(pg.evaluate("document.documentElement.scrollWidth") <= 390, "la vista de cuenta no se desborda en el celular")
    pg.screenshot(path=str(carpeta / "cuenta_movil.png"))

    # 7. Sin servidor (archivo local) la página funciona como siempre, sin cuentas
    sin = nav.new_page()
    sin.on("pageerror", lambda e: errores.append(str(e)))
    sin.goto((RAIZ / "docs/index.html").as_uri() + "#cuenta"); sin.wait_for_timeout(500)
    check("no tiene cuentas" in sin.inner_text("#app"), "sin servidor la vista de cuenta lo explica")
    check(sin.locator("#cuenta-barra").is_hidden(), "sin servidor no aparece el botón de cuenta")
    nav.close()

servidor.shutdown()
check(not errores, f"sin errores de JavaScript {errores[:3]}")
print("\nFALLAS:", len(fallos), "· captura:", carpeta / "cuenta_movil.png")
sys.exit(1 if fallos else 0)

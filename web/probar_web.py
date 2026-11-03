"""Pruebas de usuario de la página web en un navegador real (README §15).

Recorren todas las secciones en escritorio y móvil, con la IA simulada, y generan
300 rutinas con perfiles al azar para verificar que ninguna incluya un ejercicio
que cargue una zona lesionada.

Requiere Playwright y Chrome/Chromium::

    pip install playwright && playwright install chromium
    python web/probar_web.py              # capturas en web/capturas/
"""
import json
import random
import re
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

RAIZ = Path(__file__).resolve().parents[1]
pagina = RAIZ / "docs/index.html"
carpeta = RAIZ / "web/capturas"
carpeta.mkdir(exist_ok=True)
CHROME = sys.argv[1] if len(sys.argv) > 1 else None  # ruta opcional a un Chrome instalado

documento = pagina.read_text(encoding="utf-8")
fuente = documento
D = json.loads(re.search(r'<script type="application/json" id="datos">(.*?)</script>', documento, re.S).group(1).replace("<\\/", "</"))
# Versión con IA y descargas simuladas (window.claude.use)
falso = """<script>
window.__llamadas = [];
const iaFalsa = Object.assign(async (input, opts) => { window.__llamadas.push(input); opts?.onText?.({text: 'Calienta 5 minutos', delta: ''}); return {text: 'Calienta 5 minutos con movilidad.', truncated: false}; }, {
  json: async (input) => { window.__llamadas.push(input);
    const m = [...input.matchAll(/alternativas seguras permitidas: (\\d+)=/g)];
    const ok = parseInt(m[0][1]);
    return {resumen: 'Rutina equilibrada.', consejos: ['Duerme bien', 'Hidrátate'], advertencias: [],
            cambios: [{dia: 1, posicion: 0, nuevo_id: ok, motivo: 'Más simple'}, {dia: 1, posicion: 1, nuevo_id: 120, motivo: 'INSEGURO'}]}; } });
window.claude = { use: async n => n === 'sample' ? iaFalsa : n === 'downloads' ? { save: async () => ({}) } : null };
</script>"""
(carpeta / "app_ia.html").write_text(documento.replace("<body>", "<body>" + falso, 1), encoding="utf-8")
url, url_ia = pagina.as_uri(), (carpeta / "app_ia.html").as_uri()

fallos, errores = [], []
def check(c, m):
    print("OK   " if c else "FALLA", m)
    if not c: fallos.append(m)

# 0. Los índices curados apuntan al ejercicio esperado
esperado = {1382: 'bodyweight squat', 1570: 'leg press', 98: 'barbell bench press', 1059: 'pull-up', 1651: 'plank',
            654: 'dumbbell romanian deadlift', 376: 'cable seated row', 553: 'dumbbell lateral raise', 2131: 'jumping jack hd', 1920: 'glute bridge'}
check(all(D['ejercicios'][i][0].lower() == n for i, n in esperado.items()), "índices curados apuntan a los ejercicios correctos")
ids = [int(x) for x in re.findall(r"\[(\d+), '", fuente.split('const CURADOS = [')[1].split('];')[0])]
check(all(0 <= i < len(D['ejercicios']) for i in ids) and len(ids) == len(set(ids)), f"{len(ids)} ejercicios curados válidos y sin duplicar")

with sync_playwright() as p:
    nav = p.chromium.launch(**({"executable_path": CHROME} if CHROME else {}), args=["--no-sandbox"])
    def pagina_nueva(u, **kw):
        pg = nav.new_page(**({"viewport": {"width": 1280, "height": 900}} | kw))
        pg.on("pageerror", lambda e: errores.append(str(e)))
        pg.on("console", lambda m: m.type == "error" and "ERR_TUNNEL" not in m.text and "Failed to load resource" not in m.text and errores.append(m.text))
        pg.goto(u); pg.wait_for_timeout(500)
        return pg

    pg = pagina_nueva(url)
    check("cuida lo que duele" in pg.inner_text("#app"), "inicio carga")
    pg.click("text=Cómo funciona"); pg.wait_for_timeout(200)
    check("Preguntas frecuentes" in pg.inner_text("#app"), "cómo funciona carga")

    # Sustituir
    pg.click("#tabs a[data-tab=sustituir]"); pg.wait_for_timeout(200)
    check("Hip Thrust" in pg.inner_text("#resultado"), "sustituir: sentadilla·rodilla da Hip Thrust")
    check("mantienen el trabajo de glúteos" in pg.inner_text("#resultado"), "explica qué músculo se mantiene")
    pg.fill("#q", "press de banca"); pg.wait_for_timeout(150)
    check("press de banca" in pg.locator("#sug [role=option]").first.inner_text().lower(), "búsqueda en español devuelve nombre en español")
    pg.keyboard.press("Enter"); pg.wait_for_timeout(250)
    check("Press de banca con barra" in pg.inner_text(".ficha-h"), "elige con Enter")
    pg.click("text=Extensión de piernas · rodilla"); pg.wait_for_timeout(250)
    check("no hay un sustituto seguro" in pg.inner_text("#resultado").lower(), "abstención anatómica")

    # Rutina: validación
    pg.click("#tabs a[data-tab=rutina]"); pg.wait_for_timeout(200)
    pg.click("button[type=submit]"); pg.wait_for_timeout(150)
    check(pg.locator("#f-error").is_visible(), "valida edad vacía")
    pg.fill("#f-edad", "30"); pg.fill("#f-peso", "80"); pg.fill("#f-altura", "175")
    pg.select_option("#f-deporte", "futbol")
    pg.click("[data-campo=dias] [data-v='3']"); pg.click("[data-campo=minutos] [data-v='60']")
    pg.click("[data-multi=equipo] [data-v=pesas]"); pg.click("[data-multi=equipo] [data-v=gym]")
    pg.click("[data-multi=lesiones] [data-v=rodilla]")
    pg.click("button[type=submit]"); pg.wait_for_timeout(300)
    txt = pg.inner_text("#app")
    check("3 días por semana" in txt and "Día 3" in txt, "rutina de 3 días creada")
    check("Para tu deporte" in txt, "incluye complemento de fútbol")
    rutina = pg.evaluate("JSON.parse(localStorage.getItem('sustituto-seguro-v1')).rutina")
    rod = D['articulaciones'].index('rodilla')
    usados = [e['i'] for d in rutina['dias'] for e in d['ejercicios']]
    check(all(rod not in D['ejercicios'][i][5] for i in usados), f"ningún ejercicio carga la rodilla ({len(usados)} ejercicios)")
    check(all(len(d['ejercicios']) >= 5 for d in rutina['dias']), "cada día tiene al menos 5 ejercicios")
    check(not pg.locator("#ia-card").is_visible(), "sin IA: la tarjeta de IA está oculta")
    pg.locator("[data-cambiar]").first.click(); pg.wait_for_timeout(150)
    r2 = pg.evaluate("JSON.parse(localStorage.getItem('sustituto-seguro-v1')).rutina")
    check(r2['dias'][0]['ejercicios'][0]['i'] != rutina['dias'][0]['ejercicios'][0]['i'], "cambiar ejercicio funciona")

    # Barrido de seguridad del generador: 300 perfiles al azar
    random.seed(7)
    malos, vacios = 0, 0
    lesiones = ['hombro', 'codo', 'muneca', 'cuello', 'lumbar', 'cadera', 'rodilla', 'tobillo']
    perfiles = [{"edad": random.choice([12, 16, 25, 40, 65, 80]), "peso": random.choice(["", 60, 120]), "altura": random.choice(["", 170]),
                 "deporte": random.choice(['ninguno', 'futbol', 'running', 'natacion', 'combate', 'escalada', 'basquet']),
                 "nivel": random.choice(['principiante', 'intermedio', 'avanzado']), "objetivo": random.choice(['salud', 'fuerza', 'musculo', 'resistencia']),
                 "dias": random.choice([2, 3, 4, 5, 6]), "minutos": random.choice([30, 45, 60, 75]),
                 "equipo": random.sample(['pesas', 'gym', 'banda', 'balon'], random.randint(0, 3)),
                 "lesiones": random.sample(lesiones, random.randint(0, 3))} for _ in range(300)]
    LES = {'hombro': ['hombro', 'escapulotorácica'], 'codo': ['codo', 'radiocubital'], 'muneca': ['muñeca', 'muñeca/dedos', 'radiocubital'],
           'cuello': ['columna (cervical)'], 'lumbar': ['columna (lumbar)', 'columna (torácico-lumbar)'], 'cadera': ['cadera'],
           'rodilla': ['rodilla'], 'tobillo': ['tobillo', 'pie (intertarsal)']}
    for k, pf in enumerate(perfiles):
        pg.evaluate("p => localStorage.setItem('sustituto-seguro-v1', JSON.stringify({perfil: p, rutina: null, sesiones: [], pesos: []}))", pf)
        pg.reload(); pg.wait_for_timeout(30)
        pg.evaluate("location.hash = 'rutina'"); pg.wait_for_timeout(30)
        # rellenar desde perfil guardado y enviar
        pg.fill("#f-edad", str(pf['edad']))
        pg.click("button[type=submit]"); pg.wait_for_timeout(30)
        r = pg.evaluate("JSON.parse(localStorage.getItem('sustituto-seguro-v1')).rutina")
        vet = {D['articulaciones'].index(a) for l in pf['lesiones'] for a in LES[l]}
        for d in r['dias']:
            if not d['ejercicios']: vacios += 1  # ya no deberían existir
            for e in d['ejercicios']:
                if vet & set(D['ejercicios'][e['i']][5]): malos += 1
                if pf['edad'] >= 60 and e['grupo'] == 'potencia': malos += 1
                if not pf['equipo'] and D['equipos'][D['ejercicios'][e['i']][3]] not in ('peso corporal',) and e['i'] not in (1920, 2159): malos += 1
                for a in e['alternativas']:
                    if vet & set(D['ejercicios'][a][5]): malos += 1
    check(malos == 0, f"300 perfiles al azar: 0 ejercicios o alternativas que carguen lesiones (encontrados {malos})")
    print("     días sin ejercicios:", vacios)

    # Entrenar → progreso
    pg.evaluate("localStorage.clear()"); pg.goto(url + "#rutina"); pg.reload(); pg.wait_for_timeout(300)
    pg.fill("#f-edad", "35"); pg.click("[data-multi=lesiones] [data-v=hombro]"); pg.click("[data-multi=equipo] [data-v=pesas]")
    pg.click("button[type=submit]"); pg.wait_for_timeout(200)
    pg.click("text=Empezar a entrenar"); pg.wait_for_timeout(200)
    check("Te toca" in pg.inner_text("#app"), "sugiere el día que toca")
    pg.locator("[data-dia='0']").click(); pg.wait_for_timeout(200)
    kg = pg.locator("[data-kg]:not([disabled])").first
    if kg.count(): kg.fill("12"); kg.dispatch_event("change")
    pg.locator("[data-hecho]").first.click(); pg.wait_for_timeout(200)
    check(pg.locator("#timer").is_visible() and "Descanso" in pg.inner_text("#timer"), "cronómetro de descanso aparece")
    pg.locator("#timer button", has_text="Saltar").click(); pg.wait_for_timeout(100)
    check(pg.locator("#timer").is_hidden(), "saltar descanso")
    pg.locator("[data-hecho]").nth(1).click(); pg.wait_for_timeout(100)
    pg.reload(); pg.wait_for_timeout(400)
    check(pg.locator("[data-hecho][aria-pressed=true]").count() == 2, "la sesión en curso sobrevive a recargar")
    pg.click("#terminar"); pg.wait_for_timeout(200)
    pg.locator("[data-dolor=hombro]").fill("8"); pg.wait_for_timeout(100)
    check("Dolor alto" in pg.inner_text("#alerta-dolor"), "alerta de dolor alto")
    pg.click("#guardar-ses"); pg.wait_for_timeout(400)
    t = pg.inner_text("#app")
    check("Tu avance" in t and "dolor alto" in t.lower(), "progreso muestra sesión y alerta de dolor")
    check(pg.locator("svg.graf").count() >= 2, "gráficos de progreso")
    pg.fill("#p-kg", "72.5"); pg.click("#f-pesaje button"); pg.wait_for_timeout(100)
    pg.locator("[data-borrar]").first.click(); pg.wait_for_timeout(100)
    pg.locator("[data-borrar-no]").click(); pg.wait_for_timeout(100)
    check(pg.evaluate("JSON.parse(localStorage.getItem('sustituto-seguro-v1')).sesiones.length") == 1, "cancelar borrado no borra")
    pg.screenshot(path=str(carpeta / "progreso.png"), full_page=True)
    pg.goto(url + "#rutina"); pg.wait_for_timeout(300)
    pg.screenshot(path=str(carpeta / "rutina.png"), full_page=True)
    pg.goto(url + "#inicio"); pg.wait_for_timeout(300)
    pg.screenshot(path=str(carpeta / "inicio.png"))

    # IA simulada
    pi = pagina_nueva(url_ia)
    pi.evaluate("localStorage.clear()"); pi.goto(url_ia + "#rutina"); pi.reload(); pi.wait_for_timeout(400)
    pi.fill("#f-edad", "28"); pi.click("[data-multi=equipo] [data-v=pesas]"); pi.click("[data-multi=equipo] [data-v=gym]")
    pi.click("button[type=submit]"); pi.wait_for_timeout(400)
    check(pi.locator("#ia-card").is_visible(), "con IA: aparece la tarjeta")
    antes = pi.evaluate("JSON.parse(localStorage.getItem('sustituto-seguro-v1')).rutina.dias[0].ejercicios.map(e=>e.i)")
    pi.click("#ia-mejorar"); pi.wait_for_timeout(400)
    t = pi.inner_text("#ia-res")
    check("Rutina equilibrada" in t and "INSEGURO" not in t, "IA: muestra resumen y descarta el cambio no permitido")
    pi.click("#ia-aplicar"); pi.wait_for_timeout(300)
    despues = pi.evaluate("JSON.parse(localStorage.getItem('sustituto-seguro-v1')).rutina.dias[0].ejercicios.map(e=>e.i)")
    check(despues[0] != antes[0] and despues[1] == antes[1], "IA: aplica solo el cambio válido")
    pi.fill("#ia-pregunta", "¿cómo caliento?"); pi.click("#ia-preguntar"); pi.wait_for_timeout(300)
    check("Calienta" in pi.inner_text("#ia-resp"), "IA: responde preguntas")
    pi.screenshot(path=str(carpeta / "rutina_ia.png"))

    # Móvil
    m = pagina_nueva(url, viewport={"width": 390, "height": 844}, is_mobile=True, has_touch=True)
    for v in ["inicio", "sustituir", "rutina", "entrenar", "progreso", "como-funciona"]:
        m.goto(url + "#" + v); m.wait_for_timeout(250)
        w = m.evaluate("document.documentElement.scrollWidth")
        check(w <= 390, f"móvil sin scroll horizontal en #{v} ({w}px)")
    m.evaluate("localStorage.clear()"); m.goto(url + "#rutina"); m.reload(); m.wait_for_timeout(300)
    m.fill("#f-edad", "67"); m.tap("button[type=submit]"); m.wait_for_timeout(300)
    check("60 años o más" in m.inner_text("#app"), "aviso de edad 60+")
    check(m.evaluate("document.documentElement.scrollWidth") <= 390, "móvil: rutina sin scroll horizontal")
    m.screenshot(path=str(carpeta / "movil_rutina.png"), full_page=True)
    m.goto(url + "#sustituir"); m.wait_for_timeout(300)
    m.screenshot(path=str(carpeta / "movil_sustituir.png"))
    nav.close()

check(not errores, f"sin errores de JavaScript {errores[:3]}")
print("\nFALLAS:", len(fallos))

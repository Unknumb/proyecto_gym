"""Pruebas de usuario de la página web en un navegador real (README §15).

Recorren todas las secciones en escritorio y móvil, con la IA simulada, y generan
300 rutinas con perfiles y tipos de rutina al azar para verificar que ninguna
incluya un ejercicio que cargue una zona lesionada. También prueban «Qué trabaja»,
los tipos de rutina, los aportes para el modelo (corregir, calificar, descargar e
importar) y la vista «El modelo por dentro» (README §18).

Requiere Playwright y Chrome/Chromium::

    pip install playwright && playwright install chromium
    python web/probar_web.py              # capturas en web/capturas/
"""
import json
import random
import re
import sys
import tempfile
from pathlib import Path

from playwright.sync_api import sync_playwright

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "scripts"))
import importar_aportes  # noqa: E402
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

# 0. Los ejercicios curados (por ejercicio_id) apuntan al ejercicio esperado
POS = {x: k for k, x in enumerate(D['ids'])}
esperado = {'free_exercise_db:Bodyweight_Squat': 'bodyweight squat', 'free_exercise_db:Leg_Press': 'leg press', 'catalogo:0025': 'barbell bench press', 'catalogo:0652': 'pull-up', 'free_exercise_db:Plank': 'plank', 'catalogo:1459': 'dumbbell romanian deadlift', 'catalogo:0861': 'cable seated row', 'catalogo:0334': 'dumbbell lateral raise', 'wger:1314': 'jumping jack hd', 'wger:265': 'glute bridge'}
check(all(D['ejercicios'][POS[x]][0].lower() == n for x, n in esperado.items()), "ejercicios curados apuntan a los ejercicios correctos")
cur_ids = re.findall(r"\['([a-z_]+:[^']+)', '", fuente.split('const CURADOS = [')[1].split('];')[0])
ids = [POS.get(x, -1) for x in cur_ids]
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

    # Qué trabaja: músculos, articulaciones con sus ligamentos y perfil del modelo
    pg.click("text=Sentadilla · rodilla"); pg.wait_for_timeout(250)
    pg.click("#ficha-original summary"); pg.wait_for_timeout(100)
    ficha = pg.inner_text("#ficha-original")
    check("Músculos de apoyo".upper() in ficha.upper() and "LCA" in ficha and "Mecánica:" in ficha, "qué trabaja: músculos de apoyo, ligamentos y perfil")
    check(pg.locator("#ficha-original .perfil-mov .pill").count() == 3, "qué trabaja: origen de cada etiqueta")
    pg.locator("#resultado details.mini summary").first.click(); pg.wait_for_timeout(100)
    check("No carga rodilla" in pg.locator("#resultado details.mini").first.inner_text(), "alternativa: explica por qué es segura")

    # Aportes: corregir una etiqueta y calificar alternativas
    pg.evaluate("localStorage.removeItem('sustituto-seguro-v1')"); pg.reload(); pg.wait_for_timeout(300)
    pg.click("#ficha-original summary"); pg.click("#ficha-original details.corregir summary")
    pg.click("#ficha-original [data-guardar-corr]"); pg.wait_for_timeout(50)
    check("Marca al menos" in pg.inner_text("#ficha-original [data-msg-corr]"), "corregir: pide marcar algo")
    pg.click("#ficha-original [data-corr-obj=mecanica] [data-v='1']"); pg.click("#ficha-original [data-guardar-corr]"); pg.wait_for_timeout(100)
    check("Tu corrección" in pg.inner_text("#ficha-original .perfil-mov"), "corregir: muestra la corrección")
    pg.click("#ficha-original [data-corr-obj=mecanica] [data-v='0']"); pg.click("#ficha-original [data-guardar-corr]"); pg.wait_for_timeout(100)
    ap = pg.evaluate("JSON.parse(localStorage.getItem('sustituto-seguro-v1')).aportes")
    check(len(ap["correcciones"]) == 1 and ap["correcciones"][0]["k"] == 0 and ap["correcciones"][0]["id"] == "catalogo:0043",
          "corregir otra vez reemplaza el voto (por ejercicio_id)")
    pg.locator("#resultado [data-veredicto=bueno]").first.click(); pg.locator("#resultado [data-veredicto=inseguro]").nth(1).click()
    pg.locator("#resultado [data-veredicto=inseguro]").nth(1).click(); pg.wait_for_timeout(100)
    ap = pg.evaluate("JSON.parse(localStorage.getItem('sustituto-seguro-v1')).aportes")
    check([r["v"] for r in ap["revisiones"]] == ["bueno"] and ap["revisiones"][0]["art"] == "rodilla", "calificar: guarda y se puede desmarcar")
    pg.goto(url + "#aportes"); pg.wait_for_timeout(300)
    check("Sentadilla profunda con barra" in pg.inner_text("#app") and "Buena" in pg.inner_text("#app"), "enseña al modelo: lista los aportes")
    with pg.expect_download() as descarga:
        pg.click("#descargar-aportes")
    with tempfile.TemporaryDirectory() as tmp:
        archivo = Path(tmp) / "aportes.json"
        descarga.value.save_as(archivo)
        contenido = json.loads(archivo.read_text(encoding="utf-8"))
        tablas = contenido["tablas"]
        check(contenido["formato"] == importar_aportes.FORMATO_WEB
              and all(list(f) == importar_aportes.COLUMNAS[t] for t, filas in tablas.items() for f in filas),
              "descarga: mismo formato y columnas que la app de iPhone")
        nuevas = importar_aportes.importar(archivo, Path(tmp) / "destino")
        check(nuevas == {"correcciones_etiquetas": 1, "revisiones_recomendaciones": 1}, "descarga: el importador la acepta")
        check("edad" not in json.dumps(contenido) and "dolor" not in json.dumps(contenido),
              "descarga: no incluye edad ni dolor")

    # El modelo por dentro (EV3)
    pg.click("#tabs a[data-tab=modelo]"); pg.wait_for_timeout(300)
    t = pg.inner_text("#app")
    check("Siete candidatos" in t and "Decisión: mantenemos" in t and "aprende de los usuarios" in t.lower(), "el modelo: candidatos, regla y aprendizaje")
    check(pg.locator("svg.graf").first.locator("circle").count() == 7, "el modelo: 7 candidatos en el gráfico")
    datos = pg.inner_text("#datos-modelo")
    check("kappa −0,01" in datos and "788 grupos" in datos and "2.592 ejercicios" in datos, "el modelo: datos, EDA y preparación")
    check(pg.locator("#que-predice .modelo-fila").count() == 3 and "Estático 12 %" in pg.inner_text("#que-predice"), "el modelo: qué predice cada modelo y sus clases")
    pg.click("[data-obj-modelo=fuerza]"); pg.wait_for_timeout(150)
    check("random forest" in pg.inner_text("#app").lower(), "el modelo: cambia de objetivo")
    pg.screenshot(path=str(carpeta / "modelo.png"), full_page=True)

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
                 "tipo": random.choice(['auto', 'full', 'tp', 'epp', 'arnold', 'weider']),
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
                if not pf['equipo'] and D['equipos'][D['ejercicios'][e['i']][3]] not in ('peso corporal',) and e['i'] not in (POS['wger:265'], POS['wger:1408']): malos += 1
                for a in e['alternativas']:
                    if vet & set(D['ejercicios'][a][5]): malos += 1
    check(malos == 0, f"300 perfiles y tipos de rutina al azar: 0 ejercicios o alternativas que carguen lesiones (encontrados {malos})")

    # Tipos de rutina
    pg.evaluate("localStorage.clear()"); pg.goto(url + "#rutina"); pg.reload(); pg.wait_for_timeout(300)
    pg.fill("#f-edad", "25"); pg.click("[data-campo=nivel] [data-v=intermedio]"); pg.click("[data-campo=dias] [data-v='3']")
    check("Push / Pull / Legs" in pg.inner_text("#tipo-ayuda"), "tipo: 3 días con experiencia recomienda Push / Pull / Legs")
    pg.click("[data-campo=nivel] [data-v=principiante]")
    check("Cuerpo completo" in pg.inner_text("#tipo-ayuda"), "tipo: 3 días principiante recomienda cuerpo completo")
    pg.click("[data-campo=tipo] [data-v=epp]"); pg.click("button[type=submit]"); pg.wait_for_timeout(300)
    check(pg.locator(".dia-h h3").all_inner_texts() == ["Día 1 · Empuje", "Día 2 · Tracción", "Día 3 · Piernas"], "tipo: la persona elige Push / Pull / Legs")
    pg.click("[data-tipo=arnold]"); pg.wait_for_timeout(200)
    check(pg.locator(".dia-h h3").all_inner_texts()[1] == "Día 2 · Hombros y brazos", "tipo: cambio rápido a Arnold")
    check(pg.locator("[data-tipo=weider]").is_disabled(), "tipo: un grupo por día necesita 5 días")
    pg.locator(".ej details.mini summary").first.click(); pg.wait_for_timeout(100)
    check("En esta rutina cubre" in pg.locator(".ej details.mini").first.inner_text(), "rutina: qué trabaja cada ejercicio")
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

    # Respaldo: uno legítimo se recupera completo; uno alterado no inserta HTML ni ejercicios inseguros
    pg.goto(url + "#progreso"); pg.wait_for_timeout(300)
    antes_r = pg.evaluate("JSON.parse(localStorage.getItem('sustituto-seguro-v1'))")
    pg.set_input_files("#importar", files=[{"name": "r.json", "mimeType": "application/json", "buffer": json.dumps(antes_r).encode()}])
    pg.wait_for_timeout(300)
    despues_r = pg.evaluate("JSON.parse(localStorage.getItem('sustituto-seguro-v1'))")
    check(despues_r["sesiones"] == antes_r["sesiones"] and despues_r["pesos"] == antes_r["pesos"] and despues_r["perfil"] == antes_r["perfil"]
          and [[e["i"] for e in d["ejercicios"]] for d in despues_r["rutina"]["dias"]] == [[e["i"] for e in d["ejercicios"]] for d in antes_r["rutina"]["dias"]],
          "respaldo legítimo: se recupera sin perder datos")
    xss = '<img src=x onerror="window.__xss=1">'
    hombro = D['articulaciones'].index('hombro')
    inseguro = next(k for k, e in enumerate(D['ejercicios']) if hombro in e[5])
    malo = {"perfil": {"edad": 30, "dias": xss, "minutos": xss, "lesiones": ["hombro"], "equipo": [xss]},
            "rutina": {"creada": xss, "objetivo": "salud", "avisos": [["red", xss, xss]], "huecos": [xss],
                       "perfil": {"edad": 30, "lesiones": ["hombro"], "minutos": xss},
                       "dias": [{"titulo": xss, "ejercicios": [{"i": inseguro, "series": xss, "reps": xss, "descanso": xss, "alternativas": [inseguro]},
                                                               {"i": POS['free_exercise_db:Bodyweight_Squat'], "series": xss, "reps": xss,
                                                                "descanso": xss, "alternativas": [inseguro, xss]}]}]},
            "sesiones": [{"fecha": "2026-01-05", "dia": xss, "titulo": xss, "minutos": xss, "esfuerzo": xss, "notas": xss,
                          "dolor": {"hombro": xss, xss: 3}, "ejercicios": [{"i": xss, "series": []}]}],
            "pesos": [{"fecha": "2026-01-05", "kg": xss}]}
    pg.set_input_files("#importar", files=[{"name": "m.json", "mimeType": "application/json", "buffer": json.dumps(malo).encode()}])
    pg.wait_for_timeout(300)
    check("Respaldo cargado: 1 sesiones" in pg.inner_text("#msg-datos"), "respaldo alterado: se carga lo válido")
    for v in ["progreso", "rutina", "entrenar", "inicio"]:
        pg.goto(url + "#" + v); pg.wait_for_timeout(200)
    r_mala = pg.evaluate("JSON.parse(localStorage.getItem('sustituto-seguro-v1')).rutina")
    usados_m = [e["i"] for d in r_mala["dias"] for e in d["ejercicios"]] + [a for d in r_mala["dias"] for e in d["ejercicios"] for a in e["alternativas"]]
    check(not pg.evaluate("window.__xss") and pg.locator("img[src=x]").count() == 0, "respaldo alterado: no inserta HTML")
    pg.goto(url + "#progreso"); pg.wait_for_timeout(200)
    aportes_malos = dict(malo, aportes={"correcciones": [{"id": xss, "o": "mecanica", "k": 0}, {"id": "catalogo:0043", "o": "toString", "k": 0},
                                                         {"id": "catalogo:0043", "o": "fuerza", "k": 1, "previa": xss, "origen": xss}],
                                        "revisiones": [{"orig": "catalogo:0043", "alt": xss, "art": "rodilla", "perfil": 0, "v": "bueno"}]})
    pg.set_input_files("#importar", files=[{"name": "a.json", "mimeType": "application/json", "buffer": json.dumps(aportes_malos).encode()}])
    pg.wait_for_timeout(300)
    ap = pg.evaluate("JSON.parse(localStorage.getItem('sustituto-seguro-v1')).aportes")
    check(len(ap["correcciones"]) == 1 and ap["correcciones"][0]["previa"] == "" and not ap["revisiones"], "respaldo alterado: solo aportes válidos")
    pg.goto(url + "#aportes"); pg.wait_for_timeout(200)
    check(not pg.evaluate("window.__xss"), "respaldo alterado: aportes no insertan HTML")
    check(inseguro not in usados_m and usados_m, "respaldo alterado: descarta ejercicios que cargan la lesión")
    pg.evaluate("localStorage.clear()")

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
    for v in ["inicio", "sustituir", "rutina", "entrenar", "progreso", "como-funciona", "modelo", "aportes"]:
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

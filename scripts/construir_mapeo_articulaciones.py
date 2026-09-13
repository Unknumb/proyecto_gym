"""
Construcción de la tabla músculo -> articulación (prerrequisito del grafo, §7).

**Por qué esto no es un scraper de ExRx.net.** Se evaluó ExRx.net como fuente
(tabla de referencia citada en el ACSM's Resource Manual), pero:

  1. Devuelve HTTP 403 a cualquier request automatizado, con o sin cabecera de
     navegador (`curl -A "Mozilla/5.0 ..." https://exrx.net/Lists/Articulations`).
  2. Su `robots.txt` deshabilita explícitamente user-agents genéricos
     (`Wget`, `Python-urllib`, `Scrapy`...) y, en una sección aparte, a los
     crawlers de IA con nombre propio — incluido `ClaudeBot`/`anthropic-ai` —
     con `Disallow: /`.

Automatizar la extracción ahí violaría la política que el propio sitio publica,
exactamente lo que este proyecto ya se cuidó de no hacer con los medios de
Gym visual (`NOTICE.md`, README §5.2). La alternativa correcta no es un
scraper más sigiloso: es tratar a ExRx.net como lo que es, una fuente de
**consulta manual en navegador**, citada con URL y fecha — igual que se citaría
un libro.

Este script hace la mitad que sí es automatizable de forma legítima:

  1. Descarga la taxonomía de movimientos articulares desde Wikipedia
     ("List of movements of the human body" — CC BY-SA, `robots.txt`
     permisivo) a ``data/01_raw/movimientos_articulares_wikipedia.csv``.
     Sirve como catálogo de qué tipos de movimiento existen por articulación
     (flexión, extensión, abducción...) para validar la curación manual.

  2. Descarga, por API de Wikipedia (no HTML — el wikitext es más estable
     que el HTML renderizado), el campo `Action` de la ficha anatómica de
     cada músculo (plantilla `Infobox muscle`). Es una **segunda fuente
     independiente** para contrastar el borrador de anatomía estándar antes
     de gastar tiempo verificando en ExRx.net algo que ya coincide en dos
     lugares.

  3. Genera (o refresca, sin pisar filas ya completadas) la plantilla de
     curación manual en ``data/01_raw/musculo_articulacion_manual.csv``, con
     una fila por cada valor único de músculo que aparece en el catálogo
     propio (``target`` + ``secondary_muscles``) — para que el join contra
     ``exercises_clean.parquet`` sea exacto por construcción.

La persona a cargo compara `articulacion_borrador` (mi conocimiento de
kinesiología) contra `contraste_wikipedia` (la ficha anatómica real) y, solo
cuando ambas coincidan con lo que ve en ExRx.net, llena
`articulacion_principal`, `movimiento`, `fuente` y `fecha_consulta` a mano.

Uso:
    python scripts/construir_mapeo_articulaciones.py
"""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date
from io import StringIO
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
ORIGEN_CATALOGO = RAIZ / "data" / "01_raw" / "exercises.json"
DESTINO_MOVIMIENTOS = RAIZ / "data" / "01_raw" / "movimientos_articulares_wikipedia.csv"
DESTINO_PLANTILLA = RAIZ / "data" / "01_raw" / "musculo_articulacion_manual.csv"

WIKIPEDIA_URL = "https://en.wikipedia.org/wiki/List_of_movements_of_the_human_body"
WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "proyecto-gym-biomecanico/EV1 (uso educativo, Duoc UC MLY1101)"

#: Título del artículo de Wikipedia con ficha `Infobox muscle` para cada
#: músculo normalizado — de ahí sale `contraste_wikipedia`. No todos los
#: músculos del catálogo tienen artículo propio (p. ej. "grip muscles" es
#: demasiado genérico); esos simplemente no tienen entrada acá y el
#: contraste queda vacío, no se fuerza un título aproximado.
TITULO_WIKIPEDIA = {
    "abdominals": "Rectus abdominis muscle",
    "abductors": "Gluteus medius",
    "adductors": "Adductor muscles of the hip",
    "biceps": "Biceps brachii muscle",
    "brachialis": "Brachialis muscle",
    "calves": "Gastrocnemius muscle",
    "deltoids": "Deltoid muscle",
    "glutes": "Gluteus maximus",
    "groin": "Adductor muscles of the hip",
    "hamstrings": "Hamstring",
    "hip flexors": "Iliopsoas",
    "inner thighs": "Adductor muscles of the hip",
    "latissimus dorsi": "Latissimus dorsi muscle",
    "levator scapulae": "Levator scapulae muscle",
    "lower abs": "Rectus abdominis muscle",
    "lower back": "Erector spinae muscles",
    "obliques": "External abdominal oblique muscle",
    "pectorals": "Pectoralis major muscle",
    "quadriceps": "Quadriceps femoris muscle",
    "rear deltoids": "Deltoid muscle",
    "rhomboids": "Rhomboid muscles",
    "rotator cuff": "Rotator cuff",
    "serratus anterior": "Serratus anterior muscle",
    "shins": "Tibialis anterior muscle",
    "soleus": "Soleus muscle",
    "sternocleidomastoid": "Sternocleidomastoid muscle",
    "trapezius": "Trapezius muscle",
    "triceps": "Triceps brachii muscle",
    "upper chest": "Pectoralis major muscle",
    "wrist extensors": "Extensor carpi radialis longus muscle",
    "wrist flexors": "Flexor carpi radialis muscle",
}

#: Sinónimos evidentes dentro del propio catálogo: `target` usa la forma corta
#: ("abs", "quads"), `secondary_muscles` a veces usa la forma larga
#: ("abdominals", "quadriceps"). Sin esto, el join produciría dos filas
#: distintas para el mismo músculo real.
SINONIMOS = {
    "abs": "abdominals",
    "quads": "quadriceps",
    "delts": "deltoids",
    "traps": "trapezius",
    "lats": "latissimus dorsi",
}

#: Borrador de anatomía estándar (articulación, movimiento) por músculo
#: normalizado — NO es una transcripción de ExRx.net, es conocimiento de
#: kinesiología de libro de texto, para que la curación no empiece en blanco.
#: **No copiar directo a las columnas humanas.** Cada fila debe verificarse
#: contra ExRx.net (u otra fuente) y recién ahí completar
#: `articulacion_principal` / `movimiento` / `fuente` / `fecha_consulta` con
#: la fecha real de esa verificación — de lo contrario la cita sería falsa.
BORRADOR_ANATOMIA: dict[str, tuple[str, str]] = {
    "abdominals": ("columna vertebral (lumbar)", "flexión del tronco"),
    "abductors": ("cadera", "abducción de cadera"),
    "adductors": ("cadera", "aducción de cadera"),
    "ankle stabilizers": ("tobillo", "inversión/eversión — estabilización"),
    "biceps": ("codo", "flexión de codo (+ supinación del antebrazo)"),
    "brachialis": ("codo", "flexión de codo"),
    "calves": ("tobillo y rodilla (biarticular)", "flexión plantar de tobillo + flexión de rodilla (gastrocnemio)"),
    "deltoids": ("hombro", "abducción (fascículo lateral) — varía por fascículo, ver notas"),
    "forearms": ("muñeca", "flexión/extensión de muñeca y dedos (grupo general)"),
    "glutes": ("cadera", "extensión de cadera"),
    "grip muscles": ("mano / dedos", "flexión de dedos (agarre)"),
    "groin": ("cadera", "aducción de cadera (sinónimo informal de adductors)"),
    "hamstrings": ("cadera y rodilla (biarticular)", "extensión de cadera + flexión de rodilla"),
    "hip flexors": ("cadera", "flexión de cadera"),
    "inner thighs": ("cadera", "aducción de cadera (sinónimo informal de adductors)"),
    "latissimus dorsi": ("hombro", "extensión, aducción y rotación interna de hombro"),
    "levator scapulae": ("escapulotorácica", "elevación de la escápula"),
    "lower abs": ("columna vertebral (lumbar)", "flexión del tronco (énfasis recto abdominal inferior)"),
    "lower back": ("columna vertebral (lumbar)", "extensión de columna"),
    "obliques": ("columna vertebral (torácico-lumbar)", "flexión lateral y rotación del tronco"),
    "pectorals": ("hombro", "aducción horizontal y flexión de hombro"),
    "quadriceps": ("rodilla", "extensión de rodilla (rectus femoris también flexiona cadera — biarticular)"),
    "rear deltoids": ("hombro", "extensión y rotación externa de hombro"),
    "rhomboids": ("escapulotorácica", "retracción (aducción) de la escápula"),
    "rotator cuff": ("hombro", "rotación externa/interna y estabilización dinámica"),
    "serratus anterior": ("escapulotorácica", "protracción y rotación superior de la escápula"),
    "shins": ("tobillo", "dorsiflexión (tibial anterior)"),
    "soleus": ("tobillo", "flexión plantar (monoarticular, a diferencia del gastrocnemio)"),
    "sternocleidomastoid": ("cuello (columna cervical)", "flexión cervical bilateral / rotación y flexión lateral unilateral"),
    "trapezius": ("escapulotorácica", "superior=elevación, medio=retracción, inferior=depresión"),
    "triceps": ("codo", "extensión de codo"),
    "upper back": ("escapulotorácica", "retracción/estabilización escapular (región: trapecio + romboides)"),
    "upper chest": ("hombro", "flexión de hombro (fascículo clavicular del pectoral mayor)"),
    "wrist extensors": ("muñeca", "extensión de muñeca"),
    "wrist flexors": ("muñeca", "flexión de muñeca"),
}

#: Valores que aparecen en `target`/`secondary_muscles` pero que NO son un
#: músculo discreto — son una región (varios músculos, sin una articulación
#: única) o un sistema no musculoesquelético. Forzarlos a una sola
#: articulación sería inventar precisión que la fuente no tiene.
NO_ES_MUSCULO_DISCRETO = {
    "cardiovascular system": "sistema, no músculo — excluir del mapeo",
    "core": "región compuesta (recto abdominal, oblicuos, transverso...)",
    "back": "región compuesta — ver lats/traps/rhomboids/lower back por separado",
    "chest": "región compuesta — ver pectorals/upper chest",
    "shoulders": "región compuesta — ver delts/rotator cuff",
    "ankles": "articulación, no músculo",
    "feet": "región, no músculo",
    "hands": "región, no músculo",
    "wrists": "articulación, no músculo",
    "spine": "conjunto de articulaciones, no un músculo",
}


def normalizar(musculo: str) -> str:
    """Aplica los sinónimos conocidos; deja el resto sin cambio."""
    return SINONIMOS.get(musculo, musculo)


def musculos_unicos_del_catalogo() -> list[str]:
    """Valores únicos de `target` + `secondary_muscles` tal como aparecen en
    el catálogo — es la llave de join contra `exercises_clean.parquet`, así
    que se preserva la forma cruda, no la normalizada."""
    registros = json.loads(ORIGEN_CATALOGO.read_text())
    if isinstance(registros, dict):
        registros = list(registros.values())

    vistos: set[str] = set()
    for r in registros:
        vistos.add(r.get("target", ""))
        vistos.update(r.get("secondary_muscles", []) or [])
    vistos.discard("")
    return sorted(vistos)


#: Encabezados que preceden contenido irrelevante (índice, bibliografía) —
#: cualquier tabla que caiga bajo estas secciones se descarta.
SECCIONES_IGNORADAS = {None, "Contents", "References"}


def descargar_movimientos_wikipedia() -> pd.DataFrame | None:
    """Descarga y concatena las tablas de la página de Wikipedia, una por
    articulación/sección.

    Fuente elegida porque es automatizable de forma legítima: `robots.txt`
    de Wikipedia no bloquea agentes genéricos, y el contenido es CC BY-SA.

    La página tiene una tabla distinta por articulación (Shoulder, Elbow,
    Spine, Knees...); tomar solo "la tabla más larga" —el enfoque ingenuo—
    descarta silenciosamente el resto. Aquí se parte el HTML por encabezado
    (`<h2>`/`<h3>`/`<h4>`) y se etiqueta cada tabla con la sección en la que
    aparece, para no perder ninguna articulación.
    """
    request = urllib.request.Request(WIKIPEDIA_URL, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=30) as resp:
            html = resp.read().decode("utf-8")
    except (urllib.error.URLError, OSError) as exc:
        print(f"  no se pudo descargar Wikipedia: {exc}")
        print("  (revisa conexión; Databricks serverless no expone red externa,")
        print("   igual que auditar_media_fisica.py, este script corre local)")
        return None

    partes = re.split(r"(<h[234][^>]*>.*?</h[234]>)", html, flags=re.S)
    seccion_actual: str | None = None
    tablas_por_seccion = []
    for parte in partes:
        if re.match(r"^<h[234]", parte):
            seccion_actual = re.sub(r"<[^>]+>", "", parte).strip()
            continue
        if seccion_actual in SECCIONES_IGNORADAS:
            continue
        try:
            # flavor="lxml" explícito: sin esto, pandas reintenta con
            # html5lib (no instalado) en fragmentos que lxml no puede parsear
            # solo, y el error real queda enmascarado.
            tablas = pd.read_html(StringIO(parte), flavor="lxml")
        except ValueError:
            continue  # fragmento sin tablas, normal entre encabezados
        for t in tablas:
            if len(t) < 1 or t.shape[1] < 2:
                continue  # tablas de layout sin contenido tabular real
            t["articulacion_seccion"] = seccion_actual
            tablas_por_seccion.append(t)

    if not tablas_por_seccion:
        print("  no se encontraron tablas — revisar si Wikipedia cambió el formato")
        return None

    resultado = pd.concat(tablas_por_seccion, ignore_index=True)
    resultado.to_csv(DESTINO_MOVIMIENTOS, index=False)
    print(f"  {len(resultado)} filas en {resultado['articulacion_seccion'].nunique()} "
          f"secciones -> {DESTINO_MOVIMIENTOS.relative_to(RAIZ)}")
    print(f"  secciones: {', '.join(sorted(resultado['articulacion_seccion'].unique()))}")
    return resultado


def _wikitext(titulo: str) -> str:
    """Wikitext crudo de un artículo, vía API de Wikipedia (no HTML).

    Se prefiere sobre parsear el HTML renderizado: los nombres de clase CSS
    del infobox cambian con el skin y ya rompieron un intento anterior
    (`ca-view-more`, menú de UI, no el infobox). El wikitext expone el
    template `{{Infobox muscle | Action = ... }}` de forma literal.
    """
    params = urllib.parse.urlencode({
        "action": "query", "titles": titulo, "prop": "revisions",
        "rvprop": "content", "rvslots": "main", "format": "json", "redirects": 1,
    })
    request = urllib.request.Request(f"{WIKIPEDIA_API}?{params}",
                                      headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=20) as resp:
        datos = json.load(resp)
    paginas = datos.get("query", {}).get("pages", {})
    pagina = next(iter(paginas.values()), {})
    revisiones = pagina.get("revisions", [{}])
    return revisiones[0].get("slots", {}).get("main", {}).get("*", "")


def _campo_action(wikitext: str) -> str:
    """Extrae el valor de `| Action = ...` del template `Infobox muscle`,
    limpiando el markup wiki (enlaces `[[...]]`, referencias `<ref>`, listas).

    El campo suele ser multilínea (un músculo con varias cabezas, cada una
    con su propia acción) — capturar solo "el resto de esa línea" (como en
    el primer intento) truncaba `pectorals`/`upper chest` a la mitad. Se
    captura hasta el siguiente parámetro (`\\n|`) o el cierre del template
    (`}}`).
    """
    m = re.search(r"\|\s*[Aa]ctions?\s*=\s*(.*?)(?=\n\s*\||\n\}\})", wikitext, re.S)
    if not m:
        return ""
    texto = m.group(1)
    texto = re.sub(r"<ref[^>]*?/>", "", texto)
    texto = re.sub(r"<ref[^>]*?>.*?</ref>", "", texto, flags=re.S)
    texto = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]", r"\1", texto)  # [[a|b]] -> b
    texto = re.sub(r"<br\s*/?>", "; ", texto, flags=re.I)
    texto = re.sub(r"[{}*']", "", texto)
    texto = re.sub(r"\s+", " ", texto)
    return texto.strip()[:300]


def _wikitext_con_reintentos(titulo: str, intentos: int = 4) -> str:
    """Como `_wikitext`, pero reintenta con espera creciente ante HTTP 429.

    La API de Wikipedia empezó a devolver 429 a partir de la ~10ª request
    seguida con solo 0.3 s de por medio — el límite de cortesía anónimo es
    más estricto que eso. Se respeta `Retry-After` si el servidor lo manda.
    """
    for intento in range(intentos):
        try:
            return _wikitext(titulo)
        except urllib.error.HTTPError as exc:
            if exc.code != 429 or intento == intentos - 1:
                raise
            espera = int(exc.headers.get("Retry-After", 5)) * (intento + 1)
            time.sleep(espera)
    return ""  # inalcanzable, calma a los type-checkers


def descargar_contraste_wikipedia() -> dict[str, str]:
    """Ficha anatómica (`Action`) por músculo normalizado — la segunda
    fuente independiente para contrastar `BORRADOR_ANATOMIA` antes de abrir
    ExRx.net. Si un músculo no está en `TITULO_WIKIPEDIA`, o la descarga
    falla, simplemente no aparece en el resultado (no se fuerza un intento
    aproximado que podría traer el músculo equivocado).
    """
    contrastes: dict[str, str] = {}
    for normalizado, titulo in TITULO_WIKIPEDIA.items():
        try:
            texto = _campo_action(_wikitext_con_reintentos(titulo))
        except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
            print(f"  {normalizado} ({titulo}): error — {exc}")
            continue
        contrastes[normalizado] = texto if texto else "sin campo 'Action' en el infobox"
        time.sleep(1.0)  # cortesía con la API — más lento, pero sin 429
    return contrastes


def generar_plantilla_curacion(musculos: list[str], contrastes: dict[str, str]) -> None:
    """Crea o refresca la plantilla de curación manual.

    Si ya existe una plantilla con filas completadas, se preservan los campos
    que la persona llena a mano (`articulacion_principal`, `movimiento`,
    `fuente`, `url_fuente`, `fecha_consulta`, `notas`). Las columnas
    "_borrador" y `tipo`/`musculo_normalizado` se recalculan siempre —no las
    edita nadie a mano, así que mejoras futuras a `BORRADOR_ANATOMIA` se
    propagan sin perder el trabajo ya hecho.
    """
    columnas = [
        "musculo_crudo", "musculo_normalizado", "tipo",
        "articulacion_borrador", "movimiento_borrador", "contraste_wikipedia",
        "articulacion_principal", "movimiento", "fuente", "url_fuente",
        "fecha_consulta", "notas",
    ]
    columnas_humanas = [
        "articulacion_principal", "movimiento", "fuente", "url_fuente",
        "fecha_consulta", "notas",
    ]

    # dtype=str + keep_default_na=False: sin esto, pandas relee las celdas
    # vacías como NaN (float) en lugar de "", y la comparación de más abajo
    # (`== ""`) deja de detectar filas pendientes después del primer ciclo
    # de guardado — reporta "0 pendientes" aunque nada esté completado.
    previa = (
        pd.read_csv(DESTINO_PLANTILLA, dtype=str, keep_default_na=False)
        if DESTINO_PLANTILLA.exists() else None
    )
    completadas = (
        {row["musculo_crudo"]: row for _, row in previa.iterrows()}
        if previa is not None else {}
    )

    filas = []
    for musculo in musculos:
        normalizado = normalizar(musculo)
        es_revisar = musculo in NO_ES_MUSCULO_DISCRETO
        borrador_articulacion, borrador_movimiento = BORRADOR_ANATOMIA.get(
            normalizado, ("", "")
        )
        fila = {
            "musculo_crudo": musculo,
            "musculo_normalizado": normalizado,
            "contraste_wikipedia": "" if es_revisar else contrastes.get(normalizado, ""),
            "tipo": "revisar — región o sistema, no músculo discreto"
                    if es_revisar else "musculo",
            "articulacion_borrador": "" if es_revisar else borrador_articulacion,
            "movimiento_borrador": "" if es_revisar else borrador_movimiento,
        }
        if musculo in completadas:
            # Preserva solo lo que la persona pudo haber editado a mano;
            # lo derivado (arriba) siempre se recalcula fresco.
            previa_fila = completadas[musculo]
            for col in columnas_humanas:
                fila[col] = previa_fila.get(col, "")
            filas.append(fila)
            continue
        fila.update({
            "articulacion_principal": "",
            "movimiento": "",
            "fuente": "",
            "url_fuente": "",
            "fecha_consulta": "",
            "notas": NO_ES_MUSCULO_DISCRETO.get(musculo, ""),
        })
        filas.append(fila)

    tabla = pd.DataFrame(filas, columns=columnas)
    DESTINO_PLANTILLA.parent.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(DESTINO_PLANTILLA, index=False)

    pendientes = (tabla["articulacion_principal"] == "").sum()
    con_borrador = (tabla["articulacion_borrador"] != "").sum()
    print(f"  {len(tabla)} músculos únicos -> {DESTINO_PLANTILLA.relative_to(RAIZ)}")
    print(f"  {pendientes} filas pendientes de verificar y copiar a las columnas humanas")
    print(f"  {con_borrador} con borrador de anatomía estándar ya sugerido")
    print(f"  {sum(t.startswith('revisar') for t in tabla['tipo'])} marcadas "
          f"'revisar' — regiones/sistemas, no músculos discretos, sin borrador")


def main() -> None:
    print(f"Fecha de referencia para citar fuentes: {date.today().isoformat()}\n")

    print("1. Movimientos articulares (Wikipedia)")
    descargar_movimientos_wikipedia()

    print("\n2. Contraste independiente por músculo (fichas anatómicas de Wikipedia)")
    contrastes = descargar_contraste_wikipedia()
    print(f"  {len(contrastes)}/{len(TITULO_WIKIPEDIA)} fichas descargadas")

    print("\n3. Plantilla de curación músculo -> articulación")
    musculos = musculos_unicos_del_catalogo()
    generar_plantilla_curacion(musculos, contrastes)

    print(f"\nSiguiente paso: abrir {DESTINO_PLANTILLA.relative_to(RAIZ)}. Compara "
          "'articulacion_borrador' (mi conocimiento de kinesiología) contra "
          "'contraste_wikipedia' (ficha anatómica real, fuente independiente). Donde "
          "coincidan, verificarlo contra ExRx.net debería ser rápido; donde no "
          "coincidan o el campo esté vacío, ahí sí dedica el tiempo. Recién con eso "
          "copia a 'articulacion_principal'/'movimiento', citando 'fuente' y "
          "'fecha_consulta' reales de tu propia verificación. Las filas 'revisar' no "
          "tienen borrador ni contraste — decide tú cómo tratarlas.")


if __name__ == "__main__":
    main()

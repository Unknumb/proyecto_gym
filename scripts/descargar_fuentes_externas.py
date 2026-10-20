"""
Descarga de las fuentes externas de la EV2: free-exercise-db y wger.

Igual que ``auditar_media_fisica.py``, es la única pieza de su pipeline que
necesita red, así que se ejecuta aparte: deja una **instantánea** en
``data/01_raw/`` y el pipeline ``integracion_fuentes`` trabaja sobre ella sin
conexión. La instantánea queda fijada en ``fuentes_externas_manifest.json``
(commit exacto de free-exercise-db, fecha de descarga de wger y conteos), de
modo que una re-ejecución futura pueda compararse contra lo que se usó.

1. **free-exercise-db** (github.com/yuhonas/free-exercise-db, *The Unlicense*).
   876 ejercicios con ``level``, ``mechanic`` y ``force``: las variables que el
   catálogo base no trae (README §3.4). Se descarga ``dist/exercises.json`` en
   el commit vigente de ``main``, y ese SHA queda en el manifiesto.

2. **wger** (wger.de, API REST pública, licencia **por ejercicio**, casi toda
   CC-BY-SA). Se pagina ``/api/v2/exerciseinfo/`` y se guarda solo lo que el
   proyecto usa: músculos, equipamiento, categoría, grupo de variación, nombre y
   descripción en inglés, y la licencia y autoría de **cada** ficha, que la
   licencia obliga a conservar. Las imágenes y vídeos no se descargan.

Uso:
    python scripts/descargar_fuentes_externas.py
"""

from __future__ import annotations

import html
import json
import re
import urllib.request
from datetime import date
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
DESTINO_FEDB = RAIZ / "data" / "01_raw" / "free_exercise_db.json"
DESTINO_WGER = RAIZ / "data" / "01_raw" / "wger_exercises.json"
DESTINO_MANIFIESTO = RAIZ / "data" / "01_raw" / "fuentes_externas_manifest.json"

USER_AGENT = "proyecto-gym-biomecanico/EV2 (uso educativo, Duoc UC MLY1101)"
FEDB_REPO = "yuhonas/free-exercise-db"
WGER_API = "https://wger.de/api/v2/exerciseinfo/?limit=200"
WGER_IDIOMA_INGLES = 2


def _get(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as respuesta:
        return respuesta.read()


def descargar_free_exercise_db() -> dict:
    """Descarga ``exercises.json`` fijado al commit actual de ``main``."""
    commit = json.loads(_get(f"https://api.github.com/repos/{FEDB_REPO}/commits/main"))
    sha = commit["sha"]
    url = f"https://raw.githubusercontent.com/{FEDB_REPO}/{sha}/dist/exercises.json"
    ejercicios = json.loads(_get(url))
    DESTINO_FEDB.write_text(json.dumps(ejercicios, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"free-exercise-db: {len(ejercicios)} ejercicios (commit {sha[:10]})")
    return {"repositorio": f"https://github.com/{FEDB_REPO}", "commit": sha, "url": url,
            "licencia": "The Unlicense", "n_ejercicios": len(ejercicios)}


def _texto_plano(descripcion: str) -> str:
    """La descripción de wger viene en HTML; se deja solo el texto."""
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", descripcion or ""))).strip()


def _reducir_ficha_wger(ficha: dict) -> dict | None:
    """Conserva los campos que usa el proyecto. Sin traducción inglesa → None."""
    ingles = next((t for t in ficha["translations"] if t["language"] == WGER_IDIOMA_INGLES), None)
    if ingles is None:
        return None
    return {
        "id": ficha["id"],
        "uuid": ficha["uuid"],
        "nombre": ingles["name"].strip(),
        "descripcion": _texto_plano(ingles["description"]),
        "categoria": ficha["category"]["name"],
        "musculos": [m["name"] for m in ficha["muscles"]],
        "musculos_secundarios": [m["name"] for m in ficha["muscles_secondary"]],
        "equipamiento": [e["name"] for e in ficha["equipment"]],
        "grupo_variacion": ficha["variation_group"],
        "licencia": ficha["license"]["short_name"],
        "licencia_url": ficha["license"]["url"],
        "autor": ficha["license_author"],
    }


def descargar_wger() -> dict:
    """Pagina la API de wger y guarda la versión reducida de cada ficha."""
    fichas, url = [], WGER_API
    while url:
        pagina = json.loads(_get(url))
        fichas.extend(pagina["results"])
        url = pagina["next"]
    reducidas = [r for r in map(_reducir_ficha_wger, fichas) if r is not None]
    DESTINO_WGER.write_text(json.dumps(reducidas, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wger: {len(fichas)} fichas, {len(reducidas)} con traducción al inglés")
    return {"api": WGER_API.split("?")[0], "fecha_descarga": date.today().isoformat(),
            "licencia": "por ejercicio (campo 'licencia')", "n_fichas_api": len(fichas),
            "n_ejercicios": len(reducidas)}


def main() -> None:
    manifiesto = {
        "free_exercise_db": descargar_free_exercise_db(),
        "wger": descargar_wger(),
    }
    DESTINO_MANIFIESTO.write_text(json.dumps(manifiesto, ensure_ascii=False, indent=2) + "\n",
                                  encoding="utf-8")
    print(f"Manifiesto: {DESTINO_MANIFIESTO.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()

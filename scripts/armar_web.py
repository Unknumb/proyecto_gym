"""Arma la página publicable ``docs/index.html`` insertando ``web/datos_web.json`` en ``web/app.html``.

Uso, desde la raíz del repositorio::

    python scripts/armar_web.py
"""
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
FUENTE = RAIZ / "web/app.html"
DATOS = RAIZ / "web/datos_web.json"
SALIDA = RAIZ / "docs/index.html"

if __name__ == "__main__":
    # "</" se escapa para que el JSON no pueda cerrar la etiqueta <script> que lo contiene
    datos = DATOS.read_text(encoding="utf-8").replace("</", "<\\/")
    plantilla = FUENTE.read_text(encoding="utf-8")
    assert plantilla.count("__DATOS__") == 1, "web/app.html debe tener exactamente un __DATOS__"
    # Documento completo para GitHub Pages: codificación, ancho de celular y un reset mínimo
    cabecera = ('<!doctype html><html lang="es"><head><meta charset="utf-8">'
                '<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">'
                '<style>:root{padding-top:env(safe-area-inset-top,0px);padding-bottom:env(safe-area-inset-bottom,0px)}'
                'body{margin:0;font:14px system-ui,sans-serif}img{max-width:100%}[hidden]{display:none!important}</style>'
                '</head><body>\n')
    SALIDA.write_text(cabecera + plantilla.replace("__DATOS__", datos) + "\n</body></html>\n", encoding="utf-8")
    print(f"{SALIDA.relative_to(RAIZ)}: {SALIDA.stat().st_size:,} bytes")

"""Exporta los aportes guardados en la base de cuentas al formato que lee el importador (README §20.3).

Las correcciones de etiquetas y las calificaciones de alternativas que las personas hacen
con su cuenta, en la web o en el iPhone, quedan en la base de datos. Este script las saca
en el mismo archivo ``sustituto-seguro-aportes`` que descarga la página, **sin datos de la
persona**: ni correo, ni nombre, ni id. Cada fila es un voto; el consenso de §18.1 los cuenta.

Uso, desde la raíz del repositorio::

    DATABASE_URL=postgresql://… python scripts/exportar_aportes_bd.py aportes_bd.json
    python scripts/aprender_de_usuarios.py aportes_bd.json      # reentrena (§18.3)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
from api.basedatos import BaseDatos  # noqa: E402

FORMATO = "sustituto-seguro-aportes"


def datos_pagina(ruta: Path = RAIZ / "docs/index.html") -> dict:
    html = ruta.read_text(encoding="utf-8")
    m = re.search(r'<script type="application/json" id="datos">(.*?)</script>', html, re.S)
    return json.loads(m.group(1).replace("<\\/", "</"))


def exportar(url: str, pagina: dict) -> dict:
    indice = {eid: k for k, eid in enumerate(pagina["ids"])}

    def nombre(eid):
        return pagina["ejercicios"][indice[eid]][0] if eid in indice else ""

    def familia(eid):
        return pagina["familias"][pagina["ejercicios"][indice[eid]][2]] if eid in indice else ""

    with BaseDatos(url).conectar() as cx:
        correcciones = cx.todas("SELECT ejercicio_id, objetivo, etiqueta, etiqueta_previa, origen_previo, confianza_previa, fecha "
                                "FROM correcciones ORDER BY id")
        revisiones = cx.todas("SELECT original_id, alternativa_id, articulacion, perfil, veredicto, similitud, misma_familia, fecha "
                              "FROM revisiones ORDER BY id")
    return {
        "formato": FORMATO, "version": 1, "origen": "base_de_datos",
        "creado": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tablas": {
            "correcciones_etiquetas": [{
                "ejercicio_id": c["ejercicio_id"], "nombre": nombre(c["ejercicio_id"]), "objetivo": c["objetivo"],
                "etiqueta_pipeline": c["etiqueta_previa"], "origen_pipeline": c["origen_previo"],
                "confianza_pipeline": c["confianza_previa"], "etiqueta_corregida": c["etiqueta"],
                "comentario": "", "fecha": c["fecha"]} for c in correcciones],
            "revisiones_recomendaciones": [{
                "articulacion_vetada": r["articulacion"], "original": nombre(r["original_id"]),
                "familia_original": familia(r["original_id"]), "alternativa": nombre(r["alternativa_id"]),
                "familia_alternativa": familia(r["alternativa_id"]), "misma_familia": bool(r["misma_familia"]),
                "similitud": r["similitud"], "veredicto": r["veredicto"], "comentario": "",
                "original_id": r["original_id"], "alternativa_id": r["alternativa_id"], "perfil": r["perfil"],
                "fecha": r["fecha"]} for r in revisiones],
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("salida", type=Path, help="archivo .json a escribir")
    ap.add_argument("--base", default=os.environ.get("DATABASE_URL"), help="URL de la base (por defecto DATABASE_URL)")
    args = ap.parse_args()
    if not args.base:
        raise SystemExit("Falta la URL de la base: usa --base o la variable DATABASE_URL.")
    archivo = exportar(args.base, datos_pagina())
    args.salida.write_text(json.dumps(archivo, ensure_ascii=False, indent=1), encoding="utf-8")
    t = archivo["tablas"]
    print(f"✓ {len(t['correcciones_etiquetas'])} correcciones y {len(t['revisiones_recomendaciones'])} calificaciones "
          f"→ {args.salida}")


if __name__ == "__main__":
    main()

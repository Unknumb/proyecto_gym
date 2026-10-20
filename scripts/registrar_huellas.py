"""Registra la huella SHA-256 de cada archivo de ``data/01_raw/``.

Los datos crudos son inmutables: si una huella cambia, cambió la fuente y los
resultados versionados dejan de corresponder a ella. Uso::

    python scripts/registrar_huellas.py            # reescribe el registro
    python scripts/registrar_huellas.py --verificar # falla si algún archivo cambió
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
CRUDOS = RAIZ / "data" / "01_raw"
REGISTRO = CRUDOS / "huellas_sha256.txt"


def _huellas() -> dict[str, str]:
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(CRUDOS.iterdir())
            if p.is_file() and p != REGISTRO and not p.name.startswith(".")}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--verificar", action="store_true", help="comparar en vez de reescribir")
    args = parser.parse_args()
    actuales = _huellas()
    if not args.verificar:
        REGISTRO.write_text("".join(f"{h}  {n}\n" for n, h in actuales.items()))
        print(f"{len(actuales)} archivos registrados en {REGISTRO.relative_to(RAIZ)}")
        return 0
    guardadas = dict(reversed(linea.split("  ", 1)) for linea in REGISTRO.read_text().splitlines())
    distintas = sorted(n for n in actuales.keys() | guardadas.keys() if actuales.get(n) != guardadas.get(n))
    for nombre in distintas:
        print(f"cambió: {nombre}")
    print("huellas correctas" if not distintas else f"{len(distintas)} archivos distintos")
    return 1 if distintas else 0


if __name__ == "__main__":
    sys.exit(main())

"""
Auditoría física de los recursos cinemáticos (GIF) del catálogo.

Los GIF **no están versionados en este repositorio** (ver `NOTICE.md` de la
fuente: los medios son propiedad de Gym visual y clonar el repo no constituye
licencia). Este script los descarga bajo demanda desde el repositorio de origen,
mide sus propiedades físicas sobre una muestra estratificada y persiste el
resultado en ``data/08_reporting/media_sample_audit.csv``.

El notebook consume ese CSV, de modo que el EDA sigue siendo reproducible
**sin conexión a internet**: la red se toca aquí, una sola vez, y no durante
el análisis.

Uso:
    python scripts/auditar_media_fisica.py [--n 40] [--semilla 7]
"""

from __future__ import annotations

import argparse
import io
import json
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

BASE = "https://raw.githubusercontent.com/Unknumb/dataset_ejercicios/main/"
RAIZ = Path(__file__).resolve().parent.parent
ORIGEN = RAIZ / "data" / "01_raw" / "exercises.json"
DESTINO = RAIZ / "data" / "08_reporting" / "media_sample_audit.csv"
FIGURAS = RAIZ / "data" / "08_reporting" / "figures"

#: Ejercicio usado como muestra visual: su ciclo excéntrico-concéntrico es nítido.
EJEMPLO_VISUAL = "barbell full squat"

#: Umbral de intensidad (0-255) sobre el que un píxel se considera "cambiado".
UMBRAL_CAMBIO = 12


def medir_gif(contenido: bytes) -> dict[str, float]:
    """Extrae las propiedades físicas y la energía de movimiento de un GIF.

    La energía de movimiento es la fracción media de píxeles que cambian de
    forma apreciable entre fotogramas consecutivos. Sirve para distinguir una
    animación real de una secuencia estática.
    """
    im = Image.open(io.BytesIO(contenido))
    n_frames = getattr(im, "n_frames", 1)

    grises, duraciones = [], []
    for i in range(n_frames):
        im.seek(i)
        grises.append(np.asarray(im.convert("L"), dtype=np.int16))
        duraciones.append(im.info.get("duration", 0))

    pila = np.stack(grises)
    segundos = sum(duraciones) / 1000 or np.nan
    if n_frames > 1:
        cambios = (np.abs(np.diff(pila, axis=0)) > UMBRAL_CAMBIO).mean(axis=(1, 2))
        energia_media, energia_max = cambios.mean(), cambios.max()
    else:
        energia_media = energia_max = 0.0

    return {
        "ancho": im.size[0],
        "alto": im.size[1],
        "n_frames": n_frames,
        "duracion_s": round(segundos, 3),
        "fps": round(n_frames / segundos, 2) if segundos else np.nan,
        "peso_kb": round(len(contenido) / 1024, 1),
        "energia_movimiento_media": round(float(energia_media), 4),
        "energia_movimiento_max": round(float(energia_max), 4),
    }


def hoja_de_contacto(contenido: bytes, destino: Path, titulo: str) -> None:
    """Despliega los fotogramas de un GIF en una tira horizontal.

    Es la evidencia visual de dos propiedades que ninguna métrica captura: que
    la figura es una ilustración anatómica y no una persona, y que el
    movimiento se sugiere con estelas fantasma superpuestas.
    """
    im = Image.open(io.BytesIO(contenido))
    n, (w, h) = im.n_frames, im.size
    hoja = Image.new("RGB", (w * n, h), "white")
    for i in range(n):
        im.seek(i)
        hoja.paste(im.convert("RGB"), (w * i, 0))
    hoja = hoja.resize((w * n * 2, h * 2), Image.LANCZOS)
    destino.parent.mkdir(parents=True, exist_ok=True)
    hoja.save(destino)
    print(f"  hoja de contacto ({titulo}, {n} fotogramas) -> {destino.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=40,
                        help="tamaño de la muestra (por defecto 40)")
    parser.add_argument("--semilla", type=int, default=7,
                        help="semilla para que la muestra sea reproducible")
    args = parser.parse_args()

    catalogo = pd.DataFrame(json.load(open(ORIGEN)))

    # Muestreo estratificado por parte del cuerpo: evita que la muestra quede
    # dominada por las categorías sobrerrepresentadas del catálogo.
    rng = np.random.default_rng(args.semilla)
    por_grupo = max(1, args.n // catalogo["body_part"].nunique())
    indices: list[int] = []
    for _, grupo in catalogo.groupby("body_part"):
        k = min(len(grupo), por_grupo)
        indices.extend(rng.choice(grupo.index.to_numpy(), size=k, replace=False))
    muestra = catalogo.loc[indices].reset_index(drop=True)

    filas = []
    for _, r in muestra.iterrows():
        try:
            with urllib.request.urlopen(BASE + r["gif_url"], timeout=30) as resp:
                contenido = resp.read()
            fila = {"id": r["id"], "name": r["name"], "body_part": r["body_part"],
                    "gif_url": r["gif_url"], "estado": "ok", **medir_gif(contenido)}
        except (urllib.error.URLError, OSError) as exc:
            fila = {"id": r["id"], "name": r["name"], "body_part": r["body_part"],
                    "gif_url": r["gif_url"], "estado": f"error: {exc}"}
        filas.append(fila)
        print(f"  {fila['estado']:8s} {r['id']}  {r['name'][:45]}")

    # Evidencia visual: los fotogramas de un ejercicio representativo.
    ejemplo = catalogo.loc[catalogo["name"] == EJEMPLO_VISUAL]
    if not ejemplo.empty:
        try:
            with urllib.request.urlopen(BASE + ejemplo.iloc[0]["gif_url"], timeout=30) as resp:
                hoja_de_contacto(resp.read(), FIGURAS / "media_fotogramas.png",
                                 EJEMPLO_VISUAL)
        except (urllib.error.URLError, OSError) as exc:
            print(f"  no se pudo generar la hoja de contacto: {exc}")

    resultado = pd.DataFrame(filas)
    DESTINO.parent.mkdir(parents=True, exist_ok=True)
    resultado.to_csv(DESTINO, index=False)

    ok = resultado[resultado["estado"] == "ok"]
    print(f"\n{len(ok)}/{len(resultado)} GIF descargados y medidos.")
    print(f"Resultado en {DESTINO.relative_to(RAIZ)}")
    if not ok.empty:
        print("\nResumen:")
        print(f"  resolución : {ok.ancho.min()}x{ok.alto.min()} — {ok.ancho.max()}x{ok.alto.max()}")
        print(f"  frames     : {ok.n_frames.min()} — {ok.n_frames.max()} (mediana {ok.n_frames.median():.0f})")
        print(f"  fps        : {ok.fps.min()} — {ok.fps.max()}")


if __name__ == "__main__":
    main()

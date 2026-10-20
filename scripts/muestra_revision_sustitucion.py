"""Muestra de recomendaciones para revisar a mano (README §6, limitación 10).

El KPI de sustitución mide que las alternativas sean seguras y de la misma familia,
no que un entrenador las considere buenas. Este script toma 10 ejercicios al azar
por articulación vetada (rodilla, hombro y columna lumbar), pide la primera
alternativa al sustituidor y deja la muestra en
``data/01_raw/sustitucion_revision_manual.csv`` con una columna ``veredicto`` vacía
para que la complete el equipo. Si el archivo
ya existe, no lo sobrescribe: los veredictos son trabajo manual.

Veredictos: ``bueno`` (un entrenador lo propondría), ``aceptable`` (mismo músculo,
pero cambia el patrón, el equipo o la dificultad), ``malo`` (seguro pero no sirve
como reemplazo) e ``inseguro`` (carga la articulación vetada).

Uso::

    python scripts/muestra_revision_sustitucion.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import yaml

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
pd.set_option("compute.use_numexpr", False)

from gym_exercises.pipelines.grafo_biomecanico.nodes import Sustituidor  # noqa: E402

SALIDA = RAIZ / "data" / "01_raw" / "sustitucion_revision_manual.csv"
ARTICULACIONES = ["rodilla", "hombro", "columna (lumbar)"]
POR_ARTICULACION = 10
SEMILLA = 7


def main() -> int:
    if SALIDA.exists():
        print(f"{SALIDA.relative_to(RAIZ)} ya existe; no se sobrescribe.")
        return 0
    familias = pd.read_parquet(RAIZ / "data/07_model_output/familias_movimiento.parquet")
    sustituidor = Sustituidor(
        familias, pd.read_parquet(RAIZ / "models/embeddings_node2vec.parquet"),
        pd.read_csv(RAIZ / "data/03_primary/musculo_articulacion.csv"),
        yaml.safe_load((RAIZ / "conf/base/parameters.yml").read_text())["grafo_biomecanico"])
    filas = []
    for articulacion in ARTICULACIONES:
        candidatos = familias[familias["articulaciones_carga"].map(lambda a: articulacion in set(a))]
        for _, original in candidatos.sample(frac=1, random_state=SEMILLA).iterrows():
            alternativas, motivo = sustituidor.recomendar(original["ejercicio_id"], articulacion)
            if motivo or alternativas.empty:
                continue
            primera = alternativas.iloc[0]
            filas.append({"articulacion_vetada": articulacion,
                          "original": original["nombre"], "familia_original": original["familia_kmeans_node2vec"],
                          "alternativa": primera["nombre"], "familia_alternativa": primera["familia_kmeans_node2vec"],
                          "misma_familia": bool(primera["misma_familia"]), "similitud": primera["similitud"],
                          "veredicto": "", "comentario": ""})
            if sum(f["articulacion_vetada"] == articulacion for f in filas) == POR_ARTICULACION:
                break
    pd.DataFrame(filas).to_csv(SALIDA, index=False)
    print(f"{len(filas)} pares en {SALIDA.relative_to(RAIZ)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Exporta los datos de la página web (``docs/index.html``) desde las salidas del pipeline.

Ejecuta el mismo ``Sustituidor`` del pipeline ``grafo_biomecanico`` para cada
ejercicio, cada articulación que carga y cada perfil (cualquiera / principiante),
con la cuota sin equipamiento activa, y guarda el resultado en un JSON compacto.
Así la página funciona sin servidor y da exactamente las mismas recomendaciones
que el dashboard y el README (§13.5).

Uso, desde la raíz del repositorio::

    python scripts/exportar_datos_web.py          # escribe web/datos_web.json
    python scripts/armar_web.py                   # inserta los datos en docs/index.html

``--json-dir`` permite leer las tablas como JSON en lugar de parquet (útil en
entornos sin pyarrow): ``familias_movimiento.json`` y ``embeddings_node2vec.json``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
from gym_exercises.pipelines.grafo_biomecanico.nodes import Sustituidor  # noqa: E402

# Códigos de motivo que entiende la página
MOTIVO = {None: 0, "menos_de_3_alternativas": 1, "sin_sustituto_anatomico": 2,
          "musculo_con_pocos_ejercicios": 3, "no_carga_la_articulacion": 4}


def cargar(json_dir: Path | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    if json_dir:
        fam = pd.DataFrame(json.loads((json_dir / "familias_movimiento.json").read_text()))
        emb = pd.DataFrame(json.loads((json_dir / "embeddings_node2vec.json").read_text()))
    else:
        fam = pd.read_parquet(RAIZ / "data/07_model_output/familias_movimiento.parquet")
        emb = pd.read_parquet(RAIZ / "models/embeddings_node2vec.parquet")
    return fam, emb


def exportar(fam: pd.DataFrame, emb: pd.DataFrame) -> dict:
    ma = pd.read_csv(RAIZ / "data/03_primary/musculo_articulacion.csv")
    params = yaml.safe_load((RAIZ / "conf/base/parameters.yml").read_text(encoding="utf-8"))["grafo_biomecanico"]
    s = Sustituidor(fam, emb, ma, params)
    f = s.familias
    # Mismo perfil de principiante que el dashboard (app/dashboard.py)
    principiante = ((f["complejidad_tecnica"] == "baja") & ~f["impacto"].astype(bool)
                    & (f["estabilidad"] != "inestable")).to_numpy()

    articulaciones = sorted({a for c in s.carga for a in c})
    art_idx = {a: k for k, a in enumerate(articulaciones)}
    familias = sorted(f["familia_kmeans_node2vec"].unique())
    fam_idx = {x: k for k, x in enumerate(familias)}
    equipos = sorted(f["familia_equipamiento"].unique())
    eq_idx = {x: k for k, x in enumerate(equipos)}
    musculos = sorted({m for a in f["agonistas"] for m in a} | {m for a in f["musculos"] for m in a})
    mus_idx = {x: k for k, x in enumerate(musculos)}
    texto = lambda v: "" if pd.isna(v) else v  # noqa: E731

    # Cada ejercicio: [nombre, fuente, familia, equipo, agonistas, articulaciones,
    #                  complejidad, fuerza, es_estiramiento, apto_principiante]
    ejercicios = [[r["nombre"], r["fuente"], fam_idx[r["familia_kmeans_node2vec"]], eq_idx[r["familia_equipamiento"]],
                   [mus_idx[m] for m in r["agonistas"]], [art_idx[a] for a in sorted(r["articulaciones_carga"])],
                   texto(r["complejidad_tecnica"]), texto(r["fuerza"]), bool(r["es_estiramiento"]), bool(principiante[i])]
                  for i, r in f.iterrows()]

    # rec[i][articulación] = [cualquiera, principiante]; cada uno [motivo, [[j, similitud*1000, misma_familia], ...]]
    rec = []
    for eid in f["ejercicio_id"]:
        i = int(s.indice[eid])
        por_art = {}
        for a in sorted(s.carga[i]):
            par = []
            for filtro in (None, principiante):
                alt, motivo = s.recomendar(eid, a, filtro=filtro)
                filas = [] if alt.empty else [
                    [int(s.indice[x]), int(round(sim * 1000)), int(m)]
                    for x, sim, m in zip(alt["ejercicio_id"], alt["similitud"], alt["misma_familia"])]
                par.append([MOTIVO[motivo], filas])
            por_art[art_idx[a]] = par
        rec.append(por_art)

    # Verificación de seguridad antes de escribir: ninguna alternativa carga la articulación vetada
    for i, por_art in enumerate(rec):
        for a, par in por_art.items():
            vetadas = s.vetadas(articulaciones[a])
            for _, filas in par:
                malas = [j for j, _, _ in filas if vetadas & s.carga[j]]
                if malas:
                    raise SystemExit(f"Alternativa insegura para {f['ejercicio_id'].iat[i]} / {articulaciones[a]}: {malas}")

    return {"articulaciones": articulaciones, "familias": familias, "equipos": equipos,
            "musculos": musculos, "ejercicios": ejercicios, "rec": rec}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--json-dir", type=Path, help="leer las tablas como JSON en vez de parquet")
    ap.add_argument("--salida", type=Path, default=RAIZ / "web/datos_web.json")
    args = ap.parse_args()
    datos = exportar(*cargar(args.json_dir))
    args.salida.write_text(json.dumps(datos, ensure_ascii=False, separators=(",", ":"), allow_nan=False), encoding="utf-8")
    consultas = sum(len(r) * 2 for r in datos["rec"])
    print(f"{len(datos['ejercicios'])} ejercicios, {consultas} consultas → {args.salida} ({args.salida.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()

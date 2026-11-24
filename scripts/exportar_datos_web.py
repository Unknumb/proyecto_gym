"""Exporta los datos de la página web (``docs/index.html``) desde las salidas del pipeline.

Ejecuta el mismo ``Sustituidor`` del pipeline ``grafo_biomecanico`` para cada
ejercicio, cada articulación que carga y cada perfil (cualquiera / principiante),
con la cuota sin equipamiento activa, y guarda el resultado en un JSON compacto.
Así la página funciona sin servidor y da exactamente las mismas recomendaciones
que el dashboard y el README (§13.5).

Uso, desde la raíz del repositorio::

    python scripts/exportar_datos_web.py          # escribe web/datos_web.json
    python scripts/armar_web.py                   # inserta los datos en docs/index.html

Además de las recomendaciones, lleva lo que la página explica de cada ejercicio
(músculos de apoyo; mecánica, fuerza y complejidad con su origen y la confianza del
modelo), de dónde salen los datos y cómo se prepararon (``resumen_datos``), el resumen
de la EV3 (``ensambles_comparacion``) y lo que el modelo aprendió de los usuarios
(``aprendizaje_usuarios_informe`` y ``aprendizaje_simulacion``).

``--json-dir`` permite leer las tablas como JSON en lugar de parquet (útil en
entornos sin pyarrow): ``familias_movimiento.json``, ``embeddings_node2vec.json`` y
``perfil_exigencia.json``.
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


OBJETIVOS = ["mecanica", "fuerza", "complejidad_tecnica"]
# Origen de cada etiqueta (predecir_etiquetas): fuente, transferida, modelo, usuarios
ORIGEN = {"fuente": "f", "transferida": "t", "modelo": "m", "usuarios": "u"}
REPORTES = RAIZ / "data/08_reporting"


def cargar(json_dir: Path | None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if json_dir:
        fam = pd.DataFrame(json.loads((json_dir / "familias_movimiento.json").read_text()))
        emb = pd.DataFrame(json.loads((json_dir / "embeddings_node2vec.json").read_text()))
        perfil = pd.DataFrame(json.loads((json_dir / "perfil_exigencia.json").read_text()))
    else:
        fam = pd.read_parquet(RAIZ / "data/07_model_output/familias_movimiento.parquet")
        emb = pd.read_parquet(RAIZ / "models/embeddings_node2vec.parquet")
        perfil = pd.read_parquet(RAIZ / "data/07_model_output/perfil_exigencia.parquet")
    return fam, emb, perfil


def _tabla(df: pd.DataFrame) -> dict:
    """``{"columnas": [...], "filas": [[...], ...]}``, con decimales a 4 cifras y sin NaN."""
    df = df.astype(object).where(df.notna(), None)
    filas = [[round(v, 4) if isinstance(v, float) else (bool(v) if isinstance(v, (bool, np.bool_)) else v)
              for v in fila] for fila in df.itertuples(index=False)]
    return {"columnas": list(df.columns), "filas": filas}


def resumen_datos() -> dict:
    """Lo que la sección *De los datos al modelo* muestra: fuentes, hallazgos del EDA y
    preparación, calculados de las salidas del pipeline (no escritos a mano)."""
    from sklearn.metrics import cohen_kappa_score  # noqa: PLC0415

    integracion = pd.read_csv(REPORTES / "integracion_fuentes_resumen.csv")
    fuentes = integracion[integracion["dimension"] == "fuente"].set_index("grupo")["n_ejercicios"].astype(int)
    catalogo = integracion[integracion["dimension"] == "contexto_catalogo"].set_index("grupo")["n_ejercicios"]
    contexto = pd.read_csv(REPORTES / "sustitucion_contexto.csv")
    gimnasio = contexto[(contexto["metodo"] == "kmeans_node2vec") & (contexto["contexto"] == "gimnasio")].iloc[0]

    # EDA §11.5: el nivel de dificultad de free-exercise-db frente al curado de megaGymDataset
    ejercicios = pd.read_parquet(RAIZ / "data/03_primary/ejercicios.parquet")
    pares = ejercicios[(ejercicios["nivel_origen"] == "free_exercise_db") & ejercicios["nivel_megagym"].notna()]
    nivel, nivel_mg = pares["nivel"].astype(str), pares["nivel_megagym"].astype(str)

    tabla = pd.read_parquet(RAIZ / "data/05_model_input/tabla_base.parquet")
    fedb = tabla[tabla["rol"] == "entrenamiento"]
    ma = pd.read_csv(RAIZ / "data/03_primary/musculo_articulacion.csv")
    comparacion = pd.read_csv(REPORTES / "modelos_comparacion.csv")
    externa = comparacion[comparacion["conjunto"] == "externa_catalogo"].groupby("objetivo", sort=False)["n"].first()
    clases = {o: {str(k): round(float(v), 3) for k, v in fedb[o].value_counts(normalize=True).sort_index().items()}
              for o in OBJETIVOS}
    return {
        "fuentes": {"catalogo": int(fuentes["catalogo"]), "free_exercise_db": int(len(fedb)),
                    "free_exercise_db_nuevos": int(fuentes["free_exercise_db"]), "wger_nuevos": int(fuentes["wger"]),
                    "total": int(fuentes.sum()), "musculos": int(len(set(ejercicios["musculos_primarios"].explode().dropna())
                                                                  | set(ejercicios["musculos_secundarios"].explode().dropna()))),
                    "articulaciones": int(ma["articulacion"].nunique())},
        "nivel": {"n": int(len(pares)), "acuerdo": round(float((nivel == nivel_mg).mean()), 3),
                  "kappa": round(float(cohen_kappa_score(nivel, nivel_mg)), 2)},
        "gimnasio": {"pct_catalogo": round(float(catalogo["gimnasio"] / catalogo.sum()), 3),
                     "sin_equipamiento_antes": float(gimnasio["con_sin_equipamiento_antes"]),
                     "sin_equipamiento_despues": float(gimnasio["con_sin_equipamiento_despues"])},
        "preparacion": {"ejercicios": int(len(fedb)), "grupos": int(fedb["grupo"].nunique()),
                        "train": int((fedb["particion"] == "train").sum()), "test": int((fedb["particion"] == "test").sum()),
                        "musculos_modelo": int(sum(c.startswith("agonista__") for c in tabla.columns)),
                        "externa": {o: int(n) for o, n in externa.items()}},
        "clases": clases,
    }


def resumen_modelo() -> dict:
    """Lo que la vista *El modelo por dentro* muestra de la EV3: los candidatos con
    los mismos pliegues, la regla de adopción y la referencia de azar."""
    comparacion = pd.read_csv(REPORTES / "ensambles_comparacion.csv")
    columnas = ["objetivo", "modelo", "familia", "es_modelo_actual", "cv_f1_media", "cv_f1_sd",
                "f1_train", "f1_test", "f1_externa", "diferencia_media", "pct_pliegues_mejor",
                "t_nadeau_bengio", "p_valor", "adoptar"]
    modelos = pd.read_csv(REPORTES / "modelos_comparacion.csv")
    azar = modelos[modelos["modelo"] == "azar_estratificado"][["objetivo", "f1_macro"]]
    # Mejor configuración del boosting en los pliegues de su propia búsqueda (optimista)
    busqueda = pd.read_csv(REPORTES / "ensambles_busqueda_boosting.csv")
    mejor = busqueda.groupby("objetivo", sort=False)["cv_f1_macro"].max().reset_index()
    return {"candidatos": _tabla(comparacion[columnas]), "azar": _tabla(azar), "busqueda_boosting": _tabla(mejor)}


def resumen_aprendizaje() -> dict:
    """Aportes de usuarios: el informe del último ``kedro run`` y la curva simulada
    (media, mínimo y máximo del F1 externo, y fracción adoptada por el control)."""
    informe = pd.read_csv(REPORTES / "aprendizaje_usuarios_informe.csv")
    simulacion = pd.read_csv(REPORTES / "aprendizaje_simulacion.csv")
    curva = (simulacion.groupby(["objetivo", "escenario", "n_aportes"], sort=False)
             .agg(f1_externa=("f1_externa", "mean"), f1_min=("f1_externa", "min"),
                  f1_max=("f1_externa", "max"), adoptado=("adoptado", "mean"))
             .reset_index())
    return {"informe": _tabla(informe), "simulacion": _tabla(curva)}


def exportar(fam: pd.DataFrame, emb: pd.DataFrame, perfil: pd.DataFrame) -> dict:
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

    # Perfil de exigencia: por objetivo, [etiqueta final o -1, origen, predicción del
    # modelo, confianza en %]; las etiquetas, como índice en "clases".
    clases = {o: sorted(set(perfil[o].dropna()) | set(perfil[f"{o}_prediccion"].dropna())) for o in OBJETIVOS}
    fichas = perfil.set_index("ejercicio_id")

    def perfil_de(eid: str) -> list:
        fila = fichas.loc[eid]
        return [[clases[o].index(fila[o]) if pd.notna(fila[o]) else -1, ORIGEN.get(fila[f"{o}_origen"], ""),
                 clases[o].index(fila[f"{o}_prediccion"]), int(round(100 * float(fila[f"{o}_confianza"])))]
                for o in OBJETIVOS]

    # Cada ejercicio: [nombre, fuente, familia, equipo, agonistas, articulaciones,
    #                  complejidad, fuerza, es_estiramiento, apto_principiante,
    #                  sinergistas, perfil]
    ejercicios = [[r["nombre"], r["fuente"], fam_idx[r["familia_kmeans_node2vec"]], eq_idx[r["familia_equipamiento"]],
                   [mus_idx[m] for m in r["agonistas"]], [art_idx[a] for a in sorted(r["articulaciones_carga"])],
                   texto(r["complejidad_tecnica"]), texto(r["fuerza"]), bool(r["es_estiramiento"]), bool(principiante[i]),
                   [mus_idx[m] for m in sorted(set(r["musculos"]) - set(r["agonistas"]))], perfil_de(r["ejercicio_id"])]
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

    # ids: la página identifica por ejercicio_id (no por posición) los ejercicios con nombre en español
    return {"ids": f["ejercicio_id"].tolist(), "articulaciones": articulaciones, "familias": familias, "equipos": equipos,
            "musculos": musculos, "ejercicios": ejercicios, "rec": rec, "clases": clases,
            "datos": resumen_datos(), "modelo": resumen_modelo(), "aprendizaje": resumen_aprendizaje()}


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

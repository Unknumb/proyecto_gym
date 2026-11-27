"""
Revisión humana de las etiquetas del perfil en el catálogo (README §19.3).

Una muestra fija de 105 etiquetas que el modelo puso en ejercicios del catálogo (35 de
mecánica, 35 de fuerza y 35 de complejidad técnica). El equipo anota la etiqueta correcta
de cada una. Mide el KPI "acierto de lo etiquetado en el catálogo ≥ 90 %" con un
intervalo de confianza, y las correcciones reentrenan el modelo (§18).

La planilla es ``data/01_raw/revision_equipo/etiquetas.csv``: se completa a mano, en
Excel o en cualquier editor de CSV, sin cambiar las columnas.

Uso, desde la raíz del repositorio::

    python scripts/revision_etiquetas.py preparar     # (re)genera la planilla vacía
    python scripts/revision_etiquetas.py evaluar      # mide el acierto con lo ya revisado

``evaluar`` escribe ``data/08_reporting/revision_etiquetas_resultado.csv`` y deja las
etiquetas revisadas en el formato de la app (una carpeta con ``correcciones_etiquetas.csv``),
listas para importarlas y reentrenar en un comando (§18.3)::

    python scripts/aprender_de_usuarios.py data/08_reporting/revision_etiquetas_aportes

La muestra es determinista (un hash de cada ejercicio, no un sorteo): ``preparar`` da
siempre las mismas filas con los mismos datos, y ``tests/test_revision_etiquetas.py``
comprueba que la planilla versionada sea esa.
"""
from __future__ import annotations

import argparse
import hashlib
import math
import sys
from datetime import date
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
CARPETA = RAIZ / "data/01_raw/revision_equipo"
PLANILLA_ETIQUETAS = CARPETA / "etiquetas.csv"
RESULTADO = RAIZ / "data/08_reporting/revision_etiquetas_resultado.csv"
APORTES = RAIZ / "data/08_reporting/revision_etiquetas_aportes"

OBJETIVOS = ["mecanica", "fuerza", "complejidad_tecnica"]
POR_OBJETIVO = 35
SEMILLA = "eft-2026"
META_ACIERTO = 0.90

COLUMNAS_ETIQUETAS = [
    "ejercicio_id", "nombre", "objetivo", "etiqueta_pipeline", "confianza_pipeline", "opciones",
    "musculos_primarios", "musculos_secundarios", "equipamiento", "instrucciones_en",
    "etiqueta_correcta", "revisor", "comentario"]


def _clave(objetivo: str, ejercicio_id: str) -> str:
    return hashlib.sha256(f"{SEMILLA}:{objetivo}:{ejercicio_id}".encode()).hexdigest()


def _lista(valor) -> str:
    """Las listas de músculos del parquet, como texto legible: "glutes; hamstrings"."""
    if valor is None or (isinstance(valor, float) and math.isnan(valor)):
        return ""
    if isinstance(valor, str):
        return valor.strip("[]").replace("'", "").replace(", ", "; ")
    return "; ".join(map(str, valor))


def muestra_etiquetas(perfil: pd.DataFrame, ejercicios: pd.DataFrame) -> pd.DataFrame:
    """35 etiquetas por objetivo, puestas por el **modelo** en el **catálogo** (las que mide
    el KPI de acierto). Un ejercicio aparece una sola vez en toda la muestra."""
    ej = ejercicios.set_index("ejercicio_id")
    usados: set[str] = set()
    filas = []
    for objetivo in OBJETIVOS:
        candidatos = perfil[(perfil["fuente"] == "catalogo") & (perfil[f"{objetivo}_origen"] == "modelo")]
        opciones = " / ".join(sorted(perfil[objetivo].dropna().astype(str).unique()))
        orden = sorted(candidatos["ejercicio_id"], key=lambda e: _clave(objetivo, e))
        elegidos = [e for e in orden if e not in usados][:POR_OBJETIVO]
        usados.update(elegidos)
        fila = candidatos.set_index("ejercicio_id")
        for eid in elegidos:
            datos = ej.loc[eid]
            filas.append({
                "ejercicio_id": eid, "nombre": datos["nombre"], "objetivo": objetivo,
                "etiqueta_pipeline": str(fila.at[eid, objetivo]),
                "confianza_pipeline": round(float(fila.at[eid, f"{objetivo}_confianza"]), 3),
                "opciones": opciones,
                "musculos_primarios": _lista(datos["musculos_primarios"]),
                "musculos_secundarios": _lista(datos["musculos_secundarios"]),
                "equipamiento": datos["familia_equipamiento"],
                "instrucciones_en": datos["instrucciones_en"].strip() if isinstance(datos["instrucciones_en"], str) else "",
                "etiqueta_correcta": "", "revisor": "", "comentario": ""})
    return pd.DataFrame(filas, columns=COLUMNAS_ETIQUETAS)


def wilson(aciertos: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Intervalo de confianza del 95 % de una proporción (Wilson): se comporta bien con
    muestras chicas y proporciones cercanas a 1, donde el intervalo normal se sale de [0, 1]."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = aciertos / n
    centro = (p + z * z / (2 * n)) / (1 + z * z / n)
    radio = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, centro - radio), min(1.0, centro + radio))


def _texto(serie: pd.Series) -> pd.Series:
    return serie.fillna("").astype(str).str.strip()


def evaluar_etiquetas(revision: pd.DataFrame) -> pd.DataFrame:
    """Acierto por objetivo entre las etiquetas ya revisadas (las que tienen
    ``etiqueta_correcta``), con su intervalo de Wilson."""
    revisadas = revision[_texto(revision["etiqueta_correcta"]) != ""]
    filas = []
    for objetivo in OBJETIVOS + ["total"]:
        sub = revisadas if objetivo == "total" else revisadas[revisadas["objetivo"] == objetivo]
        total_muestra = len(revision) if objetivo == "total" else int((revision["objetivo"] == objetivo).sum())
        n = len(sub)
        aciertos = int((_texto(sub["etiqueta_correcta"]) == _texto(sub["etiqueta_pipeline"])).sum())
        bajo, alto = wilson(aciertos, n)
        filas.append({"grupo": objetivo, "revisadas": n, "de": total_muestra,
                      "aciertos": aciertos, "acierto": aciertos / n if n else float("nan"),
                      "ic95_bajo": bajo, "ic95_alto": alto,
                      "cumple_meta": (bajo >= META_ACIERTO) if n else None})
    return pd.DataFrame(filas)


def aportes_de_la_revision(revision: pd.DataFrame, perfil: pd.DataFrame, fecha: str) -> pd.DataFrame:
    """Las etiquetas revisadas, en el formato de ``correcciones_etiquetas`` del importador:
    una confirmación o una corrección por fila, que el modelo usa al reentrenarse (§18)."""
    revision = revision.copy()
    for columna in ["etiqueta_correcta", "revisor", "comentario"]:
        revision[columna] = _texto(revision[columna])
    revisadas = revision[revision["etiqueta_correcta"] != ""]
    origen = perfil.set_index("ejercicio_id")
    filas = []
    for _, r in revisadas.iterrows():
        autor = f"revisión del equipo ({r['revisor']})" if r["revisor"] else "revisión del equipo"
        filas.append({"ejercicio_id": r["ejercicio_id"], "nombre": r["nombre"], "objetivo": r["objetivo"],
                      "etiqueta_pipeline": r["etiqueta_pipeline"],
                      "origen_pipeline": origen.at[r["ejercicio_id"], f"{r['objetivo']}_origen"],
                      "confianza_pipeline": r["confianza_pipeline"],
                      "etiqueta_corregida": r["etiqueta_correcta"],
                      "comentario": " · ".join(filter(None, [autor, r["comentario"]])), "fecha": fecha})
    return pd.DataFrame(filas, columns=["ejercicio_id", "nombre", "objetivo", "etiqueta_pipeline",
                                        "origen_pipeline", "confianza_pipeline", "etiqueta_corregida",
                                        "comentario", "fecha"])


def _validar_revision(revision: pd.DataFrame) -> list[str]:
    problemas = []
    if list(revision.columns) != COLUMNAS_ETIQUETAS:
        return [f"{PLANILLA_ETIQUETAS.name}: las columnas cambiaron; deben ser {COLUMNAS_ETIQUETAS}"]
    for i, r in revision.iterrows():
        anotada = str(r["etiqueta_correcta"]).strip()
        if anotada and anotada not in str(r["opciones"]).split(" / "):
            problemas.append(f"{PLANILLA_ETIQUETAS.name}, fila {i + 2}: «{anotada}» no es una opción "
                             f"({r['opciones']})")
    return problemas


def _cargar_datos() -> tuple[pd.DataFrame, pd.DataFrame]:
    perfil = pd.read_parquet(RAIZ / "data/07_model_output/perfil_exigencia.parquet")
    ejercicios = pd.read_parquet(RAIZ / "data/03_primary/ejercicios.parquet")
    return perfil, ejercicios


def preparar(forzar: bool = False) -> None:
    perfil, ejercicios = _cargar_datos()
    CARPETA.mkdir(parents=True, exist_ok=True)
    if PLANILLA_ETIQUETAS.is_file() and not forzar:
        print(f"· {PLANILLA_ETIQUETAS.relative_to(RAIZ)} ya existe "
              "(no se sobrescribe; usa --forzar para empezar de cero)")
        return
    tabla = muestra_etiquetas(perfil, ejercicios)
    tabla.to_csv(PLANILLA_ETIQUETAS, index=False)
    print(f"✓ {PLANILLA_ETIQUETAS.relative_to(RAIZ)}: {len(tabla)} filas")


def evaluar() -> pd.DataFrame:
    perfil, _ = _cargar_datos()
    revision = pd.read_csv(PLANILLA_ETIQUETAS, dtype=str, keep_default_na=False)
    if problemas := _validar_revision(revision):
        for p in problemas:
            print(f"✗ {p}")
        raise SystemExit("Corrige la planilla y vuelve a evaluar.")
    resultado = evaluar_etiquetas(revision)
    RESULTADO.parent.mkdir(parents=True, exist_ok=True)
    resultado.round(4).to_csv(RESULTADO, index=False)

    print("Revisión de etiquetas del catálogo (meta: acierto ≥ 90 %, con el intervalo completo sobre la meta)")
    for _, r in resultado.iterrows():
        if not r["revisadas"]:
            print(f"  {r['grupo']:<20} sin revisar todavía (0 de {r['de']})")
            continue
        veredicto = ("certifica la meta: todo el intervalo ≥ 90 %" if r["cumple_meta"]
                     else "compatible con la meta, sin certificarla" if r["ic95_alto"] >= META_ACIERTO
                     else "bajo la meta: todo el intervalo < 90 %")
        print(f"  {r['grupo']:<20} {r['aciertos']}/{r['revisadas']} = {r['acierto']:.1%} "
              f"(IC 95 % {r['ic95_bajo']:.1%}–{r['ic95_alto']:.1%}) → {veredicto}")

    aportes = aportes_de_la_revision(revision, perfil, date.today().isoformat())
    APORTES.mkdir(parents=True, exist_ok=True)
    aportes.to_csv(APORTES / "correcciones_etiquetas.csv", index=False)
    print(f"✓ {len(aportes)} etiquetas revisadas en {(APORTES / 'correcciones_etiquetas.csv').relative_to(RAIZ)}"
          "\n  para reentrenar con ellas: python scripts/aprender_de_usuarios.py "
          f"{APORTES.relative_to(RAIZ)}")
    print(f"✓ {RESULTADO.relative_to(RAIZ)}")
    return resultado


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    sub = ap.add_subparsers(dest="accion", required=True)
    p = sub.add_parser("preparar", help="genera la planilla vacía")
    p.add_argument("--forzar", action="store_true", help="sobrescribir la planilla (se pierde lo anotado)")
    sub.add_parser("evaluar", help="mide los resultados de lo ya revisado")
    args = ap.parse_args()
    preparar(args.forzar) if args.accion == "preparar" else evaluar()


if __name__ == "__main__":
    sys.exit(main())

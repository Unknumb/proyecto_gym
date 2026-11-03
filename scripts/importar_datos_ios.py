"""Importa al proyecto los datos registrados en la app de iPhone (``ios/``).

La app exporta cinco CSV (pestaña *Mis datos* → *Preparar los CSV*). Este script
los valida contra los datos del pipeline y los agrega, sin duplicar filas, a
``data/01_raw/app_ios/``:

=============================== ======================================================
``revisiones_recomendaciones``  Calificación de alternativas. Las nueve primeras
                                columnas son las de la revisión manual del equipo
                                (``data/01_raw/sustitucion_revision_manual.csv``).
``correcciones_etiquetas``      Etiquetas del perfil corregidas a mano.
``ejercicios_nuevos``           Ejercicios fuera del catálogo, con lo que predijo el
                                modelo en el teléfono y la etiqueta final.
``entrenamiento_series``        Series de entrenamiento.
``entrenamiento_dolor``         Dolor por zona al terminar cada sesión.
=============================== ======================================================

Uso, desde la raíz del repositorio::

    python scripts/importar_datos_ios.py ~/Downloads/sustituto_seguro_1762200000
    python scripts/importar_datos_ios.py <carpeta> --solo-validar
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
DESTINO = RAIZ / "data/01_raw/app_ios"

#: Mismas columnas que ``TablaExportable.columnas`` en
#: ``ios/SustitutoSeguroCore/Sources/SustitutoSeguroCore/Exportacion.swift``
#: (``tests/test_app_ios.py`` comprueba que coincidan).
COLUMNAS = {
    "revisiones_recomendaciones": [
        "articulacion_vetada", "original", "familia_original", "alternativa", "familia_alternativa",
        "misma_familia", "similitud", "veredicto", "comentario", "original_id", "alternativa_id", "perfil", "fecha"],
    "correcciones_etiquetas": [
        "ejercicio_id", "nombre", "objetivo", "etiqueta_pipeline", "origen_pipeline", "confianza_pipeline",
        "etiqueta_corregida", "comentario", "fecha"],
    "ejercicios_nuevos": [
        "nombre", "nombre_norm", "instrucciones_en", "familia_equipamiento", "musculos_primarios",
        "musculos_secundarios", "mecanica_modelo", "mecanica_confianza", "mecanica",
        "fuerza_modelo", "fuerza_confianza", "fuerza",
        "complejidad_tecnica_modelo", "complejidad_tecnica_confianza", "complejidad_tecnica",
        "impacto", "estabilidad", "carga_graduable", "fecha"],
    "entrenamiento_series": ["fecha", "sesion", "ejercicio_id", "nombre", "serie", "peso_kg", "repeticiones",
                             "esfuerzo_1_10"],
    "entrenamiento_dolor": ["fecha", "sesion", "zona", "dolor_0_10"],
}
VEREDICTOS = {"bueno", "aceptable", "malo", "inseguro"}


def _referencias() -> dict:
    """Lo que el pipeline conoce: ejercicios, músculos, equipamiento, articulaciones y
    clases de cada modelo."""
    ejercicios = pd.read_parquet(RAIZ / "data/07_model_output/familias_movimiento.parquet")
    ma = pd.read_csv(RAIZ / "data/03_primary/musculo_articulacion.csv")
    modelos = joblib.load(RAIZ / "models/modelos_supervisados.pkl")
    return {"ids": set(ejercicios["ejercicio_id"]), "musculos": set(ma["musculo"]),
            "articulaciones": set(ma["articulacion"]),
            "familias": set(ejercicios["familia_equipamiento"]),
            "clases": {o: set(map(str, m["clases"])) for o, m in modelos.items()}}


def validar(tabla: str, df: pd.DataFrame, ref: dict) -> list[str]:
    """Problemas encontrados; una lista vacía = la tabla se puede importar."""
    problemas = []
    if list(df.columns) != COLUMNAS[tabla]:
        return [f"columnas {list(df.columns)} ≠ {COLUMNAS[tabla]}"]

    def revisar(mascara: pd.Series, que: str) -> None:
        if mascara.any():
            problemas.append(f"{int(mascara.sum())} filas con {que} (fila {int(mascara.idxmax()) + 2} del CSV)")

    if pd.to_datetime(df["fecha"], errors="coerce", utc=True).isna().any():
        revisar(pd.to_datetime(df["fecha"], errors="coerce", utc=True).isna(), "fecha inválida")
    if tabla == "revisiones_recomendaciones":
        revisar(~df["veredicto"].isin(VEREDICTOS), "veredicto desconocido")
        revisar(~df["original_id"].isin(ref["ids"]) | ~df["alternativa_id"].isin(ref["ids"]), "ejercicio desconocido")
        revisar(~df["articulacion_vetada"].isin(ref["articulaciones"]), "articulación desconocida")
    elif tabla == "correcciones_etiquetas":
        revisar(~df["ejercicio_id"].isin(ref["ids"]), "ejercicio desconocido")
        revisar(~df["objetivo"].isin(ref["clases"]), "objetivo desconocido")
        validas = [str(e) in ref["clases"].get(o, set()) for o, e in zip(df["objetivo"], df["etiqueta_corregida"])]
        revisar(~pd.Series(validas, index=df.index), "etiqueta que el modelo no conoce")
    elif tabla == "ejercicios_nuevos":
        revisar(df["nombre"].fillna("").str.strip().eq(""), "nombre vacío")
        revisar(~df["familia_equipamiento"].isin(ref["familias"]), "equipamiento desconocido")
        for columna in ["musculos_primarios", "musculos_secundarios"]:
            desconocido = df[columna].fillna("").map(lambda s: bool(set(filter(None, s.split(";"))) - ref["musculos"]))
            revisar(desconocido, f"músculo desconocido en {columna}")
        for objetivo, clases in ref["clases"].items():
            final = df[objetivo].dropna().astype(str)
            revisar(~final.isin(clases).reindex(df.index, fill_value=True), f"etiqueta desconocida en {objetivo}")
    elif tabla == "entrenamiento_series":
        revisar(~df["ejercicio_id"].isin(ref["ids"]), "ejercicio desconocido")
        revisar((df["peso_kg"] < 0) | (df["repeticiones"] < 0), "peso o repeticiones negativos")
        revisar(~df["esfuerzo_1_10"].between(1, 10), "esfuerzo fuera de 1–10")
    elif tabla == "entrenamiento_dolor":
        revisar(~df["zona"].isin(ref["articulaciones"]), "zona desconocida")
        revisar(~df["dolor_0_10"].between(0, 10), "dolor fuera de 0–10")
    return problemas


def importar(carpeta: Path, destino: Path = DESTINO, solo_validar: bool = False) -> dict[str, int]:
    """Valida los CSV de ``carpeta`` y agrega a ``destino`` las filas nuevas. Si
    alguna tabla tiene problemas, no se importa ninguna."""
    ref = _referencias()
    tablas, errores = {}, {}
    for tabla in COLUMNAS:
        ruta = carpeta / f"{tabla}.csv"
        if not ruta.is_file():
            continue
        df = pd.read_csv(ruta, dtype={"comentario": str, "instrucciones_en": str})
        if problemas := validar(tabla, df, ref):
            errores[tabla] = problemas
        tablas[tabla] = df
    if errores:
        for tabla, problemas in errores.items():
            for problema in problemas:
                print(f"✗ {tabla}: {problema}")
        raise SystemExit("No se importó nada: corrige los datos y vuelve a intentarlo.")
    if not tablas:
        raise SystemExit(f"No hay CSV de la app en {carpeta}.")

    nuevas = {}
    destino.mkdir(parents=True, exist_ok=True)
    for tabla, df in tablas.items():
        ruta = destino / f"{tabla}.csv"
        previo = pd.read_csv(ruta, dtype=str) if ruta.is_file() else pd.DataFrame(columns=COLUMNAS[tabla])
        combinado = pd.concat([previo, df.astype(str).where(df.notna(), None)], ignore_index=True)
        combinado = combinado.drop_duplicates(keep="first")
        nuevas[tabla] = len(combinado) - len(previo)
        if not solo_validar:
            combinado.to_csv(ruta, index=False)
        print(f"✓ {tabla}: {len(df)} filas, {nuevas[tabla]} nuevas"
              + ("" if solo_validar else f" → {ruta.relative_to(RAIZ) if ruta.is_relative_to(RAIZ) else ruta}"))
    return nuevas


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("carpeta", type=Path, help="carpeta con los CSV que exportó la app")
    ap.add_argument("--solo-validar", action="store_true", help="validar sin escribir nada")
    args = ap.parse_args()
    importar(args.carpeta.expanduser(), solo_validar=args.solo_validar)


if __name__ == "__main__":
    sys.exit(main())

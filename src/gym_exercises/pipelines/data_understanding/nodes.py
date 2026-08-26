"""
Nodos del pipeline Data Understanding (EDA).

Cada función es pura: recibe DataFrames / dicts y devuelve DataFrames / dicts.
No produce efectos secundarios (I/O gestionado por el catálogo de Kedro).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Auditoría estructural del dataset
# ─────────────────────────────────────────────────────────────────────────────
def audit_dataset_structure(raw_data: pd.DataFrame) -> dict[str, Any]:
    """Calcula dimensiones, nulos, duplicados y tipos de datos.

    Args:
        raw_data: DataFrame cargado directamente del JSON crudo.

    Returns:
        Diccionario con las métricas de auditoría.
    """
    n_rows, n_cols = raw_data.shape
    null_counts = raw_data.isnull().sum().to_dict()
    null_pct = raw_data.isnull().mean().mul(100).round(2).to_dict()
    duplicate_rows = int(raw_data.duplicated(subset=["id"]).sum())
    dtypes = raw_data.dtypes.astype(str).to_dict()

    report = {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "null_counts": null_counts,
        "null_pct": null_pct,
        "duplicate_rows": duplicate_rows,
        "dtypes": dtypes,
        "columns": list(raw_data.columns),
    }

    logger.info(
        "Auditoría completada: %d filas × %d columnas, %d duplicados por id.",
        n_rows,
        n_cols,
        duplicate_rows,
    )
    return report


# ─────────────────────────────────────────────────────────────────────────────
# 2. Aplanar / desanidar metadatos semi-estructurados
# ─────────────────────────────────────────────────────────────────────────────
def flatten_exercise_metadata(raw_data: pd.DataFrame) -> pd.DataFrame:
    """Desanida campos JSON y produce un DataFrame tabular limpio.

    Operaciones:
    - Extrae las instrucciones en español desde ``instructions`` (dict → str).
    - Extrae los pasos en español desde ``instruction_steps`` (dict → lista).
    - Convierte ``secondary_muscles`` (lista) en filas separadas (explode)
      y lo guarda en una columna ``secondary_muscle`` (singular).
    - Descarta columnas con objetos complejos que ya se extrajeron.

    Args:
        raw_data: DataFrame cargado directamente del JSON crudo.

    Returns:
        DataFrame tidy apto para análisis tabular y exportación a Parquet.
    """
    df = raw_data.copy()

    # -- Extraer instrucciones en español (campo tipo dict) ----------------
    def _safe_get_es(val: Any) -> str | None:
        """Obtiene la clave 'es' de un dict; devuelve None si falla."""
        if isinstance(val, dict):
            return val.get("es")
        return None

    if "instructions" in df.columns:
        df["instructions_es"] = df["instructions"].apply(_safe_get_es)

    if "instruction_steps" in df.columns:
        df["instruction_steps_es"] = df["instruction_steps"].apply(
            lambda v: v.get("es") if isinstance(v, dict) else None
        )
        # Convertir la lista de pasos a un string numerado
        df["instruction_steps_es_text"] = df["instruction_steps_es"].apply(
            lambda steps: (
                "\n".join(f"{i+1}. {s}" for i, s in enumerate(steps))
                if isinstance(steps, list)
                else None
            )
        )

    # -- Explotar secondary_muscles en filas individuales ------------------
    if "secondary_muscles" in df.columns:
        # Reemplazar listas vacías / NaN con [None] para no perder filas
        df["secondary_muscles"] = df["secondary_muscles"].apply(
            lambda v: v if isinstance(v, list) and len(v) > 0 else [None]
        )
        df = df.explode("secondary_muscles", ignore_index=True)
        df = df.rename(columns={"secondary_muscles": "secondary_muscle"})

    # -- Seleccionar columnas finales (descartar dicts anidados) -----------
    drop_cols = [
        c
        for c in [
            "instructions",
            "instruction_steps",
            "instruction_steps_es",
        ]
        if c in df.columns
    ]
    df = df.drop(columns=drop_cols)

    # -- Normalizar strings ------------------------------------------------
    str_cols = [
        "name",
        "category",
        "body_part",
        "equipment",
        "target",
        "muscle_group",
        "secondary_muscle",
    ]
    for col in str_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.lower()
            df[col] = df[col].replace({"nan": None, "none": None})

    logger.info(
        "Flatten completado: %d filas × %d columnas tras explode.",
        *df.shape,
    )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3. Métricas de distribución
# ─────────────────────────────────────────────────────────────────────────────
def compute_distribution_metrics(
    exercises_clean: pd.DataFrame,
) -> pd.DataFrame:
    """Genera conteos de frecuencia por dimensión clave.

    Calcula distribuciones para: ``body_part``, ``equipment``, ``target``,
    ``muscle_group`` y ``secondary_muscle``.

    Args:
        exercises_clean: DataFrame tidy (salida de ``flatten_exercise_metadata``).

    Returns:
        DataFrame largo con columnas ``dimension``, ``value``, ``count``.
    """
    # Usamos los IDs únicos para conteos (evitar inflar por explode)
    frames: list[pd.DataFrame] = []

    dimensions = ["body_part", "equipment", "target", "muscle_group"]
    for dim in dimensions:
        if dim not in exercises_clean.columns:
            continue
        # Contar ejercicios únicos (por id) para cada valor de dimensión
        counts = (
            exercises_clean.dropna(subset=[dim])
            .drop_duplicates(subset=["id", dim])
            .groupby(dim, as_index=False)
            .agg(count=("id", "nunique"))
            .rename(columns={dim: "value"})
            .assign(dimension=dim)
        )
        frames.append(counts)

    # secondary_muscle se cuenta directamente (ya está exploded)
    if "secondary_muscle" in exercises_clean.columns:
        sec = (
            exercises_clean.dropna(subset=["secondary_muscle"])
            .groupby("secondary_muscle", as_index=False)
            .agg(count=("id", "nunique"))
            .rename(columns={"secondary_muscle": "value"})
            .assign(dimension="secondary_muscle")
        )
        frames.append(sec)

    metrics = (
        pd.concat(frames, ignore_index=True)[["dimension", "value", "count"]]
        .sort_values(["dimension", "count"], ascending=[True, False])
        .reset_index(drop=True)
    )

    logger.info(
        "Métricas de distribución calculadas: %d filas para %d dimensiones.",
        len(metrics),
        metrics["dimension"].nunique(),
    )
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# 4. Reporte resumen EDA
# ─────────────────────────────────────────────────────────────────────────────
def save_eda_summary(
    audit_report: dict[str, Any],
    distribution_metrics: pd.DataFrame,
) -> pd.DataFrame:
    """Combina la auditoría y las métricas en un único DataFrame resumen.

    La primera sección contiene los metadatos del dataset (filas, columnas,
    duplicados, nulos). La segunda sección reproduce los top-5 por dimensión.

    Args:
        audit_report: Diccionario de auditoría (de ``audit_dataset_structure``).
        distribution_metrics: DataFrame largo de distribuciones.

    Returns:
        DataFrame resumen exportable a CSV.
    """
    # -- Sección 1: Metadatos generales ------------------------------------
    meta_rows = [
        {"section": "meta", "metric": "n_rows", "value": str(audit_report["n_rows"])},
        {"section": "meta", "metric": "n_cols", "value": str(audit_report["n_cols"])},
        {
            "section": "meta",
            "metric": "duplicate_rows",
            "value": str(audit_report["duplicate_rows"]),
        },
    ]
    # Nulos por columna
    for col, cnt in audit_report["null_counts"].items():
        pct = audit_report["null_pct"][col]
        meta_rows.append(
            {
                "section": "nulls",
                "metric": f"null_{col}",
                "value": f"{cnt} ({pct}%)",
            }
        )
    # Tipos de datos
    for col, dtype in audit_report["dtypes"].items():
        meta_rows.append(
            {"section": "dtypes", "metric": f"dtype_{col}", "value": dtype}
        )

    meta_df = pd.DataFrame(meta_rows)

    # -- Sección 2: Top-5 por dimensión ------------------------------------
    top5_rows: list[dict[str, str]] = []
    for dim in distribution_metrics["dimension"].unique():
        subset = distribution_metrics[distribution_metrics["dimension"] == dim].head(5)
        for _, row in subset.iterrows():
            top5_rows.append(
                {
                    "section": f"top5_{dim}",
                    "metric": str(row["value"]),
                    "value": str(row["count"]),
                }
            )
    top5_df = pd.DataFrame(top5_rows)

    summary = pd.concat([meta_df, top5_df], ignore_index=True)

    logger.info("Reporte EDA generado: %d filas.", len(summary))
    return summary

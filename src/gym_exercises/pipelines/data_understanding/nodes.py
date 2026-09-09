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


# ─────────────────────────────────────────────────────────────────────────────
# 5. Feature engineering a grano ejercicio (base cuantitativa del EDA)
# ─────────────────────────────────────────────────────────────────────────────
def build_exercise_features(raw_data: pd.DataFrame) -> pd.DataFrame:
    """Construye la tabla analítica cuantitativa: 1 fila = 1 ejercicio.

    Esta tabla es la base del EDA diagnóstico (IE3). Se separa deliberadamente
    de ``flatten_exercise_metadata`` porque aquel nodo aplica ``explode`` sobre
    ``secondary_muscles`` y por lo tanto opera a grano *ejercicio × músculo*,
    lo que inflaría los estadísticos univariados.

    Correcciones aplicadas respecto de la versión preliminar del equipo:

    1. ``id`` se fuerza a *string* con relleno de ceros. Leído como entero,
       ``"0001"`` colapsa a ``1`` y se rompe la llave hacia ``gif_url`` /
       ``media_id`` (``videos/0001-2gPfomN.gif``), necesaria para la fase
       cinemática posterior.
    2. ``musculo_primario`` se deriva de ``target`` y **no** de ``muscle_group``.
       Se verificó que ``muscle_group`` está contenido en ``secondary_muscles``
       en el 100% de los registros: es un músculo sinergista, no el agonista.
    3. ``n_musculos_total`` deduplica el conjunto ``{target} ∪ secondary``
       en lugar de asumir ``1 + len(secondary)``, que sobrecuenta cuando el
       músculo objetivo también aparece listado como secundario.
    4. Se descartan las columnas ``instructions`` / ``instruction_steps``
       completas (10 idiomas anidados ≈ 15.8 MB en memoria) y se conserva solo
       el texto en español ya extraído, en línea con la restricción FinOps de
       mantener livianas las capas intermedias.

    Args:
        raw_data: DataFrame cargado directamente del JSON crudo.

    Returns:
        DataFrame a grano ejercicio con variables categóricas y cuantitativas.
    """
    df = raw_data.copy()

    # -- Corrección 1: preservar el id como llave textual ------------------
    df["id"] = df["id"].astype(str).str.strip().str.zfill(4)

    # -- Extracción del contenido en español -------------------------------
    steps_es = df["instruction_steps"].apply(
        lambda v: v.get("es", []) if isinstance(v, dict) else []
    )
    text_es = df["instructions"].apply(
        lambda v: v.get("es", "") if isinstance(v, dict) else ""
    )
    df["instrucciones_es"] = text_es

    # -- Variables de volumen y complejidad textual ------------------------
    df["n_pasos"] = steps_es.apply(len)
    df["n_idiomas"] = df["instructions"].apply(
        lambda v: len(v) if isinstance(v, dict) else 0
    )
    df["n_caracteres"] = text_es.str.len()
    df["n_palabras"] = text_es.str.split().apply(len)
    df["palabras_por_paso"] = (df["n_palabras"] / df["n_pasos"]).round(3)
    df["n_palabras_paso_max"] = steps_es.apply(
        lambda pasos: max((len(p.split()) for p in pasos), default=0)
    )
    df["n_palabras_nombre"] = df["name"].str.split().apply(len)

    # -- Corrección 2: jerarquía muscular correcta -------------------------
    df["musculo_primario"] = df["target"]
    df["musculo_sinergista"] = df["muscle_group"]

    secundarios = df["secondary_muscles"].apply(
        lambda v: v if isinstance(v, list) else []
    )
    df["n_musculos_secundarios"] = secundarios.apply(len)

    # -- Corrección 3: conteo deduplicado ----------------------------------
    df["n_musculos_total"] = [
        len(set(sec) | {tgt}) for sec, tgt in zip(secundarios, df["target"])
    ]

    # -- Frecuencias de contexto (proxy de accesibilidad y de sesgo) -------
    df["frecuencia_equipamiento"] = df.groupby("equipment")["id"].transform("count")
    df["frecuencia_musculo_primario"] = df.groupby("target")["id"].transform("count")

    columnas = [
        "id",
        "name",
        "category",
        "body_part",
        "equipment",
        "musculo_primario",
        "musculo_sinergista",
        "gif_url",
        "media_id",
        "instrucciones_es",
        "n_pasos",
        "n_palabras",
        "n_caracteres",
        "palabras_por_paso",
        "n_palabras_paso_max",
        "n_palabras_nombre",
        "n_musculos_secundarios",
        "n_musculos_total",
        "n_idiomas",
        "frecuencia_equipamiento",
        "frecuencia_musculo_primario",
    ]
    features = df[columnas].copy()

    logger.info(
        "Tabla de features construida: %d filas × %d columnas (%.2f MB en memoria).",
        *features.shape,
        features.memory_usage(deep=True).sum() / 1e6,
    )
    return features


# ─────────────────────────────────────────────────────────────────────────────
# 6. IE3 — Detección de outliers por Rango Intercuartílico (IQR)
# ─────────────────────────────────────────────────────────────────────────────
#: Variables cuantitativas sobre las que se aplica el criterio de Tukey.
VARIABLES_CUANTITATIVAS: list[str] = [
    "n_pasos",
    "n_palabras",
    "n_caracteres",
    "palabras_por_paso",
    "n_palabras_paso_max",
    "n_palabras_nombre",
    "n_musculos_secundarios",
    "n_musculos_total",
    "frecuencia_equipamiento",
    "frecuencia_musculo_primario",
]


def detect_outliers_iqr(
    exercise_features: pd.DataFrame,
    factor: float = 1.5,
) -> pd.DataFrame:
    """Aplica el criterio de Tukey sobre cada variable cuantitativa.

    Para cada variable se calcula:

    .. math::

        IQR = Q_3 - Q_1

        \\text{Límite inferior} = Q_1 - k \\cdot IQR

        \\text{Límite superior} = Q_3 + k \\cdot IQR

    con :math:`k = 1.5` (valor por defecto). Todo registro fuera del intervalo
    se marca como atípico.

    Args:
        exercise_features: Tabla analítica a grano ejercicio.
        factor: Multiplicador ``k`` del IQR. 1.5 = atípico, 3.0 = extremo.

    Returns:
        DataFrame con una fila por variable y sus estadísticos de frontera.
    """
    columnas = [c for c in VARIABLES_CUANTITATIVAS if c in exercise_features.columns]
    datos = exercise_features[columnas]

    q1 = datos.quantile(0.25)
    q3 = datos.quantile(0.75)
    iqr = q3 - q1
    limite_inferior = q1 - factor * iqr
    limite_superior = q3 + factor * iqr

    # Máscara vectorizada: True donde el valor cae fuera de los bigotes.
    fuera_rango = (datos.lt(limite_inferior, axis=1)) | (
        datos.gt(limite_superior, axis=1)
    )

    reporte = pd.DataFrame(
        {
            "variable": columnas,
            "Q1": q1.values,
            "mediana": datos.median().values,
            "Q3": q3.values,
            "IQR": iqr.values,
            "limite_inferior": limite_inferior.values,
            "limite_superior": limite_superior.values,
            "n_outliers": fuera_rango.sum().values,
            "pct_outliers": (fuera_rango.mean() * 100).round(2).values,
            "n_outliers_inferiores": datos.lt(limite_inferior, axis=1).sum().values,
            "n_outliers_superiores": datos.gt(limite_superior, axis=1).sum().values,
        }
    ).round(3)

    logger.info(
        "IQR (k=%.1f) calculado sobre %d variables; %d outliers en total.",
        factor,
        len(columnas),
        int(reporte["n_outliers"].sum()),
    )
    return reporte


# ─────────────────────────────────────────────────────────────────────────────
# 7. IE3 — Estadísticos de forma de las distribuciones
# ─────────────────────────────────────────────────────────────────────────────
def compute_shape_statistics(exercise_features: pd.DataFrame) -> pd.DataFrame:
    """Calcula tendencia central, dispersión y forma de cada distribución.

    Incluye asimetría (*skewness*) y curtosis de Fisher para documentar el
    sesgo de las variables de complejidad y volumen, tal como exige el
    análisis de forma del EDA diagnóstico.

    Args:
        exercise_features: Tabla analítica a grano ejercicio.

    Returns:
        DataFrame con una fila por variable cuantitativa.
    """
    columnas = [c for c in VARIABLES_CUANTITATIVAS if c in exercise_features.columns]
    datos = exercise_features[columnas]

    asimetria = datos.skew()
    stats = pd.DataFrame(
        {
            "variable": columnas,
            "n": datos.count().values,
            "media": datos.mean().values,
            "mediana": datos.median().values,
            "desv_std": datos.std().values,
            "cv": (datos.std() / datos.mean()).values,
            "minimo": datos.min().values,
            "maximo": datos.max().values,
            "asimetria": asimetria.values,
            "curtosis": datos.kurt().values,
        }
    ).round(3)

    # Interpretación cualitativa del sesgo (regla práctica de Bulmer).
    stats["forma"] = pd.cut(
        stats["asimetria"],
        bins=[-np.inf, -1.0, -0.5, 0.5, 1.0, np.inf],
        labels=[
            "sesgo negativo fuerte",
            "sesgo negativo moderado",
            "aproximadamente simétrica",
            "sesgo positivo moderado",
            "sesgo positivo fuerte",
        ],
    ).astype(str)

    logger.info("Estadísticos de forma calculados para %d variables.", len(columnas))
    return stats


# ─────────────────────────────────────────────────────────────────────────────
# 8. IE3 — Matriz de correlación de Pearson
# ─────────────────────────────────────────────────────────────────────────────
def compute_correlation_matrix(exercise_features: pd.DataFrame) -> pd.DataFrame:
    """Calcula la matriz de correlación de Pearson entre variables cuantitativas.

    Sirve para auditar multicolinealidad antes de cualquier modelado posterior:
    pares con :math:`|r| > 0.9` indican información redundante.

    Args:
        exercise_features: Tabla analítica a grano ejercicio.

    Returns:
        Matriz cuadrada de correlaciones, con la variable como primera columna.
    """
    columnas = [c for c in VARIABLES_CUANTITATIVAS if c in exercise_features.columns]
    matriz = exercise_features[columnas].corr(method="pearson").round(4)

    # Auditoría de multicolinealidad sobre el triángulo superior.
    triangulo = matriz.where(np.triu(np.ones(matriz.shape), k=1).astype(bool))
    redundantes = (
        triangulo.stack().abs().pipe(lambda s: s[s > 0.9]).index.tolist()
    )
    if redundantes:
        logger.warning(
            "Pares con |r| > 0.9 (multicolinealidad): %s", redundantes
        )

    return matriz.reset_index(names="variable")

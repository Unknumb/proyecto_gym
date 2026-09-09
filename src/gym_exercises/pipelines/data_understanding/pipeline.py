"""
Pipeline Data Understanding — definición del DAG.

Conecta los 4 nodos de EDA:

    raw_exercises_data
        ├── audit_dataset_structure   → audit_report (MemoryDataset)
        └── flatten_exercise_metadata → intermediate_exercises_clean
                                            └── compute_distribution_metrics → distribution_metrics
                                                                                    │
    audit_report + distribution_metrics → save_eda_summary → eda_summary_report

Rama IE3 (EDA diagnóstico), a grano ejercicio:

    raw_exercises_data → build_exercise_features → intermediate_exercise_features
                                                        ├── detect_outliers_iqr
                                                        ├── compute_shape_statistics
                                                        └── compute_correlation_matrix
"""

from __future__ import annotations

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    audit_dataset_structure,
    build_exercise_features,
    compute_correlation_matrix,
    compute_distribution_metrics,
    compute_shape_statistics,
    detect_outliers_iqr,
    flatten_exercise_metadata,
    save_eda_summary,
)


def create_pipeline(**kwargs) -> Pipeline:  # noqa: ARG001
    """Crea el pipeline de Data Understanding (EDA)."""
    return pipeline(
        [
            node(
                func=audit_dataset_structure,
                inputs="raw_exercises_data",
                outputs="audit_report",
                name="audit_dataset_structure",
            ),
            node(
                func=flatten_exercise_metadata,
                inputs="raw_exercises_data",
                outputs="intermediate_exercises_clean",
                name="flatten_exercise_metadata",
            ),
            node(
                func=compute_distribution_metrics,
                inputs="intermediate_exercises_clean",
                outputs="distribution_metrics",
                name="compute_distribution_metrics",
            ),
            node(
                func=save_eda_summary,
                inputs=["audit_report", "distribution_metrics"],
                outputs="eda_summary_report",
                name="save_eda_summary",
            ),
            # ── IE3: EDA diagnóstico ──────────────────────────────────────
            node(
                func=build_exercise_features,
                inputs="raw_exercises_data",
                outputs="intermediate_exercise_features",
                name="build_exercise_features",
            ),
            node(
                func=detect_outliers_iqr,
                inputs="intermediate_exercise_features",
                outputs="outliers_iqr_report",
                name="detect_outliers_iqr",
            ),
            node(
                func=compute_shape_statistics,
                inputs="intermediate_exercise_features",
                outputs="shape_statistics_report",
                name="compute_shape_statistics",
            ),
            node(
                func=compute_correlation_matrix,
                inputs="intermediate_exercise_features",
                outputs="correlation_matrix_report",
                name="compute_correlation_matrix",
            ),
        ]
    )

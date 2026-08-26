"""
Pipeline Data Understanding — definición del DAG.

Conecta los 4 nodos de EDA:

    raw_exercises_data
        ├── audit_dataset_structure   → audit_report (MemoryDataset)
        └── flatten_exercise_metadata → intermediate_exercises_clean
                                            └── compute_distribution_metrics → distribution_metrics
                                                                                    │
    audit_report + distribution_metrics → save_eda_summary → eda_summary_report
"""

from __future__ import annotations

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    audit_dataset_structure,
    compute_distribution_metrics,
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
        ]
    )

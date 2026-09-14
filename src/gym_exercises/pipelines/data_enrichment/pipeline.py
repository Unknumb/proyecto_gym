"""
Pipeline Data Enrichment — definición del DAG.

    raw_megagym_data → preparar_megagym → intermediate_megagym
                                              │
    intermediate_exercise_features ───────────┼──→ emparejar_con_megagym → megagym_match_report
                                              │                                  │
                                              └──→ enrich_with_external_metadata ←┘
                                                        → primary_exercise_features_enriched

``intermediate_exercise_features`` lo produce el pipeline ``data_understanding``;
por eso ``kedro run`` ejecuta ambos, en ese orden.
"""

from __future__ import annotations

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import emparejar_con_megagym, enrich_with_external_metadata, preparar_megagym


def create_pipeline(**kwargs) -> Pipeline:  # noqa: ARG001
    """Crea el pipeline de enriquecimiento con megaGymDataset."""
    return pipeline(
        [
            node(
                func=preparar_megagym,
                inputs="raw_megagym_data",
                outputs="intermediate_megagym",
                name="preparar_megagym",
            ),
            node(
                func=emparejar_con_megagym,
                inputs=[
                    "intermediate_exercise_features",
                    "intermediate_megagym",
                    "params:emparejamiento_megagym",
                ],
                outputs="megagym_match_report",
                name="emparejar_con_megagym",
            ),
            node(
                func=enrich_with_external_metadata,
                inputs=[
                    "intermediate_exercise_features",
                    "intermediate_megagym",
                    "megagym_match_report",
                ],
                outputs="primary_exercise_features_enriched",
                name="enrich_with_external_metadata",
            ),
        ]
    )

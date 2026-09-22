"""
Pipeline Integración de Fuentes — definición del DAG.

    musculo_articulacion_manual ─┐
    vocabulario_muscular_externo ├→ construir_ontologia_muscular → vocabulario_muscular
    articulaciones_canonicas ────┤                               → primary_musculo_articulacion
    regiones_compuestas ─────────┘

    raw_exercises_data + primary_exercise_features_enriched → preparar_catalogo
    raw_free_exercise_db                                     → preparar_free_exercise_db
    raw_wger                                                 → preparar_wger
          │
          ├→ emparejar_free_exercise_db → match_free_exercise_db_report
          ├→ emparejar_wger            → match_wger_report
          └→ consolidar_ejercicios → primary_ejercicios + primary_ejercicio_musculo
                                   → resumir_integracion → integracion_fuentes_resumen

``primary_exercise_features_enriched`` lo produce ``data_enrichment`` (nivel de
megaGymDataset), por eso este pipeline corre después de los otros dos.
"""

from __future__ import annotations

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    consolidar_ejercicios,
    construir_ontologia_muscular,
    emparejar_free_exercise_db,
    emparejar_wger,
    preparar_catalogo,
    preparar_free_exercise_db,
    preparar_wger,
    resumir_integracion,
)


def create_pipeline(**kwargs) -> Pipeline:  # noqa: ARG001
    """Crea el pipeline que integra free-exercise-db y wger con el catálogo."""
    return pipeline(
        [
            node(
                func=construir_ontologia_muscular,
                inputs=["musculo_articulacion_manual", "vocabulario_muscular_externo",
                        "articulaciones_canonicas", "regiones_compuestas"],
                outputs=["vocabulario_muscular", "primary_musculo_articulacion"],
                name="construir_ontologia_muscular",
            ),
            node(
                func=preparar_catalogo,
                inputs=["raw_exercises_data", "primary_exercise_features_enriched",
                        "vocabulario_muscular", "equipamiento_canonico"],
                outputs="intermediate_catalogo_unificado",
                name="preparar_catalogo",
            ),
            node(
                func=preparar_free_exercise_db,
                inputs=["raw_free_exercise_db", "vocabulario_muscular", "equipamiento_canonico"],
                outputs="intermediate_free_exercise_db",
                name="preparar_free_exercise_db",
            ),
            node(
                func=preparar_wger,
                inputs=["raw_wger", "vocabulario_muscular", "equipamiento_canonico"],
                outputs="intermediate_wger",
                name="preparar_wger",
            ),
            node(
                func=emparejar_free_exercise_db,
                inputs=["intermediate_catalogo_unificado", "intermediate_free_exercise_db",
                        "primary_musculo_articulacion", "regiones_compuestas",
                        "params:emparejamiento_fuentes"],
                outputs="match_free_exercise_db_report",
                name="emparejar_free_exercise_db",
            ),
            node(
                func=emparejar_wger,
                inputs=["intermediate_catalogo_unificado", "intermediate_free_exercise_db",
                        "intermediate_wger", "primary_musculo_articulacion",
                        "regiones_compuestas", "params:emparejamiento_fuentes"],
                outputs="match_wger_report",
                name="emparejar_wger",
            ),
            node(
                func=consolidar_ejercicios,
                inputs=["intermediate_catalogo_unificado", "intermediate_free_exercise_db",
                        "intermediate_wger", "match_free_exercise_db_report",
                        "match_wger_report"],
                outputs=["primary_ejercicios", "primary_ejercicio_musculo"],
                name="consolidar_ejercicios",
            ),
            node(
                func=resumir_integracion,
                inputs=["primary_ejercicios", "primary_ejercicio_musculo",
                        "primary_musculo_articulacion"],
                outputs="integracion_fuentes_resumen",
                name="resumir_integracion",
            ),
        ]
    )

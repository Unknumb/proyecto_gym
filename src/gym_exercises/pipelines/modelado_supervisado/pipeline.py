"""
Pipeline Modelado Supervisado — definición del DAG.

    intermediate_free_exercise_db ─┐
    primary_ejercicios ────────────┼→ construir_tabla_modelado → model_input_tabla_modelado
    match_free_exercise_db_report ─┤                                   │
    primary_musculo_articulacion ──┘                                   ▼
                                        entrenar_modelos → modelos_comparacion
                                                         → modelos_predicciones_test
                                                         → modelos_supervisados (models/)
                                                                       │
                                        predecir_etiquetas ←───────────┘
                                                         → model_output_perfil_exigencia

Requiere las salidas de ``integracion_fuentes``.
"""

from __future__ import annotations

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import construir_tabla_modelado, entrenar_modelos, predecir_etiquetas


def create_pipeline(**kwargs) -> Pipeline:  # noqa: ARG001
    """Crea el pipeline que entrena y compara los modelos supervisados."""
    return pipeline(
        [
            node(
                func=construir_tabla_modelado,
                inputs=["intermediate_free_exercise_db", "primary_ejercicios",
                        "match_free_exercise_db_report", "primary_musculo_articulacion",
                        "params:modelado_supervisado"],
                outputs="model_input_tabla_modelado",
                name="construir_tabla_modelado",
            ),
            node(
                func=entrenar_modelos,
                inputs=["model_input_tabla_modelado", "params:modelado_supervisado"],
                outputs=["modelos_comparacion", "modelos_predicciones_test",
                         "modelos_supervisados"],
                name="entrenar_modelos",
            ),
            node(
                func=predecir_etiquetas,
                inputs=["model_input_tabla_modelado", "primary_ejercicios",
                        "modelos_supervisados"],
                outputs="model_output_perfil_exigencia",
                name="predecir_etiquetas",
            ),
        ]
    )

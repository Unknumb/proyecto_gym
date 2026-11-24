"""
Pipeline Modelado Supervisado — definición del DAG.

    intermediate_free_exercise_db ─┐
    primary_ejercicios ────────────┼→ construir_tabla_modelado → model_input_tabla_base
    match_free_exercise_db_report ─┤                                   │
    primary_musculo_articulacion ──┘                                   ▼
    aportes_correcciones_etiquetas ─────→ incorporar_aportes → tabla con aportes
                                                             → aportes_consolidados
                                                                       │
                                        entrenar_modelos ←─────────────┤
                                                         → modelos_comparacion
                                                         → modelos_predicciones_test
                                                                       │
                                        evaluar_aprendizaje ←──────────┘
                                                         → aprendizaje_usuarios_informe
                                                         → modelos_supervisados (models/)
                                                         → model_input_tabla_modelado
                                                                       │
                                        diagnosticar_modelos ←─────────┤
                                                         → informe: CV repetida y contraste
                                                           pareado, calibración, curva de
                                                           aprendizaje, control de fuga,
                                                           importancia, validación adversarial
                                        predecir_etiquetas ←───────────┘
                                                         → model_output_perfil_exigencia

Los aportes son las correcciones de etiquetas que hacen las personas en la página
web y en la app de iPhone (README §18). Sin aportes, ``model_input_tabla_modelado``
y ``modelos_supervisados`` son idénticos a los de la EV2 y la EV3.

Requiere las salidas de ``integracion_fuentes``.
"""

from __future__ import annotations

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (construir_tabla_modelado, diagnosticar_modelos, entrenar_modelos,
                    evaluar_aprendizaje, incorporar_aportes, predecir_etiquetas)


def create_pipeline(**kwargs) -> Pipeline:  # noqa: ARG001
    """Crea el pipeline que entrena y compara los modelos supervisados."""
    return pipeline(
        [
            node(
                func=construir_tabla_modelado,
                inputs=["intermediate_free_exercise_db", "primary_ejercicios",
                        "match_free_exercise_db_report", "primary_musculo_articulacion",
                        "params:modelado_supervisado"],
                outputs="model_input_tabla_base",
                name="construir_tabla_modelado",
            ),
            node(
                func=incorporar_aportes,
                inputs=["model_input_tabla_base", "aportes_correcciones_etiquetas",
                        "params:modelado_supervisado"],
                outputs=["model_input_tabla_con_aportes", "aportes_consolidados"],
                name="incorporar_aportes",
            ),
            node(
                func=entrenar_modelos,
                inputs=["model_input_tabla_con_aportes", "params:modelado_supervisado"],
                outputs=["modelos_comparacion", "modelos_predicciones_test",
                         "modelos_candidatos"],
                name="entrenar_modelos",
            ),
            node(
                func=evaluar_aprendizaje,
                inputs=["model_input_tabla_base", "aportes_correcciones_etiquetas",
                        "model_input_tabla_con_aportes", "modelos_candidatos",
                        "params:modelado_supervisado"],
                outputs=["aprendizaje_usuarios_informe", "modelos_supervisados",
                         "model_input_tabla_modelado"],
                name="evaluar_aprendizaje",
            ),
            node(
                func=diagnosticar_modelos,
                inputs=["model_input_tabla_modelado", "modelos_supervisados",
                        "params:modelado_supervisado"],
                outputs=["modelos_cv_repetida", "modelos_contraste_pareado",
                         "modelos_calibracion", "modelos_curva_aprendizaje",
                         "modelos_control_fuga", "modelos_importancia",
                         "validacion_adversarial"],
                name="diagnosticar_modelos",
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

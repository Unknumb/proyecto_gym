"""
Pipeline Ensambles — definición del DAG (EV3, README §16).

    model_input_tabla_modelado ─┐
    modelos_supervisados ───────┼→ comparar_ensambles → ensambles_busqueda_boosting
    params:modelado_supervisado ┘                     → ensambles_cv_repetida
                                                      → ensambles_comparacion

Solo produce informes: no cambia ``modelos_supervisados`` ni las etiquetas.
Requiere las salidas de ``modelado_supervisado``.
"""

from __future__ import annotations

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import comparar_ensambles


def create_pipeline(**kwargs) -> Pipeline:  # noqa: ARG001
    """Crea el pipeline que compara los ensambles con los modelos actuales."""
    return pipeline(
        [
            node(
                func=comparar_ensambles,
                inputs=["model_input_tabla_modelado", "modelos_supervisados",
                        "params:modelado_supervisado"],
                outputs=["ensambles_busqueda_boosting", "ensambles_cv_repetida",
                         "ensambles_comparacion"],
                name="comparar_ensambles",
            ),
        ]
    )

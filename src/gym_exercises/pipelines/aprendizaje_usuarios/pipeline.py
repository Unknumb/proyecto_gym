"""
Pipeline Aprendizaje de usuarios — definición del DAG (README §18).

    model_input_tabla_base ─┐
    modelos_supervisados ───┼→ simular_aprendizaje → aprendizaje_simulacion
    params ─────────────────┘

Solo produce un informe: no cambia ``modelos_supervisados`` ni las etiquetas. El
aprendizaje con aportes reales ocurre en ``modelado_supervisado``
(``incorporar_aportes`` y ``evaluar_aprendizaje``). Requiere sus salidas.
"""

from __future__ import annotations

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import simular_aprendizaje


def create_pipeline(**kwargs) -> Pipeline:  # noqa: ARG001
    """Crea el pipeline que simula aportes de usuarios y mide cuánto aprende el modelo."""
    return pipeline([
        node(
            func=simular_aprendizaje,
            inputs=["model_input_tabla_base", "modelos_supervisados",
                    "params:modelado_supervisado"],
            outputs="aprendizaje_simulacion",
            name="simular_aprendizaje",
        ),
    ])

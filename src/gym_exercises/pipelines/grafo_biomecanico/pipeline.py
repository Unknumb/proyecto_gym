"""
Pipeline Grafo Biomecánico — definición del DAG.

    primary_ejercicios ───────────┐
    model_output_perfil_exigencia ┤
    primary_ejercicio_musculo ────┼→ construir_grafo → grafo_biomecanico (models/)
    primary_musculo_articulacion ─┤                  → grafo_nodos_ejercicio
    regiones_compuestas ──────────┘                          │
                     ┌───────────────────────────────────────┤
                     ▼                                       ▼
    detectar_comunidades (Louvain)            entrenar_node2vec → embeddings_node2vec (models/)
                     │                                       │
                     │                        agrupar_embeddings (K-Means)
                     └──────────────┬────────────────────────┘
                                    ▼
                         consolidar_familias → model_output_familias_movimiento
                                    │        → familias_perfiles, clustering_comparacion
                                    ▼
                         evaluar_sustitucion → sustitucion_cobertura, sustitucion_contexto

Requiere las salidas de ``integracion_fuentes`` y ``modelado_supervisado``.
"""

from __future__ import annotations

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (agrupar_embeddings, consolidar_familias, construir_grafo,
                    detectar_comunidades, entrenar_node2vec, evaluar_sustitucion)


def create_pipeline(**kwargs) -> Pipeline:  # noqa: ARG001
    """Crea el pipeline que construye el grafo y agrupa los ejercicios."""
    return pipeline(
        [
            node(
                func=construir_grafo,
                inputs=["primary_ejercicios", "model_output_perfil_exigencia",
                        "primary_ejercicio_musculo", "primary_musculo_articulacion",
                        "regiones_compuestas", "params:grafo_biomecanico"],
                outputs=["grafo_biomecanico", "grafo_nodos_ejercicio"],
                name="construir_grafo",
            ),
            node(
                func=detectar_comunidades,
                inputs=["grafo_biomecanico", "grafo_nodos_ejercicio", "params:grafo_biomecanico"],
                outputs=["comunidades_louvain", "louvain_barrido"],
                name="detectar_comunidades",
            ),
            node(
                func=entrenar_node2vec,
                inputs=["grafo_biomecanico", "params:grafo_biomecanico"],
                outputs="embeddings_node2vec",
                name="entrenar_node2vec",
            ),
            node(
                func=agrupar_embeddings,
                inputs=["embeddings_node2vec", "grafo_nodos_ejercicio", "params:grafo_biomecanico"],
                outputs=["clusters_kmeans", "kmeans_barrido"],
                name="agrupar_embeddings",
            ),
            node(
                func=consolidar_familias,
                inputs=["grafo_nodos_ejercicio", "comunidades_louvain", "clusters_kmeans",
                        "embeddings_node2vec", "grafo_biomecanico", "params:grafo_biomecanico"],
                outputs=["model_output_familias_movimiento", "familias_perfiles",
                         "clustering_comparacion"],
                name="consolidar_familias",
            ),
            node(
                func=evaluar_sustitucion,
                inputs=["model_output_familias_movimiento", "embeddings_node2vec",
                        "primary_musculo_articulacion", "params:grafo_biomecanico"],
                outputs=["sustitucion_cobertura", "sustitucion_contexto"],
                name="evaluar_sustitucion",
            ),
        ]
    )

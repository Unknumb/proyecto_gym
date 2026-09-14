"""Pipeline registry — connects pipeline names to their factory functions."""

from __future__ import annotations

from kedro.pipeline import Pipeline

from gym_exercises.pipelines.data_enrichment.pipeline import (
    create_pipeline as create_data_enrichment_pipeline,
)
from gym_exercises.pipelines.data_understanding.pipeline import (
    create_pipeline as create_data_understanding_pipeline,
)


def register_pipelines() -> dict[str, Pipeline]:
    """Register all project pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    data_understanding = create_data_understanding_pipeline()
    data_enrichment = create_data_enrichment_pipeline()

    return {
        "data_understanding": data_understanding,
        "data_enrichment": data_enrichment,
        "__default__": data_understanding + data_enrichment,
    }

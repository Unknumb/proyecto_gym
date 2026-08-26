"""Pipeline registry — connects pipeline names to their factory functions."""

from __future__ import annotations

from kedro.pipeline import Pipeline

from gym_exercises.pipelines.data_understanding.pipeline import (
    create_pipeline as create_data_understanding_pipeline,
)


def register_pipelines() -> dict[str, Pipeline]:
    """Register all project pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    data_understanding = create_data_understanding_pipeline()

    return {
        "data_understanding": data_understanding,
        "__default__": data_understanding,
    }

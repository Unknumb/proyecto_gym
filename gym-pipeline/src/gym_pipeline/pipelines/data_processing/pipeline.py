from kedro.pipeline import Pipeline, node, pipeline
from .nodes import clean_exercises, enrich_with_external_metadata


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=clean_exercises,
            inputs="exercises_raw",
            outputs="exercises_limpio",
            name="clean_exercises_node",
        ),
        node(
            func=enrich_with_external_metadata,
            inputs=["exercises_limpio", "mega_gym_dataset"],
            outputs="exercise_features_enriched",
            name="enrich_with_external_metadata_node",
        ),
    ])
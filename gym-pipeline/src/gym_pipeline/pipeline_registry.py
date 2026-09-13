from gym_pipeline.pipelines import data_processing as dp


def register_pipelines():
    return {
        "__default__": dp.create_pipeline(),
    }
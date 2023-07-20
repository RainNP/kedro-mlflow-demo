"""
This is a boilerplate pipeline 'somthing'
generated using Kedro 0.18.11
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import do_rf, process_data, check, do_signature


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=process_data,
            inputs="raw_data",
            outputs="model_input_table",
            name="process_data"
        ),
        node(
            func=do_rf,
            inputs=["model_input_table", "params:model_params"],
            outputs={"clf" : "clf", "model_metrics" : "model_metrics"},
            name="do_rf"
        ),
        node(
            func=check,
            inputs=["clf", "model_input_table"],
            outputs="something",
            name="check"
        ),
        node(
            func=do_signature,
            inputs=["clf", "model_input_table"],
            outputs="done_signature",
            name="signature"
        ),
    ])

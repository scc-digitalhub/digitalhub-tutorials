from digitalhub_runtime_hera.dsl import step
from hera.workflows import DAG, Workflow


def pipeline():
    with Workflow(entrypoint="dag") as w, DAG(name="dag"):
        A = step(
            template={"action": "build"},
            function="build-function-darts",
        )
        B = step(
            template={"action": "job"},
            function="train-time-series-model",
            outputs=["model"],
        )
        C = step(
            template={
                "action": "serve",
                "init_parameters": {"model_key": "{{inputs.parameters.model}}"},
            },
            function="serve-time-series-model",
            inputs={"model": B.get_parameter("model")},
        )
        A >> B >> C
    return w

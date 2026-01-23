from digitalhub_runtime_hera.dsl import step
from hera.workflows import DAG, Workflow


def pipeline():
    with Workflow(entrypoint="dag") as w:
        with DAG(name="dag"):
            A = step(
                template={"action": "job"},
                function="train-mlflow-model",
                outputs=["model"],
            )
            B = step(
                template={
                    "action": "serve",
                    "path": "{{inputs.parameters.model}}",
                    "model_name": "iris-classifier",
                },
                function="serve-mlflow-model",
                inputs={"model": A.get_parameter("model")},
            )
            A >> B
    return w

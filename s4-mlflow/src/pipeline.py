from digitalhub_runtime_hera.dsl import step
from hera.workflows import Steps, Workflow


def pipeline():
    with Workflow(entrypoint="dag") as w:
        with Steps(name="dag"):
            A = step(template={"action": "job"}, function="train-mlflow-model", outputs=["model"])
    return w

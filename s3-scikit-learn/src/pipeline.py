from digitalhub_runtime_hera.dsl import step
from hera.workflows import DAG, Workflow


def pipeline():
    with Workflow(entrypoint="dag") as w:
        with DAG(name="dag"):
            A = step(
                template={"action": "job"},
                function="prepare-data",
                outputs=["dataset"],
            )
            B = step(
                template={
                    "action": "job",
                    "inputs": {"di": "{{inputs.parameters.di}}"},
                },
                function="train-classifier",
                inputs={"di": A.get_parameter("dataset")},
                outputs=["model"],
            )
            C = step(
                template={
                    "action": "serve",
                    "path": "{{inputs.parameters.model}}",
                },
                function="serve-classifier",
                inputs={"model": B.get_parameter("model")},
            )
            A >> B >> C
    return w

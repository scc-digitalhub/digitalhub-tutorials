from digitalhub_runtime_hera.dsl import step
from hera.workflows import DAG, Workflow


def pipeline():
    with Workflow(entrypoint="dag") as w, DAG(name="dag"):
        A = step(
            template={"action": "build"},
            function="sklearn-image",
        )
        B = step(
            template={"action": "job"},
            function="prepare-data",
            outputs=["dataset"],
        )
        C = step(
            template={
                "action": "job",
                "inputs": {"di": "{{inputs.parameters.di}}"},
            },
            function="train-classifier",
            inputs={"di": A.get_parameter("dataset")},
            outputs=["model"],
        )
        D = step(
            template={
                "action": "serve",
                "path": "{{inputs.parameters.model}}",
            },
            function="serve-classifier",
            inputs={"model": B.get_parameter("model")},
        )
        A >> B >> C >> D
    return w

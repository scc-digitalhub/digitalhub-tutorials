from digitalhub_runtime_hera.dsl import step
from hera.workflows import DAG, Workflow


def pipeline():
    with Workflow(entrypoint="dag") as w:
        with DAG(name="dag"):
            A = step(template={"action": "job"}, function="prepare-data", outputs=["dataset"])
            B = step(
                template={"action": "job", "inputs": {"di": "{{inputs.parameters.di}}"}},
                function="train-classifier",
                inputs={"di": A.get_parameter("dataset")},
            )
            A >> B
    return w

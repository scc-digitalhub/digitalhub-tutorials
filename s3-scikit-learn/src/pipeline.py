from hera.workflows import Workflow, DAG, Parameter
from digitalhub_runtime_hera.dsl import step


def pipeline():
    with Workflow(entrypoint="dag") as w:
        with DAG(name="dag"):
            A = step(template={"action":"job"},
                     function="download-prep",
                     outputs=["dataset"])
            B = step(template={"action":"job", "inputs": {"di": "{{inputs.parameters.di}}"}},
                     function="train",
                     inputs={"di": A.get_parameter("dataset")})
            A >> B
    return w

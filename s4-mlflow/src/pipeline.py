from hera.workflows import Workflow, DAG, Parameter
from digitalhub_runtime_hera.dsl import step


def pipeline():
    with Workflow(entrypoint="dag") as w:
        with DAG(name="dag"):
            A = step(template={"action":"job"}, function="train")
    return w

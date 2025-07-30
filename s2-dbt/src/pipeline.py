
from hera.workflows import Workflow, DAG, Parameter
from digitalhub_runtime_hera.dsl import step


def pipeline():
    with Workflow(entrypoint="dag", arguments=Parameter(name="di")) as w:
        with DAG(name="dag"):
            A = step(template={"action":"job",
                               "inputs": {"di": "{{workflow.parameters.di}}"}
                               "outputs": {"output_table": "department-50"}},
                     function="function-dbt",)
    return w

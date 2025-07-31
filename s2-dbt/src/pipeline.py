from digitalhub_runtime_hera.dsl import step
from hera.workflows import DAG, Parameter, Workflow


def pipeline():
    with Workflow(entrypoint="dag", arguments=Parameter(name="employees")) as w:
        with DAG(name="dag"):
            A = step(
                template={
                    "action": "transform",
                    "inputs": {"employees": "{{workflow.parameters.employees}}"},
                    "outputs": {"output_table": "department-50"},
                },
                function="transform-employees",
            )
    return w

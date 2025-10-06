from digitalhub_runtime_hera.dsl import step
from hera.workflows import DAG, Parameter, Workflow


def pipeline():
    with Workflow(entrypoint="dag", arguments=Parameter(name="url")) as w:
        with DAG(name="dag"):
            A = step(
                template={
                    "action": "job",
                    "inputs": {"url": "{{workflow.parameters.url}}"},
                },
                function="download-data",
                outputs=["dataset"],
            )
            B = step(
                template={
                    "action": "job",
                    "inputs": {"di": "{{inputs.parameters.di}}"},
                },
                function="process-spire",
                inputs={"di": A.get_parameter("dataset")},
            )
            C = step(
                template={
                    "action": "job",
                    "inputs": {"di": "{{inputs.parameters.di}}"},
                },
                function="process-measures",
                inputs={"di": A.get_parameter("dataset")},
                outputs=["dataset-measures"],
            )
            D = step(
                template={
                    "action": "serve",
                    "init_parameters": {"dataitem": "{{inputs.parameters.dataitem}}"},
                },
                function="api",
                inputs={"dataitem": C.get_parameter("dataset-measures")},
            )
            A >> [B, C]
            C >> D
    return w

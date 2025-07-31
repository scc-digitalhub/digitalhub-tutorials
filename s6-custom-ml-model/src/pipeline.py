from digitalhub_runtime_hera.dsl import step
from hera.workflows import DAG, Workflow


def pipeline():
    with Workflow(entrypoint="dag") as w:
        with DAG(name="dag"):
            A = step(
                template={
                    "action": "build",
                    "instructions": [
                        "pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu",
                        "pip3 install darts==0.30.0 patsy scikit-learn",
                    ],
                },
                function="train-time-series-model",
            )
            B = step(
                template={
                    "action": "build",
                    "instructions": [
                        "pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu",
                        "pip3 install darts==0.30.0 patsy scikit-learn",
                    ],
                },
                function="serve-time-series-model",
            )
            C = step(
                template={"action": "job"},
                function="train-time-series-model",
                outputs=["model"],
            )
            D = step(
                template={"action": "serve", "init_parameters": {"model_key": "{{inputs.parameters.model}}"}},
                function="serve-time-series-model",
                inputs={"model": C.get_parameter("model")},
            )
            [A, B] >> C >> D
    return w

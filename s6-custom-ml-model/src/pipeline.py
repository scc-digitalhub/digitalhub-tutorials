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
                function="train-darts",
            )
            B = step(template={"action": "job"}, function="train-darts", outputs=["model"])
            C = step(
                template={
                    "action": "build",
                    "instructions": [
                        "pip3 install darts==0.30.0 patsy scikit-learn torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu"
                    ],
                },
                function="serve_darts_model",
            )
            D = step(
                template={"action": "job", "init_parameters": {"model_key": "{{inputs.parameters.model}}"}},
                function="serve_darts_model",
                inputs={"model": B.get_parameter("model")},
            )
            A >> B >> C >> D
    return w

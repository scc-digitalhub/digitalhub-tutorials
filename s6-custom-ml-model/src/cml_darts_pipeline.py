from hera.workflows import Workflow, DAG, Parameter
from digitalhub_runtime_hera.dsl import step


def pipeline():
    with Workflow(entrypoint="dag") as w:
        with DAG(name="dag"):
            A = step(template={"action":"job"},
                     function="train-darts",
                     outputs=["model"])
            B = step(template={"action":"build", "instructions": ["pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu","pip3 install darts==0.30.0 patsy scikit-learn"]},
                     function="serve_darts_model")
            C = step(template={"action":"job", "init_parameters": {"model_key": "{{inputs.parameters.model}}"}},
                     function="serve_darts_model",
                     inputs={"model": A.get_parameter("model")})
            A >> B >> C
    return w

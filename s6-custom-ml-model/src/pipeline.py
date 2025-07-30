
from digitalhub_runtime_hera.dsl import pipeline_context

def pipeline():
    with pipeline_context() as pc:
        trainer = pc.step(name="train-model", function="train-darts", action="job")

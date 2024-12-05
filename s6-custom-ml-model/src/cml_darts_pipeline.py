
from digitalhub_runtime_kfp.dsl import pipeline_context

def myhandler():
    with pipeline_context() as pc:
        trainer = pc.step(name="train-model", function="train-darts", action="job")        

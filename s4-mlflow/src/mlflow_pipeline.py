
from digitalhub_runtime_kfp.dsl import pipeline_context

def myhandler(): 
    with pipeline_context() as pc:
        train = pc.step(name="train", function="train", action="job")


from digitalhub_runtime_kfp.dsl import pipeline_context

def myhandler(path): 
    with pipeline_context() as pc:
         serve = pc.step(name="serve", function="llm_classification", action="serve", inputs={"path":path})

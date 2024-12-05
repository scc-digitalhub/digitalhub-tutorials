
from digitalhub_runtime_kfp.dsl import pipeline_context

def myhandler():
    with pipeline_context() as pc:
        downloader = pc.step(name="download-data", function="data-prep", action="job", outputs={"dataset": "dataset"})
        trainer = pc.step(name="train-model", function="train", action="job", inputs={"di": downloader.outputs["dataset"]})

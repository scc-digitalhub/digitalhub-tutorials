
from digitalhub_runtime_kfp.dsl import pipeline_context

def pipeline(url):
    with pipeline_context() as pc:
        downloader = pc.step(
            name="download-data",
            function="download-data",
            action="job",
            inputs={"url": url},
            outputs={"dataset": "dataset"},
        )

        process_spire = pc.step(
            name="process-spire",
            function="process-spire",
            action="job",
            inputs={"di": downloader.outputs["dataset"]}
        )

        process_measures = pc.step(
            name="process-measures",
            function="process-measures",
            action="job",
            inputs={"di": downloader.outputs["dataset"]}
        )

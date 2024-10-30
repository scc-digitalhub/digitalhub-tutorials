
from digitalhub_runtime_kfp.dsl import pipeline_context

def myhandler(url):
    with pipeline_context() as pc:
        s1_dataset = pc.step(name="dbt", function="function-dbt", action="transform", inputs={"employees":url}, outputs={"output_table": "department-50"})

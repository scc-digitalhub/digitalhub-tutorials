# ETL (Extract transform load)

The scenario depict how to collect some data regarding traffic, analyze and transform it, then expose the resulting dataset. The 's1_etl' folder contains
a jypter notebook and a digitalhub project. 

- Jypter notebook 
	- Import the Jupyter notebook for each scenario located inside project folder in the 'Coder' instance and execute it step by step.

- Project
	
   1. Import the project inside the 'Coder' instance using the yaml file.
	```
 	import digitalhub as dh
 	proj = dh.import_project('project-etl-ci.yaml")
	```

   2.  View the project details such as pipeline name.
    ```
    print(proj)
   ```
   ```
   {'kind': 'project', 'metadata': {'project': 'project-etl-ci', 'name': 'project-etl-ci', 'created': '2024-10-24T11:20:17.542Z', 'updated': '2024-10-24T11:28:49.778Z'}, 'spec': {'context': './', 'functions': [{'id': 'cad09f32-6468-480a-b3ef-a30379fecd60', 'key': 'store://project-etl-ci/function/python/download-data:cad09f32-6468-480a-b3ef-a30379fecd60', 'kind': 'python', 'metadata': {'created': '2024-10-24T11:22:04.66Z', 'name': 'download-data', 'project': 'project-etl-ci', 'updated': '2024-10-24T11:22:04.66Z', 'ref': 'function-download-data-cad09f32-6468-480a-b3ef-a30379fecd60.yml'}, 'name': 'download-data', 'project': 'project-etl-ci'}, {'id': '9f5de914-72fe-4ade-9fdf-8007c4cd6350', 'key': 'store://project-etl-ci/function/python/process-spire:9f5de914-72fe-4ade-9fdf-8007c4cd6350', 'kind': 'python', 'metadata': {'created': '2024-10-24T11:24:23.608Z', 'name': 'process-spire', 'project': 'project-etl-ci', 'updated': '2024-10-24T11:24:23.608Z', 'ref': 'function-process-spire-9f5de914-72fe-4ade-9fdf-8007c4cd6350.yml'}, 'name': 'process-spire', 'project': 'project-etl-ci'}, {'id': '8d736080-3f52-4de5-b8be-158b9b7cec07', 'key': 'store://project-etl-ci/function/python/process-measures:8d736080-3f52-4de5-b8be-158b9b7cec07', 'kind': 'python', 'metadata': {'created': '2024-10-24T11:25:34.654Z', 'name': 'process-measures', 'project': 'project-etl-ci', 'updated': '2024-10-24T11:25:34.654Z', 'ref': 'function-process-measures-8d736080-3f52-4de5-b8be-158b9b7cec07.yml'}, 'name': 'process-measures', 'project': 'project-etl-ci'}, {'id': '02125fed-2efb-48ea-a688-f90d6d6aab21', 'key': 'store://project-etl-ci/function/python/api:02125fed-2efb-48ea-a688-f90d6d6aab21', 'kind': 'python', 'metadata': {'created': '2024-10-24T11:26:24.489Z', 'name': 'api', 'project': 'project-etl-ci', 'updated': '2024-10-24T11:26:24.489Z', 'ref': 'function-api-02125fed-2efb-48ea-a688-f90d6d6aab21.yml'}, 'name': 'api', 'project': 'project-etl-ci'}], 'artifacts': [], 'workflows': [{'id': '051163dd-cbe0-4c50-8ea2-84fd295918c9', 'key': 'store://project-etl-ci/workflow/kfp/pipeline:051163dd-cbe0-4c50-8ea2-84fd295918c9', 'kind': 'kfp', 'metadata': {'created': '2024-10-24T11:26:10.234Z', 'name': 'pipeline', 'project': 'project-etl-ci', 'updated': '2024-10-24T11:26:10.234Z', 'ref': 'workflow-pipeline-051163dd-cbe0-4c50-8ea2-84fd295918c9.yml'}, 'name': 'pipeline', 'project': 'project-etl-ci'}], 'dataitems': [], 'models': []}, 'status': {'state': 'CREATED'}, 'id': 'project-etl-ci', 'name': 'project-etl-ci', 'key': 'store://project-etl-ci'}
   ```
   
   3. Create the input parameter from url
    ```
    URL = "https://opendata.comune.bologna.it/api/explore/v2.1/catalog/datasets/rilevazione-flusso-veicoli-tramite-spire-anno-2023/exports/csv?lang=it&timezone=Europe%2FRome&use_labels=true&delimiter=%3B"
	di= project.new_dataitem(name="url_data_item",kind="table",path=URL)
	di.key
    ```
	```
	'store://project-etl-ci/dataitem/table/url_data_item:ed2c0631-8112-4fe1-b5f6-6a0d56f5872e'
	```
   
   4. Run the pipeline
    ```
    workflow_run = proj.run('pipeline', parameters={"url": di.key})
    ```
	
	

	
	



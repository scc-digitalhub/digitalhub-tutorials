# LLM Flow Model Training and Serving Scenario
This scenario depict how to create and serve LLM HuggingFace-compatible-models. Specifically, it is possible to serve directly the LLM models from the HuggingFace catalogue provided the id of the model or to serve the fine-tuned model from the specified path.The 's5-llm' folder contains a jypter notebook and a digitalhub project yaml descriptor.

- Jypter notebook 
	- Import the Jupyter notebook for each scenario located inside project folder in the 'Coder' instance and execute it step by step.

- Project
	
   1. Import the project inside the 'Coder' instance using the yaml file.
	```
 	import digitalhub as dh
	proj = dh.import_project('project-llm-ci.yml')
	```
   2.  View the project details such as pipeline name.
        ```
        proj.to_dict()
        ```
        ```
        {
        'kind': 'project',
        'metadata': {
          'project': 'llm-ci',
          'name': 'llm-ci',
        },
        'spec': {'context': './',
        'functions': [
          {
          'id': '6da5afcf-ee73-4142-a7cc-9c8cf4fdbfce',
          'key': 'store://llm-ci/function/huggingfaceserve/llm_classification:6da5afcf-ee73-4142-a7cc-9c8cf4fdbfce',
          'kind': 'huggingfaceserve',
          'metadata': {'created': '2024-10-28T08:48:35.211Z',
          'name': 'llm_classification',
          'project': 'llm-ci',
          'updated': '2024-10-28T08:48:35.211Z'},
          'name': 'llm_classification',
          'project': 'llm-ci',
          }
        ],
       'artifacts': [],
       'workflows': [
          {
          'id': '501ef375-c099-4c2f-a490-cf5a4444ed1a',
          'key': 'store://llm-ci/workflow/kfp/pipeline_llm:501ef375-c099-4c2f-a490-cf5a4444ed1a',
          'kind': 'kfp',
          'metadata': {'created': '2024-10-28T08:48:35.28Z',
          'name': 'pipeline_llm',
          'project': 'llm-ci',
          'updated': '2024-10-28T08:48:35.28Z'},
          'name': 'pipeline_llm',
          'project': 'llm-ci',
          }
        ],
       'dataitems': [],
       'models': []},
       'status': {'state': 'CREATED'},
        'id': 'llm-ci',
        'name': 'llm-ci',
        'key': 'store://llm-ci'
       }
       ```

   3. In this scenario functions are self-contained so one execute the pipeline directly.
     ```
     workflow_run = proj.run('pipeline_llm')
     ```
     ```
     workflow_run.refresh().status.state
     ```
     As a result of this scenario workflow, a serving function 'llm_classification' is created which is availabe to make an API call, it is also possible to expose the service using the KRM API gateway functionality.
	
	

	
	


	
	



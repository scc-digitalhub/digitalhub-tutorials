# ML Flow Model Training and Serving

This scenario provides a quick overview of developing and deploying a machine learning application based on model tracked with MLFlow framework using the functionalities of the platform.. The 's4-mlflow' folder contains a jypter notebook and a digitalhub project. 

- Jypter notebook 
	- Import the Jupyter notebook for each scenario located inside project folder in the 'Coder' instance and execute it step by step.

- Project
	
   1. Import the project inside the 'Coder' instance using the yaml file.
	```
 	import digitalhub as dh
 	proj = dh.import_project('project-mlflow-model-ci.yml')
	```

   2.  View the project details such as pipeline name.
    ```
    proj.to_dict()
    ```
    ```
    {
    'kind': 'project',
    'metadata': {
     'project': 'project-cml-darts-ci',
    'name': 'project-cml-darts-ci',
     },
    'spec': {
    'context': './',
    'functions': [{'id': 'e03fe55f-2007-4ad4-a7e3-367fe774b821',
    'key': 'store://project-cml-darts-ci/function/python/train-darts:e03fe55f-2007-4ad4-a7e3-367fe774b821',
    'kind': 'python',
    'metadata': {'created': '2024-10-28T09:54:41.43Z',
     'name': 'train-darts',
     'project': 'project-cml-darts-ci',
     },
    'name': 'train-darts',
    'project': 'project-cml-darts-ci',
    }],
    'artifacts': [],
    'workflows': [],
    'name': 'darts_model',
    'project': 'project-cml-darts-ci',
    }]
     },
    'status': {'state': 'CREATED'},
     'id': 'project-cml-darts-ci',
    'name': 'project-cml-darts-ci',
    'key': 'store://project-cml-darts-ci'
    }
   ```
   3. Run the pipeline
      ```
      workflow_run = proj.run('pipeline_mlflow')
      workflow_run
      ```
      Wait until workflow execution is completed. One can view the state of workflow run using console or using sdk api as shown below.
      ```
      workflow_run.refresh().status.state
      ```
      Once completed, one can explore the results of the function by fetching the list of data model using digithub sdk.
      ```
      data_model = dh.list_models(project='project-mlflow-model-ci')
      len(data_model)
      ```  
	
	

	
	



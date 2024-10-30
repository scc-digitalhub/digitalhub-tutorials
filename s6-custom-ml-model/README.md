# Custom ML FLow Model Training and Serving

This scenario provides a quick overview of developing and deploying generic machine learning applications using the functionalities of the platform. For this purpose, we use ML algorithms for the time series management provided by the Darts framework. The 's6-cml-darts-ci' folder contains a jypter notebook and a digitalhub project. 

- Jypter notebook 
	- Import the Jupyter notebook for each scenario located inside project folder in the 'Coder' instance and execute it step by step.

- Project
	
   1. Import the project inside the 'Coder' instance using the yaml file.
	```
 	import digitalhub as dh
 	proj = dh.import_project('project-cml-darts-ci.yml')
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
   3. Build the container.
      Fetch the training function and build it.
      ```
      train_fn = proj.get_function('train-darts')
      ```
      ```
      train_run =  train_fn.run(action="build", instructions=["pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu","pip3 install darts patsy scikit-learn"])
      ```      
      Wait for build to be completed. One can check the status of 'Run' using console or by typing the following command
      ```
      train_run.refresh().status.state
      ```
      Once 'Completed', proceed to next step
      
   4. Run the pipeline
      ```
      workflow_run = proj.run('pipeline_cml_darts')
      ```
      Wait for workflow pipeline to be completed. One can check the status of 'Run' using console or by typing the following command.
      ```
      workflow_run.refresh().status.state
      ```
      Once completed, one can explore the results by fetching the list of newly created data model using digithub sdk API.
      ```
      data_model = dh.list_models(project='project-cml-darts-ci')
      len(data_model)
      ```   
	
	

	
	



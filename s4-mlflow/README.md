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
      workflow_run.get_state()
      ```
      Once completed, one can explore the results of the function by fetching the list of data model using digithub sdk.
      ```
      data_model = dh.list_models(project='project-mlflow-model-ci')
      print((data_model))
      ```
      ```
      [{'kind': 'mlflow', 'metadata': {'project': 'project-mlflow-model-ci', 'name': 'model-mlflow', 'version': '3898f09a-2a88-4d58-a823-67764641c1d0', 'created': '2024-10-28T13:30:47.489Z', 'updated': '2024-10-28T13:30:47.847Z', 'created_by': 'khurshid@fbk.eu', 'updated_by': 'khurshid@fbk.eu', 'embedded': False}, 'spec': {'path': 's3://datalake/project-mlflow-model-ci/model/model-mlflow/3898f09a-2a88-4d58-a823-67764641c1d0/', 'framework': 'sklearn', 'parameters': {'n_jobs': 'None', 'refit': 'True', 'param_grid': "{'kernel': ('linear', 'rbf'), 'C': [1, 10]}", 'verbose': '0', 'pre_dispatch': '2*n_jobs', 'estimator': 'SVC()', 'return_train_score': 'False', 'cv': 'None', 'scoring': 'None', 'best_C': '1', 'best_kernel': 'linear', 'error_score': 'nan'}, 'metrics': {'training_score': 0.9933333333333333, 'training_f1_score': 0.9933326665999933, 'training_precision_score': 0.9934640522875816, 'best_cv_score': 0.9800000000000001, 'training_recall_score': 0.9933333333333333, 'training_accuracy_score': 0.9933333333333333}, 'flavor': 'sklearn', 'input_datasets': [{'name': 'dataset', 'digest': '2bf3ace3', 'profile': '{"features_shape": [150, 4], "features_size": 600, "features_nbytes": 4800, "targets_shape": [150], "targets_size": 150, "targets_nbytes": 1200}', 'source': '{"tags": {"mlflow.user": "jovyan", "mlflow.source.name": "/opt/nuclio/_nuclio_wrapper.py", "mlflow.source.type": "LOCAL"}}', 'source_type': 'code'}], 'signature': {'inputs': '[{"type": "tensor", "tensor-spec": {"dtype": "float64", "shape": [-1, 4]}}]', 'outputs': '[{"type": "tensor", "tensor-spec": {"dtype": "int64", "shape": [-1]}}]'}}, 'status': {'state': 'CREATED', 'files': [{'path': 'cv_results.csv', 'name': 'cv_results.csv', 'size': 1140, 'hash': 'md5:905b8c1e1ae2cae79f1c15b06f785ae7', 'content_type': 'text/csv', 'last_modified': '2024-10-28T13:30:47.000+00:00'}, {'path': 'training_confusion_matrix.png', 'name': 'training_confusion_matrix.png', 'size': 28733, 'hash': 'md5:e965a94e1bb1a0b76134572a9e65b1a7', 'content_type': 'image/png', 'last_modified': '2024-10-28T13:30:47.000+00:00'}, {'path': 'estimator.html', 'name': 'estimator.html', 'size': 14205, 'hash': 'md5:7ab1b5e995f7c26171b3467d9dc5fa3a', 'content_type': 'text/html', 'last_modified': '2024-10-28T13:30:47.000+00:00'}, {'path': 'best_estimator/MLmodel', 'name': 'MLmodel', 'size': 734, 'hash': 'md5:ddc0d4af9aae6419d9d476f4e55d75aa', 'content_type': 'binary/octet-stream', 'last_modified': '2024-10-28T13:30:47.000+00:00'}, {'path': 'best_estimator/conda.yaml', 'name': 'conda.yaml', 'size': 231, 'hash': 'md5:641e3a9b56caaa3bcc600e1289a9cc10', 'content_type': 'binary/octet-stream', 'last_modified': '2024-10-28T13:30:47.000+00:00'}, {'path': 'best_estimator/model.pkl', 'name': 'model.pkl', 'size': 2650, 'hash': 'md5:66b8ce6190eb7bc5609b8a404c4f0078', 'content_type': 'binary/octet-stream', 'last_modified': '2024-10-28T13:30:47.000+00:00'}, {'path': 'best_estimator/python_env.yaml', 'name': 'python_env.yaml', 'size': 123, 'hash': 'md5:3feb467e82c61b891b6e3b57b1f74878', 'content_type': 'binary/octet-stream', 'last_modified': '2024-10-28T13:30:47.000+00:00'}, {'path': 'best_estimator/requirements.txt', 'name': 'requirements.txt', 'size': 109, 'hash': 'md5:d53b69f66cf3bc970d8319eb12e20570', 'content_type': 'text/plain', 'last_modified': '2024-10-28T13:30:47.000+00:00'}, {'path': 'model/MLmodel', 'name': 'MLmodel', 'size': 725, 'hash': 'md5:1a625b76b87b26239b76ca27895eea47', 'content_type': 'binary/octet-stream', 'last_modified': '2024-10-28T13:30:47.000+00:00'}, {'path': 'model/conda.yaml', 'name': 'conda.yaml', 'size': 231, 'hash': 'md5:641e3a9b56caaa3bcc600e1289a9cc10', 'content_type': 'binary/octet-stream', 'last_modified': '2024-10-28T13:30:47.000+00:00'}, {'path': 'model/model.pkl', 'name': 'model.pkl', 'size': 5007, 'hash': 'md5:340fa334a1fcb0f6d04838aa570095d2', 'content_type': 'binary/octet-stream', 'last_modified': '2024-10-28T13:30:47.000+00:00'}, {'path': 'model/python_env.yaml', 'name': 'python_env.yaml', 'size': 123, 'hash': 'md5:3feb467e82c61b891b6e3b57b1f74878', 'content_type': 'binary/octet-stream', 'last_modified': '2024-10-28T13:30:47.000+00:00'}, {'path': 'model/requirements.txt', 'name': 'requirements.txt', 'size': 109, 'hash': 'md5:d53b69f66cf3bc970d8319eb12e20570', 'content_type': 'text/plain', 'last_modified': '2024-10-28T13:30:47.000+00:00'}]}, 'user': 'khurshid@fbk.eu', 'project': 'project-mlflow-model-ci', 'name': 'model-mlflow', 'id': '3898f09a-2a88-4d58-a823-67764641c1d0', 'key': 'store://project-mlflow-model-ci/model/mlflow/model-mlflow:3898f09a-2a88-4d58-a823-67764641c1d0'}]
      ```
	
	

	
	



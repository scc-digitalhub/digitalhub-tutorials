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
      train_fn
      ```
      ```
      {'kind': 'python', 'metadata': {'project': 'project-cml-darts-ci', 'name': 'train-darts', 'version': 'e03fe55f-2007-4ad4-a7e3-367fe774b821', 'created': '2024-10-28T10:13:59.326Z', 'updated': '2024-10-28T10:13:59.326Z', 'created_by': 'khurshid@fbk.eu', 'updated_by': 'khurshid@fbk.eu', 'embedded': False}, 'spec': {'python_version': 'PYTHON3_10', 'source': {'source': 'src/train-model.py', 'base64': 'Cgpmcm9tIGRpZ2l0YWxodWJfcnVudGltZV9weXRob24gaW1wb3J0IGhhbmRsZXIKCmltcG9ydCBwYW5kYXMgYXMgcGQKaW1wb3J0IG51bXB5IGFzIG5wCgpmcm9tIGRhcnRzIGltcG9ydCBUaW1lU2VyaWVzCmZyb20gZGFydHMuZGF0YXNldHMgaW1wb3J0IEFpclBhc3NlbmdlcnNEYXRhc2V0CmZyb20gZGFydHMubW9kZWxzIGltcG9ydCBOQkVBVFNNb2RlbApmcm9tIGRhcnRzLm1ldHJpY3MgaW1wb3J0IG1hcGUsIHNtYXBlLCBtYWUKCmZyb20gemlwZmlsZSBpbXBvcnQgWmlwRmlsZQoKQGhhbmRsZXIoKQpkZWYgdHJhaW5fbW9kZWwocHJvamVjdCk6CiAgICBzZXJpZXMgPSBBaXJQYXNzZW5nZXJzRGF0YXNldCgpLmxvYWQoKQogICAgdHJhaW4sIHZhbCA9IHNlcmllc1s6LTM2XSwgc2VyaWVzWy0zNjpdCgogICAgbW9kZWwgPSBOQkVBVFNNb2RlbCgKICAgICAgICBpbnB1dF9jaHVua19sZW5ndGg9MjQsCiAgICAgICAgb3V0cHV0X2NodW5rX2xlbmd0aD0xMiwKICAgICAgICBuX2Vwb2Nocz0yMDAsCiAgICAgICAgcmFuZG9tX3N0YXRlPTAKICAgICkKICAgIG1vZGVsLmZpdCh0cmFpbikKICAgIHByZWQgPSBtb2RlbC5wcmVkaWN0KG49MzYpCgogICAgbW9kZWwuc2F2ZSgicHJlZGljdG9yX21vZGVsLnB0IikKICAgIHdpdGggWmlwRmlsZSgicHJlZGljdG9yX21vZGVsLnB0LnppcCIsICJ3IikgYXMgejoKICAgICAgICB6LndyaXRlKCJwcmVkaWN0b3JfbW9kZWwucHQiKQogICAgICAgIHoud3JpdGUoInByZWRpY3Rvcl9tb2RlbC5wdC5ja3B0IikKICAgIG1ldHJpY3MgPSB7CiAgICAgICAgIm1hcGUiOiBtYXBlKHNlcmllcywgcHJlZCksCiAgICAgICAgInNtYXBlIjogc21hcGUoc2VyaWVzLCBwcmVkKSwKICAgICAgICAibWFlIjogbWFlKHNlcmllcywgcHJlZCkKICAgIH0KCiAgICBwcm9qZWN0LmxvZ19tb2RlbCgKICAgICAgICBuYW1lPSJkYXJ0c19tb2RlbCIsCiAgICAgICAga2luZD0ibW9kZWwiLAogICAgICAgIHNvdXJjZT0icHJlZGljdG9yX21vZGVsLnB0LnppcCIsCiAgICAgICAgYWxnb3JpdGhtPSJkYXJ0cy5tb2RlbHMuTkJFQVRTTW9kZWwiLAogICAgICAgIGZyYW1ld29yaz0iZGFydHMiLAogICAgICAgIG1ldHJpY3M9bWV0cmljcwogICAgKQo=', 'handler': 'train_model', 'lang': 'python'}}, 'status': {'state': 'CREATED'}, 'user': 'khurshid@fbk.eu', 'project': 'project-cml-darts-ci', 'name': 'train-darts', 'id': 'e03fe55f-2007-4ad4-a7e3-367fe774b821', 'key': 'store://project-cml-darts-ci/function/python/train-darts:e03fe55f-2007-4ad4-a7e3-367fe774b821'}
      ```
      ```
      train_fn.run(action="build", instructions=["pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu","pip3 install darts patsy scikit-learn"])
      ```
      wait for build to be completed. One can check the status of 'Run' using console or by typing the following command
      ```
      train_fn.get_state()
      ```
      
   5. Run the pipeline
      ```
      workflow_run = proj.run('pipeline_cml_darts')
      ```
      Wait for workflow pipeline to be completed. One can check the status of 'Run' using console or by typing the following command.
      ```
      workflow_run.get_state()
      ```
      Once completed, one can explore the results by fetching the list of newly created data model using digithub sdk API.
      ```
      data_model = dh.list_models(project='project-cml-darts-ci')
      print((data_model))
      ```
      ```
      [{'kind': 'model', 'metadata': {'project': 'project-cml-darts-ci', 'name': 'darts_model', 'version': '428d2a95-9885-4cc7-a465-cb921b902c17', 'created': '2024-10-28T10:30:15.011Z', 'updated': '2024-10-28T10:30:15.733Z', 'created_by': 'khurshid@fbk.eu', 'updated_by': 'khurshid@fbk.eu', 'embedded': False}, 'spec': {'path': 'zip+s3://datalake/project-cml-darts-ci/model/darts_model/428d2a95-9885-4cc7-a465-cb921b902c17/predictor_model.pt.zip', 'framework': 'darts', 'algorithm': 'darts.models.NBEATSModel'}, 'status': {'state': 'CREATED', 'files': [{'path': 'predictor_model.pt.zip', 'name': 'predictor_model.pt.zip', 'content_type': 'application/zip', 'size': 49758418, 'hash': 'LiteralETag:32850d806e9704208bf1176211d38445-6', 'last_modified': '2024-10-28T10:30:15+00:00'}]}, 'user': 'khurshid@fbk.eu', 'project': 'project-cml-darts-ci', 'name': 'darts_model', 'id': '428d2a95-9885-4cc7-a465-cb921b902c17', 'key': 'store://project-cml-darts-ci/model/model/darts_model:428d2a95-9885-4cc7-a465-cb921b902c17'}]
      ```
	
	

	
	



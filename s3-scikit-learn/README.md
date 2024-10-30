# Scikit Learn Scenario
This scenario provides a quick overview of developing and deploying a scikit-learn machine learning application using the functionalities of the platform. We will prepare data, train a generic model and expose it as a service.The 's2-ml' folder contains a jypter notebook and a digitalhub project yaml descriptor.

- Jypter notebook 
	- Import the Jupyter notebook for each scenario located inside project folder in the 'Coder' instance and execute it step by step.

- Project
	
   1. Import the project inside the 'Coder' instance using the yaml file.
	```
 	import digitalhub as dh
	proj = dh.import_project('project-ml-ci.yml')
	```

   2.  View the project details such as pipeline name.
     ```
     proj.to_dict()
     ```
     ```
     {
     'kind': 'project',
     'metadata': {
     'project': 'project-ml-ci',
     'name': 'project-ml-ci'
     },
     'spec': {'context': './',
     'functions': [
     {
     'id': '198e8387-0b03-4828-8eb7-914e10e4771e',
     'key': 'store://project-ml-ci/function/python/data-prep:198e8387-0b03-4828-8eb7-914e10e4771e',
     'kind': 'python',
     'metadata': {
     'name': 'data-prep',
     'project': 'project-ml-ci',
     },
     'name': 'data-prep',
     'project': 'project-ml-ci'
     },
     {
     'id': 'd1e4a950-0d3e-4898-853b-3b446124dd4c',
    'key': 'store://project-ml-ci/function/python/train:d1e4a950-0d3e-4898-853b-3b446124dd4c',
    'kind': 'python',
    'metadata': {
     'name': 'train',
     'project': 'project-ml-ci',
     'name': 'train',
     'project': 'project-ml-ci'
     },
     {
     'id': 'd3149583-07ae-4090-aa63-d91ca8614b0f',
     'key': 'store://project-ml-ci/function/sklearnserve/serve_sklearnmodel:d3149583-07ae-4090-aa63-d91ca8614b0f',
     'kind': 'sklearnserve',
     'metadata': {
     'name': 'serve_sklearnmodel',
     'project': 'project-ml-ci'
     },
     'name': 'serve_sklearnmodel',
     'project': 'project-ml-ci',
    ],
     'artifacts': [],
     'workflows': [
     {
     'id': '57197af4-39e0-426f-885e-1e122bf7102d',
     'key': 'store://project-ml-ci/workflow/kfp/pipeline_ml:57197af4-39e0-426f-885e-1e122bf7102d',
     'kind': 'kfp',
     'metadata': {
     'name': 'pipeline_ml',
     'project': 'project-ml-ci',
     },
     'name': 'pipeline_ml',
     'project': 'project-ml-ci',
     }],
     'dataitems': [],
     'models': [
     {
     'id': 'd1f3b412-92d1-4bcc-8b08-be766d540e3c',
    'key': 'store://project-ml-ci/model/sklearn/cancer_classifier:d1f3b412-92d1-4bcc-8b08-be766d540e3c',
    'kind': 'sklearn',
    'metadata': {
     'name': 'cancer_classifier',
     'project': 'project-ml-ci',
     'name': 'cancer_classifier',
    'project': 'project-ml-ci'
     }]},
     'status': {'state': 'CREATED'},
     'id': 'project-ml-ci',
     'name': 'project-ml-ci',
     'key': 'store://project-ml-ci'}
     ```

   3. In this scenario functions are self-contained so one execute the pipeline directly.
     ```
     workflow_run = proj.run('pipeline_ml')
     workflow_run
     ```
     Wait until the workflow pipeline executed completely. One can check the status of pipeline in the 'console' or using the digitalhub sdk api call.
     ```
     workflow_run.refresh().status.state
     ```
     Once completed, one can explore the results by fetching the list of data model using digithub sdk.
     ```
     data_model = dh.list_models(project='project-ml-ci')
     len(data_model)
     ```     
	
	

	
	


	
	



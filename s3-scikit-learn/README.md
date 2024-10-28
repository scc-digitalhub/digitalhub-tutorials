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
     print(proj)
     ```

   3. In this scenario functions are self-contained so one execute the pipeline directly.
     ```
     workflow_run = proj.run('pipeline_ml')
     ```
     We can now explore the results of the function. We can fetch the list of data model using digithub sdk.
     ```
     data_model = dh.list_models(project='project-ml-ci')
     ```
     ```
     print((data_model))
     ```
     ```
     [{'kind': 'sklearn', 'metadata': {'project': 'project-ml-ci', 'name': 'cancer_classifier', 'version': '6ffba1cc-1a1c-4719-a177-c2a739dc7a48', 'created': '2024-10-28T08:43:52.854Z', 'updated': '2024-10-28T08:43:53.025Z', 'created_by': 'khurshid@fbk.eu', 'updated_by': 'khurshid@fbk.eu', 'embedded': False}, 'spec': {'path': 's3://datalake/project-ml-ci/model/cancer_classifier/6ffba1cc-1a1c-4719-a177-c2a739dc7a48/', 'parameters': {}, 'metrics': {'f1_score': 0.949640287769784, 'accuracy': 0.9385964912280702, 'precision': 0.9041095890410958, 'recall': 1.0}}, 'status': {'state': 'CREATED', 'files': [{'path': 'model/cancer_classifier.pkl', 'name': 'cancer_classifier.pkl', 'content_type': 'binary/octet-stream', 'size': 34063, 'hash': 'md5:cc7991955f464545f34c515156ba4b32', 'last_modified': '2024-10-28T08:43:52+00:00'}]}, 'user': 'khurshid@fbk.eu', 'project': 'project-ml-ci', 'name': 'cancer_classifier', 'id': '6ffba1cc-1a1c-4719-a177-c2a739dc7a48', 'key': 'store://project-ml-ci/model/sklearn/cancer_classifier:6ffba1cc-1a1c-4719-a177-c2a739dc7a48'}]
     ```
	
	

	
	


	
	



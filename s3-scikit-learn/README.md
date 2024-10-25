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
     workflow_run()
     ```
	
	

	
	


	
	



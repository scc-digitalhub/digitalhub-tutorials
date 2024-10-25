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
     print(proj)
     ```

   3. In this scenario functions are self-contained so one execute the pipeline directly.
     ```
     workflow_run()
     ```
	
	

	
	


	
	



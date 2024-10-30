# DBT (Database transformation) scenario
This scenario depict how to collect some data regarding organizations, analyze and transform it, then expose the resulting dataset. 
The 's2-dbt' folder contains a jypter notebook and a digitalhub project yaml descriptor.

- Jypter notebook 
	- Import the Jupyter notebook for each scenario located inside project folder in the 'Coder' instance and execute it step by step.

- Project
	
   1. Import the project inside the 'Coder' instance using the yaml file.
	```
 	import digitalhub as dh
	proj = dh.import_project('project-dbt-ci.yml')
	```

   2.  View the project details such as pipeline name.
     ```
     print(proj)
     ```
	 
   3. Create the input parameter from url
     ```
     url = "https://gist.githubusercontent.com/kevin336/acbb2271e66c10a5b73aacf82ca82784/raw/e38afe62e088394d61ed30884dd50a6826eee0a8/employees.csv"
     di_url = proj.new_dataitem(name="url_data_item",kind="table",path=url)
     ```
	 
   4. Run the pipeline
     ```
     workflow_run = proj.run('pipeline_dbt', parameters={"url": di_url.key})
     ```
     ```
     workflow_run.refresh().status.state
     ```
     ```
     `Completed`
     ```
     Once completed, one can explore the results by fetching the list of dataitems using digithub sdk.
     ```
     data_items = dh.list_dataitems(project='project-dbt-ci')
     ```
     ```
     len(data_items)
     ```
     ```
     3
     ```
     As the result of scenario 3 data items are created in the core project. One can use the 'Core' instance to view them in the project console.
	
	

	
	


	
	



# ETL (Extract transform load)

The scenario depict how to collect some data regarding traffic, analyze and transform it, then expose the resulting dataset. The 's1_etl' folder contains
a jypter notebook and a digitalhub project. 

- Jypter notebook 
	- Import the Jupyter notebook for each scenario located inside project folder in the 'Coder' instance and execute it step by step.

- Project
	
   1. Import the project inside the 'Coder' instance using the yaml file.
	```
 	import digitalhub as dh
 	proj = dh.load_project(filename="projects-tutorial-dbt-ci.yaml")
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
	
	

	
	



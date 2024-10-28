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
     We can now explore the results of the function. We can fetch the list of dataitems using digithub sdk.
     ```
     data_items = dh.list_dataitems(project='project-dbt-ci')
     ```
     ```
     print(len(data_items))
     ```
     ```
     3
     ```
     As the result of scenario 3 data items are created in the core project. One can use the 'Core' instance to view them in the project console.
     ```
     [{'kind': 'table', 'metadata': {'project': 'project-dbt-ci', 'name': 'department-50', 'version': 'b46527f4-92ed-4fa7-b492-700485749de0', 'created': '2024-10-28T08:34:43.681Z', 'updated': '2024-10-28T08:34:43.681Z', 'created_by': 'khurshid@fbk.eu', 'updated_by': 'khurshid@fbk.eu', 'embedded': False}, 'spec': {'path': 'sql://digitalhub/public/department-50_vb46527f4-92ed-4fa7-b492-700485749de0', 'schema': {'fields': [{'name': 'EMPLOYEE_ID', 'type': 'integer'}, {'name': 'FIRST_NAME', 'type': 'string'}, {'name': 'LAST_NAME', 'type': 'string'}, {'name': 'EMAIL', 'type': 'string'}, {'name': 'PHONE_NUMBER', 'type': 'string'}, {'name': 'HIRE_DATE', 'type': 'string'}, {'name': 'JOB_ID', 'type': 'string'}, {'name': 'SALARY', 'type': 'integer'}, {'name': 'COMMISSION_PCT', 'type': 'string'}, {'name': 'MANAGER_ID', 'type': 'string'}, {'name': 'DEPARTMENT_ID', 'type': 'integer'}]}}, 'status': {'state': 'CREATED', 'files': [], 'preview': {'cols': [{'name': 'EMPLOYEE_ID', 'value': [198, 199, 120, 121, 122, 123, 124, 125, 126, 127]}, {'name': 'FIRST_NAME', 'value': ['Donald', 'Douglas', 'Matthew', 'Adam', 'Payam', 'Shanta', 'Kevin', 'Julia', 'Irene', 'James']}, {'name': 'LAST_NAME', 'value': ['OConnell', 'Grant', 'Weiss', 'Fripp', 'Kaufling', 'Vollman', 'Mourgos', 'Nayer', 'Mikkilineni', 'Landry']}, {'name': 'EMAIL', 'value': ['DOCONNEL', 'DGRANT', 'MWEISS', 'AFRIPP', 'PKAUFLIN', 'SVOLLMAN', 'KMOURGOS', 'JNAYER', 'IMIKKILI', 'JLANDRY']}, {'name': 'PHONE_NUMBER', 'value': ['650.507.9833', '650.507.9844', '650.123.1234', '650.123.2234', '650.123.3234', '650.123.4234', '650.123.5234', '650.124.1214', '650.124.1224', '650.124.1334']}, {'name': 'HIRE_DATE', 'value': ['21-JUN-07', '13-JAN-08', '18-JUL-04', '10-APR-05', '01-MAY-03', '10-OCT-05', '16-NOV-07', '16-JUL-05', '28-SEP-06', '14-JAN-07']}, {'name': 'JOB_ID', 'value': ['SH_CLERK', 'SH_CLERK', 'ST_MAN', 'ST_MAN', 'ST_MAN', 'ST_MAN', 'ST_MAN', 'ST_CLERK', 'ST_CLERK', 'ST_CLERK']}, {'name': 'SALARY', 'value': [2600, 2600, 8000, 8200, 7900, 6500, 5800, 3200, 2700, 2400]}, {'name': 'COMMISSION_PCT', 'value': [' - ', ' - ', ' - ', ' - ', ' - ', ' - ', ' - ', ' - ', ' - ', ' - ']}, {'name': 'MANAGER_ID', 'value': ['124', '124', '100', '100', '100', '100', '100', '120', '120', '120']}, {'name': 'DEPARTMENT_ID', 'value': [50, 50, 50, 50, 50, 50, 50, 50, 50, 50]}], 'rows': 23}}, 'user': 'khurshid@fbk.eu', 'project': 'project-dbt-ci', 'name': 'department-50', 'id': 'b46527f4-92ed-4fa7-b492-700485749de0', 'key': 'store://project-dbt-ci/dataitem/table/department-50:b46527f4-92ed-4fa7-b492-700485749de0'}, {'kind': 'table', 'metadata': {'project': 'project-dbt-ci', 'name': 'employees_v7455a003-e588-4f24-850a-d0b0c3723faf', 'version': '1dcf2c40-ed9b-4c30-ae58-c45a6cfe893b', 'created': '2024-10-28T08:34:41.109Z', 'updated': '2024-10-28T08:34:41.109Z', 'created_by': 'khurshid@fbk.eu', 'updated_by': 'khurshid@fbk.eu', 'embedded': False}, 'spec': {'path': 'sql://digitalhub/public/employees_v7455a003-e588-4f24-850a-d0b0c3723faf'}, 'status': {'state': 'CREATED', 'files': []}, 'user': 'khurshid@fbk.eu', 'project': 'project-dbt-ci', 'name': 'employees_v7455a003-e588-4f24-850a-d0b0c3723faf', 'id': '1dcf2c40-ed9b-4c30-ae58-c45a6cfe893b', 'key': 'store://project-dbt-ci/dataitem/table/employees_v7455a003-e588-4f24-850a-d0b0c3723faf:1dcf2c40-ed9b-4c30-ae58-c45a6cfe893b'}, {'kind': 'table', 'metadata': {'project': 'project-dbt-ci', 'name': 'url_data_item', 'version': '7455a003-e588-4f24-850a-d0b0c3723faf', 'created': '2024-10-28T08:34:21.43Z', 'updated': '2024-10-28T08:34:21.43Z', 'created_by': 'khurshid@fbk.eu', 'updated_by': 'khurshid@fbk.eu', 'embedded': False}, 'spec': {'path': 'https://gist.githubusercontent.com/kevin336/acbb2271e66c10a5b73aacf82ca82784/raw/e38afe62e088394d61ed30884dd50a6826eee0a8/employees.csv'}, 'status': {'state': 'CREATED', 'files': [{'path': 'https://gist.githubusercontent.com/kevin336/acbb2271e66c10a5b73aacf82ca82784/raw/e38afe62e088394d61ed30884dd50a6826eee0a8/employees.csv', 'name': 'employees.csv', 'size': 3778, 'content_type': 'text/plain;charset=utf-8', 'last_modified': '1969-12-31T23:59:59.999+00:00'}]}, 'user': 'khurshid@fbk.eu', 'project': 'project-dbt-ci', 'name': 'url_data_item', 'id': '7455a003-e588-4f24-850a-d0b0c3723faf', 'key': 'store://project-dbt-ci/dataitem/table/url_data_item:7455a003-e588-4f24-850a-d0b0c3723faf'}]
     ```
	
	

	
	


	
	



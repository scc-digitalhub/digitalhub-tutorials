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
     5
     ```
     As the result of scenario 5 data items are created in the core project. One can use the 'Core' instance to view them in the project console.
     ```
     [{'kind': 'table', 'metadata': {'project': 'project-dbt-ci', 'name': 'department-50', 'version': 'f0ea40e9-f6a2-4c54-b6cc-70f6918a9269', 'created': '2024-10-29T13:16:49.682Z', 'updated': '2024-10-29T13:16:49.682Z', 'created_by': 'khurshid@fbk.eu', 'updated_by': 'khurshid@fbk.eu', 'embedded': False}, 'spec': {'path': 'sql://digitalhub/public/department-50_vf0ea40e9-f6a2-4c54-b6cc-70f6918a9269', 'schema': {'fields': [{'name': 'EMPLOYEE_ID', 'type': 'integer'}, {'name': 'FIRST_NAME', 'type': 'string'}, {'name': 'LAST_NAME', 'type': 'string'}, {'name': 'EMAIL', 'type': 'string'}, {'name': 'PHONE_NUMBER', 'type': 'string'}, {'name': 'HIRE_DATE', 'type': 'string'}, {'name': 'JOB_ID', 'type': 'string'}, {'name': 'SALARY', 'type': 'integer'}, {'name': 'COMMISSION_PCT', 'type': 'string'}, {'name': 'MANAGER_ID', 'type': 'string'}, {'name': 'DEPARTMENT_ID', 'type': 'integer'}]}}, 'status': {'state': 'CREATED', 'files': [], 'preview': {'cols': [{'name': 'EMPLOYEE_ID', 'value': [198, 199, 120, 121, 122, 123, 124, 125, 126, 127]}, {'name': 'FIRST_NAME', 'value': ['Donald', 'Douglas', 'Matthew', 'Adam', 'Payam', 'Shanta', 'Kevin', 'Julia', 'Irene', 'James']}, {'name': 'LAST_NAME', 'value': ['OConnell', 'Grant', 'Weiss', 'Fripp', 'Kaufling', 'Vollman', 'Mourgos', 'Nayer', 'Mikkilineni', 'Landry']}, {'name': 'EMAIL', 'value': ['DOCONNEL', 'DGRANT', 'MWEISS', 'AFRIPP', 'PKAUFLIN', 'SVOLLMAN', 'KMOURGOS', 'JNAYER', 'IMIKKILI', 'JLANDRY']}, {'name': 'PHONE_NUMBER', 'value': ['650.507.9833', '650.507.9844', '650.123.1234', '650.123.2234', '650.123.3234', '650.123.4234', '650.123.5234', '650.124.1214', '650.124.1224', '650.124.1334']}, {'name': 'HIRE_DATE', 'value': ['21-JUN-07', '13-JAN-08', '18-JUL-04', '10-APR-05', '01-MAY-03', '10-OCT-05', '16-NOV-07', '16-JUL-05', '28-SEP-06', '14-JAN-07']}, {'name': 'JOB_ID', 'value': ['SH_CLERK', 'SH_CLERK', 'ST_MAN', 'ST_MAN', 'ST_MAN', 'ST_MAN', 'ST_MAN', 'ST_CLERK', 'ST_CLERK', 'ST_CLERK']}, {'name': 'SALARY', 'value': [2600, 2600, 8000, 8200, 7900, 6500, 5800, 3200, 2700, 2400]}, {'name': 'COMMISSION_PCT', 'value': [' - ', ' - ', ' - ', ' - ', ' - ', ' - ', ' - ', ' - ', ' - ', ' - ']}, {'name': 'MANAGER_ID', 'value': ['124', '124', '100', '100', '100', '100', '100', '120', '120', '120']}, {'name': 'DEPARTMENT_ID', 'value': [50, 50, 50, 50, 50, 50, 50, 50, 50, 50]}], 'rows': 23}}, 'user': 'khurshid@fbk.eu', 'project': 'project-dbt-ci', 'name': 'department-50', 'id': 'f0ea40e9-f6a2-4c54-b6cc-70f6918a9269', 'key': 'store://project-dbt-ci/dataitem/table/department-50:f0ea40e9-f6a2-4c54-b6cc-70f6918a9269'},{'kind': 'table', 'metadata': {'project': 'project-dbt-ci', 'name': 'employees_v1cd328e9-7e8b-454e-9377-fb09d39ff060', 'version': 'c5931b95-5160-47a9-b216-ef36ba2ea935', 'created': '2024-10-29T13:16:47.202Z', 'updated': '2024-10-29T13:16:47.202Z', 'created_by': 'khurshid@fbk.eu', 'updated_by': 'khurshid@fbk.eu', 'embedded': False}, 'spec': {'path': 'sql://digitalhub/public/employees_v1cd328e9-7e8b-454e-9377-fb09d39ff060'}, 'status': {'state': 'CREATED', 'files': []}, 'user': 'khurshid@fbk.eu', 'project': 'project-dbt-ci', 'name': 'employees_v1cd328e9-7e8b-454e-9377-fb09d39ff060', 'id': 'c5931b95-5160-47a9-b216-ef36ba2ea935', 'key': 'store://project-dbt-ci/dataitem/table/employees_v1cd328e9-7e8b-454e-9377-fb09d39ff060:c5931b95-5160-47a9-b216-ef36ba2ea935'}, {'kind': 'table', 'metadata': {'project': 'project-dbt-ci', 'name': 'url_data_item', 'version': '1cd328e9-7e8b-454e-9377-fb09d39ff060', 'created': '2024-10-29T13:16:34.936Z', 'updated': '2024-10-29T13:16:34.936Z', 'created_by': 'khurshid@fbk.eu', 'updated_by': 'khurshid@fbk.eu', 'embedded': False}, 'spec': {'path': 'https://gist.githubusercontent.com/kevin336/acbb2271e66c10a5b73aacf82ca82784/raw/e38afe62e088394d61ed30884dd50a6826eee0a8/employees.csv'}, 'status': {'state': 'CREATED', 'files': [{'path': 'https://gist.githubusercontent.com/kevin336/acbb2271e66c10a5b73aacf82ca82784/raw/e38afe62e088394d61ed30884dd50a6826eee0a8/employees.csv', 'name': 'employees.csv', 'size': 3778, 'content_type': 'text/plain;charset=utf-8', 'last_modified': '1969-12-31T23:59:59.999+00:00'}]}, 'user': 'khurshid@fbk.eu', 'project': 'project-dbt-ci', 'name': 'url_data_item', 'id': '1cd328e9-7e8b-454e-9377-fb09d39ff060', 'key': 'store://project-dbt-ci/dataitem/table/url_data_item:1cd328e9-7e8b-454e-9377-fb09d39ff060'}, {'kind': 'table', 'metadata': {'project': 'project-dbt-ci', 'name': 'employees_vd3122cfa-830d-443b-a00f-549327d3f361', 'version': '02c9be21-06c2-4c59-8739-4b12d48ec9e0', 'created': '2024-10-29T13:15:14.697Z', 'updated': '2024-10-29T13:15:14.697Z', 'created_by': 'khurshid@fbk.eu', 'updated_by': 'khurshid@fbk.eu', 'embedded': False}, 'spec': {'path': 'sql://digitalhub/public/employees_vd3122cfa-830d-443b-a00f-549327d3f361'}, 'status': {'state': 'CREATED', 'files': []}, 'user': 'khurshid@fbk.eu', 'project': 'project-dbt-ci', 'name': 'employees_vd3122cfa-830d-443b-a00f-549327d3f361', 'id': '02c9be21-06c2-4c59-8739-4b12d48ec9e0', 'key': 'store://project-dbt-ci/dataitem/table/employees_vd3122cfa-830d-443b-a00f-549327d3f361:02c9be21-06c2-4c59-8739-4b12d48ec9e0'}, {'kind': 'table', 'metadata': {'project': 'project-dbt-ci', 'name': 'employees-dbt', 'version': 'd3122cfa-830d-443b-a00f-549327d3f361', 'created': '2024-10-29T13:15:06.498Z', 'updated': '2024-10-29T13:15:06.498Z', 'created_by': 'khurshid@fbk.eu', 'updated_by': 'khurshid@fbk.eu', 'embedded': False}, 'spec': {'path': 'https://gist.githubusercontent.com/kevin336/acbb2271e66c10a5b73aacf82ca82784/raw/e38afe62e088394d61ed30884dd50a6826eee0a8/employees.csv'}, 'status': {'state': 'CREATED', 'files': [{'path': 'https://gist.githubusercontent.com/kevin336/acbb2271e66c10a5b73aacf82ca82784/raw/e38afe62e088394d61ed30884dd50a6826eee0a8/employees.csv', 'name': 'employees.csv', 'size': 3778, 'content_type': 'text/plain;charset=utf-8', 'last_modified': '1969-12-31T23:59:59.999+00:00'}]}, 'user': 'khurshid@fbk.eu', 'project': 'project-dbt-ci', 'name': 'employees-dbt', 'id': 'd3122cfa-830d-443b-a00f-549327d3f361', 'key': 'store://project-dbt-ci/dataitem/table/employees-dbt:d3122cfa-830d-443b-a00f-549327d3f361'}]
     ```
	
	

	
	


	
	



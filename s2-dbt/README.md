# DBT (Database transformation) scenario

This scenario depict how to collect some data regarding organizations, analyze and transform it, then expose the resulting dataset. The "s2-dbt" folder contains a jupyter notebook and a digitalhub project yaml descriptor.

## Jypter notebook

Import the Jupyter notebook located inside project folder in the "Coder" instance and execute it step by step.

## Project

Import the project using the yaml file.

```python
import digitalhub as dh
proj = dh.import_project("projects-project-dbt.yaml")
```

View the project details such as pipeline name.

```python
wkfl = proj.spec.workflows[0]["name"]
```

Get the input parameter for workflow (dataitem is imported above with the project)

```python
di = proj.get_dataitem("employees-dbt")
```

Run the pipeline through the project

```python
workflow_run = proj.run(wkfl, action="pipeline", parameters={"url": di.key}, wait=True)
```

# ETL (Extract transform load)

The scenario depict how to collect some data regarding traffic, analyze and transform it, then expose the resulting dataset. The "s1-etl" folder contains a jupyter notebook and a digitalhub project.

## Jypter notebook

Import the Jupyter notebook located inside project folder in the "Coder" instance and execute it step by step.

## Project

Import the project using the yaml file.

```python
import digitalhub as dh
proj = dh.import_project("projects-project-etl.yaml")
```

View the project details such as pipeline name.

```python
wkfl = proj.spec.workflows[0].name
```

Get the input parameter for workflow (dataitem is imported above with the project)

```python
di = proj.get_dataitem("url_data_item")
```

Run the pipeline through the project

```python
workflow_run = proj.run(wkfl, action="pipeline", parameters={"url": di.key}, wait=True)
```

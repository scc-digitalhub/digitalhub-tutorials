# ETL (Extract transform load)

The scenario depict how to collect some data regarding traffic, analyze and transform it, then expose the resulting dataset. The "s1-etl" folder contains a jupyter notebook and a digitalhub project.

## Jypter notebook

Import the Jupyter notebook located inside project folder in the "Coder" instance and execute it step by step.

## Project

Import the project using the yaml file.

```python
import digitalhub as dh
try:
    proj = dh.import_project("projects-tutorial-project.yaml")
except:
    proj = dh.load_project("projects-tutorial-project.yaml")
```

Get the input parameter for workflow (dataitem is imported above with the project)

```python
di = proj.get_dataitem("url-data-item")
```

Build and run the pipeline through the project:

```python
proj.run("pipeline", action="build", wait=True)
workflow_run = proj.run("pipeline", action="pipeline", parameters={"url": di.key}, wait=True)
```

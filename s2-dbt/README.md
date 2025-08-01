# DBT (Database transformation) scenario

This scenario depict how to collect some data regarding organizations, analyze and transform it, then expose the resulting dataset. The "s2-dbt" folder contains a jupyter notebook and a digitalhub project yaml descriptor.

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
di = proj.get_dataitem("employees-dbt")
```

Build and run the pipeline through the project:

```python
proj.run("dbt-pipeline", action="build", wait=True)
workflow_run = proj.run("dbt-pipeline", action="pipeline", parameters={"di": di.key}, wait=True)
```

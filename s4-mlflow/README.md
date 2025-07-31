# ML Flow Model Training and Serving

This scenario provides a quick overview of developing and deploying a machine learning application based on model tracked with MLFlow framework using the functionalities of the platform. The "s4-mlflow" folder contains a jupyter notebook and a digitalhub project.

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

View the project details such as pipeline name.

```python
wkfl = proj.spec.workflows[0]["name"]
```

Build and run the pipeline through the project:

```python
proj.run(wkfl, action="build", wait=True)
workflow_run = proj.run(wkfl, action="pipeline", wait=True)
```

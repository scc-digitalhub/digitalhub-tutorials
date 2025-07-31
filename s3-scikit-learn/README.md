# Scikit Learn Scenario

This scenario provides a quick overview of developing and deploying a scikit-learn machine learning application using the functionalities of the platform. We will prepare data, train a generic model and expose it as a service. The "s3-scikit-learn" folder contains a jupyter notebook and a digitalhub project yaml descriptor.

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

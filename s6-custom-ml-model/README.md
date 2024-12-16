# Custom ML FLow Model Training and Serving

This scenario provides a quick overview of developing and deploying generic machine learning applications using the functionalities of the platform. For this purpose, we use ML algorithms for the time series management provided by the Darts framework. The "s6-custom-ml-model" folder contains a jupyter notebook and a digitalhub project.

## Jypter notebook

Import the Jupyter notebook located inside project folder in the "Coder" instance and execute it step by step.

## Project

Import the project using the yaml file.

```python
import digitalhub as dh
proj = dh.import_project("projects-project-cml-darts-ci.yaml")
```

View the project details such as pipeline name.

```python
wkfl = proj.spec.workflows[0]["name"]
```

Build the container. Fetch the training function and build it.

```python
train_fn = proj.get_function("train-darts")
train_run =  train_fn.run(action="build", instructions=["pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu","pip3 install darts patsy scikit-learn"])
```

Run the pipeline through the project

```python
workflow_run = proj.run(wkfl, action="pipeline", parameters={"url": di.key}, wait=True)
```

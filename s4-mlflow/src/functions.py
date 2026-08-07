from urllib.parse import urlparse

import mlflow
from digitalhub_runtime_python import handler
from sklearn import datasets, svm
from sklearn.model_selection import GridSearchCV


@handler(outputs=["model"])
def train_model(project):
    """Train an SVM classifier on Iris data with MLflow logging."""
    # Enable MLflow autologging for sklearn.
    mlflow.sklearn.autolog(log_datasets=True)

    # Load the Iris dataset.
    iris = datasets.load_iris()

    # Define the grid search space.
    parameters = {"kernel": ("linear", "rbf"), "C": [1, 10]}
    svc = svm.SVC()
    clf = GridSearchCV(svc, parameters)

    # Train the model.
    clf.fit(iris.data, iris.target)

    # Get the MLflow run id.
    run_id = mlflow.last_active_run().info.run_id

    # Extract model metadata and metrics for DigitalHub.
    model_params, metrics = _from_mlflow_run(run_id)

    # Register the model in DigitalHub.
    model = project.log_mlflow(name="iris-classifier", **model_params)
    model.log_metrics(metrics)
    return model


def _from_mlflow_run(run_id: str) -> dict:
    """Extract model metadata and metrics from an MLflow run."""

    # Load the MLflow run.
    run = mlflow.MlflowClient().get_run(run_id)

    data = run.data
    parameters = data.params
    # Resolve the model source path.
    source_path = urlparse(run.info.artifact_uri).path + "/model"
    model_uri = f"runs:/{run_id}/model"
    model = mlflow.pyfunc.load_model(model_uri=model_uri)
    try:
        model_config = model.model_config
    except Exception:
        model_config = {}
    flavor = None
    for f in model.metadata.flavors:
        if f != "python_function":
            flavor = f
            break

    # Extract the model signature when available.
    try:
        mlflow_signature = model.metadata.signature
        signature = {
            "inputs": mlflow_signature.inputs.to_json()
            if mlflow_signature.inputs
            else None,
            "outputs": mlflow_signature.outputs.to_json()
            if mlflow_signature.outputs
            else None,
            "params": mlflow_signature.params.to_json()
            if mlflow_signature.params
            else None,
        }
    except Exception:
        signature = None

    # Extract dataset metadata when available.
    datasets = []
    try:
        if run.inputs and run.inputs.dataset_inputs:
            datasets = [
                {
                    "name": d.dataset.name,
                    "digest": d.dataset.digest,
                    "profile": d.dataset.profile,
                    "dataset_schema": d.dataset.schema,
                    "source": d.dataset.source,
                    "source_type": d.dataset.source_type,
                }
                for d in run.inputs.dataset_inputs
            ]
    except Exception:
        datasets = []

    # Build the DigitalHub model payload.
    model_params = {}

    model_params["source"] = source_path

    model_params["framework"] = flavor
    model_params["parameters"] = parameters

    model_params["flavor"] = flavor
    model_params["model_config"] = model_config
    model_params["input_datasets"] = datasets
    model_params["signature"] = signature

    # Return the logged metrics.
    metrics = run.data.metrics

    return model_params, metrics

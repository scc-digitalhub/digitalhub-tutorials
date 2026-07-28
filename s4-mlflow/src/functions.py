from urllib.parse import urlparse

import mlflow
from digitalhub_runtime_python import handler
from sklearn import datasets, svm
from sklearn.model_selection import GridSearchCV


@handler(outputs=["model"])
def train_model(project):
    """
    Train an SVM classifier on the Iris dataset with hyperparameter tuning using MLflow
    """
    # Enable MLflow autologging for sklearn
    mlflow.sklearn.autolog(log_datasets=True)

    # Load Iris dataset
    iris = datasets.load_iris()

    # Define hyperparameter search space
    parameters = {"kernel": ("linear", "rbf"), "C": [1, 10]}
    svc = svm.SVC()
    clf = GridSearchCV(svc, parameters)

    # Train model with grid search
    clf.fit(iris.data, iris.target)

    # Get MLflow run information
    run_id = mlflow.last_active_run().info.run_id

    # Extract MLflow run artifacts and metadata for DigitalHub integration
    model_params, metrics = _from_mlflow_run(run_id)

    # Register model in DigitalHub with MLflow metadata
    model = project.log_model(name="iris-classifier", kind="mlflow", **model_params)
    model.log_metrics(metrics)
    return model


def _from_mlflow_run(run_id: str) -> dict:
    """
    Extract from mlflow run spec for Digitalhub Model.

    Parameters
    ----------
    run_id : str
        The id of the mlflow run.

    Returns
    -------
    dict
        The extracted spec.
    """

    # Get MLFlow run
    run = mlflow.MlflowClient().get_run(run_id)

    # Extract spec
    data = run.data
    parameters = data.params
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

    # Extract signature
    try:
        mlflow_signature = model.metadata.signature
        signature = dict(
            inputs=mlflow_signature.inputs.to_json()
            if mlflow_signature.inputs
            else None,
            outputs=mlflow_signature.outputs.to_json()
            if mlflow_signature.outputs
            else None,
            params=mlflow_signature.params.to_json()
            if mlflow_signature.params
            else None,
        )
    except Exception:
        signature = None

    # Extract datasets
    datasets = []
    try:
        if run.inputs and run.inputs.dataset_inputs:
            datasets = [
                dict(
                    name=d.dataset.name,
                    digest=d.dataset.digest,
                    profile=d.dataset.profile,
                    dataset_schema=d.dataset.schema,
                    source=d.dataset.source,
                    source_type=d.dataset.source_type,
                )
                for d in run.inputs.dataset_inputs
            ]
    except Exception:
        datasets = []

    # Create model params
    model_params = {}

    # source path
    model_params["source"] = source_path

    # common properties
    model_params["framework"] = flavor
    model_params["parameters"] = parameters

    # specific to MLFlow
    model_params["flavor"] = flavor
    model_params["model_config"] = model_config
    model_params["input_datasets"] = datasets
    model_params["signature"] = signature

    metrics = run.data.metrics

    return model_params, metrics

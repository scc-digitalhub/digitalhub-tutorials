import mlflow
from digitalhub import from_mlflow_run, get_mlflow_model_metrics
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
    model_params = from_mlflow_run(run_id)
    metrics = get_mlflow_model_metrics(run_id)

    # Register model in DigitalHub with MLflow metadata
    model = project.log_model(name="iris-classifier", kind="mlflow", **model_params)
    model.log_metrics(metrics)
    return model


from digitalhub_runtime_python import handler
from digitalhub import from_mlflow_run
import mlflow

from sklearn import datasets, svm
from sklearn.model_selection import GridSearchCV

@handler(outputs=["model"])
def train(project):
    mlflow.sklearn.autolog(log_datasets=True)

    iris = datasets.load_iris()
    parameters = {"kernel": ("linear", "rbf"), "C": [1, 10]}
    svc = svm.SVC()
    clf = GridSearchCV(svc, parameters)

    clf.fit(iris.data, iris.target)
    run_id = mlflow.last_active_run().info.run_id

    # utility to map mlflow run artifacts to model metadata
    model_params = from_mlflow_run(run_id)

    return (project.log_model(
        name="model-mlflow",
        kind="mlflow",
        **model_params
))

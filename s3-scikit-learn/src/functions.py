import os
from pickle import dump

import pandas as pd
import sklearn.metrics
from digitalhub_runtime_python import handler
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


@handler(outputs=["dataset"])
def data_generator():
    """
    A function which generates the breast cancer dataset from scikit-learn
    """
    breast_cancer = load_breast_cancer()
    breast_cancer_dataset = pd.DataFrame(
        data=breast_cancer.data, columns=breast_cancer.feature_names
    )
    breast_cancer_labels = pd.DataFrame(data=breast_cancer.target, columns=["target"])
    breast_cancer_dataset = pd.concat(
        [breast_cancer_dataset, breast_cancer_labels], axis=1
    )
    return breast_cancer_dataset


@handler(outputs=["model"])
def train_model(project, di):
    """
    Train an SVM classifier on the breast cancer dataset and log metrics
    """
    df_cancer = di.as_df()
    X = df_cancer.drop(["target"], axis=1)
    y = df_cancer["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=5
    )
    svc_model = SVC()
    svc_model.fit(X_train, y_train)
    y_predict = svc_model.predict(X_test)

    if not os.path.exists("model"):
        os.makedirs("model")

    with open("model/breast_cancer_classifier.pkl", "wb") as f:
        dump(svc_model, f, protocol=5)

    metrics = {
        "f1_score": sklearn.metrics.f1_score(y_test, y_predict),
        "accuracy": sklearn.metrics.accuracy_score(y_test, y_predict),
        "precision": sklearn.metrics.precision_score(y_test, y_predict),
        "recall": sklearn.metrics.recall_score(y_test, y_predict),
    }
    model = project.log_model(
        name="breast_cancer_classifier", kind="sklearn", source="./model/"
    )
    model.log_metrics(metrics)
    return model

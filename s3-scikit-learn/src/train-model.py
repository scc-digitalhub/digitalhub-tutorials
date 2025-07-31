import os
from pickle import dump

import sklearn.metrics
from digitalhub_runtime_python import handler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


@handler(outputs=["model"])
def train(project, di):
    df_cancer = di.as_df()
    X = df_cancer.drop(["target"], axis=1)
    y = df_cancer["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=5)
    svc_model = SVC()
    svc_model.fit(X_train, y_train)
    y_predict = svc_model.predict(X_test)

    if not os.path.exists("model"):
        os.makedirs("model")

    with open("model/cancer_classifier.pkl", "wb") as f:
        dump(svc_model, f, protocol=5)

    metrics = {
        "f1_score": sklearn.metrics.f1_score(y_test, y_predict),
        "accuracy": sklearn.metrics.accuracy_score(y_test, y_predict),
        "precision": sklearn.metrics.precision_score(y_test, y_predict),
        "recall": sklearn.metrics.recall_score(y_test, y_predict),
    }
    model = project.log_model(name="cancer_classifier", kind="sklearn", source="./model/")
    model.log_metrics(metrics)
    return model

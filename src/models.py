import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from datetime import datetime
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import hashlib


from src.metrics import get_metrics


def get_file_hash(filepath):
    with open(filepath, 'rb') as f:
        file_hash = hashlib.md5(f.read()).hexdigest()
    return file_hash


def train_and_log_logistic_regression(X_train, X_test, y_train, y_test, preprocessor, data_path=None, **params):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if mlflow.active_run():
        mlflow.end_run()

    with mlflow.start_run():
        mlflow.set_tag("model_type", f"logistic_regression_{timestamp}")

        if data_path:
            data_hash = get_file_hash(data_path)
            mlflow.set_tag("data_path", data_path)
            mlflow.set_tag("data_hash", data_hash)
            mlflow.log_artifact(data_path, artifact_path="dataset")

        pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                    ('logregressor', LogisticRegression(**params))])
        mlflow.log_params(params)

        model = pipeline.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_pred_prob = model.predict_proba(X_test)[:, 1]
        metrics = get_metrics(y_test, y_pred, y_pred_prob)

        input_example = X_test[:5]
        signature = infer_signature(X_test, y_pred)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            signature=signature,
            input_example=input_example,
            registered_model_name="LogisticRegressionModel"
        ) # type: ignore
    return model
    
    
def train_and_log_random_forest(X_train, X_test, y_train, y_test, preprocessor, data_path=None, **params):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if mlflow.active_run():
        mlflow.end_run()

    with mlflow.start_run():
        mlflow.set_tag("model_type", f"random_forest_{timestamp}")

        if data_path:
            data_hash = get_file_hash(data_path)
            mlflow.set_tag("data_path", data_path)
            mlflow.set_tag("data_hash", data_hash)
            mlflow.log_artifact(data_path, artifact_path="dataset")

        pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                    ('randomforest', RandomForestClassifier(**params))])
        mlflow.log_params(params)

        model = pipeline.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_pred_prob = model.predict_proba(X_test)[:, 1]
        metrics = get_metrics(y_test, y_pred, y_pred_prob)
        
        input_example = X_test[:5]
        signature = infer_signature(X_test, y_pred)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            signature=signature,
            input_example=input_example,
            registered_model_name="RandomForestClassifier"
        ) # type: ignore
    return model
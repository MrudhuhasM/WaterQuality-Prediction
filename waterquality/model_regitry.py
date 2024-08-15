"""
This module contains tasks for registering and testing models in MLFlow.

"""

import mlflow
import pandas as pd
import sklearn
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from mlflow.pyfunc import PyFuncModel
from prefect import task
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)


@task(name="Check Model Metrics", log_prints=True)
def check_model_with_registered_model(
    model_name: str, new_accuracy: float
) -> bool:
    """
    Check if the new model is better than the current champion

    Args:
        model_name   (str)   : Name of the model.
        new_accuracy (float) : Accuracy of the new model.

    Returns:
        bool: True if the new model is better, False otherwise.
    """
    client = MlflowClient()
    try:
        model_version = client.get_model_version_by_alias(
            model_name, "champion"
        )
        if model_version:
            run_id = model_version.run_id
            metrics = client.get_run(run_id).data.metrics
            accuracy = metrics.get("f1")
            if new_accuracy >= accuracy:
                return True
            print(f"Current champion model f1 score: {accuracy*100: .3f}%")
            print(f"New model f1 score: {new_accuracy*100: .3f}%")
            print("Model is not better than the current champion model")
        return False
    except MlflowException as e:
        if "RESOURCE_DOES_NOT_EXIST" in str(e):
            print("No registered model found. This might be the first run.")
            return True
        else:
            raise


@task(name="Register Model")
def register_model(
    model: sklearn.base.BaseEstimator,
    model_name: str,
    input_example: pd.DataFrame,
) -> None:
    """
    Register the model in MLFlow

    Args:
        model         (sklearn.base.BaseEstimator): The trained model.
        model_name    (str): Name of the model.
        input_example (pd.DataFrame): Example input for the model.
    """
    client = MlflowClient()
    model_info = mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        input_example=input_example,
        registered_model_name=model_name,
    )
    client.set_registered_model_alias(
        model_name, "champion", model_info.registered_model_version
    )


@task(name="Load Registered Model")
def load_registered_model(
    model_name: str, alias: str = "champion"
) -> PyFuncModel:
    """
    Load the registered model from MLFlow.

    Args:
        model_name (str): Name of the registered model.
        alias (str, optional): Alias of the registered model.
                               Defaults to "champion".

    Returns:
        PyFuncModel: The loaded registered model.
    """
    model_uri = f"models:/{model_name}@{alias}"  # noqa: E231
    loaded_model = mlflow.pyfunc.load_model(model_uri)
    return loaded_model


@task(name="Testing Registered Model", log_prints=True)
def test_register_model(
    model: PyFuncModel, X_test: pd.DataFrame, y_test: pd.Series
) -> dict:
    """
    Test the registered model

    Args:
        model_name (PyFuncModel): The registered model.
        X_test (pd.DataFrame): The test features.
        y_test (pd.Series)   : The test target.

    Returns:
        dict: The accuracy, f1, precision, and recall scores.
    """
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    return {
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }

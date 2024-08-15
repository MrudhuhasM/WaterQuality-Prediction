"""
Description: This module contains the flow for training the model.

"""

import os

import mlflow
import optuna
import pandas as pd
import sklearn
from dotenv import load_dotenv
from prefect import flow, task
from prefect.artifacts import create_table_artifact
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

from waterquality.data_handling import load_data, split_data
from waterquality.hyperparameter_tuning import tune_model
from waterquality.model_regitry import (
    check_model_with_registered_model,
    load_registered_model,
    register_model,
    test_register_model,
)

load_dotenv()


def mlflow_dataset(X: pd.DataFrame, y: pd.Series) -> mlflow.data.Dataset:
    """
    Create a MLFlow dataset from the features and target.

    Args:
        X (pd.DataFrame): Dataframe containing the features.
        y (pd.Series): Series containing the target.

    Returns:
        mlflow.data.Dataset: MLFlow dataset containing the data.
    """
    return mlflow.data.from_pandas(
        pd.concat([X, y], axis=1), targets="is_safe", name="train_data"
    )


@task(name="Initiate MLFlow")
def initiate_mlflow(experiment_name: str) -> int:
    """
    Create a new experiment in MLFlow if it doesn't exist

    Args:
        experiment_name (str): Name of the experiment.

    Returns:
        int: The experiment ID.
    """
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")

    mlflow.set_tracking_uri(tracking_uri)
    if experiment := mlflow.get_experiment_by_name(experiment_name):
        experiment_id = experiment.experiment_id
    else:
        experiment_id = mlflow.create_experiment(name=experiment_name)
    mlflow.set_experiment(experiment_id=experiment_id)
    return experiment_id


@task(name="Fit Tuned Model")
def fit_tuned_model(
    study: optuna.study.Study,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[sklearn.base.BaseEstimator, float]:
    """
    Fit the tuned model and log the metrics

    Args:
        study (optuna.study.Study): The optuna study object.
        X_train (_type_): The training features.
        y_train (_type_): The training target.
        X_test (_type_): The test features.
        y_test (_type_): The test target.

    Returns:
        tuple: The trained model and the F1 score.
    """
    model = RandomForestClassifier(**study.best_params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    mlflow.log_params(study.best_params)
    mlflow.set_tags(
        {
            "classifier": model.__class__.__name__,
            "Optimizer": "optuna",
        }
    )
    mlflow.log_input(mlflow_dataset(X_train, y_train), context="train")
    mlflow.log_input(mlflow_dataset(X_test, y_test), context="test")
    mlflow.log_metrics(
        {
            "accuracy": accuracy,
            "f1": f1,
            "precision": precision,
            "recall": recall,
        }
    )
    return (model, f1)


@task
def table_task(metrics: dict[str, float]) -> None:
    """
    Create a table artifact with the model metrics
    """
    table = [
        {"Metric": "Accuracy", "Value": metrics["accuracy"]},
        {"Metric": "F1 Score", "Value": metrics["f1"]},
        {"Metric": "Precision", "Value": metrics["precision"]},
        {"Metric": "Recall", "Value": metrics["recall"]},
    ]
    create_table_artifact(
        key="model-metrics",
        table=table,
        description="# Model Metrics for this run",
    )


@flow(name="Training Model", log_prints=True)
def main(trails: int) -> None:
    """
    Main flow for training the model
    """
    experiment_name = os.getenv("EXPERIMENT_NAME", default="waterquality")
    data_path = os.getenv("DATA_PATH", default="data/water_potability.csv")
    model_name = os.getenv("REGISTER_MODEL_NAME", default="waterquality")

    experiment_id = initiate_mlflow(experiment_name)
    with mlflow.start_run(experiment_id=experiment_id, nested=True):
        data = load_data(data_path)
        X_train, X_test, y_train, y_test = split_data(data, "is_safe")
        study = tune_model(X_train, y_train, trails)
        model, accuracy = fit_tuned_model(
            study, X_train, y_train, X_test, y_test
        )

        input_example = X_train.iloc[0:1]
        if check_model_with_registered_model(model_name, accuracy):
            print("Registering new model")
            register_model(model, model_name, input_example)
    model = load_registered_model(model_name)
    metrics = test_register_model(model, X_test, y_test)

    table_task(metrics)


if __name__ == "__main__":
    main.serve(  # type: ignore
        name    ="Training Model",
        tags=["training", "model"],
        parameters={"trails": 30},
        version="0.1.0",
    )

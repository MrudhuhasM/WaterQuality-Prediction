"""
Module for hyperparameter optimization using Optuna.

"""

import mlflow
import optuna
import pandas as pd
from prefect import task
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def objective(trial: optuna.Trial, X: pd.DataFrame, y: pd.Series) -> float:
    """
    Optuna objective function for hyperparameter optimization

    Args:
        trial (optuna.Trial) : The optuna trial object.
        X     (pd.DataFrame) : Dataframe containing the features.
        y     (pd.Series)    : Series containing the target.

    Returns:
        float: The average accuracy of the model.
    """
    with mlflow.start_run(nested=True):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 2, 150),
            "criterion": trial.suggest_categorical(
                "criterion", ["gini", "entropy"]
            ),
            "max_depth": trial.suggest_int("max_depth", 1, 32),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical(
                "max_features", [0.2, 0.5, 0.7, "sqrt", "log2"]
            ),
            "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
        }
        model = RandomForestClassifier(**params)

        normalizer = ColumnTransformer(
            transformers=[("num", StandardScaler(), X.columns)]
        )
        pipeline = Pipeline([("normalizer", normalizer), ("model", model)])

        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_validate(
            pipeline, X, y, cv=cv, n_jobs=-1, scoring=["f1_weighted"]
        )

        accuracy = scores["test_f1_weighted"].mean()
        mlflow.log_params(trial.params)
        mlflow.log_metric("Average fold accuracy", accuracy)
        return accuracy


@task(name="Hyperparameter Optimization")
def tune_model(
    X: pd.DataFrame, y: pd.Series, trails: int = 20
) -> optuna.study.Study:
    """
    Perform hyperparameter optimization using Optuna.

    Args:
        X (pd.DataFrame): Dataframe containing the features.
        y (pd.Series)   : Dataframe containing the target.

    Returns:
        optuna.study.Study: The optuna study object.
    """
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trail: objective(trail, X, y), n_trials=trails)
    return study

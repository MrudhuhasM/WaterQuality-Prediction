"""
This module contains functions for loading and splitting data.
"""

from typing import Tuple

import pandas as pd
from prefect import task
from sklearn.model_selection import train_test_split


@task(name="Load Data")
def load_data(path: str) -> pd.DataFrame:
    """
    Load data from a file.

    Args:
        path (str): Path to the file.

    Returns:
        pd.DataFrame: Dataframe containing the data.
    """
    data = pd.read_csv(path)
    data["ammonia"] = pd.to_numeric(data["ammonia"], errors="coerce")
    data["is_safe"] = pd.to_numeric(data["is_safe"], errors="coerce")
    data.dropna(inplace=True)
    return data


@task(name="Split Data")
def split_data(
    data: pd.DataFrame,
    target: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split the data into features and target.

    Args:
        data   (pd.DataFrame): Dataframe containing the data.
        target (str)         : Name of the target column.
        test_size (float)    : Size of the test set.
        random_state (int)   : Random state for reproducibility.
        subset (str)         : Subset of data to return.
                               Options: 'train', 'test', 'all'

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
        Subset of the data based on the input.
    """
    X = data.drop(target, axis=1)
    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test

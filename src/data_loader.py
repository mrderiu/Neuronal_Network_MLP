from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from src import config


def load_data(path: Path = config.DATA_PATH) -> pd.DataFrame:
    """Carga el CSV del dataset Adult Census Income."""
    return pd.read_csv(path)


def split_data(
    df: pd.DataFrame,
) -> Tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series
]:
    """
    Separa el DataFrame en features (X) y target (y), y los divide en
    train (70%) / validation (15%) / test (15%), con stratify sobre y.

    Returns
    -------
    X_train, X_validation, X_test, y_train, y_validation, y_test
    """
    X = df.drop(columns=[config.TARGET_COLUMN])
    y = df[config.TARGET_COLUMN].astype(int)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=config.TEST_SIZE,
        stratify=y,
        random_state=config.RANDOM_STATE,
    )
    X_validation, X_test, y_validation, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=config.VALIDATION_TEST_SPLIT,
        stratify=y_temp,
        random_state=config.RANDOM_STATE,
    )

    return X_train, X_validation, X_test, y_train, y_validation, y_test

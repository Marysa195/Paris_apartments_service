"""
Program: Splitting data into train/test
Version: 1.0
"""

from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split


def split_train_test(dataset: pd.DataFrame, **kwargs):
    """
    Splitting data into train/test and then saving
    :param dataset: dataset
    :return: train/test datasets
    """
    # Split in train/test
    df_train, df_test = train_test_split(
        dataset, test_size=kwargs["test_size"], random_state=kwargs["random_state"],
    )
    df_train.to_csv(kwargs["train_path_proc"], index=False)
    df_test.to_csv(kwargs["test_path_proc"], index=False)
    return df_train, df_test


def get_train_test_data(
    data_train: pd.DataFrame, data_test: pd.DataFrame, target: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Retrieving train/test data broken down into feature objects and target
    variable separately
    :param data_train: train dataset
    :param data_test: test dataset
    :param target: target feature name
    :return: train/test datasets
    """
    x_train, x_test = (
        data_train.drop(target, axis=1),
        data_test.drop(target, axis=1),
    )
    y_train, y_test = (
        data_train.loc[:, target],
        data_test.loc[:, target],
    )
    return x_train, x_test, y_train, y_test


def split_train__val(df_train: pd.DataFrame, **kwargs):
    """
    Splitting data into train/validation and then saving
    :param df_train: train dataset
    :return: train_/val datasets
    """
    # Split in train/validation
    df_train_, df_val = train_test_split(
        df_train,
        train_size=kwargs["validation_size"],
        random_state=kwargs["random_state"],
    )
    df_train_.to_csv(kwargs["train__path_proc"], index=False)
    df_val.to_csv(kwargs["val_path_proc"], index=False)
    return df_train_, df_val


def get_train__val_data(
    data_train_: pd.DataFrame, data_val: pd.DataFrame, target: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Retrieving train_/val data broken down into feature objects and target
    variable separately
    :param data_train_: train_ dataset
    :param data_val: validation dataset
    :param target: target feature name
    :return: train_/val datasets
    """
    x_train_, x_val = (
        data_train_.drop(target, axis=1),
        data_val.drop(target, axis=1),
    )
    y_train_, y_val = (
        data_train_.loc[:, target],
        data_val.loc[:, target],
    )
    return x_train_, x_val, y_train_, y_val

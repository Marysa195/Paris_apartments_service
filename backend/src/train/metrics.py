"""
Program: Getting Metrics
Version: 1.0
"""
import json
import numpy as np
import yaml
import pandas as pd
from numpy import ndarray

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def r2_adjusted(y_true: pd.Series, y_predict: pd.Series, x_test: pd.DataFrame) -> float:
    """
    This function returns the adjustment for the Coefficient of Determination
    that takes into account the number of variables
    in a data set
    param y_true: target feature real value
    param y_predict: target feature predicted value
    param x_test: array with other features values
    return: float
    """
    n_objects = len(y_true)
    n_features = x_test.shape[1]
    r2 = r2_score(y_true, y_predict)
    return 1 - (1 - r2) * (n_objects - 1) / (n_objects - n_features - 1)


def mpe(y_true: pd.Series, y_predict: pd.Series) -> ndarray:
    """
    This function returns the mean percentage error
    param y_true: target feature real value,
    param y_predict: target feature predicted value
    return: float
    """
    return np.mean((y_true - y_predict) / y_true)


def mape(y_true: pd.Series, y_predict: pd.Series) -> ndarray:
    """
    This function returns the mean absolute percentage error
    param y_true: target feature real value,
    param y_predict: target feature predicted value
    return: float
    """
    return np.mean(np.abs((y_predict - y_true) / y_true))


def wape(y_true: pd.Series, y_predict: pd.Series) -> float:
    """
    This function returns The wape metric (the sum of the absolute error
    normalized by the total demand)
    param y_true: target feature real value,
    param y_predict: target feature predicted value
    return: float
    """
    return np.sum(np.abs(y_predict - y_true)) / np.sum(y_true)


def create_dict_metrics(
    y_true: pd.Series, y_predict: pd.Series, x_test: pd.DataFrame
) -> dict:
    """
    This function returns the dictionary with main regression metrics
    param y_true: target feature real value,
    param y_predict: target feature predicted value,
    param x_test: array with other features values,
    param delta: loss function regularization coefficient
    return: dictionary
    """

    dict_metrics = {
        "mae": round(mean_absolute_error(y_true, y_predict), 0),
        "mse": round(mean_squared_error(y_true, y_predict), 0),
        "rmse": round(np.sqrt(mean_squared_error(y_true, y_predict)), 0),
        "r2_adjusted": round(r2_adjusted(y_true, y_predict, x_test), 3),
        "mpe_%": round(mpe(y_true, y_predict) * 100, 0),
        "mape_%": round(mape(y_true, y_predict) * 100, 0),
        "wape_%": round(wape(y_true, y_predict) * 100, 0),
    }

    return dict_metrics


def save_metrics(
    data_x: pd.DataFrame, data_y: pd.Series, model: classmethod, metric_path: str
) -> None:
    """
    Getting and saving metrics
    :param data_x: attribute object
    :param data_y: target variable
    :param model: model
    :param metric_path: path to store metrics
    """
    result_metrics = create_dict_metrics(
        y_true=data_y, y_predict=model.predict(data_x), x_test=data_x
    )
    with open(metric_path, "w") as file:
        json.dump(result_metrics, file)


def load_metrics(config_path: str) -> dict:
    """
    Getting metrics from a file
    :param config_path: path to the configuration file
    :return: metrics
    """
    # get params
    with open(config_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    with open(config["train"]["metrics_path"]) as json_file:
        metrics = json.load(json_file)

    return metrics

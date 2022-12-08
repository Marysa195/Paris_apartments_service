"""
Program: Data training
Version: 1.0
"""

import lightgbm
import numpy as np
import optuna
import pandas as pd
from numpy import ndarray
from optuna import Study
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold

from ..data.split_dataset import get_train__val_data
from ..data.split_dataset import get_train_test_data
from ..train.metrics import save_metrics


def objective(
    trial: str, x: pd.DataFrame, y: pd.Series, n_folds: int, random_state: int
) -> ndarray:
    """
    Define an objective function to be minimized
    param trial: Optuna parametr meaning a single execution of the objective function
    param x: array with features values except target feature
    param y: list with target features values
    param N_FOLDS: number of equal sized subsamples we divide our data
    param RANDOM_STATE: the lot number of the set generated randomly in an operation
    """
    param_grid = {
        "n_estimators": trial.suggest_categorical("n_estimators", [100]),
        "random_state": trial.suggest_categorical("random_state", [random_state]),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "num_leaves": trial.suggest_int("num_leaves", 20, 1000, step=20),
        "max_depth": trial.suggest_int("max_depth", 4, 15, step=1),
        # regularization
        "lambda_l1": trial.suggest_int("lambda_l1", 0, 100),
        "lambda_l2": trial.suggest_int("lambda_l2", 0, 100),
        "min_gain_to_split": trial.suggest_int("min_gain_to_split", 0, 20),
        # proportion of objects during training in the tree
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.2, 1.0),
        "bagging_freq": trial.suggest_categorical("bagging_freq", [1]),
        # proportion of features during training in the tree
        "feature_fraction": trial.suggest_float("feature_fraction", 0.2, 1.0),
    }

    cv = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    cv_predicts = np.empty(n_folds)
    for idx, (train_idx, test_idx) in enumerate(cv.split(x, y)):
        x_train, x_test = x.iloc[train_idx], x.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        pruning_callback = optuna.integration.LightGBMPruningCallback(trial, "l1")

        model = lightgbm.LGBMRegressor(**param_grid)
        model.fit(
            x_train,
            y_train,
            eval_set=[(x_test, y_test)],
            eval_metric="MAE",
            early_stopping_rounds=100,
            callbacks=[pruning_callback],
            verbose=-1,
        )

        preds = model.predict(x_test)
        cv_predicts[idx] = mean_absolute_error(y_test, preds)

    return np.mean(cv_predicts)


def find_optimal_params(
    data_train: pd.DataFrame, data_test: pd.DataFrame, **kwargs
) -> Study:
    """
    Pipeline for model training
    :param data_train: train dataset
    :param data_test: test dataset
    :return:  Study
    """
    x_train, x_test, y_train, y_test = get_train_test_data(
        data_train=data_train, data_test=data_test, target=kwargs["target_column"]
    )

    study = optuna.create_study(direction="minimize", study_name="LGB")
    function = lambda trial: objective(
        trial, x_train, y_train, kwargs["n_folds"], kwargs["random_state"]
    )
    study.optimize(function, n_trials=kwargs["n_trials"], show_progress_bar=True)
    return study


def train_model(
    data_train: pd.DataFrame,
    data_test: pd.DataFrame,
    data_train_: pd.DataFrame,
    data_val: pd.DataFrame,
    study: Study,
    target: str,
    metric_path: str,
) -> lightgbm.LGBMRegressor:
    """
    Model training on the best parameters
    :param data_train: training dataset
    :param data_test: test dataset
    :param study: study optuna
    :param target: target variable name
    :param metric_path: path to metrics folder
    :return: LGBMRegressor
    """
    # get data
    x_train, x_test, y_train, y_test = get_train_test_data(
        data_train=data_train, data_test=data_test, target=target
    )

    x_train_, x_val, y_train_, y_val = get_train__val_data(
        data_train_=data_train_, data_val=data_val, target=target
    )

    # training optimal params
    eval_set = [(x_val, y_val)]
    clf = lightgbm.LGBMRegressor(**study.best_params, silent=True, verbose=-1)
    clf.fit(
        x_train_,
        y_train_,
        eval_metric="mae",
        eval_set=eval_set,
        verbose=False,
        early_stopping_rounds=100,
    )

    # save metrics
    save_metrics(data_x=x_test, data_y=y_test, model=clf, metric_path=metric_path)
    return clf

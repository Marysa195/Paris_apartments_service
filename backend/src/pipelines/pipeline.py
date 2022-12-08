"""
Program: Assembly line for model training
Version: 1.0
"""

import os
import joblib
import yaml

from ..data.split_dataset import split_train_test, split_train__val
from ..train.train import find_optimal_params, train_model
from ..data.get_data import get_dataset
from ..transform.transform import pipeline_preprocess


def pipeline_training(config_path: str) -> None:
    """
    Full cycle of data acquisition, preprocessing and model training
    :param config_path: path to configuration file
    :return: None
    """
    # get params
    with open(config_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    preprocessing_config = config["preprocessing"]
    train_config = config["train"]

    # get data
    train_data = get_dataset(dataset_path=preprocessing_config["train_path"])

    # preprocessing
    train_data = pipeline_preprocess(
        data=train_data, flg_evaluate=False, **preprocessing_config
    )
    # save preprocessed train data for eda graphs (frontend)
    train_data.to_csv(preprocessing_config["train_eda_path"], index=False)

    # split data to train/test
    df_train, df_test = split_train_test(dataset=train_data, **preprocessing_config)

    # split data to train_/validation
    df_train_, df_val = split_train__val(df_train=df_train, **preprocessing_config)

    # find optimal params
    study = find_optimal_params(data_train=df_train, data_test=df_test, **train_config)

    # train with optimal params
    clf = train_model(
        data_train=df_train,
        data_test=df_test,
        data_train_=df_train_,
        data_val=df_val,
        study=study,
        target=preprocessing_config["target_column"],
        metric_path=train_config["metrics_path"],
    )

    # save result (study, model)
    joblib.dump(clf, os.path.join(train_config["model_path"]))
    joblib.dump(study, os.path.join(train_config["study_path"]))

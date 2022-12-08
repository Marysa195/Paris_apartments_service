"""
Program: Getting a Prediction Based on a Trained Model
Version: 1.0
"""

import os
import yaml
import joblib
import pandas as pd
from ..data.get_data import get_dataset
from ..transform.transform import pipeline_preprocess


def pipeline_evaluate(
    config_path, dataset: pd.DataFrame = None, data_path: str = None
) -> list:
    """
    Preprocessing input data and getting predictions
    :param dataset: dataset without target
    :param config_path: path to the configuration file
    :param data_path: path to data file
    :return: list of predictions
    """
    # get params
    with open(config_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    preprocessing_config = config["preprocessing"]
    train_config = config["train"]

    # preprocessing
    if data_path:
        dataset = get_dataset(dataset_path=data_path)

    dataset = pipeline_preprocess(data=dataset, **preprocessing_config)

    model = joblib.load(os.path.join(train_config["model_path"]))
    prediction = model.predict(dataset).tolist()

    return prediction

"""
Program: Getting data from a file
Version: 1.0
"""

from typing import Text
import pandas as pd


def get_dataset(dataset_path: Text) -> pd.DataFrame:
    """
    Retrieving data from a given path
    :param dataset_path: path to data
    :return: dataset
    """
    return pd.read_csv(dataset_path)

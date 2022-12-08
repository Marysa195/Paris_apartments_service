"""
Program: Getting data along the way and reading
Version: 1.0
"""

from io import BytesIO
import io
from typing import Dict, Tuple
import streamlit as st
import pandas as pd


def get_dataset(dataset_path: str) -> pd.DataFrame:
    """
    Getting data from a given path
    :param dataset_path: path to data
    :return: dataset
    """
    return pd.read_csv(dataset_path)


def load_data(
    data: str, type_data: str
) -> Tuple[pd.DataFrame, Dict[str, Tuple[str, BytesIO, str]]]:
    """
    Getting data and converting to BytesIO type for processing in streamlit
    :param data: dataset
    :param type_data: dataset type (train/test)
    :return: dataset, dataset in BytesIO format
    """
    dataset = pd.read_csv(data)
    st.write("Dataset load")
    st.write(dataset.head())

    # Convert dataframe to BytesIO object (for later parsing as a file in FastAPI)
    dataset_bytes_obj = io.BytesIO()
    # write to BytesIO buffer
    dataset.to_csv(dataset_bytes_obj, index=False)
    # Reset pointer to avoid empty data error
    dataset_bytes_obj.seek(0)

    files = {
        "file": (f"{type_data}_dataset.csv", dataset_bytes_obj, "multipart/form-data")
    }
    return dataset, files

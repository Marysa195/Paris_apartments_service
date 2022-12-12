"""
Program: Data preprocessing
Version: 1.0
"""

import json
import warnings
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")


def change_to_numerical(data: pd.DataFrame, numerical_columns: list) -> None:
    """
    This function changes values in given columns to numerical and sets
     invalid parsing as NaN
    :param data: data frame
    :param numerical_columns: list with columns to to be converted
    :return: changed Data Frame
    """
    data[numerical_columns] = data[numerical_columns].apply(
        pd.to_numeric, errors="coerce"
    )


def fillna_with_mode(data: pd.DataFrame, fillna_with_mode_columns: list) -> None:
    """
    This function fills in NaN values in given columns with mode
    :param data: data frame
    :param fillna_with_mode_columns: list with columns to to be converted
    return: changed Data Frame
    """
    for elem in fillna_with_mode_columns:
        data[elem] = data[elem].fillna(data[elem].mode()[0])


def fillna_groupby_category_mode(
    data: pd.DataFrame, column_to_change: list, column_to_groupby: list
) -> None:
    """
    This function fills empty values in a column with the mode received
     grouping data on another column
    :param data: data frame
    :param column_to_change: the list of columns we fill empty values in
    :param column_to_groupby: the list of columns by which we group
    """

    for elem_ch, elem_gb in zip(column_to_change, column_to_groupby):
        data[elem_ch] = data.groupby(elem_gb)[elem_ch].transform(
            lambda x: x.fillna(x.mode()[0])
        )


def transform_types(data: pd.DataFrame, change_type_columns: dict) -> pd.DataFrame:
    """
    Converting features to a given data type
    :param data: data frame
    :param change_type_columns: dictionary with features and data types
    :return: data frame with changed types
    """
    return data.astype(change_type_columns, errors="raise")


def get_bins(
    data: (int, float), first_val: (int, float), second_val: (int, float)
) -> str:
    """
    Bins creation for different features
    :param data: dataset
    :param first_val: first value threshold for binning
    :param second_val: second value threshold for binning
    :return: dataset
    """
    assert isinstance(data, (int, float)), "Problem with data type in feature"
    result = (
        "small"
        if data <= first_val
        else "medium"
        if first_val < data <= second_val
        else "large"
    )
    return result


def save_unique_train_data(
    data: pd.DataFrame, target_column: str, unique_values_path: str,
) -> None:
    """
    Saving a Dictionary with Features and Unique Values
    :param data: dataset
    :param target_column: target feature
    :param unique_values_path: path to file with the dictionary
    :return: None
    """
    unique_df = data.drop(columns=[target_column], axis=1, errors="ignore")
    dict_unique = {key: unique_df[key].unique().tolist() for key in unique_df.columns}
    with open(unique_values_path, "w") as file:
        json.dump(dict_unique, file)


def check_columns_evaluate(data: pd.DataFrame, unique_values_path: str) -> pd.DataFrame:
    """
    Checking for train features fullness and ordering features according to
    train
    :param data: test dataset
    :param unique_values_path: path to the list with train features for
     comparison
    :return: test dataset
    """
    with open(unique_values_path) as json_file:
        unique_values = json.load(json_file)

    column_sequence = unique_values.keys()

    assert set(column_sequence) == set(data.columns), "Different features"
    return data[column_sequence]


def pipeline_preprocess(
    data: pd.DataFrame, flg_evaluate: bool = True, **kwargs
) -> pd.DataFrame:
    """
    Preprocessing pipeline
    :param data: dataset
    :param flg_evaluate: evaluate flag
    :return: dataset
    """
    # drop columns
    data = data.drop(kwargs["drop_columns"], axis=1, errors="ignore")
    data = data.drop(kwargs["empty_columns"], axis=1, errors="ignore")

    # change columns names: switch '.' to '_'
    data = data.rename(columns=lambda col: col.replace(".", "_"))

    # delete unreal values
    data[kwargs["column_with_unreal_values"]].where(
        ~(data[kwargs["column_with_unreal_values"]] == 999), other=np.nan, inplace=True
    )

    # change columns from object to numerical
    if flg_evaluate:
        change_to_numerical(
            data=data, numerical_columns=kwargs["numerical_columns_evaluate"]
        )

    else:
        change_to_numerical(
            data=data, numerical_columns=kwargs["numerical_columns_train"]
        )
        data = data.drop(np.where(data[kwargs["target_column"]].isnull())[0])

    # fill empty values with mode
    fillna_with_mode(
        data=data, fillna_with_mode_columns=kwargs["fillna_with_mode_columns"]
    )

    # fill empty values in a column with the mode
    # received grouping data on another column
    fillna_groupby_category_mode(
        data=data,
        column_to_change=kwargs["column_to_change"],
        column_to_groupby=kwargs["column_to_groupby"],
    )

    # checking the dataset for a match with features from train
    # or saving unique data with features from train
    if flg_evaluate:
        data = check_columns_evaluate(
            data=data, unique_values_path=kwargs["unique_values_path"]
        )
    else:
        save_unique_train_data(
            data=data,
            target_column=kwargs["target_column"],
            unique_values_path=kwargs["unique_values_path"],
        )

    # transform data types
    data = transform_types(data=data, change_type_columns=kwargs["change_type_columns"])

    assert isinstance(
        kwargs["map_bins_columns"], dict
    ), "Submit data for binarization in dict format"
    # bins
    for key in kwargs["map_bins_columns"].keys():
        data[f"{key}_bins"] = data[key].apply(
            lambda x: get_bins(
                x,
                first_val=kwargs["map_bins_columns"][key][0],
                second_val=kwargs["map_bins_columns"][key][1],
            )
        )

    # change category types
    dict_category = {key: "category" for key in data.select_dtypes(["object"]).columns}
    data = transform_types(data=data, change_type_columns=dict_category)

    return data

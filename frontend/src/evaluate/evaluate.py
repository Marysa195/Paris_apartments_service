"""
Program: Drawing sliders and buttons for data entry
with further prediction based on the entered values
Version: 1.0
"""

import json
from io import BytesIO
import pandas as pd
import requests
import streamlit as st


def evaluate_input(unique_data_path: str, endpoint: object) -> None:
    """
    Getting input data by typing in UI -> displaying result
    :param unique_data_path: path to unique values
    :param endpoint: endpoint
    """
    with open(unique_data_path) as file:
        unique_df = json.load(file)

    # data entry fields, use unique values
    city = st.sidebar.selectbox("city", (sorted(unique_df["city"])))
    surface_area = st.sidebar.slider(
        "surfaceArea",
        min_value=min(unique_df["surfaceArea"]),
        max_value=max(unique_df["surfaceArea"]),
    )
    rooms_quantity = st.sidebar.selectbox(
        "roomsQuantity", (sorted(unique_df["roomsQuantity"]))
    )
    bedrooms_quantity = st.sidebar.selectbox(
        "bedroomsQuantity", (sorted(unique_df["bedroomsQuantity"]))
    )
    floor_quantity = st.sidebar.selectbox(
        "floorQuantity", (sorted(unique_df["floorQuantity"]))
    )
    floor = st.sidebar.selectbox("floor", (sorted(unique_df["floor"])))
    is_in_condominium = st.sidebar.selectbox(
        "isInCondominium", (unique_df["isInCondominium"])
    )

    district_libelle = st.sidebar.selectbox(
        "district_libelle", (sorted(unique_df["district_libelle"]))
    )
    dict_data = {
        "city": city,
        "surfaceArea": surface_area,
        "roomsQuantity": rooms_quantity,
        "bedroomsQuantity": bedrooms_quantity,
        "floorQuantity": floor_quantity,
        "floor": floor,
        "isInCondominium": is_in_condominium,
        "district_libelle": district_libelle,
    }

    st.write(
        f"""### Apartment information:\n
    1) City area: {dict_data['city']}
    2) Surface area: {dict_data['surfaceArea']}
    3) Rooms quantity: {dict_data['roomsQuantity']}
    4) Bedrooms quantity: {dict_data['bedroomsQuantity']}
    5) Floor quantity: {dict_data['floorQuantity']}
    6) Floor: {dict_data['floor']}
    7) Is in Condominium: {dict_data['isInCondominium']}
    8) District name: {dict_data['district_libelle']}
    """
    )

    # evaluate and return prediction (text)
    button_ok = st.button("Predict")
    if button_ok:
        result = requests.post(endpoint, timeout=8000, json=dict_data)
        json_str = json.dumps(result.json())
        output = json.loads(json_str)
        st.write(f"{output}")
        st.success("Success!")


def evaluate_from_file(data: pd.DataFrame, endpoint: object, files: BytesIO):
    """
    Getting input data as a file -> outputting the result as a table
    :param data: data set
    :param endpoint: endpoint
    :param files:
    """
    button_ok = st.button("Predict")
    if button_ok:
        # stub because we do not display all predictions
        data_ = data[:5]
        output = requests.post(endpoint, files=files, timeout=8000)
        data_["predict"] = output.json()["prediction"]
        st.write(data_.head())

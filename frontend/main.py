"""
Program: Frontend part of the project
Version: 1.0
"""

import os

import yaml
import streamlit as st
from src.data.get_data import load_data, get_dataset
from src.plotting.charts import boxplot, kdeplotting
from src.train.training import start_training
from src.evaluate.evaluate import evaluate_input, evaluate_from_file

CONFIG_PATH = "../config/params.yml"


def main_page():
    """
    Project description page
    """
    st.image(
        "https://images.squarespace-cdn.com/content/v1/5a8fa2373c3a5393819df189"
        "/be69821f-02e9-4c08-bea6-a43c62a78aa6/belle_vue_rebonjourlafrance_copyright-3."
        "jpg?format=1500w",
        width=350,
    )

    st.markdown("# Project description")
    st.title("MLOps project:  Paris apartments price prediction 🏠 ")
    st.write(
        """
        The goal of this pet-project was to create a web-service which will
        use ML-model that can predict the cost of an apartment in Paris, based
        on the provided information  The data for this project was obtained by
        parsing the site www.bienici.com - a French site for finding real estate.
    """
    )

    # name of the columns
    st.markdown(
        "\n"
        "        ### Features description \n"
        "            City - Paris + area number; \n"
        "            Surface area - surface area of the property in square meters;\n"
        "            Rooms quantity - quantity of rooms;\n"
        "            Bedrooms quantity - quantity of bedrooms;\n"
        "            Floor quantity - quantity of floors in the building;\n"
        "            Floor - property floor;\n"
        "            Is in condominium - if apartment is in condominium or not;\n"
        "            District libelle - name of the district (is smaller then area number.\n"
        "            One area contains few districts);\n"
        "            Price - price in euro (ML-model prediction).\n"
        "        "
    )


def exploratory():
    """
    Exploratory data analysis
    """
    st.markdown("# Exploratory data analysis️")

    with open(CONFIG_PATH) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    if os.path.exists(config["preprocessing"]["train_eda_path"]):
        data = get_dataset(dataset_path=(config["preprocessing"]["train_eda_path"]))
        st.write(data.head())
        st.write(
            """
            Surface area bins:\n
            small - apartments with surface area less then 60 sq meters,\n
            medium - apartments with surface area more then 60 and less then 100 sq meters,\n
            large - apartments with surface area more then 100 sq meters 
            """
        )

        # plotting with checkbox
        price_surface_area_bins = st.sidebar.checkbox(
            "Price - Surface area distribution"
        )
        price_city_surface_area_bins = st.sidebar.checkbox(
            "Price - city area - Surface area"
        )

        if price_surface_area_bins:
            st.pyplot(
                kdeplotting(
                    data=data,
                    data_x="price",
                    hue="surfaceArea_bins",
                    title="Price - Surface area distribution",
                )
            )
        if price_city_surface_area_bins:
            st.pyplot(
                boxplot(
                    data=data,
                    col_x="price",
                    col_y="city",
                    hue="surfaceArea_bins",
                    title="Price - city area - Surface area",
                )
            )

    else:
        st.error("First train the model")


def training():
    """
    Model training
    """
    st.markdown("# Training model LightGBM")
    # get params
    with open(CONFIG_PATH) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    # endpoint
    endpoint = config["endpoints"]["train"]

    if st.button("Start training"):
        start_training(config=config, endpoint=endpoint)


def prediction():
    """
    Getting predictions by data input
    """
    st.markdown("# Prediction")
    with open(CONFIG_PATH) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    endpoint = config["endpoints"]["prediction_input"]
    unique_data_path = config["preprocessing"]["unique_values_path"]

    # checking for a saved model
    if os.path.exists(config["train"]["model_path"]):
        evaluate_input(unique_data_path=unique_data_path, endpoint=endpoint)
    else:
        st.error("First train the model")


def prediction_from_file():
    """
    Getting predictions from a data file
    """
    st.markdown("# Prediction")
    with open(CONFIG_PATH) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    endpoint = config["endpoints"]["prediction_from_file"]

    upload_file = st.file_uploader(
        "", type=["csv", "xlsx"], accept_multiple_files=False
    )
    # check if file is loaded
    if upload_file:
        dataset_csv_df, files = load_data(data=upload_file, type_data="Test")
        # checking for a saved model
        if os.path.exists(config["train"]["model_path"]):
            evaluate_from_file(data=dataset_csv_df, endpoint=endpoint, files=files)
        else:
            st.error("First train the model")


def main():
    """
    Building a pipeline in one block
    """
    page_names_to_funcs = {
        "Project description": main_page,
        "Exploratory data analysis": exploratory,
        "Training model": training,
        "Prediction": prediction,
        "Prediction from file": prediction_from_file,
    }
    selected_page = st.sidebar.selectbox("Select an item", page_names_to_funcs.keys())
    page_names_to_funcs[selected_page]()


if __name__ == "__main__":
    main()

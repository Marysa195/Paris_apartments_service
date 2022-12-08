"""
Program: A model for predicting prices to buy apartments in Paris
Version: 1.0
"""

import warnings
import optuna
import pandas as pd
import uvicorn
from fastapi import FastAPI
from fastapi import File
from fastapi import UploadFile
from pydantic import BaseModel

from src.evaluate.evaluate import pipeline_evaluate
from src.pipelines.pipeline import pipeline_training
from src.train.metrics import load_metrics

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

app = FastAPI()
CONFIG_PATH = "../config/params.yml"


class ApartmentsFeatures(BaseModel):
    """
    Features for getting prediction
    """

    city: object
    surfaceArea: float
    roomsQuantity: float
    bedroomsQuantity: float
    floorQuantity: float
    floor: float
    isInCondominium: bool
    district_libelle: object


@app.get("/hello")
def welcome():
    """
    Hello
    :return: None
    """
    return {"message": "Hello Dear Customer!"}


@app.post("/train")
def training():
    """
    Model training, metrics logging
    """
    pipeline_training(config_path=CONFIG_PATH)
    metrics = load_metrics(config_path=CONFIG_PATH)

    return {"metrics": metrics}


@app.post("/predict")
def prediction(file: UploadFile = File(...)):
    """
    Model prediction from data file
    param file: file with dataset without target we upload to make predictions
    """
    result = pipeline_evaluate(config_path=CONFIG_PATH, data_path=file.file.readable())
    assert isinstance(result, list), "Result does not match type list"
    # stub because we do not display all the predictions, otherwise it will hang
    return {"prediction": result[:5]}


@app.post("/predict_input")
def prediction_input(item: ApartmentsFeatures):
    """
    Model prediction from input data
    """
    features = [
        [
            item.city,
            item.surfaceArea,
            item.roomsQuantity,
            item.bedroomsQuantity,
            item.floorQuantity,
            item.floor,
            item.isInCondominium,
            item.district_libelle,
        ]
    ]

    cols = [
        "city",
        "surfaceArea",
        "roomsQuantity",
        "bedroomsQuantity",
        "floorQuantity",
        "floor",
        "isInCondominium",
        "district_libelle",
    ]

    data = pd.DataFrame(features, columns=cols)
    predictions = pipeline_evaluate(config_path=CONFIG_PATH, dataset=data)[0]
    return round(predictions, 0)


if __name__ == "__main__":
    # Start the server using the given host and port
    uvicorn.run(app, host="127.0.0.1", port=80)

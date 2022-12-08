"""
Program: Training the model on the backend, displaying metrics and
on-screen learning curves
Version: 1.0
"""

import os
import json
import joblib
import requests
import streamlit as st
from optuna.visualization import plot_param_importances, plot_optimization_history


def start_training(config: dict, endpoint: object) -> None:
    """
    Model training with output
    :param config: configuration file
    :param endpoint: endpoint
    """
    # Last metrics
    if os.path.exists(config["train"]["metrics_path"]):
        with open(config["train"]["metrics_path"]) as json_file:
            old_metrics = json.load(json_file)
    else:
        # if the model has not been trained before and there are no past metric values
        old_metrics = {
            "mae": 0,
            "mse": 0,
            "rmse": 0,
            "r2_adjusted": 0,
            "mpe_%": 0,
            "mape_%": 0,
            "wape_%": 0,
        }

    # Train
    with st.spinner("The model selects parameters..."):
        output = requests.post(endpoint, timeout=8000)
    st.success("Success!")

    new_metrics = output.json()['metrics']

    # diff metrics
    mae, mse, rmse, r2_adjusted, mpe, mape, wape = st.columns(7)
    mae.metric(
        "mae", new_metrics["mae"], f"{new_metrics['mae'] - old_metrics['mae']:.3f}"
    )
    mse.metric(
        "mse", new_metrics["mse"], f"{new_metrics['mse'] - old_metrics['mse']:.3f}"
    )
    rmse.metric(
        "rmse", new_metrics["rmse"], f"{new_metrics['rmse'] - old_metrics['rmse']:.3f}"
    )
    r2_adjusted.metric(
        "r2_adjusted",
        new_metrics["r2_adjusted"],
        f"{new_metrics['r2_adjusted'] - old_metrics['r2_adjusted']:.3f}",
    )
    mpe.metric(
        "mpe_%",
        new_metrics["mpe_%"],
        f"{new_metrics['mpe_%'] - old_metrics['mpe_%']:.3f}",
    )
    mape.metric(
        "mape_%",
        new_metrics["mape_%"],
        f"{new_metrics['mape_%'] - old_metrics['mape_%']:.3f}",
    )
    wape.metric(
        "wape_%",
        new_metrics["wape_%"],
        f"{new_metrics['wape_%'] - old_metrics['wape_%']:.3f}",
    )

    # plot study
    study = joblib.load(os.path.join(config["train"]["study_path"]))
    fig_imp = plot_param_importances(study)
    fig_history = plot_optimization_history(study)

    st.plotly_chart(fig_imp, use_container_width=True)
    st.plotly_chart(fig_history, use_container_width=True)

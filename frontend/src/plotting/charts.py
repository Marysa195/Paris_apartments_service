"""
Program: Drawing charts
Version: 1.0
"""

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns


def boxplot(
    data: pd.DataFrame, col_x: str, col_y: str, hue: str, title: str
) -> matplotlib.figure.Figure:
    """
    Boxplot drawing
    :param data: dataset
    :param col_x: feature which distribution we want to see
    :param col_y: feature with which we set individual boxplots for our target
    :param hue: feature with which we set individual boxplots for our
    target divided by col_y
    :param title: plot title
    :return: plot figure
    """

    sns.set_style("whitegrid")
    fig = plt.figure(figsize=(15, 30))

    sns.boxplot(data=data, y=col_y, x=col_x, hue=hue, palette="rocket", orient='horizontal')

    plt.title(title, fontsize=20)
    plt.ylabel(col_y, fontsize=14)
    plt.xlabel(col_x, fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    return fig


def kdeplotting(
    data: pd.DataFrame, data_x: str, hue: str, title: str
) -> matplotlib.figure.Figure:
    """
    Kdeplot drawing
    :param data: dataset
    :param data_x: axis OX
    :param hue: grouping by feature
    :param title: plot title
    :return: plot figure
    """
    sns.set_style("whitegrid")

    fig = plt.figure(figsize=(15, 7))

    sns.kdeplot(
        data=data, x=data_x, hue=hue, palette="rocket", common_norm=False, fill=True
    )
    plt.title(title, fontsize=20)
    plt.ylabel("Percentage", fontsize=14)
    plt.xlabel(data_x, fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    return fig

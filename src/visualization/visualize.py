"""Visualization."""

from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import auc, confusion_matrix, roc_curve

matplotlib.use("agg")


def plot_data(
    x_data: Any,
    y_data: Any,
    x_label: str,
    y_label: str,
    title: str,
    path: str,
) -> None:
    """Plot data using to matplotlib.pylot.plot.

    Args:
        x_data (Any): X-axis data
        y_data (Any): Y-axis data
        x_label (str): X label
        y_label (str): Y label
        title (str): Title
        path (str): Path to save the plot

    Raises:
        Exception
    """
    try:
        fig = plt.figure()
        plt.plot(x_data, y_data)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        if path is not None:
            fig.savefig(path)

    except Exception as exception:
        raise Exception from exception


def plot_roc_auc_curve(
    y_test: Any, y_test_prob: Any, title: str, path: str
) -> None:
    """Plot roc auc curve.

    Args:
        y_test (Any): actual labels
        y_test_prob (Any): probabilities
        title (str): Title
        path (str): Path to save the plot

    Raises:
        Exception
    """
    try:
        fig = plt.figure()
        fpr, tpr, thresholds = roc_curve(y_test, y_test_prob)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        auc_ = auc(fpr, tpr)
        plt.plot(
            fpr,
            tpr,
            label=f"auc:{auc_: .2f}, optimum_thr:{optimal_threshold: .2f}",
        )
        plt.xlabel("False positive rate")
        plt.ylabel("True positive rate")
        plt.title(title)
        plt.legend(loc="best")
        if path is not None:
            fig.savefig(path)

    except Exception as exception:
        raise Exception from exception


def plot_confusion_matrix(
    y_true: Any, y_pred: Any, title: str, path: str, labels: list
) -> None:
    """Plot confusion matrix.

    Args:
        y_true (Any): actual labels
        y_pred (Any): predicted labels
        title (str): Title
        path (str): Path to save the plot
        labels (list): Labels

    Raises:
        Exception
    """
    try:
        fig = plt.figure()
        confusion_mat = confusion_matrix(
            y_true=y_true, y_pred=y_pred, labels=labels
        )

        # classes
        classes = [
            "True Negative",
            "False Positive",
            "False Negative",
            "True Positive",
        ]

        # values
        values = [f"{x:.0f}" for x in confusion_mat.flatten()]

        # Find percentages and set format
        percentages = [
            f"{x:.2%}" for x in confusion_mat.flatten() / np.sum(confusion_mat)
        ]

        # Combine classes, values and percentages
        data = [
            f"{i}\n{j}\n{k}" for i, j, k in zip(classes, values, percentages)
        ]
        data = np.asarray(data).reshape(2, 2)

        axes = sns.heatmap(confusion_mat, annot=data, fmt="", cmap="YlGnBu")

        # title and x, y axis of plot
        axes.set_title(title)
        axes.set_xlabel("Predicted Values")
        axes.set_ylabel("Actual Values")
        if path is not None:
            fig.savefig(path)

    except Exception as exception:
        raise Exception from exception

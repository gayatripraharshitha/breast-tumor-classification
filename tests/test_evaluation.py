import numpy as np

from src.evaluation import (
    calculate_specificity,
    evaluate_predictions,
)


def test_calculate_specificity():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])

    specificity = calculate_specificity(
        y_true,
        y_pred
    )

    assert specificity == 0.5


def test_evaluate_predictions():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.2, 0.8, 0.9])

    results = evaluate_predictions(
        y_true,
        y_pred,
        probabilities
    )

    assert results["Accuracy"] == 1.0
    assert results["Sensitivity"] == 1.0
    assert results["Specificity"] == 1.0
    assert results["ROC-AUC"] == 1.0
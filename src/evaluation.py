"""Evaluation utilities for breast tumor classification models."""

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def calculate_specificity(y_true, y_pred):
    """Calculate specificity (true negative rate)."""
    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    ).ravel()

    return tn / (tn + fp)


def evaluate_predictions(y_true, y_pred, probabilities):
    """Calculate the main classification metrics."""
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Sensitivity": recall_score(y_true, y_pred, zero_division=0),
        "Specificity": calculate_specificity(y_true, y_pred),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_true, probabilities),
        "PR-AUC": average_precision_score(y_true, probabilities),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


def evaluate_model(model, X_test, y_test):
    """Generate predictions and evaluate a fitted classification model."""
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.50).astype(int)

    metrics = evaluate_predictions(
        y_test,
        predictions,
        probabilities
    )

    return metrics, predictions, probabilities


def brier_score(y_true, probabilities):
    """Calculate the Brier score for binary predictions."""
    return np.mean((probabilities - np.asarray(y_true)) ** 2)


def threshold_analysis(y_true, probabilities, thresholds=None):
    """Evaluate classification behavior across probability thresholds."""
    if thresholds is None:
        thresholds = np.arange(0.10, 0.91, 0.05)

    results = []

    for threshold in thresholds:
        predictions = (probabilities >= threshold).astype(int)

        results.append({
            "Threshold": threshold,
            "Accuracy": accuracy_score(y_true, predictions),
            "Sensitivity": recall_score(
                y_true,
                predictions,
                zero_division=0
            ),
            "Specificity": calculate_specificity(
                y_true,
                predictions
            ),
            "Precision": precision_score(
                y_true,
                predictions,
                zero_division=0
            ),
            "F1": f1_score(
                y_true,
                predictions,
                zero_division=0
            )
        })

    return pd.DataFrame(results)
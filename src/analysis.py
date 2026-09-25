"""Analysis utilities for the breast tumor classification study."""

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.inspection import permutation_importance


def calculate_permutation_importance(
    model,
    X_train,
    y_train,
    feature_names,
    cv,
    scoring="roc_auc",
    n_repeats=20,
    random_state=42,
):
    """
    Calculate permutation importance on validation folds.

    The model is fitted independently on each training fold and evaluated
    on the corresponding validation fold.
    """
    fold_importances = []

    for fold, (train_idx, validation_idx) in enumerate(
        cv.split(X_train, y_train),
        start=1
    ):
        X_fold_train = X_train.iloc[train_idx]
        X_fold_validation = X_train.iloc[validation_idx]

        y_fold_train = y_train.iloc[train_idx]
        y_fold_validation = y_train.iloc[validation_idx]

        fold_model = clone(model)

        fold_model.fit(
            X_fold_train,
            y_fold_train
        )

        permutation = permutation_importance(
            fold_model,
            X_fold_validation,
            y_fold_validation,
            scoring=scoring,
            n_repeats=n_repeats,
            random_state=random_state,
        )

        fold_importances.append(
            pd.Series(
                permutation.importances_mean,
                index=feature_names,
                name=f"Fold_{fold}",
            )
        )

    return pd.concat(fold_importances, axis=1)


def summarize_feature_stability(feature_importances):
    """Summarize mean, variation, and fold consistency of feature importance."""

    summary = pd.DataFrame({
        "Mean_Importance": feature_importances.mean(axis=1),
        "Std_Importance": feature_importances.std(axis=1),
        "Positive_Folds": (feature_importances > 0).sum(axis=1),
    })

    return summary.sort_values(
        "Mean_Importance",
        ascending=False
    )


def calculate_prediction_disagreement(
    probability_data,
    y_true,
    threshold=0.50,
):
    """
    Measure disagreement between multiple model predictions.

    probability_data should contain one malignant-class probability
    column per model.
    """
    prediction_matrix = probability_data >= threshold

    malignant_votes = prediction_matrix.sum(axis=1)
    total_models = prediction_matrix.shape[1]
    benign_votes = total_models - malignant_votes

    disagreement = (
        1
        - abs(malignant_votes - benign_votes) / total_models
    )

    majority_prediction = (
        malignant_votes >= (total_models / 2)
    ).astype(int)

    result = probability_data.copy()

    result["Malignant_Votes"] = malignant_votes
    result["Benign_Votes"] = benign_votes
    result["Prediction_Disagreement"] = disagreement
    result["Majority_Prediction_Correct"] = (
        majority_prediction == np.asarray(y_true)
    )
    result["True_Class"] = np.asarray(y_true)

    return result.sort_values(
        "Prediction_Disagreement",
        ascending=False
    )


def calculate_abstention_analysis(
    y_true,
    probabilities,
    thresholds=None,
):
    """Calculate coverage and accepted-case accuracy at confidence thresholds."""

    if thresholds is None:
        thresholds = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95]

    predictions = (probabilities >= 0.50).astype(int)

    confidence = np.maximum(
        probabilities,
        1 - probabilities
    )

    results = []

    for threshold in thresholds:
        accepted = confidence >= threshold
        coverage = accepted.mean()

        if accepted.sum() > 0:
            accepted_accuracy = np.mean(
                predictions[accepted]
                == np.asarray(y_true)[accepted]
            )
        else:
            accepted_accuracy = np.nan

        results.append({
            "Confidence_Threshold": threshold,
            "Coverage": coverage,
            "Abstention_Rate": 1 - coverage,
            "Accepted_Accuracy": accepted_accuracy,
            "Accepted_Cases": accepted.sum(),
        })

    return pd.DataFrame(results)
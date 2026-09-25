"""Preprocessing utilities for the breast tumor classification project."""

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42


def encode_target(target):
    """
    Convert the original WBC target labels to binary labels.

    Original UCI labels:
        2 -> benign (0)
        4 -> malignant (1)
    """
    return target.map({2: 0, 4: 1})


def split_data(X, y, test_size=0.20):
    """Create a stratified train-test split."""
    return train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=RANDOM_STATE
    )


def create_scaled_preprocessor():
    """Create the preprocessing pipeline used by scaled models."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])


def create_tree_preprocessor():
    """Create the preprocessing pipeline used by tree-based models."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])
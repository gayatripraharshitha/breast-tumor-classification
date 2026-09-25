from src.models import create_models


def test_create_models():
    models = create_models()

    expected_models = {
        "Logistic Regression",
        "SVM",
        "Decision Tree",
        "Random Forest",
    }

    assert set(models.keys()) == expected_models


def test_models_are_pipelines():
    models = create_models()

    for model in models.values():
        assert hasattr(model, "fit")
        assert hasattr(model, "predict")
        assert hasattr(model, "predict_proba")
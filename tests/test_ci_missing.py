import pandas as pd
import pytest

from src.data.preprocess import preprocess
from src.model import load_model, train_and_evaluate


@pytest.fixture(scope="module")
def trained_result():
    return train_and_evaluate()


def test_preprocess_creates_binary_target_and_no_missing_values():
    df = preprocess()

    assert set(df["target"].unique()).issubset({0, 1})
    assert df.isnull().sum().sum() == 0
    assert df.shape[0] > 0
    assert df.shape[1] > 15


def test_train_and_evaluate_returns_model_artifacts(trained_result):
    assert trained_result["best_model"] in {"logistic_regression", "random_forest_tuned"}
    assert trained_result["saved_pipeline_path"].endswith(".joblib")
    assert trained_result["split"]["train_shape"][0] > 0
    assert trained_result["split"]["test_shape"][0] > 0


def test_saved_model_can_predict_single_record(trained_result):
    model = load_model(trained_result["saved_pipeline_path"])
    sample = pd.DataFrame([
        {
            "age": 63,
            "sex": 1,
            "cp": 3,
            "trestbps": 145,
            "chol": 233,
            "fbs": 1,
            "restecg": 0,
            "thalach": 150,
            "exang": 0,
            "oldpeak": 2.3,
            "slope": 0,
            "ca": 0,
            "thal": 1,
        }
    ])

    preds = model.predict(sample)

    assert preds.shape[0] == 1
    assert set(preds).issubset({0, 1})

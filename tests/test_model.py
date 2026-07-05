import pandas as pd
from src.data.preprocess import preprocess
from src.model import load_model, train_and_evaluate


def test_preprocess():
    df = preprocess()
    assert 'target' in df.columns
    assert df.dropna().shape[0] == df.shape[0]


def test_train_and_save_load():
    result = train_and_evaluate()
    assert result['saved_pipeline_path'].endswith('.joblib')

    loaded = load_model(result['saved_pipeline_path'])
    sample = pd.DataFrame([
        {
            'age': 63,
            'sex': 1,
            'cp': 3,
            'trestbps': 145,
            'chol': 233,
            'fbs': 1,
            'restecg': 0,
            'thalach': 150,
            'exang': 0,
            'oldpeak': 2.3,
            'slope': 0,
            'ca': 0,
            'thal': 1,
        }
    ])

    predictions = loaded.predict(sample)
    assert predictions.shape[0] == 1
    assert set(predictions).issubset({0, 1})

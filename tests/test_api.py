from fastapi.testclient import TestClient

from src.api.main import app


def test_predict_returns_prediction_and_confidence():
    with TestClient(app) as client:
        payload = {
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

        response = client.post('/predict', json=payload)

        assert response.status_code == 200
        body = response.json()
        assert 'prediction' in body
        assert body['prediction'] in {0, 1}
        assert 'confidence' in body

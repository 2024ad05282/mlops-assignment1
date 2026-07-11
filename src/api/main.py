import logging
import time
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Histogram,
    generate_latest,
    start_http_server,
)

from ..model import COLUMN_NAMES, load_model


FEATURE_COLUMNS = [column for column in COLUMN_NAMES if column != 'target']


def _build_feature_vector(payload: dict) -> list[float]:
    return [
        payload['age'],
        payload['sex'],
        payload['cp'],
        payload['trestbps'],
        payload['chol'],
        payload['fbs'],
        payload['restecg'],
        payload['thalach'],
        payload['exang'],
        payload['oldpeak'],
        payload['slope'],
        payload['ca'],
        payload['thal'],
    ]

app = FastAPI(title='Heart Disease Prediction API')

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger('heart_api')

REQUEST_COUNT = Counter(
    'api_request_count', 'Total API requests', ['method', 'endpoint', 'http_status']
)
REQUEST_LATENCY = Histogram(
    'api_request_latency_seconds', 'API request latency seconds', ['endpoint']
)

MODEL = None


@app.on_event('startup')
async def startup_event():
    global MODEL
    MODEL = load_model()
    start_http_server(8001)
    logger.info('Model loaded and Prometheus exporter started on port 8001')


@app.middleware('http')
async def log_requests(request: Request, call_next):
    start_time = time.perf_counter()
    logger.info('Incoming request: %s %s', request.method, request.url.path)

    try:
        response = await call_next(request)
    except Exception:
        REQUEST_COUNT.labels(request.method, request.url.path, '500').inc()
        REQUEST_LATENCY.labels(endpoint=request.url.path).observe(time.perf_counter() - start_time)
        logger.exception('Request failed: %s %s', request.method, request.url.path)
        raise

    latency = time.perf_counter() - start_time
    REQUEST_COUNT.labels(request.method, request.url.path, str(response.status_code)).inc()
    REQUEST_LATENCY.labels(endpoint=request.url.path).observe(latency)
    logger.info('Completed %s %s -> %s in %.4fs', request.method, request.url.path, response.status_code, latency)
    return response


@app.get('/metrics')
async def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get('/health')
async def health():
    return {'status': 'ok'}


@app.post('/predict')
async def predict(payload: dict):
    try:
        features = _build_feature_vector(payload)
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=f'Missing field: {exc.args[0]}')

    if MODEL is None:
        raise HTTPException(status_code=503, detail='Model is not ready yet.')

    feature_frame = pd.DataFrame([features], columns=FEATURE_COLUMNS)
    prediction = MODEL.predict(feature_frame)[0]
    confidence = 0.0

    if hasattr(MODEL, 'predict_proba'):
        probabilities = MODEL.predict_proba(feature_frame)[0]
        confidence = round(float(max(probabilities)), 4)

    logger.info('Prediction completed: %s (confidence=%.4f)', int(prediction), confidence)

    return {
        'prediction': int(prediction),
        'confidence': confidence,
    }

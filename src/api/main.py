import logging
from fastapi import FastAPI, HTTPException, Request, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Histogram,
    generate_latest,
    start_http_server,
)

from ..model import load_model

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
    logger.info(
        'Model loaded and Prometheus exporter started on port 8001'
    )


@app.middleware('http')
async def log_requests(request: Request, call_next):
    logger.info(f'Request: {request.method} {request.url.path}')
    with REQUEST_LATENCY.labels(endpoint=request.url.path).time():
        response = await call_next(request)
    REQUEST_COUNT.labels(
        request.method, request.url.path, str(response.status_code)
    ).inc()
    logger.info(f'Response status: {response.status_code}')
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
        features = [
            payload['age'], payload['sex'], payload['cp'], payload['trestbps'],
            payload['chol'], payload['fbs'], payload['restecg'],
            payload['thalach'], payload['exang'], payload['oldpeak'],
            payload['slope'], payload['ca'], payload['thal'],
        ]
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=f'Missing field: {exc.args[0]}')

    prediction = MODEL.predict([features])[0]
    return {'prediction': int(prediction)}

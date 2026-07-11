# Heart Disease MLOps Project

## Overview
This repository contains a Heart Disease prediction pipeline with data ingestion, EDA, modelling, API deployment, monitoring, and CI/CD.

## Structure
- `data/` - raw and processed datasets
- `notebooks/` - EDA and model training notebooks
- `src/` - source code for API, model, and utilities
- `tests/` - unit tests
- `deployment/` - Docker, Helm, and manifests
- `.github/workflows/` - CI/CD YAML
- `report/` - final report and screenshots

## Setup
1. Create a conda env or use pip:
   - `python -m venv venv`
   - `venv\Scripts\activate`
   - `pip install -r requirements.txt`

2. Run preprocessing:
   - `python src/data/download_data.py`
   - `python src/data/preprocess.py`

3. Train model:
   - `python src/train.py`

4. Start API:
   - `uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000`

5. Monitoring and logging:
   - API logs are emitted for each request and response.
   - Prometheus metrics are exposed at `http://localhost:8001/metrics`.
   - Prometheus config is available at `deployment/monitoring/prometheus.yml`.
   - A sample Grafana dashboard JSON is available at `deployment/monitoring/grafana-dashboard.json`.

## Notes
- Add dataset and report files in the appropriate folders.
- Use `docs/` or `report/` for the final PDF or Markdown report.
- `report/reproducibility.md` documents how to reproduce training and inference.
- Saved model artifacts are written into `models/`, including `best_pipeline.joblib`, `best_pipeline.pkl`, and `metadata.json`.

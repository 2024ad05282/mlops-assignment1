$env:MLFLOW_ALLOW_FILE_STORE = "true"
python -m mlflow ui --backend-store-uri "sqlite:///mlflow.db" --default-artifact-root "file:./mlruns" --host 127.0.0.1 --port 5001

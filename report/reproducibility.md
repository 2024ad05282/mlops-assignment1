# Reproducibility Guide

## Project Structure

- `data/` - raw dataset files and preprocessing helpers
- `src/` - core application code
  - `src/model.py` - training, evaluation, model export, and inference utilities
  - `src/train.py` - training entry point
  - `src/api/main.py` - FastAPI inference service
- `models/` - saved model artifacts and metadata
- `reports/` - generated evaluation reports and figures
- `report/` - project documentation
- `requirements.txt` - required Python dependencies

## Training Workflow

1. Load raw data from `data/raw/heart_disease.csv` using `load_raw_data()`.
2. Convert the target column to binary using `convert_target(...)`.
3. Split the dataset into stratified training and test sets with `split_data(...)`.
4. Automatically infer numeric and categorical feature types with `infer_feature_types(...)`.
5. Build a scikit-learn pipeline using `build_pipeline(...)`, which includes:
   - a `ColumnTransformer` preprocessing step
   - a classifier (`LogisticRegression` or `RandomForestClassifier`)
6. Evaluate both models using 5-fold stratified cross-validation.
7. Tune the random forest model with `GridSearchCV` optimizing ROC-AUC.
8. Fit the final selected pipeline and evaluate on the hold-out test set.
9. Save the selected pipeline and export model artifacts.

## Preprocessing Pipeline

Preprocessing is handled by a scikit-learn `ColumnTransformer` that contains:

- Numeric pipeline:
  - `SimpleImputer(strategy='median')`
  - `StandardScaler()`
- Categorical pipeline:
  - `SimpleImputer(strategy='most_frequent')`
  - `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`

This preprocessing is fitted only on training data, and the complete pipeline is then used end-to-end for evaluation and inference.

## Inference Workflow

The saved model is a full pipeline containing both preprocessing and the trained classifier. That means:

- No manual preprocessing is required before prediction.
- Raw input can be passed directly into the loaded pipeline.

### Load the saved pipeline

Use `src/model.py`:

```python
from src.model import load_pipeline
pipeline = load_pipeline()
```

This loads `models/best_pipeline.joblib` by default.

### Make predictions

Use the `predict(...)` utility in `src/model.py`:

```python
from src.model import predict
sample = {
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
    'thal': 1
}
prediction = predict(sample)
print(prediction)
```

If you have multiple samples, pass a list of dictionaries.

## Model Export and Packaging

The training workflow saves the selected pipeline in multiple reusable formats:

- `models/best_pipeline.joblib`
- `models/best_pipeline.pkl`
- `models/mlflow_model/` (local MLflow model export)
- `models/metadata.json`

If `skl2onnx` is installed, an optional ONNX export is also generated at `models/best_pipeline.onnx`.

## Metadata File

`models/metadata.json` contains:

- `model_name`
- `algorithm`
- `training_date`
- `sklearn_version`
- `python_version`
- `best_hyperparameters`
- `evaluation_metrics`
- `feature_names`
- `target_column`
- saved file locations

## Reproducing Training

To reproduce training, run:

```bash
pip install -r requirements.txt
python src/train.py
```

This will:

- train models
- tune the random forest
- evaluate the final model
- save the best pipeline in the `models/` directory
- generate reports in `reports/`
- save metadata in `models/metadata.json`

## Notes

- The project is designed so the saved pipeline is inference-ready.
- Any raw sample with the expected input fields can be passed directly to `predict(...)`.
- The API at `src/api/main.py` also loads the saved model and serves predictions without separate preprocessing.

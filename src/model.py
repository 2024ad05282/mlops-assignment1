import datetime
import json
import mlflow
import mlflow.sklearn
import os
import pickle
import platform
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import __version__ as sklearn_version
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    precision_recall_curve,
    roc_curve,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    ONNX_ENABLED = True
except ImportError:
    ONNX_ENABLED = False

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RAW_FILE = os.path.join(ROOT_DIR, 'data', 'raw', 'heart_disease.csv')
MODEL_DIR = os.path.join(ROOT_DIR, 'models')
MODEL_FILE_JOBLIB = 'best_pipeline.joblib'
MODEL_FILE_PICKLE = 'best_pipeline.pkl'
METADATA_FILE = 'metadata.json'
ONNX_FILE = 'best_pipeline.onnx'
MLFLOW_MODEL_DIR = os.path.join(MODEL_DIR, 'mlflow_model')
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILE_JOBLIB)
PICKLE_MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILE_PICKLE)
METADATA_PATH = os.path.join(MODEL_DIR, METADATA_FILE)
ONNX_PATH = os.path.join(MODEL_DIR, ONNX_FILE)

os.makedirs(MODEL_DIR, exist_ok=True)

COLUMN_NAMES = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
    'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
]


def load_raw_data():
    return pd.read_csv(RAW_FILE, header=None, names=COLUMN_NAMES, na_values='?')


def convert_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['target'] = (df['target'] > 0).astype(int)
    return df


def infer_feature_types(
    X: pd.DataFrame,
    unique_threshold: int = 10
) -> tuple[list[str], list[str]]:
    """Infer numeric and categorical features automatically.

    The heuristic treats object and categorical dtypes as categorical by default.
    Numeric columns are treated as categorical when they have low cardinality,
    because many ordinal or encoded features are stored as numbers but behave
    like categories.

    Limitations:
    - A numeric feature with few unique values may be categorical, but it could
      also be a truly ordinal numeric variable.
    - A high-cardinality categorical feature will remain numeric if its dtype is
      numeric.
    - The heuristic relies on the training set only, so it stays consistent with
      production preprocessing.
    """
    object_columns = X.select_dtypes(
        include=['object', 'category', 'bool']
    ).columns.tolist()
    numeric_columns = X.select_dtypes(include=[np.number]).columns.tolist()

    categorical_from_numeric = []
    for col in numeric_columns:
        unique_values = X[col].nunique(dropna=False)
        if unique_values <= unique_threshold:
            categorical_from_numeric.append(col)

    categorical_features = sorted(
        set(object_columns + categorical_from_numeric)
    )
    numeric_features = [
        col for col in numeric_columns if col not in categorical_features
    ]

    return numeric_features, categorical_features


def build_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str]
) -> ColumnTransformer:
    """Create a preprocessing pipeline for numeric and categorical features."""
    numeric_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    transformers = []
    if numeric_features:
        transformers.append(('numeric', numeric_pipeline, numeric_features))
    if categorical_features:
        transformers.append(('categorical', categorical_pipeline, categorical_features))

    return ColumnTransformer(transformers, remainder='drop')


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split raw data into stratified training and test sets before preprocessing."""
    df = convert_target(df)
    X = df.drop(columns=['target'])
    y = df['target']
    return train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )


def build_pipeline(
    model,
    numeric_features: list[str],
    categorical_features: list[str]
) -> Pipeline:
    """Assemble the preprocessing and model steps into a single pipeline."""
    return Pipeline([
        ('preprocessor', build_preprocessor(numeric_features, categorical_features)),
        ('model', model)
    ])


def _get_preprocessed_feature_names(
    preprocessor: ColumnTransformer,
    numeric_features: list[str],
    categorical_features: list[str]
) -> list[str]:
    """Return feature names after preprocessing for plotting and interpretation."""
    feature_names: list[str] = []
    if numeric_features:
        feature_names.extend(numeric_features)

    if categorical_features:
        categorical_transformer = preprocessor.named_transformers_['categorical']
        onehot = categorical_transformer.named_steps['onehot']
        feature_names.extend(onehot.get_feature_names_out(categorical_features).tolist())

    return feature_names


def _ensure_directory(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def cross_validate_model(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    cv,
) -> dict[str, tuple[float, float]]:
    """Evaluate a pipeline using stratified cross-validation."""
    scoring = {
        'accuracy': 'accuracy',
        'precision': 'precision',
        'recall': 'recall',
        'f1': 'f1',
        'roc_auc': 'roc_auc'
    }
    results = cross_validate(
        pipeline,
        X,
        y,
        cv=cv,
        scoring=scoring,
        return_train_score=False,
        n_jobs=-1
    )

    return {
        key: (results[f'test_{key}'].mean(), results[f'test_{key}'].std())
        for key in scoring
    }


def evaluate_on_test(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, object]:
    """Evaluate a trained pipeline on the test set and return metrics."""
    y_pred = pipeline.predict(X_test)
    if hasattr(pipeline, 'predict_proba'):
        y_scores = pipeline.predict_proba(X_test)[:, 1]
    elif hasattr(pipeline, 'decision_function'):
        y_scores = pipeline.decision_function(X_test)
    else:
        y_scores = None

    evaluation = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': (
            roc_auc_score(y_test, y_scores) if y_scores is not None else None
        ),
        'classification_report': classification_report(
            y_test, y_pred, output_dict=True
        ),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
        'y_test': y_test,
        'y_scores': y_scores
    }
    return evaluation


def plot_confusion_matrix(
    cm: list[list[int]],
    model_name: str,
    output_dir: str
) -> str:
    """Save a confusion matrix plot for a model."""
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        cbar=False,
        ax=ax
    )
    ax.set_title(f'Confusion Matrix - {model_name}')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    path = os.path.join(output_dir, f'confusion_matrix_{model_name}.png')
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def plot_roc_curve(
    y_test: pd.Series,
    y_scores: np.ndarray,
    model_name: str,
    output_dir: str
) -> str:
    """Save an ROC curve plot for a model."""
    fpr, tpr, _ = roc_curve(y_test, y_scores)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(
        fpr,
        tpr,
        label=f'ROC curve (AUC = {roc_auc_score(y_test, y_scores):.3f})',
        color='darkorange',
    )
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray')
    ax.set_title(f'ROC Curve - {model_name}')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend(loc='lower right')
    path = os.path.join(output_dir, f'roc_curve_{model_name}.png')
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def plot_precision_recall_curve(
    y_test: pd.Series,
    y_scores: np.ndarray,
    model_name: str,
    output_dir: str
) -> str:
    """Save a precision-recall curve plot for a model."""
    precision, recall, _ = precision_recall_curve(y_test, y_scores)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(recall, precision, color='navy')
    ax.set_title(f'Precision-Recall Curve - {model_name}')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_ylim([0.0, 1.05])
    ax.set_xlim([0.0, 1.0])
    path = os.path.join(output_dir, f'precision_recall_{model_name}.png')
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def plot_feature_importance(
    pipeline: Pipeline,
    numeric_features: list[str],
    categorical_features: list[str],
    output_dir: str
) -> str:
    """Save a feature importance plot for the tuned random forest model."""
    model = pipeline.named_steps['model']
    preprocessor = pipeline.named_steps['preprocessor']
    importance = model.feature_importances_
    feature_names = _get_preprocessed_feature_names(
        preprocessor, numeric_features, categorical_features
    )

    importance_df = pd.DataFrame(
        {'feature': feature_names, 'importance': importance}
    )
    importance_df = importance_df.sort_values('importance', ascending=False).head(20)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.barplot(
        x='importance', y='feature', data=importance_df,
        ax=ax, color='royalblue'
    )
    ax.set_title('Random Forest Feature Importances')
    ax.set_xlabel('Importance')
    ax.set_ylabel('Feature')
    path = os.path.join(output_dir, 'feature_importance_random_forest.png')
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def _print_classification_report(report: dict, model_name: str) -> None:
    df_report = pd.DataFrame(report).transpose()
    print(f'Classification Report for {model_name}')
    print(df_report)


def _print_confusion_matrix(cm: list[list[int]], model_name: str) -> None:
    print(f'Confusion Matrix for {model_name}')
    print(pd.DataFrame(
        cm,
        index=['Actual 0', 'Actual 1'],
        columns=['Predicted 0', 'Predicted 1'],
    ))


def _print_cv_summary(
    cv_results: dict[str, tuple[float, float]],
    model_name: str,
) -> None:
    print(f'Cross-validation summary for {model_name}')
    for metric, (mean_score, std_score) in cv_results.items():
        print(f'  {metric}: {mean_score:.4f} ± {std_score:.4f}')


def _save_comparison_table(comparison: pd.DataFrame, output_path: str) -> str:
    comparison.to_csv(output_path, index=False)
    return output_path


def _save_classification_report(
    report: dict,
    model_name: str,
    output_path: str,
) -> str:
    df_report = pd.DataFrame(report).transpose()
    lines = [f'# Classification Report - {model_name}', '']
    lines.extend(df_report.to_string().splitlines())
    with open(output_path, 'w', encoding='utf-8') as handle:
        handle.write('\n'.join(lines))
    return output_path


def _log_mlflow_run(
    run_name: str,
    pipeline: Pipeline,
    model_name: str,
    params: dict[str, object],
    cv_results: dict[str, tuple[float, float]],
    test_results: dict[str, object],
    artifact_files: list[str],
    register_model: bool = False,
) -> None:
    mlflow.set_experiment('Heart Disease Classification')
    with mlflow.start_run(run_name=run_name):
        mlflow.log_param('model_name', model_name)
        for key, value in params.items():
            mlflow.log_param(key, value)

        mlflow.log_param('cv_folds', 5)

        for metric, (mean_score, std_score) in cv_results.items():
            mlflow.log_metric(f'mean_cv_{metric}', mean_score)
            mlflow.log_metric(f'std_cv_{metric}', std_score)

        mlflow.log_metric('accuracy', test_results['accuracy'])
        mlflow.log_metric('precision', test_results['precision'])
        mlflow.log_metric('recall', test_results['recall'])
        mlflow.log_metric('f1', test_results['f1'])
        if test_results['roc_auc'] is not None:
            mlflow.log_metric('roc_auc', test_results['roc_auc'])

        for artifact in artifact_files:
            if os.path.isdir(artifact):
                mlflow.log_artifacts(artifact)
            elif os.path.isfile(artifact):
                mlflow.log_artifact(artifact)

        mlflow.sklearn.log_model(
            pipeline,
            artifact_path='model',
            serialization_format='cloudpickle'
        )

        if register_model:
            try:
                model_uri = f'runs:/{mlflow.active_run().info.run_id}/model'
                mlflow.register_model(model_uri, 'HeartDiseaseClassificationModel')
            except Exception as err:
                print('MLflow model registration failed:', err)


def _save_model_selection_report(
    results: dict,
    output_path: str,
) -> None:
    comparison = results['comparison'].reset_index()
    comparison_lines = []
    comparison_lines.append('| model | accuracy | precision | recall | f1 | roc_auc |')
    comparison_lines.append('|---|---|---|---|---|---|')
    for _, row in comparison.iterrows():
        comparison_lines.append(
            f"| {row['model']} | {row['accuracy']:.4f} | "
            f"{row['precision']:.4f} | {row['recall']:.4f} | "
            f"{row['f1']:.4f} | {row['roc_auc']:.4f} |"
        )

    lines = [
        '# Model Selection Report',
        '',
        '## Models evaluated',
        '- Logistic Regression',
        '- Random Forest',
        '',
        '## Cross-validation strategy',
        '- 5-fold stratified cross-validation on training data',
        '',
        '## Hyperparameter tuning process',
        '- Random Forest tuned with `GridSearchCV` optimizing ROC-AUC',
        '- Search grid included `n_estimators`, `max_depth`, '
        '`min_samples_split`, and `min_samples_leaf`',
        '',
        '## Best Random Forest parameters',
        f'- {results["cross_validation"]["best_random_forest_params"]}',
        '',
        '## Evaluation metrics used',
        '- Accuracy',
        '- Precision',
        '- Recall',
        '- F1 Score',
        '- ROC-AUC',
        '',
        '## Comparison of Logistic Regression vs Random Forest',
        *comparison_lines,
        '',
        '## Why ROC-AUC was chosen',
        '- ROC-AUC provides a threshold-independent measure of ranking '
        'quality and is robust when the classes are imbalanced.',
        '',
        '## Final model selection',
        f'- Selected `{results["best_model"]}` based on the highest '
        'ROC-AUC on the test set.',
        ''
    ]
    with open(output_path, 'w', encoding='utf-8') as handle:
        handle.write('\n'.join(lines))


def tune_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    numeric_features: list[str],
    categorical_features: list[str]
) -> GridSearchCV:
    pipeline = build_pipeline(
        RandomForestClassifier(random_state=42),
        numeric_features,
        categorical_features,
    )
    param_grid = {
        'model__n_estimators': [100, 200],
        'model__max_depth': [None, 5, 10],
        'model__min_samples_split': [2, 4],
        'model__min_samples_leaf': [1, 2]
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = {
        'accuracy': 'accuracy',
        'roc_auc': 'roc_auc'
    }
    search = GridSearchCV(
        pipeline,
        param_grid,
        cv=cv,
        scoring=scoring,
        refit='roc_auc',
        n_jobs=-1
    )
    search.fit(X_train, y_train)
    return search


def save_model(pipeline: Pipeline, path: str = MODEL_PATH) -> str:
    joblib.dump(pipeline, path)
    return path


def save_pickle_model(pipeline: Pipeline, path: str = PICKLE_MODEL_PATH) -> str:
    with open(path, 'wb') as handle:
        pickle.dump(pipeline, handle)
    return path


def save_onnx_model(
    pipeline: Pipeline,
    raw_feature_names: list[str],
    path: str = ONNX_PATH,
) -> str | None:
    if not ONNX_ENABLED:
        return None

    try:
        initial_type = [
            (name, FloatTensorType([None, 1])) for name in raw_feature_names
        ]
        onnx_model = convert_sklearn(pipeline, initial_types=initial_type)
        with open(path, 'wb') as handle:
            handle.write(onnx_model.SerializeToString())
        return path
    except Exception as err:
        print('Warning: ONNX export failed:', err)
        return None


def save_metadata(metadata: dict[str, object], path: str = METADATA_PATH) -> str:
    with open(path, 'w', encoding='utf-8') as handle:
        json.dump(metadata, handle, indent=2)
    return path


def load_pipeline(path: str = MODEL_PATH) -> Pipeline:
    return joblib.load(path)


def load_model(path: str = MODEL_PATH) -> Pipeline:
    try:
        return load_pipeline(path)
    except Exception as err:
        print(
            f'Warning: failed to load model from {path} due to {type(err).__name__}: {err}'
        )
        print('Retraining a fresh model to restore a compatible pipeline...')
        train_and_evaluate()
        return load_pipeline(path)


def predict(
    sample: dict[str, object] | pd.DataFrame | list[dict[str, object]],
    path: str = MODEL_PATH,
) -> np.ndarray:
    pipeline = load_pipeline(path)
    if isinstance(sample, list):
        sample = pd.DataFrame(sample)
    elif isinstance(sample, dict):
        sample = pd.DataFrame([sample])
    return pipeline.predict(sample)


def train_and_evaluate() -> dict[str, object]:
    df = load_raw_data()
    X_train, X_test, y_train, y_test = split_data(df)

    numeric_features, categorical_features = infer_feature_types(X_train, unique_threshold=10)

    report_dir = os.path.join(ROOT_DIR, 'reports')
    figures_dir = os.path.join(report_dir, 'figures')
    _ensure_directory(report_dir)
    _ensure_directory(figures_dir)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    logistic_pipeline = build_pipeline(
        LogisticRegression(max_iter=1000, random_state=42),
        numeric_features,
        categorical_features
    )
    random_forest_pipeline = build_pipeline(
        RandomForestClassifier(random_state=42),
        numeric_features,
        categorical_features
    )

    logistic_cv = cross_validate_model(logistic_pipeline, X_train, y_train, cv)
    rf_cv = cross_validate_model(random_forest_pipeline, X_train, y_train, cv)

    print('Cross-validation results:')
    _print_cv_summary(logistic_cv, 'Logistic Regression')
    _print_cv_summary(rf_cv, 'Random Forest')

    rf_search = tune_random_forest(X_train, y_train, numeric_features, categorical_features)
    best_random_forest = rf_search.best_estimator_

    print('\nBest Random Forest hyperparameters:')
    print(rf_search.best_params_)

    logistic_pipeline.fit(X_train, y_train)
    logistic_test = evaluate_on_test(logistic_pipeline, X_test, y_test)

    random_forest_pipeline.fit(X_train, y_train)
    rf_baseline_test = evaluate_on_test(random_forest_pipeline, X_test, y_test)

    rf_test = evaluate_on_test(best_random_forest, X_test, y_test)

    print('\nLogistic Regression test set evaluation:')
    _print_classification_report(logistic_test['classification_report'], 'Logistic Regression')
    _print_confusion_matrix(logistic_test['confusion_matrix'], 'Logistic Regression')

    print('\nRandom Forest baseline test set evaluation:')
    _print_classification_report(rf_baseline_test['classification_report'], 'Random Forest (pre-tuning)')
    _print_confusion_matrix(rf_baseline_test['confusion_matrix'], 'Random Forest (pre-tuning)')

    print('\nRandom Forest tuned test set evaluation:')
    _print_classification_report(rf_test['classification_report'], 'Random Forest (tuned)')
    _print_confusion_matrix(rf_test['confusion_matrix'], 'Random Forest (tuned)')

    comparison = pd.DataFrame([
        {
            'model': 'logistic_regression',
            'accuracy': logistic_test['accuracy'],
            'precision': logistic_test['precision'],
            'recall': logistic_test['recall'],
            'f1': logistic_test['f1'],
            'roc_auc': logistic_test['roc_auc']
        },
        {
            'model': 'random_forest_pre_tuning',
            'accuracy': rf_baseline_test['accuracy'],
            'precision': rf_baseline_test['precision'],
            'recall': rf_baseline_test['recall'],
            'f1': rf_baseline_test['f1'],
            'roc_auc': rf_baseline_test['roc_auc']
        },
        {
            'model': 'random_forest_tuned',
            'accuracy': rf_test['accuracy'],
            'precision': rf_test['precision'],
            'recall': rf_test['recall'],
            'f1': rf_test['f1'],
            'roc_auc': rf_test['roc_auc']
        }
    ]).set_index('model')

    best_model_name = comparison['roc_auc'].idxmax()
    best_pipeline = logistic_pipeline if best_model_name == 'logistic_regression' else best_random_forest
    save_model(best_pipeline)
    save_pickle_model(best_pipeline)

    _ensure_directory(MLFLOW_MODEL_DIR)
    try:
        mlflow.sklearn.save_model(
            best_pipeline,
            MLFLOW_MODEL_DIR,
            serialization_format='cloudpickle'
        )
        mlflow_model_path = MLFLOW_MODEL_DIR
    except Exception as err:
        print('Warning: local MLflow model save failed:', err)
        mlflow_model_path = None

    onnx_path = save_onnx_model(best_pipeline, numeric_features + categorical_features)

    metadata = {
        'model_name': 'HeartDiseaseClassificationModel',
        'algorithm': 'Random Forest' if best_model_name != 'logistic_regression' else 'Logistic Regression',
        'training_date': datetime.datetime.now().isoformat(),
        'sklearn_version': sklearn_version,
        'python_version': platform.python_version(),
        'best_hyperparameters': (
            rf_search.best_params_
            if best_model_name != 'logistic_regression'
            else {'max_iter': 1000, 'random_state': 42}
        ),
        'evaluation_metrics': {
            'accuracy': float(
                rf_test['accuracy']
                if best_model_name != 'logistic_regression'
                else logistic_test['accuracy']
            ),
            'precision': float(
                rf_test['precision']
                if best_model_name != 'logistic_regression'
                else logistic_test['precision']
            ),
            'recall': float(
                rf_test['recall']
                if best_model_name != 'logistic_regression'
                else logistic_test['recall']
            ),
            'f1': float(
                rf_test['f1']
                if best_model_name != 'logistic_regression'
                else logistic_test['f1']
            ),
            'roc_auc': float(
                rf_test['roc_auc']
                if best_model_name != 'logistic_regression'
                else logistic_test['roc_auc']
            ),
        },
        'feature_names': numeric_features + categorical_features,
        'target_column': 'target',
        'saved_files': {
            'joblib': MODEL_PATH,
            'pickle': PICKLE_MODEL_PATH,
            'mlflow_model': mlflow_model_path,
            'onnx': onnx_path
        }
    }
    save_metadata(metadata)

    roc_lr_path = None
    pr_lr_path = None
    roc_rf_pre_path = None
    pr_rf_pre_path = None
    roc_rf_tuned_path = None
    pr_rf_tuned_path = None

    if logistic_test['y_scores'] is not None:
        roc_lr_path = plot_roc_curve(
            y_test, logistic_test['y_scores'], 'logistic_regression', figures_dir
        )
        pr_lr_path = plot_precision_recall_curve(
            y_test, logistic_test['y_scores'], 'logistic_regression', figures_dir
        )
    if rf_baseline_test['y_scores'] is not None:
        roc_rf_pre_path = plot_roc_curve(
            y_test, rf_baseline_test['y_scores'],
            'random_forest_pre_tuning', figures_dir
        )
        pr_rf_pre_path = plot_precision_recall_curve(
            y_test, rf_baseline_test['y_scores'],
            'random_forest_pre_tuning', figures_dir
        )
    if rf_test['y_scores'] is not None:
        roc_rf_tuned_path = plot_roc_curve(
            y_test, rf_test['y_scores'], 'random_forest_tuned', figures_dir
        )
        pr_rf_tuned_path = plot_precision_recall_curve(
            y_test, rf_test['y_scores'], 'random_forest_tuned', figures_dir
        )

    confusion_matrix_lr = plot_confusion_matrix(
        logistic_test['confusion_matrix'], 'logistic_regression', figures_dir
    )
    confusion_matrix_rf_pre = plot_confusion_matrix(
        rf_baseline_test['confusion_matrix'], 'random_forest_pre_tuning',
        figures_dir
    )
    confusion_matrix_rf_tuned = plot_confusion_matrix(
        rf_test['confusion_matrix'], 'random_forest_tuned', figures_dir
    )
    feature_importance_path = plot_feature_importance(
        best_random_forest, numeric_features, categorical_features, figures_dir
    )

    classification_report_lr_path = os.path.join(
        report_dir, 'classification_report_logistic_regression.txt'
    )
    classification_report_rf_pre_path = os.path.join(
        report_dir, 'classification_report_random_forest_pre_tuning.txt'
    )
    classification_report_rf_tuned_path = os.path.join(
        report_dir, 'classification_report_random_forest_tuned.txt'
    )
    _save_classification_report(
        logistic_test['classification_report'],
        'Logistic Regression',
        classification_report_lr_path
    )
    _save_classification_report(
        rf_baseline_test['classification_report'],
        'Random Forest (pre-tuning)',
        classification_report_rf_pre_path
    )
    _save_classification_report(
        rf_test['classification_report'],
        'Random Forest (tuned)',
        classification_report_rf_tuned_path
    )

    model_selection_report_path = os.path.join(report_dir, 'model_selection.md')
    _save_model_selection_report({
        'comparison': comparison,
        'best_model': best_model_name,
        'cross_validation': {
            'best_random_forest_params': rf_search.best_params_
        }
    }, model_selection_report_path)

    comparison_path = os.path.join(report_dir, 'comparison_table.csv')
    _save_comparison_table(comparison.reset_index(), comparison_path)

    _log_mlflow_run(
        run_name='Logistic Regression',
        pipeline=logistic_pipeline,
        model_name='Logistic Regression',
        params={
            'random_state': 42,
            'max_iter': 1000
        },
        cv_results=logistic_cv,
        test_results=logistic_test,
        artifact_files=[
            confusion_matrix_lr,
            roc_lr_path,
            pr_lr_path,
            classification_report_lr_path,
            model_selection_report_path,
            comparison_path
        ],
        register_model=False
    )

    _log_mlflow_run(
        run_name='Random Forest (pre-tuning)',
        pipeline=random_forest_pipeline,
        model_name='Random Forest (pre-tuning)',
        params={
            'random_state': 42
        },
        cv_results=rf_cv,
        test_results=rf_baseline_test,
        artifact_files=[
            confusion_matrix_rf_pre,
            roc_rf_pre_path,
            pr_rf_pre_path,
            classification_report_rf_pre_path,
            model_selection_report_path,
            comparison_path
        ],
        register_model=False
    )

    _log_mlflow_run(
        run_name='Random Forest (tuned)',
        pipeline=best_random_forest,
        model_name='Random Forest (tuned)',
        params={
            'random_state': 42,
            **rf_search.best_params_
        },
        cv_results=rf_cv,
        test_results=rf_test,
        artifact_files=[
            confusion_matrix_rf_tuned,
            roc_rf_tuned_path,
            pr_rf_tuned_path,
            feature_importance_path,
            classification_report_rf_tuned_path,
            model_selection_report_path,
            comparison_path
        ],
        register_model=True
    )

    result = {
        'split': {
            'train_shape': X_train.shape,
            'test_shape': X_test.shape
        },
        'feature_types': {
            'numeric': numeric_features,
            'categorical': categorical_features
        },
        'cross_validation': {
            'logistic_regression': logistic_cv,
            'random_forest': rf_cv,
            'best_random_forest_params': rf_search.best_params_
        },
        'test_results': {
            'logistic_regression': logistic_test,
            'random_forest_pre_tuning': rf_baseline_test,
            'random_forest_tuned': rf_test
        },
        'comparison': comparison,
        'best_model': best_model_name,
        'saved_pipeline_path': MODEL_PATH,
        'saved_pickle_path': PICKLE_MODEL_PATH,
        'saved_mlflow_model_path': mlflow_model_path,
        'saved_onnx_path': onnx_path,
        'metadata_path': METADATA_PATH,
        'report_paths': {
            'model_selection': model_selection_report_path,
            'comparison_table': comparison_path,
            'figures_dir': figures_dir
        }
    }
    return result


if __name__ == '__main__':
    results = train_and_evaluate()
    print('Training and evaluation complete.')
    print('Train shape:', results['split']['train_shape'])
    print('Test shape:', results['split']['test_shape'])
    print('\nBest model:', results['best_model'])
    print('\nComparison table:')
    print(results['comparison'])
    print('\nBest random forest parameters:')
    print(results['cross_validation']['best_random_forest_params'])
    print('\nSaved pipeline:', results['saved_pipeline_path'])
    print('\nGenerated report:', results['report_paths']['model_selection'])
    print('Saved figures in:', results['report_paths']['figures_dir'])

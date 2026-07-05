# Feature Engineering & Model Development

## Preprocessing Approach
The raw dataset is loaded from `data/raw/heart_disease.csv` and the target label is converted to binary using `target = (target > 0).astype(int)`. Feature types are detected automatically from the training data using pandas data types and a low-cardinality heuristic for numeric columns.

## Training/Test Split
The dataset is split into training and test sets with an 80/20 ratio using stratified sampling on the target label to preserve class balance and prevent selection bias.

## Preprocessing Pipeline
A `ColumnTransformer` handles preprocessing without hardcoded feature names:
- Numerical features: `SimpleImputer(strategy='median')` followed by `StandardScaler()`
- Categorical features: `SimpleImputer(strategy='most_frequent')` followed by `OneHotEncoder(handle_unknown='ignore')`

The preprocessing pipeline is fit only on training data to prevent data leakage and then applied to the test set.

## Models
Two classification models were evaluated:
- Logistic Regression
- Random Forest Classifier

## Cross-Validation Strategy
Both models are validated using 5-fold stratified cross-validation on the training set. This preserves the target class distribution across folds and provides stable performance estimates.

## Hyperparameter Tuning
The Random Forest model is tuned with `GridSearchCV` optimizing ROC-AUC. The search grid includes:
- `n_estimators`: 100, 200
- `max_depth`: None, 5, 10
- `min_samples_split`: 2, 4
- `min_samples_leaf`: 1, 2

## Evaluation Metrics
Models are evaluated on the hold-out test set using:
- Accuracy
- Precision
- Recall
- ROC-AUC
- Classification report
- Confusion matrix

## Final Model Selection
The model with the highest ROC-AUC on the test set is selected as the final model. The complete fitted pipeline (preprocessing + model) is saved to `models/heart_pipeline.joblib`.

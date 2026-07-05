# Model Selection Report

## Models evaluated
- Logistic Regression
- Random Forest

## Cross-validation strategy
- 5-fold stratified cross-validation on training data

## Hyperparameter tuning process
- Random Forest tuned with `GridSearchCV` optimizing ROC-AUC
- Search grid included `n_estimators`, `max_depth`, `min_samples_split`, and `min_samples_leaf`

## Best Random Forest parameters
- {'model__max_depth': 10, 'model__min_samples_leaf': 2, 'model__min_samples_split': 2, 'model__n_estimators': 100}

## Evaluation metrics used
- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

## Comparison of Logistic Regression vs Random Forest
| model | accuracy | precision | recall | f1 | roc_auc |
|---|---|---|---|---|---|
| logistic_regression | 0.8852 | 0.8387 | 0.9286 | 0.8814 | 0.9665 |
| random_forest_pre_tuning | 0.8689 | 0.8125 | 0.9286 | 0.8667 | 0.9453 |
| random_forest_tuned | 0.9180 | 0.8710 | 0.9643 | 0.9153 | 0.9535 |

## Why ROC-AUC was chosen
- ROC-AUC provides a threshold-independent measure of ranking quality and is robust when the classes are imbalanced.

## Final model selection
- Selected `logistic_regression` based on the highest ROC-AUC on the test set.

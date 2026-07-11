import os
from model import train_and_evaluate

if __name__ == '__main__':
    print('Starting feature engineering and model development...')
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

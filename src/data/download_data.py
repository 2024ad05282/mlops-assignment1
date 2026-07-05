import os
import urllib.request

DATA_DIR = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, 'data', 'raw')
URL = (
    'https://archive.ics.uci.edu/ml/machine-learning-databases/'
    'heart-disease/processed.cleveland.data'
)
OUTPUT_FILE = os.path.join(DATA_DIR, 'heart_disease.csv')

os.makedirs(DATA_DIR, exist_ok=True)

if __name__ == '__main__':
    print(f'Downloading dataset to {OUTPUT_FILE}')
    urllib.request.urlretrieve(URL, OUTPUT_FILE)
    print('Download complete.')

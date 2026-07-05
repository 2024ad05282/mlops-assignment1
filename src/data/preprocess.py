import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
RAW_DIR = os.path.join(ROOT_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(ROOT_DIR, "data", "processed")

INPUT_FILE = os.path.join(RAW_DIR, "heart_disease.csv")
OUTPUT_FILE = os.path.join(PROCESSED_DIR, "heart_disease_clean.csv")

COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak",
    "slope", "ca", "thal", "target"
]

os.makedirs(PROCESSED_DIR, exist_ok=True)


def load_data():
    """Load raw dataset."""
    return pd.read_csv(
        INPUT_FILE,
        header=None,
        names=COLUMNS,
        na_values="?"
    )


def preprocess():

    df = load_data()

    print("=" * 40)
    print("Original Shape:", df.shape)
    print("=" * 40)

    # -------------------------------------------------
    # 1. Convert all columns to numeric
    # -------------------------------------------------
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # -------------------------------------------------
    # 2. Handle Missing Values
    # Only 'ca' and 'thal' contain missing values
    # -------------------------------------------------
    df["ca"].fillna(df["ca"].median(), inplace=True)
    df["thal"].fillna(df["thal"].mode()[0], inplace=True)

    # -------------------------------------------------
    # 3. Remove Duplicate Rows
    # -------------------------------------------------
    before = len(df)
    df.drop_duplicates(inplace=True)
    print(f"Removed {before - len(df)} duplicate rows.")

    # -------------------------------------------------
    # 4. Convert Target to Binary
    # -------------------------------------------------
    df["target"] = (df["target"] > 0).astype(int)

    # -------------------------------------------------
    # 5. Handle Outliers using IQR Capping
    # -------------------------------------------------
    numerical_cols = [
        "age",
        "trestbps",
        "chol",
        "thalach",
        "oldpeak"
    ]

    for col in numerical_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        df[col] = np.clip(df[col], lower, upper)

    # -------------------------------------------------
    # 6. One-Hot Encode Categorical Features
    # -------------------------------------------------
    categorical_cols = [
        "cp",
        "restecg",
        "slope",
        "thal"
    ]

    df = pd.get_dummies(
        df,
        columns=categorical_cols,
        drop_first=True,
        dtype=int
    )

    # -------------------------------------------------
    # 7. Standardize Numerical Features
    # -------------------------------------------------
    scaler = StandardScaler()

    numerical_cols = [
        "age",
        "trestbps",
        "chol",
        "thalach",
        "oldpeak"
    ]

    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    print("Final Shape:", df.shape)
    print("Missing Values Remaining:", df.isnull().sum().sum())

    return df


if __name__ == "__main__":

    processed_df = preprocess()

    processed_df.to_csv(OUTPUT_FILE, index=False)

    print(f"\nProcessed dataset saved to:\n{OUTPUT_FILE}")
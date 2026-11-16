"""Train a RandomForest regression model for workout calorie estimation.

Usage:
    python tools/train_calorie_model.py

This script trains a calorie burn estimator using the provided dataset and saves:
    - models/calories_model_final.pkl
    - models/scaler_final.pkl
    - models/features_list.pkl

The model uses StandardScaler normalization and a RandomForestRegressor.
"""

import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DATASET_PATH = "data/final_huge_dataset.csv"
MODEL_DIR = "models"
FEATURES = ["Gender", "Age", "Height", "Weight", "Duration", "Heart_Rate"]
TARGET = "Calories"

CLEANING_BOUNDS = {
    "Gender": (0, 1),
    "Age": (10, 100),
    "Height": (100, 220),
    "Weight": (30, 200),
    "Duration": (1, 300),
    "Heart_Rate": (40, 220),
    "Calories": (0, 10000),
}


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Clean calorie dataset and enforce numeric ranges."""
    df = df.copy()
    required_columns = FEATURES + [TARGET]
    df = df.dropna(subset=required_columns)

    # Normalize gender encoding from text if needed
    if df["Gender"].dtype == object:
        df["Gender"] = df["Gender"].map({"Male": 0, "Female": 1})

    for col in required_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=required_columns)

    for col, (low, high) in CLEANING_BOUNDS.items():
        if col not in df.columns:
            continue
        df = df[(df[col] >= low) & (df[col] <= high)]

    df = df.reset_index(drop=True)
    return df


def train_calorie_model():
    os.makedirs(MODEL_DIR, exist_ok=True)

    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

    print(f"Loading dataset: {DATASET_PATH}")
    df = pd.read_csv(DATASET_PATH)
    print(f"Initial rows: {len(df):,}")

    df = clean_dataset(df)
    print(f"Rows after cleaning: {len(df):,}")
    print(df[FEATURES + [TARGET]].head(5).to_string(index=False))

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("rf", RandomForestRegressor(random_state=42, n_jobs=-1)),
        ]
    )

    param_grid = {
        "rf__n_estimators": [100, 200],
        "rf__max_depth": [8, 12, 16],
        "rf__min_samples_leaf": [5, 10, 20],
        "rf__min_samples_split": [5, 10, 15],
    }

    print("Running GridSearchCV on training split to reduce overfitting...")
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="r2",
        cv=5,
        n_jobs=-1,
        verbose=1,
    )
    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    print("Best hyperparameters:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")
    print(f"Best CV R² on training split: {best_score:.4f}")

    best_pipeline = grid_search.best_estimator_
    y_test_pred = best_pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, y_test_pred)
    mse = mean_squared_error(y_test, y_test_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_test_pred)

    print("\nFinal evaluation on holdout set:")
    print(f"  MAE:  {mae:.2f} kcal")
    print(f"  RMSE: {rmse:.2f} kcal")
    print(f"  R²:   {r2:.4f}")

    model_path = os.path.join(MODEL_DIR, "calories_model_final.pkl")
    scaler_path = os.path.join(MODEL_DIR, "scaler_final.pkl")
    features_path = os.path.join(MODEL_DIR, "features_list.pkl")

    joblib.dump(best_pipeline.named_steps["rf"], model_path)
    joblib.dump(best_pipeline.named_steps["scaler"], scaler_path)
    joblib.dump(FEATURES, features_path)

    print("\nSaved calibrated model artifacts:")
    print(f"  - {model_path}")
    print(f"  - {scaler_path}")
    print(f"  - {features_path}")
    print(f"  - training date: {datetime.now().isoformat()}")

    # Return the trained estimator and scaler from the best pipeline
    return (
        best_pipeline.named_steps["rf"],
        best_pipeline.named_steps["scaler"],
        FEATURES,
    )


if __name__ == "__main__":
    train_calorie_model()

"""Check calorie regression overfitting using cross-validation and train/test split."""

import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    make_scorer,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DATASET_PATH = "data/final_huge_dataset.csv"
CLEANING_BOUNDS = {
    "Gender": (0, 1),
    "Age": (10, 100),
    "Height": (100, 220),
    "Weight": (30, 200),
    "Duration": (1, 300),
    "Heart_Rate": (40, 220),
    "Calories": (0, 10000),
}

if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

print(f"Loading dataset: {DATASET_PATH}")
df = pd.read_csv(DATASET_PATH)
print(f"Initial rows: {len(df):,}")

cols = ["Gender", "Age", "Height", "Weight", "Duration", "Heart_Rate", "Calories"]
df = df[cols].copy()
if df["Gender"].dtype == object:
    df["Gender"] = df["Gender"].map({"Male": 0, "Female": 1})
for c in cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df = df.dropna(subset=cols)
for col, (low, high) in CLEANING_BOUNDS.items():
    df = df[(df[col] >= low) & (df[col] <= high)]

print(f"Rows after cleaning: {len(df):,}")
X = df[["Gender", "Age", "Height", "Weight", "Duration", "Heart_Rate"]]
y = df["Calories"]

pipeline = Pipeline(
    [
        ("scaler", StandardScaler()),
        (
            "rf",
            RandomForestRegressor(
                n_estimators=200,
                max_depth=20,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1,
            ),
        ),
    ]
)

print("\nRunning 5-fold cross-validation...")
cv = cross_validate(
    pipeline,
    X,
    y,
    cv=5,
    scoring={
        "r2": make_scorer(r2_score),
        "mae": make_scorer(mean_absolute_error, greater_is_better=False),
        "mse": make_scorer(mean_squared_error, greater_is_better=False),
    },
    return_train_score=True,
    n_jobs=-1,
)

print("train_r2 mean:", np.mean(cv["train_r2"]))
print("test_r2 mean:", np.mean(cv["test_r2"]))
print("train_mae mean:", -np.mean(cv["train_mae"]))
print("test_mae mean:", -np.mean(cv["test_mae"]))
print("train_rmse mean:", np.sqrt(-np.mean(cv["train_mse"])))
print("test_rmse mean:", np.sqrt(-np.mean(cv["test_mse"])))

print("\nTrain/test split comparison:")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)
pipeline.fit(X_train, y_train)

y_train_pred = pipeline.predict(X_train)
y_test_pred = pipeline.predict(X_test)
print("train r2:", r2_score(y_train, y_train_pred))
print("test r2:", r2_score(y_test, y_test_pred))
print("train mae:", mean_absolute_error(y_train, y_train_pred))
print("test mae:", mean_absolute_error(y_test, y_test_pred))
print("train rmse:", np.sqrt(mean_squared_error(y_train, y_train_pred)))
print("test rmse:", np.sqrt(mean_squared_error(y_test, y_test_pred)))

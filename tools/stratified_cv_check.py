"""Run StratifiedKFold (on binned Calories) and compare with regular KFold."""

import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PATH = "data/final_huge_dataset.csv"
if not os.path.exists(PATH):
    raise SystemExit("Dataset not found")

df = pd.read_csv(PATH)
cols = ["Gender", "Age", "Height", "Weight", "Duration", "Heart_Rate", "Calories"]
df = df[cols].copy()
# Normalize and numeric
if df["Gender"].dtype == object:
    df["Gender"] = df["Gender"].map({"Male": 0, "Female": 1})
for c in cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df = df.dropna(subset=cols)
# Drop exact duplicates
before = len(df)
df = df.drop_duplicates()
after = len(df)
print(f"Rows before: {before}, after dropping duplicates: {after}")

X = df[["Gender", "Age", "Height", "Weight", "Duration", "Heart_Rate"]]
y = df["Calories"]

pipeline = Pipeline(
    [
        ("scaler", StandardScaler()),
        (
            "rf",
            RandomForestRegressor(
                n_estimators=200,
                max_depth=16,
                min_samples_leaf=5,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1,
            ),
        ),
    ]
)

# Regular KFold
kf = KFold(n_splits=5, shuffle=True, random_state=42)
print("\nRunning regular 5-fold KFold CV...")
cv_kf = cross_validate(
    pipeline,
    X,
    y,
    cv=kf,
    scoring=["r2", "neg_mean_absolute_error", "neg_mean_squared_error"],
    return_train_score=True,
    n_jobs=-1,
)
print("KF train R2 mean:", np.mean(cv_kf["train_r2"]))
print("KF test R2 mean:", np.mean(cv_kf["test_r2"]))
print("KF test MAE mean:", -np.mean(cv_kf["test_neg_mean_absolute_error"]))
print("KF test RMSE mean:", np.sqrt(-np.mean(cv_kf["test_neg_mean_squared_error"])))

# Stratified by binned target (quantiles)
try:
    bins = pd.qcut(y, q=10, duplicates="drop")
    strata = bins.codes if hasattr(bins, "codes") else pd.factorize(bins)[0]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    print("\nRunning Stratified (by Calories quantile) 5-fold CV...")
    cv_skf = cross_validate(
        pipeline,
        X,
        y,
        cv=skf.split(X, strata),
        scoring=["r2", "neg_mean_absolute_error", "neg_mean_squared_error"],
        return_train_score=True,
        n_jobs=-1,
    )
    print("Strat train R2 mean:", np.mean(cv_skf["train_r2"]))
    print("Strat test R2 mean:", np.mean(cv_skf["test_r2"]))
    print("Strat test MAE mean:", -np.mean(cv_skf["test_neg_mean_absolute_error"]))
    print(
        "Strat test RMSE mean:",
        np.sqrt(-np.mean(cv_skf["test_neg_mean_squared_error"])),
    )
except Exception as e:
    print("Stratified CV failed:", e)

# Option: log-transform target
print("\nRunning KFold CV with log(target+1) transform...")
y_log = np.log1p(y)
cv_kf_log = cross_validate(
    pipeline,
    X,
    y_log,
    cv=kf,
    scoring=["r2", "neg_mean_absolute_error", "neg_mean_squared_error"],
    return_train_score=True,
    n_jobs=-1,
)
print("log KF train R2 mean:", np.mean(cv_kf_log["train_r2"]))
print("log KF test R2 mean:", np.mean(cv_kf_log["test_r2"]))
print(
    "log KF test MAE mean (in log units):",
    -np.mean(cv_kf_log["test_neg_mean_absolute_error"]),
)
print("Done.")

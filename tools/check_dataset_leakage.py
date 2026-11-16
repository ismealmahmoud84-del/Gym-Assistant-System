"""Check dataset for duplicates, grouping columns, and basic leakage signals.
Prints:
 - total rows, columns
 - exact duplicate row count and sample
 - candidate id/session/timestamp columns and value counts
 - duplicates when excluding target
"""

import os

import pandas as pd

PATH = "data/final_huge_dataset.csv"
if not os.path.exists(PATH):
    raise SystemExit(f"Dataset not found: {PATH}")

df = pd.read_csv(PATH)
print("rows, cols:", df.shape)
print("columns:", df.columns.tolist())

# Basic exact duplicates
dup_exact = df.duplicated(keep=False)
print("exact duplicate rows count:", int(dup_exact.sum()))
if dup_exact.any():
    print("\nSample exact duplicates:")
    print(df[dup_exact].head(10).to_string(index=False))

# Duplicates excluding target (Calories)
if "Calories" in df.columns:
    features_only = df.drop(columns=["Calories"])
else:
    features_only = df

dup_feat = features_only.duplicated(keep=False)
print("\nduplicate rows across features (excluding target):", int(dup_feat.sum()))
if dup_feat.any():
    print("\nSample feature-duplicates:")
    print(df[dup_feat].head(10).to_string(index=False))

# Candidate id/session/timestamp columns
candidates = [
    c
    for c in df.columns
    if any(
        k in c.lower()
        for k in ["id", "user", "member", "session", "timestamp", "time", "date"]
    )
]
print("\ncandidate id/session/timestamp columns:", candidates)
for c in candidates:
    print(
        f"-- {c}: unique={df[c].nunique()}, nulls={df[c].isna().sum()}, sample_values=",
        df[c].dropna().unique()[:5],
    )

# Check for near-duplicates using subset of columns
subset_cols = [c for c in df.columns if c not in ("Calories",)]
if len(subset_cols) > 0:
    group_counts = df.groupby(subset_cols).size().sort_values(ascending=False).head(10)
    print("\nTop groups (by identical feature sets):")
    print(group_counts.to_string())

# Simple stats for target vs duplicates
if "Calories" in df.columns:
    print("\nTarget (Calories) basic stats:")
    print(df["Calories"].describe())

print("\nDone.")

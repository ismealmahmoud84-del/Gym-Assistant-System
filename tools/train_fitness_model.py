"""
Retrain the fitness level classifier from the gym members dataset.

Usage (from project root):
    python tools/train_fitness_model.py
"""

import os
from datetime import datetime

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

EXPERIENCE_LEVEL_LABELS = {
    1: "مبتدئ",
    2: "متوسط",
    3: "متقدم",
    4: "محترف",
}

# الحد الأدنى لبيانات التدريب
MIN_TRAINING_SAMPLES = 10000
RECOMMENDED_TRAINING_SAMPLES = 16000
TARGET_CLASS_SIZE = 2500

FEATURES = [
    "Age",
    "Gender",
    "Weight (kg)",
    "Height (m)",
    "BMI",
    "Resting_BPM",
    "Workout_Frequency (days/week)",
    "Water_Intake (liters)",
    "Fat_Percentage",
    "Workout_Type_Code",
]

WORKOUT_TYPE_MAPPING = {
    "Cardio": 0,
    "Strength": 1,
    "HIIT": 2,
    "Yoga": 3,
    "Mixed": 4,
    "Circuit": 4,
    "Pilates": 4,
    "CrossFit": 4,
    "Other": 5,
}

CLEANING_BOUNDS = {
    "Age": (13, 100),
    "Weight (kg)": (30, 300),
    "Height (m)": (1.4, 2.5),
    "Resting_BPM": (30, 120),
    "Workout_Frequency (days/week)": (0, 7),
    "Water_Intake (liters)": (0.5, 10),
    "BMI": (10, 60),
    "Fat_Percentage": (3, 70),
}


def clean_dataset(df):
    """Clean and normalize the dataset before model training."""
    required_columns = [
        "Age",
        "Gender",
        "Weight (kg)",
        "Height (m)",
        "BMI",
        "Resting_BPM",
        "Workout_Frequency (days/week)",
        "Water_Intake (liters)",
        "Fat_Percentage",
        "Workout_Type",
        "Experience_Level",
    ]

    df = df.copy()
    df = df.dropna(subset=required_columns)

    df["Gender"] = df["Gender"].map({"Male": 0, "Female": 1})
    df = df[df["Gender"].isin([0, 1])]

    df["Workout_Type"] = df["Workout_Type"].fillna("Other").astype(str)
    df["Workout_Type_Code"] = df["Workout_Type"].map(WORKOUT_TYPE_MAPPING)
    df["Workout_Type_Code"] = (
        df["Workout_Type_Code"].fillna(WORKOUT_TYPE_MAPPING["Other"]).astype(int)
    )

    for col, (low, high) in CLEANING_BOUNDS.items():
        if col not in df.columns:
            continue
        median_value = df.loc[df[col].between(low, high), col].median()
        if pd.isna(median_value):
            median_value = (low + high) / 2
        df.loc[df[col] < low, col] = median_value
        df.loc[df[col] > high, col] = median_value

    # Enforce valid experience labels only.
    df = df[df["Experience_Level"].isin([1, 2, 3, 4])]

    return df


def balance_dataset(df, label_col="Experience_Level", target_count=TARGET_CLASS_SIZE):
    """Balance the dataset across all labels using resampling."""
    balanced_frames = []
    present_labels = sorted(df[label_col].unique())

    for label in present_labels:
        subset = df[df[label_col] == label]
        if len(subset) == 0:
            continue

        if len(subset) < target_count:
            subset = resample(
                subset,
                replace=True,
                n_samples=target_count,
                random_state=42,
            )
        else:
            subset = subset.sample(n=target_count, random_state=42)

        balanced_frames.append(subset)

    balanced_df = pd.concat(balanced_frames, ignore_index=True)
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
    return balanced_df


def main():
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)

    dataset_path = "data/gym_members_exercise_tracking.cv.csv"
    print(f"Loading dataset: {dataset_path}")
    df = pd.read_csv(dataset_path)

    print(f"\n📊 Initial rows: {len(df):,}")
    df = clean_dataset(df)
    print(f"📊 Rows after cleaning: {len(df):,}")
    print("📌 Class distribution after cleaning:")
    print(df["Experience_Level"].value_counts().to_string())

    if len(df) < MIN_TRAINING_SAMPLES:
        print(f"⚠️ بيانات التدريب أقل من المتطلبات ({len(df):,} صف).")
        print(
            "   سيتم الاستمرار في التدريب على البيانات المتاحة "
            "مع توازن عبر الفئات الحالية."
        )

    balanced_df = balance_dataset(df)
    print(f"📊 Rows after balancing: {len(balanced_df):,}")
    print("📌 Class distribution after balancing:")
    print(balanced_df["Experience_Level"].value_counts().to_string())

    X = balanced_df[FEATURES]
    y = balanced_df["Experience_Level"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("\nTraining fitness level classifier with Random Forest...")
    fitness_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        random_state=42,
        oob_score=True,
        n_jobs=-1,
    )
    fitness_model.fit(X_train, y_train)

    y_pred = fitness_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Fitness model accuracy: {accuracy * 100:.2f}%")
    print(f"OOB score: {fitness_model.oob_score_ * 100:.2f}%")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred))

    model_path = os.path.join(models_dir, "fitness_level_model.joblib")
    joblib.dump(fitness_model, model_path)

    preprocessing_config = {
        "version": "1.1",
        "training_date": datetime.now().isoformat(),
        "gender_mapping": {"Male": 0, "Female": 1},
        "workout_type_mapping": WORKOUT_TYPE_MAPPING,
        "fitness_labels": fitness_model.classes_.tolist(),
        "fitness_label_names": {
            int(level): EXPERIENCE_LEVEL_LABELS.get(int(level), str(level))
            for level in fitness_model.classes_.tolist()
        },
        "feature_names": FEATURES,
        "feature_dtypes": {col: str(X[col].dtype) for col in FEATURES},
        "bmr_formula": "Mifflin-St Jeor",
        "activity_multipliers": {
            "sedentary": 1.2,
            "light": 1.375,
            "moderate": 1.55,
            "active": 1.725,
            "very_active": 1.9,
        },
        "target_class_size": TARGET_CLASS_SIZE,
    }

    preprocessing_path = os.path.join(models_dir, "preprocessing_config.joblib")
    joblib.dump(preprocessing_config, preprocessing_path)

    print("Saved:")
    print(f"  - {model_path}")
    print(f"  - {preprocessing_path}")
    print(f"  - Labels: {preprocessing_config['fitness_label_names']}")


if __name__ == "__main__":
    main()

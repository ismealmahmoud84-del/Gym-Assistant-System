"""
قياس accuracy موديل التنبؤ بالسعرات الحرارية
"""

import os
import sys

import django

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "gym_config.settings")
django.setup()

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from fitness_api.model_manager import ModelRegistry
from fitness_api.utils import prepare_calorie_model_input

# حمّل البيانات
dataset_path = "data/final_huge_dataset.csv"
df = pd.read_csv(dataset_path)

# إعدادات الحد الأدنى للبيانات
MIN_TRAINING_SAMPLES = 10000
EXPECTED_TRAINING_SAMPLES = 16000

print("=" * 60)
print("📊 قياس Accuracy موديل السعرات الحرارية")
print("=" * 60)
print("\n📋 معلومات البيانات المحلية:")
print(f"   حجم البيانات المحلية: {len(df):,} صف")
print(f"   ✅ حجم بيانات التدريب الأصلي: {EXPECTED_TRAINING_SAMPLES:,} صف")
print(f"   ⚠️  الحد الأدنى للبيانات: {MIN_TRAINING_SAMPLES:,} صف")

if len(df) < MIN_TRAINING_SAMPLES:
    print(
        f"\n❌ تحذير: البيانات المحلية ({len(df):,}) أقل من الحد الأدنى ({MIN_TRAINING_SAMPLES:,})"
    )
    print("   يجب استخدام بيانات تدريب كافية!")
else:
    print("\n✅ البيانات المحلية كافية للاختبار")

predictions = []
actuals = []

scaler = ModelRegistry.get_model("calorie_scaler")
model = ModelRegistry.get_model("calorie")
features = ModelRegistry.get_model("calorie_features")

for idx, row in df.iterrows():
    try:
        gender = row["Gender"]
        if isinstance(gender, str):
            gender = 0 if gender.strip().lower() == "male" else 1

        input_data = prepare_calorie_model_input(
            gender=int(gender),
            age=float(row["Age"]),
            height_cm=float(row["Height"]),
            weight_kg=float(row["Weight"]),
            duration_min=float(row["Duration"]),
            heart_rate=float(row["Heart_Rate"]),
        )

        input_scaled = scaler.transform(input_data[features])
        pred = float(model.predict(input_scaled)[0])

        predictions.append(pred)
        actuals.append(float(row["Calories"]))
    except Exception as e:
        print(f"⚠️  صف {idx}: {str(e)}")
        continue

# حساب المقاييس
predictions = np.array(predictions)
actuals = np.array(actuals)

mae = mean_absolute_error(actuals, predictions)
mse = mean_squared_error(actuals, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(actuals, predictions)

print(f"\n📈 النتائج على {len(predictions)} صفوف:")
print(f"   R² Score:           {r2:.4f} ({r2*100:.2f}%)")
print(f"   MAE (Mean Absolute Error):  {mae:.2f} kcal")
print(f"   RMSE (Root Mean Square):    {rmse:.2f} kcal")
print(f"   MSE:                {mse:.2f}")

print("\n📊 إحصائيات المتنبأ به:")
print(f"   الحد الأدنى:        {predictions.min():.0f} kcal")
print(f"   الحد الأقصى:        {predictions.max():.0f} kcal")
print(f"   المتوسط:           {predictions.mean():.0f} kcal")

print("\n📊 إحصائيات القيمة الفعلية:")
print(f"   الحد الأدنى:        {actuals.min():.0f} kcal")
print(f"   الحد الأقصى:        {actuals.max():.0f} kcal")
print(f"   المتوسط:           {actuals.mean():.0f} kcal")

print("\n" + "=" * 60)
print("✅ التقييم:")
if r2 >= 0.8:
    print("   موديل ممتاز جداً! 🎯")
elif r2 >= 0.6:
    print("   موديل جيد 👍")
elif r2 >= 0.4:
    print("   موديل متوسط ⚠️")
else:
    print("   موديل ضعيف، يحتاج تحسين ❌")
print("=" * 60)

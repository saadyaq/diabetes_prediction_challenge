import json

notebook = {
    "cells": [],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"}
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

def add_markdown_cell(text):
    notebook["cells"].append({"cell_type": "markdown", "metadata": {}, "source": [text]})

def add_code_cell(code):
    notebook["cells"].append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [code]})

add_markdown_cell("# Diabetes Prediction - Medical Feature Engineering\n\n**Stuck at 62%** - Need domain knowledge features")

add_code_cell("""import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')""")

add_markdown_cell("# 1. Load Data")
add_code_cell("""df_train = pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")

test_ids = df_test['id']
target = df_train['diagnosed_diabetes']

df_train = df_train.drop(['id', 'diagnosed_diabetes'], axis=1)
df_test = df_test.drop(['id'], axis=1)""")

add_markdown_cell("# 2. Medical Domain Features")
add_code_cell("""def create_medical_features(df):
    df = df.copy()

    # Metabolic Syndrome indicators
    df['metabolic_syndrome'] = (
        (df['bmi'] >= 30) &
        (df['waist_to_hip_ratio'] > 0.9) &
        (df['triglycerides'] > 150)
    ).astype(int)

    # Cardiovascular risk
    df['cvd_risk_score'] = (
        df['age'] * 0.01 +
        df['systolic_bp'] * 0.005 +
        df['cholesterol_total'] * 0.002 +
        df['bmi'] * 0.05 +
        df['hypertension_history'] * 0.2
    )

    # Lipid ratios (strong diabetes indicators)
    df['tg_hdl_ratio'] = df['triglycerides'] / (df['hdl_cholesterol'] + 1)
    df['tc_hdl_ratio'] = df['cholesterol_total'] / (df['hdl_cholesterol'] + 1)
    df['ldl_hdl_ratio'] = df['ldl_cholesterol'] / (df['hdl_cholesterol'] + 1)

    # High-risk BMI
    df['obesity_severity'] = np.where(df['bmi'] >= 35, 2,
                              np.where(df['bmi'] >= 30, 1, 0))

    # Age risk groups
    df['age_risk'] = np.where(df['age'] >= 60, 2,
                      np.where(df['age'] >= 45, 1, 0))

    # Combined family history + age
    df['genetic_age_risk'] = df['family_history_diabetes'] * df['age_risk']

    # Blood pressure categories
    df['bp_category'] = np.where(df['systolic_bp'] >= 140, 2,
                        np.where(df['systolic_bp'] >= 130, 1, 0))

    # Unhealthy lifestyle score
    df['poor_lifestyle'] = (
        (df['physical_activity_minutes_per_week'] < 50) +
        (df['diet_score'] < 5) +
        (df['sleep_hours_per_day'] < 6) +
        (df['screen_time_hours_per_day'] > 8) +
        (df['alcohol_consumption_per_week'] > 3)
    )

    return df

df_train = create_medical_features(df_train)
df_test = create_medical_features(df_test)

print(f"Features created: {df_train.shape[1]}")""")

add_markdown_cell("# 3. Encode Categorical")
add_code_cell("""categorical = df_train.select_dtypes(include=['object']).columns.tolist()
df_train = pd.get_dummies(df_train, columns=categorical, drop_first=True)
df_test = pd.get_dummies(df_test, columns=categorical, drop_first=True)

df_train, df_test = df_train.align(df_test, join='left', axis=1, fill_value=0)
print(f"Final features: {df_train.shape[1]}")""")

add_markdown_cell("# 4. Train Single Best Model")
add_code_cell("""# Single XGBoost with optimal settings
model = xgb.XGBClassifier(
    n_estimators=1000,
    max_depth=6,
    learning_rate=0.01,
    min_child_weight=3,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0.1,
    reg_alpha=0.5,
    reg_lambda=2.0,
    scale_pos_weight=0.6,  # Adjust for class imbalance
    random_state=42,
    n_jobs=-1
)

model.fit(df_train, target, verbose=False)
print("Model trained!")""")

add_markdown_cell("# 5. Generate Submission")
add_code_cell("""y_pred = model.predict(df_test)

submission = pd.DataFrame({
    'id': test_ids,
    'diagnosed_diabetes': y_pred
})

submission.to_csv('submission.csv', index=False)
print("Submission created!")
print(f"Prediction distribution:\\n{submission['diagnosed_diabetes'].value_counts(normalize=True)}")""")

with open('main.ipynb', 'w') as f:
    json.dump(notebook, f, indent=2)

print("Medical features notebook created!")

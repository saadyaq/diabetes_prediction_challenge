import json

# Create simplified notebook focusing on what works
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
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [text]
    })

def add_code_cell(code):
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [code]
    })

# Imports
add_markdown_cell("# Diabetes Prediction - Simplified Approach\n\nGoal: Beat 67.83% baseline and reach 70%+")

add_code_cell("""import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))""")

# Data Loading
add_markdown_cell("# 1. Data Loading")
add_code_cell("""df_train = pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")

print(f"Train shape: {df_train.shape}")
print(f"Test shape: {df_test.shape}")""")

# Preprocessing
add_markdown_cell("# 2. Preprocessing - Keep It Simple")
add_code_cell("""# One-hot encode categorical features
categorical_features = df_train.select_dtypes(include=['object']).columns.tolist()
df_train = pd.get_dummies(df_train, columns=categorical_features, drop_first=True)
df_test = pd.get_dummies(df_test, columns=categorical_features, drop_first=True)

# Align train and test
df_train, df_test = df_train.align(df_test, join='left', axis=1, fill_value=0)

print(f"After encoding - Train: {df_train.shape}, Test: {df_test.shape}")""")

# Train/Val Split
add_markdown_cell("# 3. Train/Validation Split")
add_code_cell("""target = "diagnosed_diabetes"
X = df_train.drop(columns=[target, "id"], errors="ignore")
y = df_train[target]

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_test = df_test[X.columns]

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")""")

# XGBoost Model
add_markdown_cell("# 4. XGBoost Model - Optimized")
add_code_cell("""model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=7,
    learning_rate=0.05,
    min_child_weight=2,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0.1,
    reg_alpha=0.1,
    reg_lambda=1.5,
    random_state=42,
    eval_metric='logloss',
    n_jobs=-1
)

# Train
model.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    verbose=50
)

# Validate
y_pred_val = model.predict(X_val)
val_acc = accuracy_score(y_val, y_pred_val)

print(f"\\n{'='*50}")
print(f"Validation Accuracy: {val_acc:.4f}")
print(f"{'='*50}")
print(f"\\nClassification Report:")
print(classification_report(y_val, y_pred_val))""")

# Test Predictions
add_markdown_cell("# 5. Generate Submission")
add_code_cell("""# Predict on test
y_test_pred = model.predict(X_test)

# Create submission
submission = pd.DataFrame({
    'id': df_test['id'],
    'diagnosed_diabetes': y_test_pred
})

submission.to_csv('submission.csv', index=False)
print("Submission created!")
print(f"Predictions: {submission['diagnosed_diabetes'].value_counts()}")""")

# Save
with open('main.ipynb', 'w') as f:
    json.dump(notebook, f, indent=2)

print("Simplified notebook created!")

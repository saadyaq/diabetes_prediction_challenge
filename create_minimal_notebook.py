import json

# Create minimal notebook focusing on what matters
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

# Title
add_markdown_cell("# Diabetes Prediction - Minimal Approach\n\n**Problem:** Complex models stuck at 62%\n\n**New Strategy:** Focus on top features only + single strong model")

# Imports
add_code_cell("""import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')""")

# Load Data
add_markdown_cell("# 1. Load Data")
add_code_cell("""df_train = pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")

print(f"Train: {df_train.shape}, Test: {df_test.shape}")""")

# Minimal Preprocessing
add_markdown_cell("# 2. Minimal Feature Engineering")
add_code_cell("""# Save IDs
test_ids = df_test['id'].copy()

# Target
y = df_train['diagnosed_diabetes'].copy()

# Drop ID and target
X_train = df_train.drop(['id', 'diagnosed_diabetes'], axis=1)
X_test = df_test.drop(['id'], axis=1)

# Only encode categorical - no other feature engineering
categorical = X_train.select_dtypes(include=['object']).columns.tolist()
X_train = pd.get_dummies(X_train, columns=categorical, drop_first=True)
X_test = pd.get_dummies(X_test, columns=categorical, drop_first=True)

# Align
X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

print(f"Features: {X_train.shape[1]}")""")

# Feature Importance Analysis
add_markdown_cell("# 3. Find Top Features")
add_code_cell("""# Train quick model to find important features
X_tr, X_val, y_tr, y_val = train_test_split(X_train, y, test_size=0.2, random_state=42, stratify=y)

quick_model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    random_state=42,
    n_jobs=-1
)
quick_model.fit(X_tr, y_tr)

# Get feature importance
importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': quick_model.feature_importances_
}).sort_values('importance', ascending=False)

print("Top 15 features:")
print(importance.head(15))

# Select top 20 features only
top_features = importance.head(20)['feature'].tolist()
print(f"\\nUsing top {len(top_features)} features")""")

# Train on Top Features Only
add_markdown_cell("# 4. Train XGBoost on Top Features Only")
add_code_cell("""# Use only top features
X_train_top = X_train[top_features]
X_test_top = X_test[top_features]

# Train with strong regularization for generalization
model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    min_child_weight=5,
    subsample=0.7,
    colsample_bytree=0.7,
    gamma=0.2,
    reg_alpha=1.0,
    reg_lambda=3.0,
    scale_pos_weight=1,
    random_state=42,
    n_jobs=-1
)

# Train on full training data
model.fit(X_train_top, y, verbose=False)

# Quick validation check
X_tr_top, X_val_top, y_tr, y_val = train_test_split(
    X_train_top, y, test_size=0.2, random_state=42, stratify=y
)
model.fit(X_tr_top, y_tr)
y_pred = model.predict(X_val_top)
val_acc = accuracy_score(y_val, y_pred)

print(f"Validation accuracy: {val_acc:.4f}")
print(f"(This is just for reference, not indicative of test score)")""")

# Retrain on Full Data
add_markdown_cell("# 5. Retrain on Full Data")
add_code_cell("""# Retrain on ALL training data
model.fit(X_train_top, y, verbose=False)
print("Model trained on full training set")""")

# Generate Submission
add_markdown_cell("# 6. Generate Submission")
add_code_cell("""# Predict on test
y_pred_test = model.predict(X_test_top)

# Create submission
submission = pd.DataFrame({
    'id': test_ids,
    'diagnosed_diabetes': y_pred_test
})

submission.to_csv('submission.csv', index=False)
print("Submission created!")
print(f"\\nPrediction distribution:")
print(submission['diagnosed_diabetes'].value_counts(normalize=True))""")

# Alternative: Tune threshold
add_markdown_cell("# 7. Alternative: Try Different Threshold")
add_code_cell("""# Get probabilities instead
y_proba_test = model.predict_proba(X_test_top)[:, 1]

# Try threshold 0.48 (slightly favor positive class)
y_pred_048 = (y_proba_test > 0.48).astype(int)

submission_048 = pd.DataFrame({
    'id': test_ids,
    'diagnosed_diabetes': y_pred_048
})

submission_048.to_csv('submission_048.csv', index=False)
print("Alternative submission (threshold=0.48) created!")
print(f"Distribution:")
print(submission_048['diagnosed_diabetes'].value_counts(normalize=True))""")

# Save
with open('main.ipynb', 'w') as f:
    json.dump(notebook, f, indent=2)

print("Minimal notebook created!")

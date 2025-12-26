import json

# Create optimized ensemble notebook
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
add_markdown_cell("# Diabetes Prediction - Optimized Ensemble\n\n**Best so far:** 62.48% → **Target:** 71%\n\n**Strategy:** Better model diversity + optimized blending")

# Imports
add_code_cell("""import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.calibration import CalibratedClassifierCV
import warnings
warnings.filterwarnings('ignore')

print("Libraries loaded!")""")

# Load Data
add_markdown_cell("# 1. Load and Prepare Data")
add_code_cell("""df_train = pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")

# Save IDs
test_ids = df_test['id'].copy()
target = df_train['diagnosed_diabetes'].copy()

# Drop ID and target
df_train = df_train.drop(['id', 'diagnosed_diabetes'], axis=1)
df_test = df_test.drop(['id'], axis=1)

# Encode categorical
categorical = df_train.select_dtypes(include=['object']).columns.tolist()
df_train = pd.get_dummies(df_train, columns=categorical, drop_first=False)  # Keep all dummies
df_test = pd.get_dummies(df_test, columns=categorical, drop_first=False)

# Align
df_train, df_test = df_train.align(df_test, join='left', axis=1, fill_value=0)

# Add only proven features
df_train['age_bmi'] = df_train['age'] * df_train['bmi']
df_train['family_age'] = df_train['family_history_diabetes'] * df_train['age']

df_test['age_bmi'] = df_test['age'] * df_test['bmi']
df_test['family_age'] = df_test['family_history_diabetes'] * df_test['age']

print(f"Train: {df_train.shape}, Test: {df_test.shape}")
print(f"Target balance: {target.value_counts(normalize=True)}")""")

# CV Setup
add_markdown_cell("# 2. Cross-Validation with Diverse Models")
add_code_cell("""n_folds = 7  # More folds = more robust
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

# OOF predictions for 4 different models
oof_xgb = np.zeros(len(df_train))
oof_lgb = np.zeros(len(df_train))
oof_cat = np.zeros(len(df_train))
oof_rf = np.zeros(len(df_train))

# Test predictions
test_xgb = np.zeros(len(df_test))
test_lgb = np.zeros(len(df_test))
test_cat = np.zeros(len(df_test))
test_rf = np.zeros(len(df_test))

print(f"Using {n_folds}-fold CV")""")

# Train Models
add_markdown_cell("## 2.1 Train All Models")
add_code_cell("""for fold, (train_idx, val_idx) in enumerate(skf.split(df_train, target)):
    print(f"\\nFold {fold + 1}/{n_folds}")

    X_tr, X_val = df_train.iloc[train_idx], df_train.iloc[val_idx]
    y_tr, y_val = target.iloc[train_idx], target.iloc[val_idx]

    # XGBoost - Deeper trees, less regularization
    xgb_model = xgb.XGBClassifier(
        n_estimators=400,
        max_depth=8,
        learning_rate=0.02,
        min_child_weight=1,
        subsample=0.85,
        colsample_bytree=0.85,
        gamma=0,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42 + fold,
        n_jobs=-1
    )
    xgb_model.fit(X_tr, y_tr, verbose=False)
    oof_xgb[val_idx] = xgb_model.predict_proba(X_val)[:, 1]
    test_xgb += xgb_model.predict_proba(df_test)[:, 1] / n_folds

    # LightGBM - More leaves
    lgb_model = lgb.LGBMClassifier(
        n_estimators=400,
        max_depth=8,
        learning_rate=0.02,
        num_leaves=64,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42 + fold,
        n_jobs=-1,
        verbose=-1
    )
    lgb_model.fit(X_tr, y_tr)
    oof_lgb[val_idx] = lgb_model.predict_proba(X_val)[:, 1]
    test_lgb += lgb_model.predict_proba(df_test)[:, 1] / n_folds

    # CatBoost - Different depth
    cat_model = CatBoostClassifier(
        iterations=400,
        depth=7,
        learning_rate=0.02,
        l2_leaf_reg=3,
        random_state=42 + fold,
        verbose=False
    )
    cat_model.fit(X_tr, y_tr)
    oof_cat[val_idx] = cat_model.predict_proba(X_val)[:, 1]
    test_cat += cat_model.predict_proba(df_test)[:, 1] / n_folds

    # Random Forest - Different algorithm entirely
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_split=10,
        min_samples_leaf=4,
        max_features='sqrt',
        random_state=42 + fold,
        n_jobs=-1
    )
    rf_model.fit(X_tr, y_tr)
    oof_rf[val_idx] = rf_model.predict_proba(X_val)[:, 1]
    test_rf += rf_model.predict_proba(df_test)[:, 1] / n_folds

print("\\nAll models trained!")""")

# Model Comparison
add_markdown_cell("## 2.2 Individual Model Performance")
add_code_cell("""acc_xgb = accuracy_score(target, (oof_xgb > 0.5).astype(int))
acc_lgb = accuracy_score(target, (oof_lgb > 0.5).astype(int))
acc_cat = accuracy_score(target, (oof_cat > 0.5).astype(int))
acc_rf = accuracy_score(target, (oof_rf > 0.5).astype(int))

print(f"{'='*50}")
print("OOF Accuracy:")
print(f"{'='*50}")
print(f"XGBoost:      {acc_xgb:.4f}")
print(f"LightGBM:     {acc_lgb:.4f}")
print(f"CatBoost:     {acc_cat:.4f}")
print(f"RandomForest: {acc_rf:.4f}")
print(f"{'='*50}")""")

# Optimized Blending
add_markdown_cell("# 3. Optimized Blending (Grid Search)")
add_code_cell("""from itertools import product

# Try different weight combinations
best_score = 0
best_weights = None

# Grid search over weights (coarse)
weight_options = [0.1, 0.2, 0.25, 0.3, 0.35, 0.4]

print("Searching for best ensemble weights...")
for w1 in weight_options:
    for w2 in weight_options:
        for w3 in weight_options:
            w4 = 1.0 - w1 - w2 - w3
            if w4 < 0 or w4 > 0.5:
                continue

            blend = w1 * oof_xgb + w2 * oof_lgb + w3 * oof_cat + w4 * oof_rf
            acc = accuracy_score(target, (blend > 0.5).astype(int))

            if acc > best_score:
                best_score = acc
                best_weights = (w1, w2, w3, w4)

print(f"\\nBest weights: XGB={best_weights[0]:.2f}, LGB={best_weights[1]:.2f}, CAT={best_weights[2]:.2f}, RF={best_weights[3]:.2f}")
print(f"Best OOF accuracy: {best_score:.4f}")""")

# Final Predictions
add_markdown_cell("# 4. Generate Final Predictions")
add_code_cell("""# Apply best weights to test predictions
final_test_proba = (
    best_weights[0] * test_xgb +
    best_weights[1] * test_lgb +
    best_weights[2] * test_cat +
    best_weights[3] * test_rf
)

# Try multiple thresholds
for thresh in [0.47, 0.48, 0.49, 0.5, 0.51, 0.52, 0.53]:
    preds = (final_test_proba > thresh).astype(int)
    submission = pd.DataFrame({
        'id': test_ids,
        'diagnosed_diabetes': preds
    })
    filename = f'submission_{int(thresh*100)}.csv'
    submission.to_csv(filename, index=False)
    print(f"{filename}: {preds.mean():.3f} positive rate")

print("\\nAll submissions created!")
print("Try submission_50.csv first, then others if needed")""")

# Save
with open('main.ipynb', 'w') as f:
    json.dump(notebook, f, indent=2)

print("Optimized ensemble notebook created!")

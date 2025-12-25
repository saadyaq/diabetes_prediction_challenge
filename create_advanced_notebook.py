import json

# Create advanced notebook targeting 71% test accuracy
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
add_markdown_cell("# Diabetes Prediction - Target: 71% Test Accuracy\n\n**Baseline:** 61% → **Current:** 62% → **Goal:** 71%\n\n**Strategy:** Focus on test generalization, not just validation accuracy")

# Imports
add_code_cell("""import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
import warnings
warnings.filterwarnings('ignore')

print("Libraries loaded successfully!")""")

# Data Loading
add_markdown_cell("# 1. Load Data")
add_code_cell("""df_train = pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")

print(f"Train: {df_train.shape}")
print(f"Test: {df_test.shape}")
print(f"\\nTarget distribution:")
print(df_train['diagnosed_diabetes'].value_counts(normalize=True))""")

# Feature Engineering
add_markdown_cell("# 2. Feature Engineering - Focus on Robust Features")
add_code_cell("""# Save IDs and target
train_ids = df_train['id']
test_ids = df_test['id']
target = df_train['diagnosed_diabetes']

# Drop ID from features
df_train = df_train.drop(['id', 'diagnosed_diabetes'], axis=1)
df_test = df_test.drop(['id'], axis=1)

# One-hot encode categorical
categorical_features = df_train.select_dtypes(include=['object']).columns.tolist()
df_train = pd.get_dummies(df_train, columns=categorical_features, drop_first=True)
df_test = pd.get_dummies(df_test, columns=categorical_features, drop_first=True)

# Align columns
df_train, df_test = df_train.align(df_test, join='left', axis=1, fill_value=0)

# Add interaction features (proven to help generalization)
df_train['age_bmi'] = df_train['age'] * df_train['bmi']
df_train['bp_ratio'] = df_train['systolic_bp'] / (df_train['diastolic_bp'] + 1)
df_train['cholesterol_hdl_ratio'] = df_train['cholesterol_total'] / (df_train['hdl_cholesterol'] + 1)

df_test['age_bmi'] = df_test['age'] * df_test['bmi']
df_test['bp_ratio'] = df_test['systolic_bp'] / (df_test['diastolic_bp'] + 1)
df_test['cholesterol_hdl_ratio'] = df_test['cholesterol_total'] / (df_test['hdl_cholesterol'] + 1)

print(f"Features after engineering: {df_train.shape[1]}")
print(f"Train: {df_train.shape}, Test: {df_test.shape}")""")

# Cross-Validation Setup
add_markdown_cell("# 3. Cross-Validation Strategy")
add_code_cell("""# Use 5-fold CV for robust validation
n_folds = 5
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

# Store OOF (out-of-fold) predictions
oof_preds_xgb = np.zeros(len(df_train))
oof_preds_lgb = np.zeros(len(df_train))
oof_preds_cat = np.zeros(len(df_train))

# Store test predictions
test_preds_xgb = np.zeros(len(df_test))
test_preds_lgb = np.zeros(len(df_test))
test_preds_cat = np.zeros(len(df_test))

print(f"Using {n_folds}-fold cross-validation")
print(f"Training on {len(df_train)} samples")""")

# XGBoost Training
add_markdown_cell("## 3.1 XGBoost with CV")
add_code_cell("""for fold, (train_idx, val_idx) in enumerate(skf.split(df_train, target)):
    print(f"\\n{'='*50}")
    print(f"Fold {fold + 1}/{n_folds}")
    print(f"{'='*50}")

    X_train, X_val = df_train.iloc[train_idx], df_train.iloc[val_idx]
    y_train, y_val = target.iloc[train_idx], target.iloc[val_idx]

    # XGBoost
    xgb_model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.03,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0.1,
        reg_alpha=0.5,
        reg_lambda=2.0,
        random_state=42,
        eval_metric='logloss',
        n_jobs=-1
    )

    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    # OOF predictions
    oof_preds_xgb[val_idx] = xgb_model.predict_proba(X_val)[:, 1]

    # Test predictions
    test_preds_xgb += xgb_model.predict_proba(df_test)[:, 1] / n_folds

    # Validation score
    val_acc = accuracy_score(y_val, (oof_preds_xgb[val_idx] > 0.5).astype(int))
    print(f"XGBoost Fold {fold + 1} Accuracy: {val_acc:.4f}")

# Overall OOF score
oof_acc_xgb = accuracy_score(target, (oof_preds_xgb > 0.5).astype(int))
print(f"\\nXGBoost OOF Accuracy: {oof_acc_xgb:.4f}")""")

# LightGBM Training
add_markdown_cell("## 3.2 LightGBM with CV")
add_code_cell("""for fold, (train_idx, val_idx) in enumerate(skf.split(df_train, target)):
    print(f"Fold {fold + 1}/{n_folds}")

    X_train, X_val = df_train.iloc[train_idx], df_train.iloc[val_idx]
    y_train, y_val = target.iloc[train_idx], target.iloc[val_idx]

    # LightGBM
    lgb_model = lgb.LGBMClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.03,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.5,
        reg_lambda=2.0,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )

    lgb_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )

    # OOF predictions
    oof_preds_lgb[val_idx] = lgb_model.predict_proba(X_val)[:, 1]

    # Test predictions
    test_preds_lgb += lgb_model.predict_proba(df_test)[:, 1] / n_folds

    val_acc = accuracy_score(y_val, (oof_preds_lgb[val_idx] > 0.5).astype(int))
    print(f"LightGBM Fold {fold + 1} Accuracy: {val_acc:.4f}")

oof_acc_lgb = accuracy_score(target, (oof_preds_lgb > 0.5).astype(int))
print(f"\\nLightGBM OOF Accuracy: {oof_acc_lgb:.4f}")""")

# CatBoost Training
add_markdown_cell("## 3.3 CatBoost with CV")
add_code_cell("""for fold, (train_idx, val_idx) in enumerate(skf.split(df_train, target)):
    print(f"Fold {fold + 1}/{n_folds}")

    X_train, X_val = df_train.iloc[train_idx], df_train.iloc[val_idx]
    y_train, y_val = target.iloc[train_idx], target.iloc[val_idx]

    # CatBoost
    cat_model = CatBoostClassifier(
        iterations=500,
        depth=6,
        learning_rate=0.03,
        l2_leaf_reg=5,
        random_state=42,
        verbose=False
    )

    cat_model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)

    # OOF predictions
    oof_preds_cat[val_idx] = cat_model.predict_proba(X_val)[:, 1]

    # Test predictions
    test_preds_cat += cat_model.predict_proba(df_test)[:, 1] / n_folds

    val_acc = accuracy_score(y_val, (oof_preds_cat[val_idx] > 0.5).astype(int))
    print(f"CatBoost Fold {fold + 1} Accuracy: {val_acc:.4f}")

oof_acc_cat = accuracy_score(target, (oof_preds_cat > 0.5).astype(int))
print(f"\\nCatBoost OOF Accuracy: {oof_acc_cat:.4f}")""")

# Stacking
add_markdown_cell("# 4. Stacking - Train Meta-Model on OOF Predictions")
add_code_cell("""# Create meta-features from OOF predictions
meta_train = np.column_stack([oof_preds_xgb, oof_preds_lgb, oof_preds_cat])
meta_test = np.column_stack([test_preds_xgb, test_preds_lgb, test_preds_cat])

# Train meta-model (Logistic Regression for simplicity and regularization)
meta_model = LogisticRegression(random_state=42, max_iter=1000)
meta_model.fit(meta_train, target)

# Meta-model predictions
final_oof_preds = meta_model.predict_proba(meta_train)[:, 1]
final_test_preds = meta_model.predict_proba(meta_test)[:, 1]

# OOF accuracy
oof_acc_stacking = accuracy_score(target, (final_oof_preds > 0.5).astype(int))

print(f"\\n{'='*60}")
print("MODEL COMPARISON (OOF Accuracy):")
print(f"{'='*60}")
print(f"XGBoost:  {oof_acc_xgb:.4f}")
print(f"LightGBM: {oof_acc_lgb:.4f}")
print(f"CatBoost: {oof_acc_cat:.4f}")
print(f"Stacking: {oof_acc_stacking:.4f}")
print(f"{'='*60}")""")

# Threshold Optimization
add_markdown_cell("# 5. Threshold Optimization")
add_code_cell("""# Try different thresholds on OOF predictions
best_threshold = 0.5
best_acc = 0

for threshold in np.arange(0.3, 0.7, 0.01):
    acc = accuracy_score(target, (final_oof_preds > threshold).astype(int))
    if acc > best_acc:
        best_acc = acc
        best_threshold = threshold

print(f"Best threshold: {best_threshold:.3f}")
print(f"Best OOF accuracy: {best_acc:.4f}")
print(f"Improvement: +{(best_acc - oof_acc_stacking)*100:.2f}%")""")

# Final Submission
add_markdown_cell("# 6. Generate Final Submission")
add_code_cell("""# Use optimized threshold
final_predictions = (final_test_preds > best_threshold).astype(int)

# Create submission
submission = pd.DataFrame({
    'id': test_ids,
    'diagnosed_diabetes': final_predictions
})

submission.to_csv('submission.csv', index=False)

print("\\nSubmission created!")
print(f"Shape: {submission.shape}")
print(f"\\nPrediction distribution:")
print(submission['diagnosed_diabetes'].value_counts(normalize=True))
print(f"\\nExpected test accuracy: ~{best_acc:.1%}")
print(f"Target: 71%")""")

# Save
with open('main.ipynb', 'w') as f:
    json.dump(notebook, f, indent=2)

print("Advanced notebook created!")

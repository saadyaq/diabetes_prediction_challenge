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

add_markdown_cell("# Diabetes Prediction - Final Push to 71%\n\n**Current:** 64% → **Target:** 71% (Need +7%)\n\n**Strategy:** Medical features + Multi-model ensemble + Threshold tuning")

add_code_cell("""import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

print("Libraries loaded!")""")

add_markdown_cell("# 1. Load Data")
add_code_cell("""df_train = pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")

test_ids = df_test['id']
target = df_train['diagnosed_diabetes']

df_train = df_train.drop(['id', 'diagnosed_diabetes'], axis=1)
df_test = df_test.drop(['id'], axis=1)

print(f"Train: {df_train.shape}, Test: {df_test.shape}")""")

add_markdown_cell("# 2. Enhanced Medical Features (What Got Us to 64%)")
add_code_cell("""def create_medical_features(df):
    df = df.copy()

    # Proven features from 64% model
    df['metabolic_syndrome'] = (
        (df['bmi'] >= 30) &
        (df['waist_to_hip_ratio'] > 0.9) &
        (df['triglycerides'] > 150)
    ).astype(int)

    df['cvd_risk_score'] = (
        df['age'] * 0.01 +
        df['systolic_bp'] * 0.005 +
        df['cholesterol_total'] * 0.002 +
        df['bmi'] * 0.05 +
        df['hypertension_history'] * 0.2
    )

    df['tg_hdl_ratio'] = df['triglycerides'] / (df['hdl_cholesterol'] + 1)
    df['tc_hdl_ratio'] = df['cholesterol_total'] / (df['hdl_cholesterol'] + 1)
    df['ldl_hdl_ratio'] = df['ldl_cholesterol'] / (df['hdl_cholesterol'] + 1)

    df['obesity_severity'] = np.where(df['bmi'] >= 35, 2,
                              np.where(df['bmi'] >= 30, 1, 0))

    df['age_risk'] = np.where(df['age'] >= 60, 2,
                      np.where(df['age'] >= 45, 1, 0))

    df['genetic_age_risk'] = df['family_history_diabetes'] * df['age_risk']

    df['bp_category'] = np.where(df['systolic_bp'] >= 140, 2,
                        np.where(df['systolic_bp'] >= 130, 1, 0))

    df['poor_lifestyle'] = (
        (df['physical_activity_minutes_per_week'] < 50) +
        (df['diet_score'] < 5) +
        (df['sleep_hours_per_day'] < 6) +
        (df['screen_time_hours_per_day'] > 8) +
        (df['alcohol_consumption_per_week'] > 3)
    )

    # NEW: Additional medical features
    df['insulin_resistance_proxy'] = (
        df['tg_hdl_ratio'] * df['bmi'] * 0.01
    )

    df['diabetes_risk_score'] = (
        df['age'] * 0.02 +
        df['bmi'] * 0.1 +
        df['family_history_diabetes'] * 2 +
        df['tg_hdl_ratio'] * 0.5 +
        df['metabolic_syndrome'] * 1.5
    )

    df['high_risk_combo'] = (
        (df['family_history_diabetes'] == 1) &
        (df['age'] >= 45) &
        (df['bmi'] >= 30)
    ).astype(int)

    return df

df_train = create_medical_features(df_train)
df_test = create_medical_features(df_test)

print(f"Medical features created: {df_train.shape[1]}")""")

add_markdown_cell("# 3. Encode Categorical")
add_code_cell("""categorical = df_train.select_dtypes(include=['object']).columns.tolist()
df_train = pd.get_dummies(df_train, columns=categorical, drop_first=True)
df_test = pd.get_dummies(df_test, columns=categorical, drop_first=True)

df_train, df_test = df_train.align(df_test, join='left', axis=1, fill_value=0)
print(f"Total features: {df_train.shape[1]}")""")

add_markdown_cell("# 4. Train Ensemble with 5-Fold CV")
add_code_cell("""n_folds = 5
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

oof_xgb = np.zeros(len(df_train))
oof_lgb = np.zeros(len(df_train))
oof_cat = np.zeros(len(df_train))

test_xgb = np.zeros(len(df_test))
test_lgb = np.zeros(len(df_test))
test_cat = np.zeros(len(df_test))

for fold, (train_idx, val_idx) in enumerate(skf.split(df_train, target)):
    print(f"Fold {fold + 1}/{n_folds}")

    X_tr, X_val = df_train.iloc[train_idx], df_train.iloc[val_idx]
    y_tr, y_val = target.iloc[train_idx], target.iloc[val_idx]

    # XGBoost - Tuned for medical features
    xgb_model = xgb.XGBClassifier(
        n_estimators=1500,
        max_depth=7,
        learning_rate=0.008,
        min_child_weight=2,
        subsample=0.85,
        colsample_bytree=0.85,
        gamma=0.05,
        reg_alpha=0.3,
        reg_lambda=1.5,
        scale_pos_weight=0.62,
        random_state=42 + fold,
        n_jobs=-1
    )
    xgb_model.fit(X_tr, y_tr, verbose=False)
    oof_xgb[val_idx] = xgb_model.predict_proba(X_val)[:, 1]
    test_xgb += xgb_model.predict_proba(df_test)[:, 1] / n_folds

    # LightGBM
    lgb_model = lgb.LGBMClassifier(
        n_estimators=1500,
        max_depth=7,
        learning_rate=0.008,
        num_leaves=50,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.3,
        reg_lambda=1.5,
        random_state=42 + fold,
        n_jobs=-1,
        verbose=-1
    )
    lgb_model.fit(X_tr, y_tr)
    oof_lgb[val_idx] = lgb_model.predict_proba(X_val)[:, 1]
    test_lgb += lgb_model.predict_proba(df_test)[:, 1] / n_folds

    # CatBoost
    cat_model = CatBoostClassifier(
        iterations=1500,
        depth=7,
        learning_rate=0.008,
        l2_leaf_reg=3,
        random_state=42 + fold,
        verbose=False
    )
    cat_model.fit(X_tr, y_tr)
    oof_cat[val_idx] = cat_model.predict_proba(X_val)[:, 1]
    test_cat += cat_model.predict_proba(df_test)[:, 1] / n_folds

print("\\nAll models trained!")""")

add_markdown_cell("# 5. Find Optimal Blend")
add_code_cell("""best_score = 0
best_weights = (0.33, 0.33, 0.34)

for w1 in np.arange(0.2, 0.5, 0.05):
    for w2 in np.arange(0.2, 0.5, 0.05):
        w3 = 1.0 - w1 - w2
        if w3 < 0.2 or w3 > 0.5:
            continue

        blend = w1 * oof_xgb + w2 * oof_lgb + w3 * oof_cat
        acc = accuracy_score(target, (blend > 0.5).astype(int))

        if acc > best_score:
            best_score = acc
            best_weights = (w1, w2, w3)

print(f"Best weights: XGB={best_weights[0]:.2f}, LGB={best_weights[1]:.2f}, CAT={best_weights[2]:.2f}")
print(f"OOF Accuracy: {best_score:.4f}")""")

add_markdown_cell("# 6. Threshold Optimization")
add_code_cell("""final_blend = (
    best_weights[0] * test_xgb +
    best_weights[1] * test_lgb +
    best_weights[2] * test_cat
)

# Generate multiple submissions with different thresholds
for thresh in [0.47, 0.48, 0.49, 0.50, 0.51, 0.52]:
    preds = (final_blend > thresh).astype(int)

    submission = pd.DataFrame({
        'id': test_ids,
        'diagnosed_diabetes': preds
    })

    filename = f'submission_{int(thresh*100)}.csv'
    submission.to_csv(filename, index=False)
    print(f"{filename}: {preds.sum()}/{len(preds)} positive ({preds.mean():.3f})")

print("\\nTry submission_50.csv first, then adjust threshold based on result!")""")

with open('main.ipynb', 'w') as f:
    json.dump(notebook, f, indent=2)

print("Final push notebook created!")

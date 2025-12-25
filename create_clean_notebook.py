import json

# Create a clean, well-organized notebook
notebook = {
    "cells": [],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0"
        }
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

# Cell 1: Imports
add_markdown_cell("# Diabetes Prediction - Improved Model\n\nTarget: Achieve 70%+ accuracy on leaderboard")

add_code_cell("""# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings('ignore')

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))""")

# Cell 2: Data Loading
add_markdown_cell("# 1. Data Loading")

add_code_cell("""df_train = pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")
df_sample = pd.read_csv("/kaggle/input/playground-series-s5e12/sample_submission.csv")

print(f"Train set shape: {df_train.shape}")
print(f"Test set shape: {df_test.shape}")
print(f"\\nFirst few rows:")
print(df_train.head())""")

# Cell 3: Basic EDA
add_markdown_cell("# 2. Exploratory Data Analysis")

add_code_cell("""# Check for missing values
print("Missing values in train:", df_train.isnull().sum().sum())
print("Missing values in test:", df_test.isnull().sum().sum())

# Target distribution
print("\\nTarget distribution:")
print(df_train['diagnosed_diabetes'].value_counts())

# Basic statistics
print("\\nBasic statistics:")
print(df_train.describe())""")

# Cell 4: Feature Engineering - Basic
add_markdown_cell("# 3. Feature Engineering")

add_code_cell("""# Identify feature types
numerical_features = df_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = df_train.select_dtypes(include=['object']).columns.tolist()

# Remove id and target from features
if 'diagnosed_diabetes' in numerical_features:
    numerical_features.remove('diagnosed_diabetes')
if 'id' in numerical_features:
    numerical_features.remove('id')

print(f"Numerical features: {len(numerical_features)}")
print(f"Categorical features: {len(categorical_features)}")

# One-hot encoding for categorical features
categorical_for_ohe = [col for col in categorical_features if df_train[col].nunique() > 2]
df_train = pd.get_dummies(df_train, columns=categorical_for_ohe, drop_first=True)
df_test = pd.get_dummies(df_test, columns=categorical_for_ohe, drop_first=True)

# Align train and test columns
df_train, df_test = df_train.align(df_test, join='left', axis=1, fill_value=0)

print(f"\\nAfter encoding - Train shape: {df_train.shape}")
print(f"After encoding - Test shape: {df_test.shape}")""")

# Cell 5: Train/Val Split
add_markdown_cell("# 4. Train/Validation Split")

add_code_cell("""target_col = "diagnosed_diabetes"
X = df_train.drop(columns=[target_col, "id"], errors="ignore")
y = df_train[target_col]

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

X_test = df_test[X_train.columns]

print(f"Train set: {X_train.shape}")
print(f"Validation set: {X_val.shape}")
print(f"Test set: {X_test.shape}")""")

# Cell 6: Feature Selection
add_markdown_cell("# 5. Feature Selection with Lasso")

add_code_cell("""# Scale features for Lasso
scaler_fs = StandardScaler()
X_train_scaled = scaler_fs.fit_transform(X_train)
X_val_scaled = scaler_fs.transform(X_val)
X_test_scaled = scaler_fs.transform(X_test)

# Lasso feature selection
lasso = LassoCV(cv=5, random_state=42)
lasso.fit(X_train_scaled, y_train)

lasso_coef = pd.DataFrame({
    'feature': X_train.columns,
    'coefficient': np.abs(lasso.coef_)
}).sort_values('coefficient', ascending=False)

selected_features = lasso_coef[lasso_coef['coefficient'] > 0]['feature'].tolist()

print(f"Selected features: {len(selected_features)}")
print(f"\\nTop 10 features:\\n{lasso_coef.head(10)}")

# Select features
X_train_selected = X_train[selected_features]
X_val_selected = X_val[selected_features]
X_test_selected = X_test[selected_features]""")

# Cell 7: Advanced Feature Engineering
add_markdown_cell("# 6. Advanced Feature Engineering")

add_code_cell("""def add_engineered_features(df_original, df_processed):
    \"\"\"Add domain-specific engineered features\"\"\"
    df_eng = df_processed.copy()

    # Cholesterol ratios (important diabetes indicators)
    df_eng['cholesterol_ratio'] = df_original['cholesterol_total'] / df_original['hdl_cholesterol']
    df_eng['ldl_hdl_ratio'] = df_original['ldl_cholesterol'] / df_original['hdl_cholesterol']
    df_eng['tg_hdl_ratio'] = df_original['triglycerides'] / df_original['hdl_cholesterol']

    # Blood pressure features
    df_eng['bp_diff'] = df_original['systolic_bp'] - df_original['diastolic_bp']
    df_eng['mean_arterial_pressure'] = (df_original['systolic_bp'] + 2 * df_original['diastolic_bp']) / 3

    # BMI features
    df_eng['bmi_squared'] = df_original['bmi'] ** 2
    df_eng['is_obese'] = (df_original['bmi'] >= 30).astype(int)
    df_eng['is_overweight'] = ((df_original['bmi'] >= 25) & (df_original['bmi'] < 30)).astype(int)

    # Age interactions
    df_eng['age_bmi_interaction'] = df_original['age'] * df_original['bmi']
    df_eng['age_squared'] = df_original['age'] ** 2

    # Lifestyle score
    df_eng['lifestyle_score'] = (
        df_original['diet_score'] +
        df_original['physical_activity_minutes_per_week'] / 100 -
        df_original['alcohol_consumption_per_week'] -
        df_original['screen_time_hours_per_day']
    )

    # Health risk score
    df_eng['health_risk_score'] = (
        df_original['bmi'] / 10 +
        df_original['systolic_bp'] / 40 +
        df_original['cholesterol_total'] / 50 +
        df_original['age'] / 20
    )

    # Waist to hip ratio category
    df_eng['high_waist_hip_ratio'] = (df_original['waist_to_hip_ratio'] > 0.9).astype(int)

    return df_eng

# Apply feature engineering
X_train_enhanced = add_engineered_features(df_train.loc[X_train_selected.index], X_train_selected)
X_val_enhanced = add_engineered_features(df_train.loc[X_val_selected.index], X_val_selected)
X_test_enhanced = add_engineered_features(df_test, X_test_selected)

print(f"Enhanced features shape: {X_train_enhanced.shape}")
print(f"Added {X_train_enhanced.shape[1] - X_train_selected.shape[1]} new features")""")

# Cell 8: XGBoost
add_markdown_cell("# 7. Model Training\n\n## 7.1 Optimized XGBoost")

add_code_cell("""xgb_model = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=8,
    learning_rate=0.05,
    min_child_weight=3,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0.1,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    eval_metric='logloss',
    n_jobs=-1
)

xgb_model.fit(
    X_train_enhanced,
    y_train,
    eval_set=[(X_val_enhanced, y_val)],
    verbose=50
)

y_pred_xgb = xgb_model.predict(X_val_enhanced)
y_pred_xgb_proba = xgb_model.predict_proba(X_val_enhanced)[:, 1]

xgb_accuracy = accuracy_score(y_val, y_pred_xgb)
print(f"\\n{'='*50}")
print(f"XGBoost Validation Accuracy: {xgb_accuracy:.4f}")
print(f"{'='*50}")""")

# Cell 9: LightGBM
add_markdown_cell("## 7.2 LightGBM")

add_code_cell("""lgb_model = lgb.LGBMClassifier(
    n_estimators=500,
    max_depth=8,
    learning_rate=0.05,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)

lgb_model.fit(
    X_train_enhanced,
    y_train,
    eval_set=[(X_val_enhanced, y_val)],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(50)]
)

y_pred_lgb = lgb_model.predict(X_val_enhanced)
y_pred_lgb_proba = lgb_model.predict_proba(X_val_enhanced)[:, 1]

lgb_accuracy = accuracy_score(y_val, y_pred_lgb)
print(f"\\n{'='*50}")
print(f"LightGBM Validation Accuracy: {lgb_accuracy:.4f}")
print(f"{'='*50}")""")

# Cell 10: CatBoost
add_markdown_cell("## 7.3 CatBoost")

add_code_cell("""cat_model = CatBoostClassifier(
    iterations=500,
    depth=8,
    learning_rate=0.05,
    l2_leaf_reg=3,
    random_state=42,
    verbose=50,
    task_type='CPU'
)

cat_model.fit(
    X_train_enhanced,
    y_train,
    eval_set=(X_val_enhanced, y_val),
    verbose=50
)

y_pred_cat = cat_model.predict(X_val_enhanced)
y_pred_cat_proba = cat_model.predict_proba(X_val_enhanced)[:, 1]

cat_accuracy = accuracy_score(y_val, y_pred_cat)
print(f"\\n{'='*50}")
print(f"CatBoost Validation Accuracy: {cat_accuracy:.4f}")
print(f"{'='*50}")""")

# Cell 11: Ensemble
add_markdown_cell("# 8. Ensemble - Weighted Average")

add_code_cell("""# Ensemble weights
weights = {
    'xgb': 0.35,
    'lgb': 0.35,
    'cat': 0.30
}

# Weighted average of probabilities
ensemble_proba = (
    weights['xgb'] * y_pred_xgb_proba +
    weights['lgb'] * y_pred_lgb_proba +
    weights['cat'] * y_pred_cat_proba
)

# Convert to predictions
y_pred_ensemble = (ensemble_proba >= 0.5).astype(int)

# Evaluate ensemble
ensemble_accuracy = accuracy_score(y_val, y_pred_ensemble)

# Compare all models
print(f"\\n{'='*60}")
print("MODEL COMPARISON:")
print(f"{'='*60}")
print(f"XGBoost:  {xgb_accuracy:.4f}")
print(f"LightGBM: {lgb_accuracy:.4f}")
print(f"CatBoost: {cat_accuracy:.4f}")
print(f"Ensemble: {ensemble_accuracy:.4f}")
print(f"{'='*60}")

print(f"\\nClassification Report (Ensemble):")
print(classification_report(y_val, y_pred_ensemble))""")

# Cell 12: Test Predictions
add_markdown_cell("# 9. Generate Test Predictions and Submission")

add_code_cell("""# Generate predictions on test set
test_pred_xgb_proba = xgb_model.predict_proba(X_test_enhanced)[:, 1]
test_pred_lgb_proba = lgb_model.predict_proba(X_test_enhanced)[:, 1]
test_pred_cat_proba = cat_model.predict_proba(X_test_enhanced)[:, 1]

# Ensemble test predictions
test_ensemble_proba = (
    weights['xgb'] * test_pred_xgb_proba +
    weights['lgb'] * test_pred_lgb_proba +
    weights['cat'] * test_pred_cat_proba
)

test_ensemble_pred = (test_ensemble_proba >= 0.5).astype(int)

# Create submission file
submission = pd.DataFrame({
    'id': df_test['id'],
    'diagnosed_diabetes': test_ensemble_pred
})

# Save submission
submission.to_csv('submission.csv', index=False)
print("Submission file created!")
print(f"Shape: {submission.shape}")
print(f"\\nPrediction distribution:")
print(submission['diagnosed_diabetes'].value_counts())
print(f"\\nFirst few predictions:")
print(submission.head(10))""")

# Cell 13: Additional Tips
add_markdown_cell("""# 10. Additional Improvement Ideas

If you need to improve accuracy further:

1. **Adjust prediction threshold** - Try values between 0.45-0.55 instead of 0.5
2. **Optimize ensemble weights** - Test different weight combinations
3. **Use SMOTE** - Train on balanced data (currently created but not used)
4. **Cross-validation** - Use StratifiedKFold for more robust validation
5. **Hyperparameter tuning** - Use Optuna or GridSearchCV for thorough optimization
6. **More feature engineering** - Create polynomial features, binning, etc.
7. **Stacking** - Train a meta-learner on top of base model predictions""")

# Write the notebook
with open('main.ipynb', 'w') as f:
    json.dump(notebook, f, indent=2)

print("Clean notebook created successfully!")

# Water Potability Prediction: Full Reproducible Analysis
# ---------------------------------------------------------------------------
# Part of the code accompanying the M.Tech. thesis "An Explainable Machine
# Learning Approach to Water Potability Prediction" (Anas Umar Aliyu).
#
# What this script does:
#   1. Exploratory data analysis (EDA), run before any modelling.
#   2. Leakage-free preprocessing: median imputation is fit on training data
#      only (inside a Pipeline for cross-validation, and on X_train only for
#      the held-out test evaluation).
#   3. Random Forest and XGBoost: 5-fold cross-validation, test-set metrics,
#      confusion matrices, ROC curves, and a side-by-side comparison.
#   4. SHAP explanations for the model with the higher cross-validated ROC-AUC.
#
# Usage (Google Colab or any machine with internet access):
#   pip install -r requirements.txt
#   python water_potability_full_analysis_v2.py
# Place water_potability.csv in the same folder first. Random seed: 42.

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (classification_report, roc_auc_score, confusion_matrix,
                              roc_curve, auc)
import xgboost as xgb
import shap

sns.set_theme(style='whitegrid', font_scale=1.0)
RANDOM_STATE = 42

# ===========================================================================
# 1. LOAD DATA
# ===========================================================================
import os
if not os.path.exists('water_potability.csv'):
    raise SystemExit("water_potability.csv not found. Download it from "
                     "https://www.kaggle.com/datasets/adityakadiwal/water-potability "
                     "and place it in the same folder as this script.")
df = pd.read_csv('water_potability.csv')
X = df.drop(columns=['Potability'])
y = df['Potability']

print("=" * 70)
print("SECTION 1: DATA LOADED")
print("=" * 70)
print(f"Shape: {df.shape}")
print(f"Class balance:\n{y.value_counts()}\n{y.value_counts(normalize=True).round(3)}")

# ===========================================================================
# 2. EXPLORATORY DATA ANALYSIS (EDA) — before any modelling
# ===========================================================================
print("\n" + "=" * 70)
print("SECTION 2: EXPLORATORY DATA ANALYSIS")
print("=" * 70)

# 2.1 Descriptive statistics
desc = df.describe().T[['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']]
desc.to_csv('eda_descriptive_stats.csv')
print("\nDescriptive statistics saved to eda_descriptive_stats.csv")
print(desc.round(2))

# 2.2 Missingness
miss = df.isna().sum()
miss_pct = (miss / len(df) * 100).round(1)
miss_table = pd.DataFrame({'Missing Count': miss, 'Missing %': miss_pct})
miss_table = miss_table[miss_table['Missing Count'] > 0]
miss_table.to_csv('eda_missingness.csv')
print("\nMissingness summary:")
print(miss_table)

fig, ax = plt.subplots(figsize=(7, 3.5))
miss_sorted = miss[miss > 0].sort_values(ascending=True)
miss_pct_sorted = (miss_sorted / len(df) * 100)
bars = ax.barh(miss_sorted.index, miss_pct_sorted.values, color='#e67e22')
for b, (cnt, pct) in zip(bars, zip(miss_sorted.values, miss_pct_sorted.values)):
    ax.text(pct + 0.5, b.get_y() + b.get_height() / 2, f'{cnt} ({pct:.1f}%)', va='center', fontsize=10)
ax.set_xlabel('Missing (%)', fontsize=12)
ax.set_title('Missing Values by Feature', fontsize=14, weight='bold')
ax.set_xlim(0, max(miss_pct_sorted.values) * 1.3)
plt.tight_layout()
plt.savefig('eda_missingness.png', dpi=200)
plt.close()
print("Saved eda_missingness.png")

# 2.3 Class balance
fig, ax = plt.subplots(figsize=(6, 4.5))
counts = y.value_counts().sort_index()
labels = ['Not Potable', 'Potable']
colors = ['#e74c3c', '#2ecc71']
bars = ax.bar(labels, counts.values, color=colors, width=0.55)
for b, v in zip(bars, counts.values):
    ax.text(b.get_x() + b.get_width() / 2, v + 30, f'{v}\n({v/len(df)*100:.1f}%)', ha='center', fontsize=11)
ax.set_ylabel('Number of Samples', fontsize=12)
ax.set_title('Class Distribution: Water Potability', fontsize=14, weight='bold')
ax.set_ylim(0, max(counts.values) * 1.2)
plt.tight_layout()
plt.savefig('eda_class_balance.png', dpi=200)
plt.close()
print("Saved eda_class_balance.png")

# 2.4 Correlation heatmap
fig, ax = plt.subplots(figsize=(9, 7))
corr = df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            square=True, linewidths=0.5, cbar_kws={'shrink': 0.8}, ax=ax)
ax.set_title('Correlation Heatmap of Water Quality Parameters', fontsize=14, weight='bold', pad=15)
plt.tight_layout()
plt.savefig('eda_correlation_heatmap.png', dpi=200)
plt.close()
print("Saved eda_correlation_heatmap.png")

# 2.5 Distributions by class (top-correlated-with-nothing, but informative) + boxplots
df_melt = df.melt(id_vars='Potability', var_name='Feature', value_name='Value')
df_melt['Potability'] = df_melt['Potability'].map({0: 'Not Potable', 1: 'Potable'})
g = sns.FacetGrid(df_melt, col='Feature', col_wrap=3, sharey=False, sharex=False, height=3.2, aspect=1.1)
g.map_dataframe(sns.boxplot, x='Potability', y='Value', hue='Potability',
                 palette={'Potable': '#2ecc71', 'Not Potable': '#e74c3c'}, legend=False, width=0.5)
g.set_titles("{col_name}", fontweight='bold')
g.fig.suptitle('Feature Distributions by Potability (Boxplots)', fontsize=16, weight='bold', y=1.02)
plt.tight_layout()
plt.savefig('eda_boxplots_by_class.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved eda_boxplots_by_class.png")

# 2.6 Outlier screening (IQR method) — informs preprocessing decisions
print("\nOutlier screening (IQR method):")
outlier_summary = {}
for col in X.columns:
    Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    n_outliers = df[(df[col] < lower) | (df[col] > upper)][col].count()
    outlier_summary[col] = {'n_outliers': n_outliers, 'pct': round(n_outliers / len(df) * 100, 2)}
    print(f"  {col}: {n_outliers} ({outlier_summary[col]['pct']}%)")

# 2.7 Multicollinearity check
feature_corr = corr.drop(columns=['Potability']).drop(index=['Potability']).abs().to_numpy(copy=True)
np.fill_diagonal(feature_corr, 0)
print(f"\nMax absolute pairwise correlation among features: {feature_corr.max():.3f}")

# ===========================================================================
# 3. PREPROCESSING + MODEL TRAINING (LEAKAGE-FREE)
# ===========================================================================
print("\n" + "=" * 70)
print("SECTION 3: MODEL TRAINING (LEAKAGE-FREE IMPUTATION)")
print("=" * 70)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

results = {}

# ---- Random Forest (imputer INSIDE pipeline: refit per fold, train-only for test) ----
rf_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('model', RandomForestClassifier(n_estimators=300, class_weight='balanced', random_state=RANDOM_STATE))
])
rf_cv = cross_validate(rf_pipeline, X, y, cv=skf, scoring=scoring)
rf_pipeline.fit(X_train, y_train)  # imputer fit on X_train only
rf_preds = rf_pipeline.predict(X_test)
rf_proba = rf_pipeline.predict_proba(X_test)[:, 1]
rf_cm = confusion_matrix(y_test, rf_preds)

print("\n--- Random Forest: 5-fold CV (leakage-free) ---")
for m in scoring:
    print(f"  {m}: {rf_cv['test_'+m].mean():.3f} +/- {rf_cv['test_'+m].std():.3f}")
print("\n--- Random Forest: Test set ---")
print("ROC-AUC:", round(roc_auc_score(y_test, rf_proba), 3))
print(classification_report(y_test, rf_preds))
print("Confusion matrix:\n", rf_cm)

results['Random Forest'] = {
    'cv_accuracy': rf_cv['test_accuracy'].mean(), 'cv_accuracy_std': rf_cv['test_accuracy'].std(),
    'cv_precision': rf_cv['test_precision'].mean(), 'cv_precision_std': rf_cv['test_precision'].std(),
    'cv_recall': rf_cv['test_recall'].mean(), 'cv_recall_std': rf_cv['test_recall'].std(),
    'cv_f1': rf_cv['test_f1'].mean(), 'cv_f1_std': rf_cv['test_f1'].std(),
    'cv_roc_auc': rf_cv['test_roc_auc'].mean(), 'cv_roc_auc_std': rf_cv['test_roc_auc'].std(),
    'test_roc_auc': roc_auc_score(y_test, rf_proba),
    'confusion_matrix': rf_cm.tolist(),
}

# Random Forest ROC curve + confusion matrix figures
fpr, tpr, _ = roc_curve(y_test, rf_proba)
fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(fpr, tpr, color='#1a7a4c', linewidth=2.5, label=f'Random Forest (AUC = {auc(fpr,tpr):.2f})')
ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random guess (AUC = 0.50)')
ax.fill_between(fpr, tpr, alpha=0.15, color='#1a7a4c')
ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curve — Random Forest', fontsize=14, weight='bold')
ax.legend(loc='lower right')
plt.tight_layout(); plt.savefig('rf_roc_curve.png', dpi=200); plt.close()

fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(rf_cm, annot=True, fmt='d', cmap='Greens', cbar=False,
            xticklabels=['Not Potable', 'Potable'], yticklabels=['Not Potable', 'Potable'],
            ax=ax, annot_kws={'size': 14})
ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
ax.set_title('Confusion Matrix — Random Forest', fontsize=14, weight='bold')
plt.tight_layout(); plt.savefig('rf_confusion_matrix.png', dpi=200); plt.close()
print("Saved rf_roc_curve.png, rf_confusion_matrix.png")

# ---- XGBoost (same leakage-free pattern) ----
xgb_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('model', xgb.XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
        random_state=RANDOM_STATE, eval_metric='logloss'
    ))
])
xgb_cv = cross_validate(xgb_pipeline, X, y, cv=skf, scoring=scoring)
xgb_pipeline.fit(X_train, y_train)  # imputer fit on X_train only
xgb_preds = xgb_pipeline.predict(X_test)
xgb_proba = xgb_pipeline.predict_proba(X_test)[:, 1]
xgb_cm = confusion_matrix(y_test, xgb_preds)

print("\n--- XGBoost: 5-fold CV (leakage-free) ---")
for m in scoring:
    print(f"  {m}: {xgb_cv['test_'+m].mean():.3f} +/- {xgb_cv['test_'+m].std():.3f}")
print("\n--- XGBoost: Test set ---")
print("ROC-AUC:", round(roc_auc_score(y_test, xgb_proba), 3))
print(classification_report(y_test, xgb_preds))
print("Confusion matrix:\n", xgb_cm)

results['XGBoost'] = {
    'cv_accuracy': xgb_cv['test_accuracy'].mean(), 'cv_accuracy_std': xgb_cv['test_accuracy'].std(),
    'cv_precision': xgb_cv['test_precision'].mean(), 'cv_precision_std': xgb_cv['test_precision'].std(),
    'cv_recall': xgb_cv['test_recall'].mean(), 'cv_recall_std': xgb_cv['test_recall'].std(),
    'cv_f1': xgb_cv['test_f1'].mean(), 'cv_f1_std': xgb_cv['test_f1'].std(),
    'cv_roc_auc': xgb_cv['test_roc_auc'].mean(), 'cv_roc_auc_std': xgb_cv['test_roc_auc'].std(),
    'test_roc_auc': roc_auc_score(y_test, xgb_proba),
    'confusion_matrix': xgb_cm.tolist(),
}

fpr2, tpr2, _ = roc_curve(y_test, xgb_proba)
fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(fpr2, tpr2, color='#2563a8', linewidth=2.5, label=f'XGBoost (AUC = {auc(fpr2,tpr2):.2f})')
ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random guess (AUC = 0.50)')
ax.fill_between(fpr2, tpr2, alpha=0.15, color='#2563a8')
ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curve — XGBoost', fontsize=14, weight='bold')
ax.legend(loc='lower right')
plt.tight_layout(); plt.savefig('xgb_roc_curve.png', dpi=200); plt.close()

fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(xgb_cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Not Potable', 'Potable'], yticklabels=['Not Potable', 'Potable'],
            ax=ax, annot_kws={'size': 14})
ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
ax.set_title('Confusion Matrix — XGBoost', fontsize=14, weight='bold')
plt.tight_layout(); plt.savefig('xgb_confusion_matrix.png', dpi=200); plt.close()
print("Saved xgb_roc_curve.png, xgb_confusion_matrix.png")

# ---- Combined ROC comparison plot ----
fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(fpr, tpr, color='#1a7a4c', linewidth=2.5, label=f'Random Forest (AUC = {auc(fpr,tpr):.2f})')
ax.plot(fpr2, tpr2, color='#2563a8', linewidth=2.5, label=f'XGBoost (AUC = {auc(fpr2,tpr2):.2f})')
ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random guess (AUC = 0.50)')
ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curve Comparison: Random Forest vs. XGBoost', fontsize=14, weight='bold')
ax.legend(loc='lower right')
plt.tight_layout(); plt.savefig('model_comparison_roc.png', dpi=200); plt.close()
print("Saved model_comparison_roc.png")

# ===========================================================================
# 4. MODEL COMPARISON TABLE
# ===========================================================================
print("\n" + "=" * 70)
print("SECTION 4: MODEL COMPARISON")
print("=" * 70)
comparison_df = pd.DataFrame(results).T[['cv_accuracy', 'cv_precision', 'cv_recall', 'cv_f1', 'cv_roc_auc', 'test_roc_auc']]
comparison_df.to_csv('model_comparison.csv')
print(comparison_df.round(3))

# ===========================================================================
# 5. SHAP EXPLAINABILITY (applied to the better-performing model by ROC-AUC)
# ===========================================================================
print("\n" + "=" * 70)
print("SECTION 5: SHAP EXPLAINABILITY")
print("=" * 70)

best_model_name = max(results, key=lambda k: results[k]['cv_roc_auc'])
print(f"Best model by CV ROC-AUC: {best_model_name}")

# Impute X_test using the SAME imputer fit on X_train (leakage-free) for SHAP input
imputer_final = SimpleImputer(strategy='median').fit(X_train)
X_test_imputed = pd.DataFrame(imputer_final.transform(X_test), columns=X.columns)

if best_model_name == 'XGBoost':
    explainer = shap.TreeExplainer(xgb_pipeline.named_steps['model'])
else:
    explainer = shap.TreeExplainer(rf_pipeline.named_steps['model'])
shap_values = explainer.shap_values(X_test_imputed)
# Different SHAP versions return different shapes for a binary classifier:
#   list of arrays -> take class 1; 3D array -> take [:, :, 1]; 2D -> use as is.
if isinstance(shap_values, list):
    shap_values = shap_values[1]
elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
    shap_values = shap_values[:, :, 1]

plt.figure()
shap.summary_plot(shap_values, X_test_imputed, show=False)
plt.tight_layout(); plt.savefig('shap_summary_plot.png', dpi=200, bbox_inches='tight'); plt.close()

plt.figure()
shap.summary_plot(shap_values, X_test_imputed, plot_type='bar', show=False)
plt.tight_layout(); plt.savefig('shap_bar_plot.png', dpi=200, bbox_inches='tight'); plt.close()

mean_abs_shap = np.abs(shap_values).mean(axis=0)
top_features = X_test_imputed.columns[np.argsort(mean_abs_shap)[::-1][:4]]
for feat in top_features:
    plt.figure()
    shap.dependence_plot(feat, shap_values, X_test_imputed, show=False)
    plt.tight_layout(); plt.savefig(f'shap_dependence_{feat}.png', dpi=200, bbox_inches='tight'); plt.close()

print(f"Saved SHAP plots for best model ({best_model_name})")

# ===========================================================================
# 6. PACKAGE VERSIONS (for reproducibility appendix)
# ===========================================================================
print("\n" + "=" * 70)
print("SECTION 6: PACKAGE VERSIONS")
print("=" * 70)
import sklearn
import sys
print(f"python: {sys.version.split()[0]}")
print(f"pandas: {pd.__version__}")
print(f"numpy: {np.__version__}")
print(f"scikit-learn: {sklearn.__version__}")
print(f"xgboost: {xgb.__version__}")
print(f"shap: {shap.__version__}")

print("\nAll done. Figures and CSV summaries were saved in the current folder.")

# SHAP for XGBoost, compared with Random Forest.
# Part of the code accompanying the M.Tech. thesis "An Explainable Machine
# Learning Approach to Water Potability Prediction" (Anas Umar Aliyu).
#
# What this script does:
#   1. Refits Random Forest and XGBoost with the same split and seed (42) as the
#      main analysis, with leakage-free median imputation (fit on training data only).
#   2. Computes SHAP values for both models on the test set.
#   3. Compares the two feature rankings (share of total mean |SHAP|, rank, and
#      Spearman rank correlation).
#   4. Draws the XGBoost SHAP summary plot and the XGBoost dependence plots for
#      the same four features used for Random Forest.
#
# Note: Random Forest SHAP values are on the probability scale and XGBoost SHAP
# values are on the log-odds scale, so raw magnitudes are not comparable. The
# comparison therefore uses each feature's share of the total, and its rank.
#
# Usage: place water_potability.csv in the same folder, then run
#   pip install -r requirements.txt
#   python xgboost_shap_comparison.py

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import xgboost as xgb
import shap

RANDOM_STATE = 42

PALETTE = {'navy': '#1E2A3A', 'rf': '#1a7a4c', 'xgb': '#2563a8',
           'accent_orange': '#e67e22', 'muted': '#6b7785', 'grid': '#e3e6ea'}
plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'axes.titleweight': 'bold', 'axes.titlesize': 13,
    'axes.titlecolor': PALETTE['navy'], 'axes.labelcolor': PALETTE['navy'],
    'xtick.color': PALETTE['muted'], 'ytick.color': PALETTE['muted'],
    'axes.spines.top': False, 'axes.spines.right': False,
    'figure.facecolor': 'white', 'axes.edgecolor': PALETTE['grid'],
})
custom_cmap = LinearSegmentedColormap.from_list(
    'thesis_theme', [PALETTE['xgb'], '#f2f2f2', PALETTE['accent_orange']])

# ---------------- Data and models (same as the main analysis) ----------------
if not os.path.exists('water_potability.csv'):
    raise SystemExit("water_potability.csv not found. Download it from "
                     "https://www.kaggle.com/datasets/adityakadiwal/water-potability "
                     "and place it in the same folder as this script.")
df = pd.read_csv('water_potability.csv')
X = df.drop(columns=['Potability'])
y = df['Potability']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

rf_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('model', RandomForestClassifier(n_estimators=300, class_weight='balanced',
                                     random_state=RANDOM_STATE)),
])
xgb_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('model', xgb.XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
        random_state=RANDOM_STATE, eval_metric='logloss')),
])
rf_pipeline.fit(X_train, y_train)
xgb_pipeline.fit(X_train, y_train)

# Test set imputed with the imputer fit on the training data only
imputer_final = SimpleImputer(strategy='median').fit(X_train)
X_test_imputed = pd.DataFrame(imputer_final.transform(X_test), columns=X.columns)


def class1_shap(model):
    """SHAP values for the potable class, whatever shape this SHAP version returns."""
    values = shap.TreeExplainer(model).shap_values(X_test_imputed)
    if isinstance(values, list):
        return values[1]
    if isinstance(values, np.ndarray) and values.ndim == 3:
        return values[:, :, 1]
    return values


rf_shap = class1_shap(rf_pipeline.named_steps['model'])
xgb_shap = class1_shap(xgb_pipeline.named_steps['model'])

# ---------------- Ranking comparison ----------------
rf_mean = pd.Series(np.abs(rf_shap).mean(axis=0), index=X.columns)
xgb_mean = pd.Series(np.abs(xgb_shap).mean(axis=0), index=X.columns)
table = pd.DataFrame({
    'RF share of total (%)': rf_mean / rf_mean.sum() * 100,
    'XGBoost share of total (%)': xgb_mean / xgb_mean.sum() * 100,
})
table['RF rank'] = table['RF share of total (%)'].rank(ascending=False).astype(int)
table['XGBoost rank'] = table['XGBoost share of total (%)'].rank(ascending=False).astype(int)
table = table.sort_values('RF rank')
table.round(1).to_csv('shap_importance_rf_vs_xgboost.csv')

spearman = np.corrcoef(table['RF rank'], table['XGBoost rank'])[0, 1]
print('\nSHAP importance, Random Forest vs. XGBoost (test set):')
print(table.round(1).to_string())
print(f'\nSpearman rank correlation between the two rankings: {spearman:.2f}')
print('Top 3 Random Forest:', list(table.sort_values("RF rank").index[:3]))
print('Top 3 XGBoost:      ', list(table.sort_values("XGBoost rank").index[:3]))

# ---------------- Figure 1: ranking comparison (grouped bars) ----------------
plot_df = table.sort_values('RF share of total (%)', ascending=True)
fig, ax = plt.subplots(figsize=(8.5, 5.8))
pos = np.arange(len(plot_df))
h = 0.38
ax.barh(pos - h / 2, plot_df['RF share of total (%)'], height=h, color=PALETTE['rf'], label='Random Forest')
ax.barh(pos + h / 2, plot_df['XGBoost share of total (%)'], height=h, color=PALETTE['xgb'], label='XGBoost')
ax.set_yticks(pos)
ax.set_yticklabels(plot_df.index)
ax.set_xlabel('Share of total mean |SHAP value| (%)')
ax.set_title('SHAP Feature Importance: Random Forest vs. XGBoost', fontsize=14, pad=12)
ax.legend(loc='lower right', frameon=False)
ax.grid(axis='x', alpha=0.4)
plt.tight_layout()
plt.savefig('shap_rf_vs_xgboost_share.png', dpi=200)
plt.close()

# ---------------- Figure 2: XGBoost SHAP summary ----------------
plt.figure()
shap.summary_plot(xgb_shap, X_test_imputed, show=False, cmap=custom_cmap)
plt.gcf().axes[0].set_title('SHAP Summary: XGBoost', fontsize=13, fontweight='bold',
                            color=PALETTE['navy'], pad=10)
plt.tight_layout()
plt.savefig('shap_summary_plot_xgboost_v2.png', dpi=200, bbox_inches='tight')
plt.close()

# ---------------- Figure 3: XGBoost dependence plots, same four features as Random Forest ----------------
rf_top4 = list(rf_mean.sort_values(ascending=False).index[:4])
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
for ax, feat in zip(axes.flat, rf_top4):
    shap.dependence_plot(feat, xgb_shap, X_test_imputed, ax=ax, show=False, cmap=custom_cmap)
    ax.set_title(feat, fontsize=12, fontweight='bold', color=PALETTE['navy'])
fig.suptitle('SHAP Dependence Plots: Same Four Features (XGBoost)', fontsize=15,
             fontweight='bold', color=PALETTE['navy'], y=1.02)
plt.tight_layout()
plt.savefig('shap_dependence_grid_xgboost.png', dpi=200, bbox_inches='tight')
plt.close()

print('\nSaved: shap_rf_vs_xgboost_share.png, shap_summary_plot_xgboost_v2.png,')
print('shap_dependence_grid_xgboost.png, shap_importance_rf_vs_xgboost.csv')

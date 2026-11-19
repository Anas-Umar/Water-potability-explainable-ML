# Redraws the ROC comparison plot and the SHAP plots in the thesis figure style
# (navy / green / blue / orange). Self-contained: it refits both models from
# scratch with the same split and seed (42) as the main analysis script.
# Usage: place water_potability.csv in the same folder, then run
#   pip install -r requirements.txt
#   python regenerate_themed_plots.py

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_curve, auc
import xgboost as xgb
import shap

RANDOM_STATE = 42

# ---------- Shared theme (matches the rest of the thesis figures) ----------
PALETTE = {
    'navy': '#1E2A3A', 'rf': '#1a7a4c', 'xgb': '#2563a8',
    'accent_orange': '#e67e22', 'muted': '#6b7785', 'grid': '#e3e6ea',
}
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.titleweight': 'bold',
    'axes.titlesize': 13,
    'axes.titlecolor': PALETTE['navy'],
    'axes.labelcolor': PALETTE['navy'],
    'xtick.color': PALETTE['muted'],
    'ytick.color': PALETTE['muted'],
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.facecolor': 'white',
    'axes.edgecolor': PALETTE['grid'],
})

# ---------- Reload data and refit both models (identical to the main script) ----------
import os
if not os.path.exists('water_potability.csv'):
    raise SystemExit("water_potability.csv not found. Download it from "
                     "https://www.kaggle.com/datasets/adityakadiwal/water-potability "
                     "and place it in the same folder as this script.")
df = pd.read_csv('water_potability.csv')
X = df.drop(columns=['Potability'])
y = df['Potability']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)

rf_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('model', RandomForestClassifier(n_estimators=300, class_weight='balanced', random_state=RANDOM_STATE))
])
rf_pipeline.fit(X_train, y_train)
rf_proba = rf_pipeline.predict_proba(X_test)[:, 1]

xgb_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('model', xgb.XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
        random_state=RANDOM_STATE, eval_metric='logloss'
    ))
])
xgb_pipeline.fit(X_train, y_train)
xgb_proba = xgb_pipeline.predict_proba(X_test)[:, 1]

print("Models refit. Random Forest and XGBoost probabilities ready.")

# ---------- 1. Themed ROC comparison plot ----------
fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_proba)
fpr_xgb, tpr_xgb, _ = roc_curve(y_test, xgb_proba)
auc_rf = auc(fpr_rf, tpr_rf)
auc_xgb = auc(fpr_xgb, tpr_xgb)

fig, ax = plt.subplots(figsize=(7.5, 6.2))
ax.plot(fpr_rf, tpr_rf, color=PALETTE['rf'], linewidth=2.5, label=f'Random Forest (AUC = {auc_rf:.2f})')
ax.plot(fpr_xgb, tpr_xgb, color=PALETTE['xgb'], linewidth=2.5, label=f'XGBoost (AUC = {auc_xgb:.2f})')
ax.plot([0, 1], [0, 1], linestyle='--', color=PALETTE['muted'], linewidth=1.3, label='Random guess (AUC = 0.50)')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curve Comparison: Random Forest vs. XGBoost', fontsize=14, pad=12)
ax.legend(loc='lower right', frameon=False)
ax.grid(alpha=0.4)
plt.tight_layout()
plt.savefig('model_comparison_roc_v2.png', dpi=200)
plt.close()
print("Saved model_comparison_roc_v2.png")

# ---------- 2. Themed SHAP plots (best model = Random Forest) ----------
imputer_final = SimpleImputer(strategy='median').fit(X_train)
X_test_imputed = pd.DataFrame(imputer_final.transform(X_test), columns=X.columns)

explainer = shap.TreeExplainer(rf_pipeline.named_steps['model'])
shap_values = explainer.shap_values(X_test_imputed)
if isinstance(shap_values, list):
    shap_values = shap_values[1]
elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
    shap_values = shap_values[:, :, 1]

# Custom colormap: navy (low) -> white -> accent orange (high), replacing
# SHAP's default blue/red so it matches the rest of the thesis figures.
from matplotlib.colors import LinearSegmentedColormap
custom_cmap = LinearSegmentedColormap.from_list(
    'thesis_theme', [PALETTE['xgb'], '#f2f2f2', PALETTE['accent_orange']]
)

plt.figure()
shap.summary_plot(shap_values, X_test_imputed, show=False, cmap=custom_cmap)
plt.gcf().axes[0].set_title('SHAP Summary \u2014 Random Forest', fontsize=13, fontweight='bold', color=PALETTE['navy'], pad=10)
plt.tight_layout()
plt.savefig('shap_summary_plot_v2.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved shap_summary_plot_v2.png")

plt.figure()
shap.summary_plot(shap_values, X_test_imputed, plot_type='bar', show=False, color=PALETTE['rf'])
plt.gcf().axes[0].set_title('Mean |SHAP Value| \u2014 Random Forest', fontsize=13, fontweight='bold', color=PALETTE['navy'], pad=10)
plt.tight_layout()
plt.savefig('shap_bar_plot_v2.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved shap_bar_plot_v2.png")

# ---------- 3. Themed SHAP dependence plots (2x2 grid, single figure) ----------
mean_abs_shap = np.abs(shap_values).mean(axis=0)
top_features = X_test_imputed.columns[np.argsort(mean_abs_shap)[::-1][:4]]

fig, axes = plt.subplots(2, 2, figsize=(13, 9))
for ax, feat in zip(axes.flat, top_features):
    shap.dependence_plot(feat, shap_values, X_test_imputed, ax=ax, show=False, cmap=custom_cmap)
    ax.set_title(feat, fontsize=12, fontweight='bold', color=PALETTE['navy'])
fig.suptitle('SHAP Dependence Plots \u2014 Top 4 Features (Random Forest)', fontsize=15, fontweight='bold', color=PALETTE['navy'], y=1.02)
plt.tight_layout()
plt.savefig('shap_dependence_grid_v2.png', dpi=200, bbox_inches='tight')
plt.close()
print(f"Saved shap_dependence_grid_v2.png for: {list(top_features)}")

print("\nDone. Saved model_comparison_roc_v2.png, shap_summary_plot_v2.png,")
print("shap_bar_plot_v2.png and shap_dependence_grid_v2.png.")

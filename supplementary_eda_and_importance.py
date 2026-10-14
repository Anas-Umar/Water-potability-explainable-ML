# Supplementary analysis script for the thesis.
# Covers: (1) the restyled exploratory figures, (2) Random Forest permutation and
# native importance with the comparison figure, (3) the combined confusion matrix panel.
# Same data, same split, same seed (42) and the same leakage-free Pipeline as
# water_potability_full_analysis_v2.py.
#
# Usage: place water_potability.csv in the same folder, then run
#   pip install -r requirements.txt
#   python supplementary_eda_and_importance.py
# xgboost is only needed for the combined confusion matrix panel.

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance
from sklearn.metrics import confusion_matrix

RANDOM_STATE = 42
PALETTE = {'navy': '#1E2A3A', 'rf': '#1a7a4c', 'xgb': '#2563a8', 'accent_orange': '#e67e22',
           'not_potable': '#c0392b', 'potable': '#1a7a4c', 'grid': '#e3e6ea', 'muted': '#6b7785'}
plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'axes.titleweight': 'bold', 'axes.titlesize': 13,
    'axes.titlecolor': PALETTE['navy'], 'axes.labelcolor': PALETTE['navy'],
    'xtick.color': PALETTE['muted'], 'ytick.color': PALETTE['muted'],
    'axes.spines.top': False, 'axes.spines.right': False,
    'figure.facecolor': 'white', 'axes.edgecolor': PALETTE['grid'],
})

import os
if not os.path.exists('water_potability.csv'):
    raise SystemExit("water_potability.csv not found. Download it from "
                     "https://www.kaggle.com/datasets/adityakadiwal/water-potability "
                     "and place it in the same folder as this script.")
df = pd.read_csv('water_potability.csv')
X = df.drop(columns=['Potability'])
y = df['Potability']

# ---------------- 1. Exploratory figures ----------------
# 1a. Missingness + class balance panel
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
miss = df.isna().sum()
miss = miss[miss > 0].sort_values(ascending=True)
miss_pct = miss / len(df) * 100
bars = axes[0].barh(miss.index, miss_pct.values, color=PALETTE['accent_orange'], height=0.55)
for b, cnt, pct in zip(bars, miss.values, miss_pct.values):
    axes[0].text(pct + 0.6, b.get_y() + b.get_height() / 2, f'{cnt} ({pct:.1f}%)', va='center', fontsize=9.5)
axes[0].set_xlabel('Missing (%)'); axes[0].set_title('Missing Values by Feature')
axes[0].set_xlim(0, miss_pct.max() * 1.35); axes[0].grid(axis='x', alpha=0.5)
counts = y.value_counts().sort_index()
bars2 = axes[1].bar(['Not Potable', 'Potable'], counts.values,
                    color=[PALETTE['not_potable'], PALETTE['potable']], width=0.5)
for b, v in zip(bars2, counts.values):
    axes[1].text(b.get_x() + b.get_width() / 2, v + 40, f'{v}\n({v / len(df) * 100:.1f}%)', ha='center', fontsize=9.5)
axes[1].set_ylabel('Number of Samples'); axes[1].set_title('Class Distribution')
axes[1].set_ylim(0, counts.max() * 1.22); axes[1].grid(axis='y', alpha=0.5)
fig.suptitle('Exploratory Data Overview', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout(); plt.savefig('eda_overview_combined.png', dpi=200, bbox_inches='tight'); plt.close()

# 1b. Correlation heatmap
fig, ax = plt.subplots(figsize=(8.5, 6.8))
corr = df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r', center=0, square=True,
            linewidths=0.6, linecolor='white', cbar_kws={'shrink': 0.8}, ax=ax, annot_kws={'size': 9})
ax.set_title('Correlation Heatmap of Water Quality Parameters', fontsize=14, pad=14)
plt.tight_layout(); plt.savefig('eda_correlation_heatmap_v2.png', dpi=200); plt.close()

# 1c. Boxplots by class
df_melt = df.melt(id_vars='Potability', var_name='Feature', value_name='Value')
df_melt['Potability'] = df_melt['Potability'].map({0: 'Not Potable', 1: 'Potable'})
g = sns.FacetGrid(df_melt, col='Feature', col_wrap=3, sharey=False, sharex=False, height=2.9, aspect=1.15)
g.map_dataframe(sns.boxplot, x='Potability', y='Value', hue='Potability',
                palette={'Potable': PALETTE['potable'], 'Not Potable': PALETTE['not_potable']},
                legend=False, width=0.5, linewidth=1)
g.set_titles('{col_name}', fontweight='bold', color=PALETTE['navy'])
g.fig.suptitle('Feature Distributions by Potability', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout(); plt.savefig('eda_boxplots_by_class_v2.png', dpi=200, bbox_inches='tight'); plt.close()
print('Saved exploratory figures.')

# ---------------- 2. Random Forest permutation + native importance ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
rf_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('model', RandomForestClassifier(n_estimators=300, class_weight='balanced', random_state=RANDOM_STATE)),
])
rf_pipeline.fit(X_train, y_train)

perm = permutation_importance(rf_pipeline, X_test, y_test, n_repeats=30,
                              random_state=RANDOM_STATE, scoring='roc_auc')
perm_imp = pd.Series(perm.importances_mean, index=X.columns).sort_values(ascending=False)
native_imp = pd.Series(rf_pipeline.named_steps['model'].feature_importances_,
                       index=X.columns).sort_values(ascending=False)
print('\nPermutation importance (ROC-AUC drop):\n', perm_imp.round(4))
print('\nNative Random Forest importance (Gini):\n', native_imp.round(4))

native_norm = native_imp / native_imp.sum()
perm_norm = (perm_imp - perm_imp.min()) / (perm_imp.max() - perm_imp.min())
compare = pd.DataFrame({'Random Forest (native)': native_norm,
                        'Permutation (ROC-AUC drop)': perm_norm}).reindex(native_imp.index)
compare = compare.sort_values('Random Forest (native)', ascending=True)
fig, ax = plt.subplots(figsize=(8.5, 5.8))
compare.plot(kind='barh', ax=ax, color=[PALETTE['rf'], PALETTE['accent_orange']], width=0.72)
ax.set_xlabel('Normalized importance')
ax.set_title('Feature Importance: Random Forest Native vs. Permutation', fontsize=14, pad=12)
ax.legend(loc='lower right', frameon=False); ax.grid(axis='x', alpha=0.4)
plt.tight_layout(); plt.savefig('rf_importance_comparison_v2.png', dpi=200); plt.close()
print('Saved rf_importance_comparison_v2.png')

# ---------------- 3. Combined confusion matrix panel ----------------
rf_cm = confusion_matrix(y_test, rf_pipeline.predict(X_test))
try:
    import xgboost as xgb
    xgb_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('model', xgb.XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
            random_state=RANDOM_STATE, eval_metric='logloss')),
    ])
    xgb_pipeline.fit(X_train, y_train)
    xgb_cm = confusion_matrix(y_test, xgb_pipeline.predict(X_test))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, cm, cmap, title in [(axes[0], rf_cm, 'Greens', 'Random Forest'),
                                (axes[1], xgb_cm, 'Blues', 'XGBoost')]:
        sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, cbar=False,
                    xticklabels=['Not Potable', 'Potable'], yticklabels=['Not Potable', 'Potable'],
                    ax=ax, annot_kws={'size': 15, 'color': PALETTE['navy']}, linewidths=1, linecolor='white')
        ax.set_xlabel('Predicted'); ax.set_ylabel('Actual'); ax.set_title(title)
    fig.suptitle('Confusion Matrices: Random Forest vs. XGBoost (Test Set)', fontsize=15,
                 fontweight='bold', y=1.03)
    plt.tight_layout(); plt.savefig('confusion_matrices_combined.png', dpi=200, bbox_inches='tight'); plt.close()
    print('Saved confusion_matrices_combined.png')
    print('RF confusion matrix:\n', rf_cm, '\nXGBoost confusion matrix:\n', xgb_cm)
except ImportError:
    print('xgboost is not installed: skipped the combined confusion matrix panel.')
    print('RF confusion matrix:\n', rf_cm)

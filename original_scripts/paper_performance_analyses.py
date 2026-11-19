# Extra analyses for the research paper: performance, uncertainty, and cost-based model choice.
# Companion to the M.Tech. thesis code (same data, same 80/20 split, seed 42, leakage-free pipelines).
#
# What it adds to the thesis:
#   1. Bootstrap 95% confidence intervals for every test metric, and paired bootstrap
#      differences (Random Forest vs. XGBoost, and each vs. a logistic regression baseline).
#   2. Repeated random train/test splits, to show whether the model ranking is stable.
#   3. A label-shuffling test: how high can AUC get when the labels carry no signal?
#   4. Light, equal-budget hyperparameter tuning of both ensembles on the training set only.
#   5. ROC and precision-recall curves.
#   6. Cost analysis: how the preferable model and decision threshold change when passing
#      contaminated water as safe costs more than rejecting safe water. Thresholds are chosen
#      from training-set cross-validated predictions and then evaluated once on the test set.
#
# Usage: place water_potability.csv in the same folder, then run
#   pip install -r requirements.txt
#   python paper_performance_analyses.py            (full run, several minutes)
#   python paper_performance_analyses.py --quick    (small settings, for testing only)

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.model_selection import (train_test_split, StratifiedKFold,
                                     RandomizedSearchCV, cross_val_predict)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (roc_auc_score, recall_score, precision_score, f1_score,
                             roc_curve, precision_recall_curve, average_precision_score)
import xgboost as xgb

QUICK = '--quick' in sys.argv
SEED = 42
N_BOOT = 200 if QUICK else 2000
N_SPLITS = 4 if QUICK else 20
N_SHUFFLE = 5 if QUICK else 200
N_ITER = 3 if QUICK else 20
COST_RATIOS = [1, 2, 5, 10, 20]   # cost of passing contaminated water, relative to rejecting safe water

PALETTE = {'navy': '#1E2A3A', 'rf': '#1a7a4c', 'xgb': '#2563a8', 'lr': '#e67e22',
           'muted': '#6b7785', 'grid': '#e3e6ea'}
COLORS = {'Random Forest': PALETTE['rf'], 'XGBoost': PALETTE['xgb'], 'Logistic regression': PALETTE['lr']}
plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'axes.titleweight': 'bold', 'axes.titlesize': 12,
    'axes.titlecolor': PALETTE['navy'], 'axes.labelcolor': PALETTE['navy'],
    'xtick.color': PALETTE['muted'], 'ytick.color': PALETTE['muted'],
    'axes.spines.top': False, 'axes.spines.right': False,
    'figure.facecolor': 'white', 'axes.edgecolor': PALETTE['grid'],
})

if not os.path.exists('water_potability.csv'):
    raise SystemExit("water_potability.csv not found. Download it from "
                     "https://www.kaggle.com/datasets/adityakadiwal/water-potability "
                     "and place it in the same folder as this script.")
df = pd.read_csv('water_potability.csv')
X = df.drop(columns=['Potability'])
y = df['Potability']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED)
y_arr = y_test.to_numpy()
n_test = len(y_arr)


def make_models(y_tr):
    """The three models: thesis settings for the ensembles, plus a simple baseline."""
    spw = (y_tr == 0).sum() / (y_tr == 1).sum()
    return {
        'Logistic regression': Pipeline([
            ('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler()),
            ('model', LogisticRegression(class_weight='balanced', max_iter=1000))]),
        'Random Forest': Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('model', RandomForestClassifier(n_estimators=300, class_weight='balanced', random_state=SEED))]),
        'XGBoost': Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('model', xgb.XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                                        scale_pos_weight=spw, random_state=SEED, eval_metric='logloss'))]),
    }


def metrics(y_true, p, thr=0.5):
    pred = (p >= thr).astype(int)
    return {'AUC': roc_auc_score(y_true, p),
            'Recall (potable)': recall_score(y_true, pred, pos_label=1, zero_division=0),
            'Recall (not potable)': recall_score(y_true, pred, pos_label=0, zero_division=0),
            'Precision (potable)': precision_score(y_true, pred, zero_division=0),
            'F1 (potable)': f1_score(y_true, pred, zero_division=0)}


# ---------------- 0. Fit the three models on the training set ----------------
models = make_models(y_train)
probs = {}
for name, pipe in models.items():
    pipe.fit(X_train, y_train)
    probs[name] = pipe.predict_proba(X_test)[:, 1]
observed = pd.DataFrame({m: metrics(y_arr, p) for m, p in probs.items()}).T
print('\nObserved test-set metrics (threshold 0.5):')
print(observed.round(3).to_string())
observed.round(4).to_csv('paper_observed_metrics.csv')

# ---------------- 1. Bootstrap confidence intervals and paired differences ----------------
rng = np.random.default_rng(SEED)
boot_idx = [rng.integers(0, n_test, n_test) for _ in range(N_BOOT)]
boot = {m: pd.DataFrame([metrics(y_arr[ix], p[ix]) for ix in boot_idx]) for m, p in probs.items()}

ci_rows = []
for m, b in boot.items():
    for col in b.columns:
        ci_rows.append({'Model': m, 'Metric': col, 'Estimate': observed.loc[m, col],
                        'CI low (2.5%)': b[col].quantile(0.025), 'CI high (97.5%)': b[col].quantile(0.975)})
ci = pd.DataFrame(ci_rows)
ci.round(3).to_csv('paper_bootstrap_ci.csv', index=False)
print(f'\nBootstrap 95% CIs ({N_BOOT} resamples of the test set):')
print(ci.round(3).to_string(index=False))

pairs = [('Random Forest', 'XGBoost'), ('Random Forest', 'Logistic regression'), ('XGBoost', 'Logistic regression')]
diff_rows = []
for a, b_ in pairs:
    for col in boot[a].columns:
        d = boot[a][col].to_numpy() - boot[b_][col].to_numpy()
        p_two = min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))
        diff_rows.append({'Comparison': f'{a} minus {b_}', 'Metric': col,
                          'Difference': observed.loc[a, col] - observed.loc[b_, col],
                          'CI low': np.quantile(d, 0.025), 'CI high': np.quantile(d, 0.975),
                          'Approx. two-sided p': p_two})
diffs = pd.DataFrame(diff_rows)
diffs.round(3).to_csv('paper_paired_differences.csv', index=False)
print('\nPaired bootstrap differences:')
print(diffs.round(3).to_string(index=False))

# ---------------- 2. Repeated random splits ----------------
split_rows = []
for s in range(N_SPLITS):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=1000 + s)
    for name, pipe in make_models(ytr).items():
        pipe.fit(Xtr, ytr)
        r = metrics(yte.to_numpy(), pipe.predict_proba(Xte)[:, 1])
        r.update({'Split seed': 1000 + s, 'Model': name})
        split_rows.append(r)
splits = pd.DataFrame(split_rows)
splits.round(4).to_csv('paper_repeated_splits.csv', index=False)
summary = splits.groupby('Model').agg(['mean', 'std']).drop(columns='Split seed')
print('\nRepeated splits (%d random 80/20 splits), mean and sd:' % N_SPLITS)
print(summary.round(3).to_string())
wide = splits.pivot(index='Split seed', columns='Model', values='AUC')
print('\nShare of splits where Random Forest AUC > XGBoost AUC: %.2f' % (wide['Random Forest'] > wide['XGBoost']).mean())
print('Mean AUC difference (RF - XGBoost): %.3f (sd %.3f)' % ((wide['Random Forest'] - wide['XGBoost']).mean(),
                                                             (wide['Random Forest'] - wide['XGBoost']).std()))

# ---------------- 3. Label-shuffling test ----------------
shuffle_rows = []
for name in ['Random Forest', 'XGBoost']:
    perm_aucs = []
    for k in range(N_SHUFFLE):
        pipe = clone(models[name])
        pipe.fit(X_train, rng.permutation(y_train.to_numpy()))
        perm_aucs.append(roc_auc_score(y_arr, pipe.predict_proba(X_test)[:, 1]))
    perm_aucs = np.array(perm_aucs)
    obs = observed.loc[name, 'AUC']
    shuffle_rows.append({'Model': name, 'Observed AUC': obs, 'Shuffled AUC mean': perm_aucs.mean(),
                         'Shuffled AUC 95th pct': np.quantile(perm_aucs, 0.95), 'Shuffled AUC max': perm_aucs.max(),
                         'Empirical p': (1 + (perm_aucs >= obs).sum()) / (1 + len(perm_aucs))})
shuf = pd.DataFrame(shuffle_rows)
shuf.round(4).to_csv('paper_label_shuffle.csv', index=False)
print('\nLabel-shuffling test (%d shuffles per model):' % N_SHUFFLE)
print(shuf.round(3).to_string(index=False))

# ---------------- 4. Equal-budget tuning on the training set only ----------------
cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
param_spaces = {
    'Random Forest': {'model__n_estimators': [200, 300, 500], 'model__max_depth': [None, 5, 10, 20],
                      'model__min_samples_leaf': [1, 2, 5, 10], 'model__max_features': ['sqrt', 0.5]},
    'XGBoost': {'model__n_estimators': [200, 300, 500], 'model__max_depth': [3, 4, 5, 6],
                'model__learning_rate': [0.01, 0.03, 0.05, 0.1], 'model__subsample': [0.7, 0.85, 1.0],
                'model__colsample_bytree': [0.7, 0.85, 1.0], 'model__min_child_weight': [1, 3, 5]},
}
tuned_probs, tune_rows = {}, []
for name, space in param_spaces.items():
    base = clone(models[name])
    if name == 'XGBoost':
        base.set_params(model__n_jobs=1)
    search = RandomizedSearchCV(base, space, n_iter=N_ITER, scoring='roc_auc', cv=cv5,
                                random_state=SEED, n_jobs=-1, refit=True)
    search.fit(X_train, y_train)
    p = search.best_estimator_.predict_proba(X_test)[:, 1]
    tuned_probs[name + ' (tuned)'] = p
    m = metrics(y_arr, p)
    tune_rows.append({'Model': name, 'Best CV AUC (training set)': search.best_score_,
                      'Default test AUC': observed.loc[name, 'AUC'], 'Tuned test AUC': m['AUC'],
                      'Tuned recall (potable)': m['Recall (potable)'],
                      'Tuned recall (not potable)': m['Recall (not potable)'],
                      'Best parameters': str(search.best_params_)})
tune = pd.DataFrame(tune_rows)
tune.to_csv('paper_tuning_results.csv', index=False)
print('\nTuning (%d random configurations per model, 5-fold CV on the training set):' % N_ITER)
print(tune.drop(columns='Best parameters').round(3).to_string(index=False))

# ---------------- 5. ROC and precision-recall curves ----------------
fig, axes = plt.subplots(1, 2, figsize=(11.5, 5))
for name, p in probs.items():
    fpr, tpr, _ = roc_curve(y_arr, p)
    lo, hi = ci[(ci.Model == name) & (ci.Metric == 'AUC')][['CI low (2.5%)', 'CI high (97.5%)']].iloc[0]
    axes[0].plot(fpr, tpr, color=COLORS[name], lw=2.2,
                 label=f'{name}: AUC {observed.loc[name, "AUC"]:.2f} ({lo:.2f} to {hi:.2f})')
    prec, rec, _ = precision_recall_curve(y_arr, p)
    axes[1].plot(rec, prec, color=COLORS[name], lw=2.2,
                 label=f'{name}: AP {average_precision_score(y_arr, p):.2f}')
axes[0].plot([0, 1], [0, 1], '--', color=PALETTE['muted'], lw=1.2)
axes[0].set(xlabel='False positive rate', ylabel='True positive rate', title='ROC curves (test set)')
axes[0].legend(loc='lower right', frameon=False, fontsize=8.5)
axes[1].axhline(y_arr.mean(), ls='--', color=PALETTE['muted'], lw=1.2, label='No-skill (prevalence)')
axes[1].set(xlabel='Recall (potable)', ylabel='Precision (potable)', title='Precision-recall curves (test set)')
axes[1].legend(loc='upper right', frameon=False, fontsize=8.5)
plt.tight_layout(); plt.savefig('paper_roc_pr_panel.png', dpi=200); plt.close()

# ---------------- 6. Cost analysis (thresholds chosen on training data only) ----------------
thr_grid = np.linspace(0.02, 0.98, 97)


def errors(y_true, p, t):
    pred = p >= t
    return ((pred == 1) & (y_true == 0)).sum(), ((pred == 0) & (y_true == 1)).sum()   # (contaminated passed, safe rejected)


cost_models = {**{m: models[m] for m in ['Random Forest', 'XGBoost', 'Logistic regression']}}
oof = {}
for name, pipe in cost_models.items():
    oof[name] = cross_val_predict(clone(pipe), X_train, y_train, cv=cv5, method='predict_proba')[:, 1]
y_tr_arr = y_train.to_numpy()

cost_rows = []
for name in cost_models:
    for r in COST_RATIOS:
        oof_cost = [(r * errors(y_tr_arr, oof[name], t)[0] + errors(y_tr_arr, oof[name], t)[1]) / len(y_tr_arr)
                    for t in thr_grid]
        t_star = thr_grid[int(np.argmin(oof_cost))]
        fp5, fn5 = errors(y_arr, probs[name], 0.5)
        fps, fns = errors(y_arr, probs[name], t_star)
        pred_star = probs[name] >= t_star
        gains = []
        for ix in boot_idx:
            yb, pb = y_arr[ix], pred_star[ix]
            cost_b = (r * ((pb == 1) & (yb == 0)).sum() + ((pb == 0) & (yb == 1)).sum()) / n_test
            gains.append((yb == 1).sum() / n_test - cost_b)
        cost_rows.append({'Model': name, 'Cost ratio (pass:reject)': r,
                          'Test cost at 0.5': (r * fp5 + fn5) / n_test, 'Threshold from training CV': t_star,
                          'Test cost at chosen threshold': (r * fps + fns) / n_test,
                          'Contaminated passed (test)': int(fps), 'Safe rejected (test)': int(fns),
                          'Gain vs. reject-all': float(np.mean(gains)),
                          'Gain CI low': float(np.quantile(gains, 0.025)),
                          'Gain CI high': float(np.quantile(gains, 0.975))})
cost = pd.DataFrame(cost_rows)
cost.round(3).to_csv('paper_cost_analysis.csv', index=False)
print('\nCost analysis (cost per test sample; contaminated-passed costs r, safe-rejected costs 1):')
print(cost.round(3).to_string(index=False))
n_pot, n_non = int((y_arr == 1).sum()), int((y_arr == 0).sum())
baseline = pd.DataFrame([{'Cost ratio (pass:reject)': r,
                          'Reject-all cost (test)': n_pot / n_test,
                          'Accept-all cost (test)': r * n_non / n_test} for r in COST_RATIOS])
baseline.round(3).to_csv('paper_cost_baselines.csv', index=False)
best = cost.loc[cost.groupby('Cost ratio (pass:reject)')['Test cost at chosen threshold'].idxmin(),
                ['Cost ratio (pass:reject)', 'Model', 'Test cost at chosen threshold']]
best = best.merge(baseline, on='Cost ratio (pass:reject)')
best = best.merge(cost[['Model', 'Cost ratio (pass:reject)', 'Gain CI low', 'Gain CI high']],
                  on=['Model', 'Cost ratio (pass:reject)'])
best['Gain vs. best trivial policy'] = (best[['Reject-all cost (test)', 'Accept-all cost (test)']].min(axis=1)
                                        - best['Test cost at chosen threshold'])
print('\nLowest test cost at each cost ratio (thresholds from training CV), against the two trivial')
print('policies (reject all water, accept all water). A model only adds value if the gain is clearly above zero:')
print(best.round(3).to_string(index=False))

fig, axes = plt.subplots(1, 3, figsize=(14, 4.3), sharey=False)
for ax, r in zip(axes, [1, 5, 20]):
    for name, p in probs.items():
        c = [(r * errors(y_arr, p, t)[0] + errors(y_arr, p, t)[1]) / n_test for t in thr_grid]
        ax.plot(thr_grid, c, color=COLORS[name], lw=2, label=name)
    ax.axvline(0.5, ls=':', color=PALETTE['muted'])
    ax.axhline(n_pot / n_test, ls='--', color='#c0392b', lw=1.2, label='Reject all water')
    ax.set(xlabel='Decision threshold (probability of potable)', title=f'Cost ratio {r}:1')
axes[0].set_ylabel('Expected cost per sample')
axes[0].legend(frameon=False, fontsize=8.5)
fig.suptitle('Expected cost by threshold (test set): passing contaminated water costs r, rejecting safe water costs 1',
             fontsize=11, fontweight='bold', color=PALETTE['navy'], y=1.03)
plt.tight_layout(); plt.savefig('paper_cost_curves.png', dpi=200, bbox_inches='tight'); plt.close()

# ---------------- 8. Paper-ready figures (Frontiers: 180 mm wide, 300 dpi, RGB, text of 8 pt or more) ----------------
import io
from PIL import Image

MM = 1 / 25.4
PAPER_COLORS = {'Random Forest': '#E69F00', 'XGBoost': '#0072B2', 'Logistic regression': '#555555'}   # colour-blind safe
PAPER_LINES = {'Random Forest': '-', 'XGBoost': '--', 'Logistic regression': ':'}                     # never rely on colour alone


def paper_rc():
    plt.rcParams.update({
        'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'Liberation Sans', 'DejaVu Sans'],
        'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
        'legend.fontsize': 8, 'axes.linewidth': 0.8, 'lines.linewidth': 2, 'axes.titleweight': 'bold',
        'axes.spines.top': False, 'axes.spines.right': False, 'figure.facecolor': 'white',
        'axes.facecolor': 'white', 'axes.edgecolor': '#222222', 'axes.labelcolor': '#111111',
        'xtick.color': '#222222', 'ytick.color': '#222222'})


def save_paper(fig, name):
    """Vector PDF for editing, plus a 300 dpi RGB TIFF as Frontiers requires."""
    fig.savefig(name + '.pdf')
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=300)
    buf.seek(0)
    Image.open(buf).convert('RGB').save(name + '.tiff', dpi=(300, 300), compression='tiff_lzw')
    plt.close(fig)


paper_rc()
# Figure 1: ROC and precision-recall curves
fig, axes = plt.subplots(1, 2, figsize=(180 * MM, 78 * MM), layout='constrained')
for name, p in probs.items():
    fpr, tpr, _ = roc_curve(y_arr, p)
    lo, hi = ci[(ci.Model == name) & (ci.Metric == 'AUC')][['CI low (2.5%)', 'CI high (97.5%)']].iloc[0]
    axes[0].plot(fpr, tpr, color=PAPER_COLORS[name], ls=PAPER_LINES[name],
                 label=f'{name}: AUC {observed.loc[name, "AUC"]:.2f} ({lo:.2f} to {hi:.2f})')
    prec, rec, _ = precision_recall_curve(y_arr, p)
    axes[1].plot(rec, prec, color=PAPER_COLORS[name], ls=PAPER_LINES[name],
                 label=f'{name}: AP {average_precision_score(y_arr, p):.2f}')
axes[0].plot([0, 1], [0, 1], color='#999999', ls='-.')
axes[0].set(xlabel='False positive rate', ylabel='True positive rate')
axes[0].set_title('(A)', loc='left')
axes[0].legend(loc='lower right', frameon=False)
axes[1].axhline(y_arr.mean(), color='#999999', ls='-.', label='No skill (share of potable)')
axes[1].set(xlabel='Recall (potable)', ylabel='Precision (potable)')
axes[1].set_title('(B)', loc='left')
axes[1].legend(loc='upper right', frameon=False)
save_paper(fig, 'fig1_roc_pr')

# Figure 2: expected cost by decision threshold
fig, axes = plt.subplots(1, 3, figsize=(180 * MM, 76 * MM), layout='constrained')
for i, (ax, r, ylim) in enumerate(zip(axes, [1, 5, 20], [(0.30, 0.64), (0.30, 2.0), (0.30, 2.0)])):
    for name, p in probs.items():
        c = [(r * errors(y_arr, p, t)[0] + errors(y_arr, p, t)[1]) / n_test for t in thr_grid]
        ax.plot(thr_grid, c, color=PAPER_COLORS[name], ls=PAPER_LINES[name], label=name)
    ax.axvline(0.5, color='#999999', ls=':')
    ax.axhline(n_pot / n_test, color='black', ls='-.', label='Reject all water')
    ax.set_ylim(*ylim)
    ax.set_xlabel('Decision threshold')
    ax.set_title(f'({chr(65 + i)}) r = {r}', loc='left')
axes[0].set_ylabel('Expected cost per sample')
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='outside lower center', ncol=4, frameon=False)
save_paper(fig, 'fig2_cost_curves')
print('\nSaved paper figures: fig1_roc_pr and fig2_cost_curves (.pdf and 300 dpi .tiff)')

# ---------------- 7. Precision among the most confident "potable" predictions ----------------
top_rows = []
for name, p in probs.items():
    order = np.argsort(-p)
    for k in (25, 50, 100):
        top_rows.append({'Model': name, 'Top-k most confident potable predictions': k,
                         'Truly potable': int(y_arr[order[:k]].sum()),
                         'Precision': y_arr[order[:k]].mean()})
top = pd.DataFrame(top_rows)
top.round(3).to_csv('paper_top_confidence_precision.csv', index=False)
print('\nPrecision among the k test samples each model is most confident are potable '
      f'(base rate of potable water is {y_arr.mean():.2f}):')
print(top.round(3).to_string(index=False))

print('\nSaved: paper_observed_metrics.csv, paper_bootstrap_ci.csv, paper_paired_differences.csv,')
print('paper_repeated_splits.csv, paper_label_shuffle.csv, paper_tuning_results.csv,')
print('paper_cost_analysis.csv, paper_cost_baselines.csv, paper_top_confidence_precision.csv, paper_roc_pr_panel.png, paper_cost_curves.png')
import platform, sklearn, scipy, matplotlib
print('\nEnvironment: python', platform.python_version(), '| numpy', np.__version__, '| pandas', pd.__version__,
      '| scikit-learn', sklearn.__version__, '| scipy', scipy.__version__, '| xgboost', xgb.__version__,
      '| matplotlib', matplotlib.__version__, '| pillow', Image.__version__)
if QUICK:
    print('\n(--quick was used: settings are small, so these results are for testing only.)')

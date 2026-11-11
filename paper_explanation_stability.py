# Extra analyses for the research paper: do the explanations hold up?
# Companion to the M.Tech. thesis code (same data, same models, leakage-free pipelines).
#
# What it adds to the thesis:
#   1. SHAP feature rankings for Random Forest and XGBoost on the main split (seed 42)
#      and on repeated random splits, with rank means, spreads, and top-3 frequencies.
#   2. Agreement between the two models' rankings on every split (Spearman rank
#      correlation, and whether the same top two and top three features appear).
#   3. A numerical check of the "U-shaped" and "hump" patterns in the SHAP dependence
#      plots: a quadratic fit of SHAP value on feature value for Sulfate, Hardness,
#      Chloramines (expected U shape) and pH (expected hump), with the sign of the curvature,
#      the gain in R-squared over a straight line, and the same results after trimming the
#      outer 5% of feature values. If a pattern survives trimming, it is not driven by extremes.
#
# Random Forest SHAP values are on the probability scale and XGBoost values on the
# log-odds scale, so only ranks and shares of the total are compared, never raw sizes.
#
# Usage: place water_potability.csv in the same folder, then run
#   pip install -r requirements.txt
#   python paper_explanation_stability.py            (full run, a few minutes)
#   python paper_explanation_stability.py --quick    (small settings, for testing only)

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import io
import xgboost as xgb
import shap
from PIL import Image
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats

QUICK = '--quick' in sys.argv
N_SPLITS = 4 if QUICK else 20
SEED = 42
SEEDS = [SEED] + [2000 + s for s in range(N_SPLITS)]   # the main split first, then repeated splits
EXPECTED = {'Sulfate': 'U shape', 'Hardness': 'U shape', 'Chloramines': 'U shape', 'ph': 'hump'}
TRIM = 0.025   # trim the outer 2.5% at each end of the feature range (5% in total)

PALETTE = {'navy': '#1E2A3A', 'rf': '#1a7a4c', 'xgb': '#2563a8', 'muted': '#6b7785', 'grid': '#e3e6ea'}
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
FEATURES = list(X.columns)


def class1_shap(model, X_te):
    values = shap.TreeExplainer(model).shap_values(X_te)
    if isinstance(values, list):
        return values[1]
    if isinstance(values, np.ndarray) and values.ndim == 3:
        return values[:, :, 1]
    return values


def r_squared(y_true, y_hat):
    ss_res = ((y_true - y_hat) ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    return 1 - ss_res / ss_tot if ss_tot > 0 else np.nan


def curvature(x, s):
    """Quadratic coefficient (on standardised x) and R-squared gain over a straight line."""
    z = (x - x.mean()) / x.std()
    lin = np.polyval(np.polyfit(z, s, 1), z)
    quad_coef = np.polyfit(z, s, 2)
    quad = np.polyval(quad_coef, z)
    return quad_coef[0], r_squared(s, quad) - r_squared(s, lin)


rank_rows, u_rows, agree_rows = [], [], []
main_store = {}   # SHAP values and imputed test data for the main split (seed 42)
for seed in SEEDS:
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)
    imputer = SimpleImputer(strategy='median').fit(X_tr)
    X_te_imp = pd.DataFrame(imputer.transform(X_te), columns=FEATURES)
    fitted = {
        'Random Forest': Pipeline([('imputer', SimpleImputer(strategy='median')),
                                   ('model', RandomForestClassifier(n_estimators=300, class_weight='balanced',
                                                                    random_state=seed))]),
        'XGBoost': Pipeline([('imputer', SimpleImputer(strategy='median')),
                             ('model', xgb.XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                                                         scale_pos_weight=(y_tr == 0).sum() / (y_tr == 1).sum(),
                                                         random_state=seed, eval_metric='logloss'))]),
    }
    ranks = {}
    for name, pipe in fitted.items():
        pipe.fit(X_tr, y_tr)
        sv = class1_shap(pipe.named_steps['model'], X_te_imp)
        if seed == SEED:
            main_store[name] = (sv, X_te_imp.copy())
        mean_abs = pd.Series(np.abs(sv).mean(axis=0), index=FEATURES)
        share = mean_abs / mean_abs.sum() * 100
        rk = share.rank(ascending=False)
        ranks[name] = rk
        for f in FEATURES:
            rank_rows.append({'Split seed': seed, 'Model': name, 'Feature': f,
                              'Share of total (%)': share[f], 'Rank': rk[f]})
        for f, expected in EXPECTED.items():
            x = X_te_imp[f].to_numpy()
            s = sv[:, FEATURES.index(f)]
            a2, dr2 = curvature(x, s)
            lo, hi = np.quantile(x, [TRIM, 1 - TRIM])
            keep = (x >= lo) & (x <= hi)
            a2_t, dr2_t = curvature(x[keep], s[keep])
            u_rows.append({'Split seed': seed, 'Model': name, 'Feature': f, 'Expected': expected,
                           'Curvature': a2, 'R2 gain (quadratic vs. linear)': dr2,
                           'Curvature (trimmed)': a2_t, 'R2 gain (trimmed)': dr2_t})
    r_rf, r_xgb = ranks['Random Forest'], ranks['XGBoost']
    agree_rows.append({'Split seed': seed,
                       'Spearman (RF vs. XGBoost)': np.corrcoef(r_rf.to_numpy(), r_xgb.to_numpy())[0, 1],
                       'Same top 2 features': set(r_rf.nsmallest(2).index) == set(r_xgb.nsmallest(2).index),
                       'Same top 3 features': set(r_rf.nsmallest(3).index) == set(r_xgb.nsmallest(3).index)})

ranks_long = pd.DataFrame(rank_rows)
u_long = pd.DataFrame(u_rows)
agree = pd.DataFrame(agree_rows)
ranks_long.round(3).to_csv('paper_shap_ranks_long.csv', index=False)
u_long.round(4).to_csv('paper_u_shape_long.csv', index=False)

# ---------------- Rank stability ----------------
rank_summary = (ranks_long.groupby(['Model', 'Feature'])
                .agg(mean_rank=('Rank', 'mean'), sd_rank=('Rank', 'std'),
                     top3=('Rank', lambda r: (r <= 3).mean() * 100),
                     rank1=('Rank', lambda r: (r == 1).mean() * 100),
                     mean_share=('Share of total (%)', 'mean'))
                .reset_index())
rank_summary.columns = ['Model', 'Feature', 'Mean rank', 'SD of rank', '% of splits in top 3',
                        '% of splits ranked 1st', 'Mean share of total (%)']
rank_summary = rank_summary.sort_values(['Model', 'Mean rank'])
rank_summary.round(2).to_csv('paper_shap_rank_stability.csv', index=False)
print(f'\nSHAP rank stability over {len(SEEDS)} splits (main split plus {N_SPLITS} random splits):')
print(rank_summary.round(2).to_string(index=False))

rest = agree[agree['Split seed'] != SEED]
print('\nAgreement between the two models:')
print(f'  Main split (seed {SEED}): Spearman {agree.iloc[0]["Spearman (RF vs. XGBoost)"]:.2f}, '
      f'same top 3: {bool(agree.iloc[0]["Same top 3 features"])}')
print(f'  Random splits: mean Spearman {rest["Spearman (RF vs. XGBoost)"].mean():.2f} '
      f'(min {rest["Spearman (RF vs. XGBoost)"].min():.2f}); '
      f'same top 2 in {rest["Same top 2 features"].mean() * 100:.0f}% of splits; '
      f'same top 3 in {rest["Same top 3 features"].mean() * 100:.0f}% of splits')
agree.round(3).to_csv('paper_model_agreement.csv', index=False)

# ---------------- U-shape / hump quantification ----------------
def share_expected(df_, col):
    sign = np.where(df_['Expected'] == 'U shape', 1, -1)
    return ((df_[col] * sign) > 0).mean() * 100


u_rows_out = []
for (model, feat), g in u_long.groupby(['Model', 'Feature']):
    u_rows_out.append({'Model': model, 'Feature': feat, 'Expected': g['Expected'].iloc[0],
                       '% splits with expected curvature': share_expected(g, 'Curvature'),
                       'Median R2 gain': g['R2 gain (quadratic vs. linear)'].median(),
                       '% splits with expected curvature (trimmed)': share_expected(g, 'Curvature (trimmed)'),
                       'Median R2 gain (trimmed)': g['R2 gain (trimmed)'].median()})
u_summary = pd.DataFrame(u_rows_out).sort_values(['Feature', 'Model'])
u_summary.round(3).to_csv('paper_u_shape_summary.csv', index=False)
print('\nShape of the SHAP dependence curves (quadratic fit; trimmed = outer 5% of feature values removed):')
print(u_summary.round(3).to_string(index=False))
print('\nReading guide: a positive curvature means a U shape and a negative one a hump. A pattern that keeps')
print('its expected curvature after trimming, in most splits, is not driven by a few extreme values.')

# ---------------- Model-free checks of the same patterns ----------------
# These use only the raw data (no model, no SHAP), to see whether the U and hump shapes are already
# in the data, and whether imputation or extreme values could explain them.
print('\nModel-free checks on the raw data:')
spread_rows = []
for f in FEATURES:
    a, b = df.loc[y == 1, f].dropna(), df.loc[y == 0, f].dropna()
    spread_rows.append({'Feature': f, 'SD potable': a.std(), 'SD not potable': b.std(),
                        'SD ratio (potable / not potable)': a.std() / b.std(),
                        'Brown-Forsythe p': stats.levene(a, b, center='median').pvalue})
spread = pd.DataFrame(spread_rows).sort_values('SD ratio (potable / not potable)', ascending=False)
spread.round(4).to_csv('paper_model_free_spread.csv', index=False)
print('\n1. Spread of each feature by class (a ratio above 1 means potable samples are more spread out):')
print(spread.round(3).to_string(index=False))

decile_rows = []
for f in ['Sulfate', 'Hardness', 'Chloramines', 'ph']:
    d = df[[f, 'Potability']].dropna()
    d['Decile'] = pd.qcut(d[f], 10, labels=False, duplicates='drop') + 1
    rates = d.groupby('Decile')['Potability'].mean()
    decile_rows.append({'Feature': f, **{f'D{int(k)}': v for k, v in rates.items()}})
deciles = pd.DataFrame(decile_rows)
deciles.round(3).to_csv('paper_model_free_deciles.csv', index=False)
print('\n2. Share of potable samples in each decile of the feature (measured values only, D1 = lowest):')
print(deciles.round(2).to_string(index=False))

miss_rows = []
for f in FEATURES:
    m = df[f].isna()
    if m.sum() > 0:
        tab = pd.crosstab(m, y)
        miss_rows.append({'Feature': f, 'Missing n': int(m.sum()), 'Potable rate when missing': y[m].mean(),
                          'Potable rate when measured': y[~m].mean(), 'Chi-square p': stats.chi2_contingency(tab)[1]})
miss = pd.DataFrame(miss_rows)
miss.round(4).to_csv('paper_model_free_missingness.csv', index=False)
print('\n3. Is missingness informative about potability?')
print(miss.round(3).to_string(index=False))

# ---------------- Figure ----------------
order = rank_summary[rank_summary.Model == 'Random Forest'].sort_values('Mean rank')['Feature'].tolist()
fig, axes = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={'width_ratios': [1.5, 1]})
ypos = np.arange(len(order))[::-1]
for model, color, off in [('Random Forest', PALETTE['rf'], 0.13), ('XGBoost', PALETTE['xgb'], -0.13)]:
    g = rank_summary[rank_summary.Model == model].set_index('Feature').loc[order]
    axes[0].errorbar(g['Mean rank'], ypos + off, xerr=g['SD of rank'].fillna(0), fmt='o', color=color,
                     capsize=3, lw=1.4, label=model)
axes[0].set_yticks(ypos)
axes[0].set_yticklabels(order)
axes[0].invert_xaxis()
axes[0].set_xlabel('Mean SHAP rank across splits (1 = most important), bars show 1 SD')
axes[0].set_title('Stability of SHAP feature ranks')
axes[0].legend(frameon=False, loc='upper left')
axes[0].grid(axis='x', alpha=0.4)
axes[1].hist(rest['Spearman (RF vs. XGBoost)'], bins=10, color=PALETTE['muted'], edgecolor='white')
axes[1].set_xlabel('Spearman rank correlation, Random Forest vs. XGBoost')
axes[1].set_ylabel('Number of splits')
axes[1].set_title('Agreement between the two models')
plt.tight_layout()
plt.savefig('paper_explanation_stability.png', dpi=200)
plt.close()

# ---------------- Paper-ready figures (Frontiers: 180 mm wide, 300 dpi, RGB, text of 8 pt or more) ----------------
MM = 1 / 25.4
ORANGE, BLUE = '#E69F00', '#0072B2'   # colour-blind safe; the two models also differ in marker shape
DISPLAY = {'ph': 'pH', 'Organic_carbon': 'Organic carbon', 'Trihalomethanes': 'Trihalomethanes'}
FOUR = ['Sulfate', 'ph', 'Hardness', 'Chloramines']
paper_cmap = LinearSegmentedColormap.from_list('paper', [BLUE, '#f2f2f2', ORANGE])


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
# Figure 3: stability of SHAP ranks and agreement between the two models
order = rank_summary[rank_summary.Model == 'Random Forest'].sort_values('Mean rank')['Feature'].tolist()
fig, axes = plt.subplots(1, 2, figsize=(180 * MM, 88 * MM), layout='constrained',
                         gridspec_kw={'width_ratios': [1.5, 1]})
ypos = np.arange(len(order))[::-1]
for model, color, marker, off in [('Random Forest', ORANGE, 'o', 0.14), ('XGBoost', BLUE, 's', -0.14)]:
    g = rank_summary[rank_summary.Model == model].set_index('Feature').loc[order]
    axes[0].errorbar(g['Mean rank'], ypos + off, xerr=g['SD of rank'].fillna(0), fmt=marker, color=color,
                     capsize=3, ms=5, label=model)
axes[0].set_yticks(ypos)
axes[0].set_yticklabels([DISPLAY.get(f, f.capitalize()) for f in order])
axes[0].invert_xaxis()
axes[0].set_xlabel('Mean SHAP rank (1 = most important), bars show 1 SD')
axes[0].set_title('(A)', loc='left')
axes[0].legend(frameon=False, loc='upper left')
axes[1].hist(rest['Spearman (RF vs. XGBoost)'], bins=8, color='#999999', edgecolor='white')
axes[1].set_xlabel('Rank correlation between models')
axes[1].set_ylabel('Number of splits')
axes[1].set_title('(B)', loc='left')
save_paper(fig, 'fig3_explanation_stability')


def dependence_figure(sv, X_imp, name):
    paper_rc()
    fig, axes = plt.subplots(2, 2, figsize=(180 * MM, 150 * MM), layout='constrained')
    for i, (ax, feat) in enumerate(zip(axes.flat, FOUR)):
        shap.dependence_plot(feat, sv, X_imp, ax=ax, show=False, cmap=paper_cmap)
        ax.set_title(f'({chr(65 + i)}) {DISPLAY.get(feat, feat)}', loc='left')
    for a in fig.axes:
        a.tick_params(labelsize=8)
        a.xaxis.label.set_size(8)
        a.yaxis.label.set_size(8)
    save_paper(fig, name)


# Figure 4: Random Forest dependence plots; Figure S1: the same for XGBoost
dependence_figure(*main_store['Random Forest'], 'fig4_dependence_random_forest')
dependence_figure(*main_store['XGBoost'], 'figS1_dependence_xgboost')

# Figure 5: share of potable samples by decile of the feature (raw data, measured values only)
paper_rc()
fig, axes = plt.subplots(2, 2, figsize=(180 * MM, 120 * MM), layout='constrained', sharey=True)
base_rate = y.mean()
for i, (ax, feat) in enumerate(zip(axes.flat, FOUR)):
    d = df[[feat, 'Potability']].dropna().copy()
    d['Decile'] = pd.qcut(d[feat], 10, labels=False, duplicates='drop') + 1
    grp = d.groupby('Decile')['Potability'].agg(['mean', 'count'])
    half = 1.96 * np.sqrt(grp['mean'] * (1 - grp['mean']) / grp['count'])
    ax.errorbar(grp.index, grp['mean'], yerr=half, fmt='o-', color=BLUE, capsize=3, ms=5)
    ax.axhline(base_rate, color='#999999', ls='-.')
    ax.set_xticks(range(1, 11))
    ax.set_xlabel('Decile of ' + DISPLAY.get(feat, feat) + ' (1 = lowest)')
    ax.set_title(f'({chr(65 + i)}) {DISPLAY.get(feat, feat)}', loc='left')
    if i % 2 == 0:
        ax.set_ylabel('Share of potable samples')
save_paper(fig, 'fig5_raw_data_deciles')
print('\nSaved paper figures: fig3_explanation_stability, fig4_dependence_random_forest, figS1_dependence_xgboost,')
print('fig5_raw_data_deciles (.pdf and 300 dpi .tiff)')

print('\nSaved: paper_shap_rank_stability.csv, paper_model_agreement.csv, paper_u_shape_summary.csv,')
print('paper_shap_ranks_long.csv, paper_u_shape_long.csv, paper_explanation_stability.png,')
print('paper_model_free_spread.csv, paper_model_free_deciles.csv, paper_model_free_missingness.csv')
import platform, sklearn, scipy, matplotlib
print('\nEnvironment: python', platform.python_version(), '| numpy', np.__version__, '| pandas', pd.__version__,
      '| scikit-learn', sklearn.__version__, '| scipy', scipy.__version__, '| xgboost', xgb.__version__,
      '| shap', shap.__version__, '| matplotlib', matplotlib.__version__, '| pillow', Image.__version__)
if QUICK:
    print('\n(--quick was used: settings are small, so these results are for testing only.)')

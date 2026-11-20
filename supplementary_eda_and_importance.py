"""Corrected supplementary EDA and feature-importance analysis.

This is a separate audit copy. The original supplementary_eda_and_importance.py
is intentionally left unchanged.

Run from any working directory:
    python supplementary_eda_and_importance.py
The CSV is expected beside this script. Outputs are saved beside this script.
"""
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance
from sklearn.metrics import confusion_matrix

RANDOM_STATE = 42
N_PERMUTATION_REPEATS = 30
HERE = Path(__file__).resolve().parent
DATA_PATH = HERE / "water_potability.csv"

PALETTE = {
    "navy": "#1E2A3A", "rf": "#1a7a4c", "xgb": "#2563a8",
    "accent_orange": "#e67e22", "not_potable": "#c0392b",
    "potable": "#1a7a4c", "grid": "#e3e6ea", "muted": "#6b7785",
}
plt.rcParams.update({
    "font.family": "DejaVu Sans", "axes.titleweight": "bold",
    "axes.titlesize": 13, "axes.titlecolor": PALETTE["navy"],
    "axes.labelcolor": PALETTE["navy"], "xtick.color": PALETTE["muted"],
    "ytick.color": PALETTE["muted"], "axes.spines.top": False,
    "axes.spines.right": False, "figure.facecolor": "white",
    "axes.edgecolor": PALETTE["grid"],
})

if not DATA_PATH.exists():
    raise SystemExit(
        f"Dataset not found at {DATA_PATH}. Place water_potability.csv beside this script."
    )

df = pd.read_csv(DATA_PATH)
if "Potability" not in df.columns:
    raise ValueError("The dataset must contain a 'Potability' target column.")
if df["Potability"].isna().any():
    raise ValueError("The Potability target contains missing values; inspect before modelling.")
feature_cols = [c for c in df.columns if c != "Potability"]
X = df[feature_cols]
y = df["Potability"]
if not set(pd.unique(y)).issubset({0, 1}):
    raise ValueError("Potability must be encoded as binary values 0 and 1.")

# 1. Exploratory figures ----------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
miss = df[feature_cols].isna().sum()
miss = miss[miss > 0].sort_values(ascending=True)
if not miss.empty:
    miss_pct = miss / len(df) * 100
    bars = axes[0].barh(miss.index, miss_pct.values,
                        color=PALETTE["accent_orange"], height=0.55)
    for bar, count, pct in zip(bars, miss.values, miss_pct.values):
        axes[0].text(pct + 0.6, bar.get_y() + bar.get_height() / 2,
                     f"{count} ({pct:.1f}%)", va="center", fontsize=9)
    axes[0].set_xlim(0, miss_pct.max() * 1.35)
else:
    axes[0].text(0.5, 0.5, "No missing feature values", ha="center", va="center")
axes[0].set_xlabel("Missing (%)")
axes[0].set_title("Missing Values by Feature")
axes[0].grid(axis="x", alpha=0.5)

counts = y.value_counts().reindex([0, 1], fill_value=0)
bars2 = axes[1].bar(["Not Potable", "Potable"], counts.values,
                    color=[PALETTE["not_potable"], PALETTE["potable"]], width=0.5)
for bar, value in zip(bars2, counts.values):
    axes[1].text(bar.get_x() + bar.get_width() / 2, value + max(counts) * 0.025,
                 f"{value}\n({value / len(df) * 100:.1f}%)", ha="center", fontsize=9)
axes[1].set_ylabel("Number of Samples")
axes[1].set_title("Class Distribution")
axes[1].set_ylim(0, max(counts) * 1.22)
axes[1].grid(axis="y", alpha=0.5)
fig.suptitle("Exploratory Data Overview", fontsize=15, fontweight="bold", y=1.02)
plt.tight_layout()
fig.savefig(HERE / "eda_overview_combined_corrected.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# Correlation heatmap intentionally includes only the nine predictor variables.
fig, ax = plt.subplots(figsize=(8.5, 6.8))
corr = df[feature_cols].corr(numeric_only=True)
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
            square=True, linewidths=0.6, linecolor="white",
            cbar_kws={"shrink": 0.8}, ax=ax, annot_kws={"size": 9})
ax.set_title("Correlation Heatmap of Water-Quality Features", fontsize=14, pad=14)
plt.tight_layout()
fig.savefig(HERE / "eda_correlation_heatmap_corrected.png", dpi=200)
plt.close(fig)

# Melt predictor columns only: Potability is the grouping variable, never a feature plot.
box_data = df[feature_cols + ["Potability"]].melt(
    id_vars="Potability", var_name="Feature", value_name="Value"
)
box_data["Potability"] = box_data["Potability"].map({0: "Not Potable", 1: "Potable"})
g = sns.FacetGrid(box_data, col="Feature", col_wrap=3, sharey=False,
                  sharex=False, height=2.9, aspect=1.15)
g.map_dataframe(
    sns.boxplot, x="Potability", y="Value", hue="Potability",
    palette={"Potable": PALETTE["potable"], "Not Potable": PALETTE["not_potable"]},
    legend=False, width=0.5, linewidth=1,
)
g.set_titles("{col_name}", fontweight="bold", color=PALETTE["navy"])
g.fig.suptitle("Feature Distributions by Potability", fontsize=15,
                fontweight="bold", y=1.02)
plt.tight_layout()
g.savefig(HERE / "eda_boxplots_by_class_corrected.png", dpi=200, bbox_inches="tight")
plt.close(g.fig)
print("Saved corrected exploratory figures.")

# 2. Random Forest native and permutation importance -----------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)
rf_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("model", RandomForestClassifier(
        n_estimators=300, class_weight="balanced", random_state=RANDOM_STATE,
        n_jobs=-1,
    )),
])
rf_pipeline.fit(X_train, y_train)

perm = permutation_importance(
    rf_pipeline, X_test, y_test, n_repeats=N_PERMUTATION_REPEATS,
    random_state=RANDOM_STATE, scoring="roc_auc", n_jobs=-1,
)
importance = pd.DataFrame({
    "feature": feature_cols,
    "permutation_auc_drop_mean": perm.importances_mean,
    "permutation_auc_drop_sd_across_repeats": perm.importances_std,
    "native_gini_importance": rf_pipeline.named_steps["model"].feature_importances_,
})
importance["permutation_rank"] = importance["permutation_auc_drop_mean"].rank(
    ascending=False, method="min"
).astype(int)
importance["native_gini_rank"] = importance["native_gini_importance"].rank(
    ascending=False, method="min"
).astype(int)
importance = importance.sort_values(
    ["native_gini_rank", "feature"]
).reset_index(drop=True)
importance.to_csv(HERE / "rf_importance_values_and_ranks_corrected.csv", index=False)
print("\nRandom Forest importance values and ranks:")
print(importance.round(5).to_string(index=False))

# Compare ranks, not incompatible magnitudes. Rank 1 is most important.
rank_plot = importance.sort_values("native_gini_rank", ascending=True)
y_pos = np.arange(len(rank_plot))
fig, ax = plt.subplots(figsize=(9, 5.8))
ax.scatter(rank_plot["native_gini_rank"], y_pos, label="Native Gini rank",
           color=PALETTE["rf"], s=55, zorder=3)
ax.scatter(rank_plot["permutation_rank"], y_pos, label="Permutation rank",
           color=PALETTE["accent_orange"], s=55, zorder=3)
for _, row in rank_plot.iterrows():
    ax.plot([row["native_gini_rank"], row["permutation_rank"]],
            [list(rank_plot["feature"]).index(row["feature"])] * 2,
            color=PALETTE["grid"], linewidth=1, zorder=1)
ax.set_yticks(y_pos)
ax.set_yticklabels(rank_plot["feature"])
ax.invert_yaxis()
ax.set_xticks(range(1, len(feature_cols) + 1))
ax.set_xlim(0.5, len(feature_cols) + 0.5)
ax.set_xlabel("Importance rank (1 = highest importance)")
ax.set_title("Random Forest Feature-Importance Rank Comparison")
ax.grid(axis="x", alpha=0.4)
ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2)
plt.tight_layout()
fig.savefig(HERE / "rf_importance_rank_comparison_corrected.png", dpi=200)
plt.close(fig)

# 3. Confusion matrices at the default 0.5 decision threshold --------------
rf_pred = rf_pipeline.predict(X_test)
rf_cm = confusion_matrix(y_test, rf_pred, labels=[0, 1])
try:
    import xgboost as xgb
except ImportError:
    xgb = None

if xgb is not None:
    xgb_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", xgb.XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
            random_state=RANDOM_STATE, eval_metric="logloss", n_jobs=-1,
        )),
    ])
    xgb_pipeline.fit(X_train, y_train)
    xgb_pred = xgb_pipeline.predict(X_test)
    xgb_cm = confusion_matrix(y_test, xgb_pred, labels=[0, 1])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    panels = [(axes[0], rf_cm, "Greens", "Random Forest"),
              (axes[1], xgb_cm, "Blues", "XGBoost")]
    for ax, cm, cmap, title in panels:
        sns.heatmap(cm, annot=True, fmt="d", cmap=cmap, cbar=False,
                    xticklabels=["Not Potable", "Potable"],
                    yticklabels=["Not Potable", "Potable"], ax=ax,
                    annot_kws={"size": 15, "color": PALETTE["navy"]},
                    linewidths=1, linecolor="white")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(title)
    fig.suptitle("Confusion Matrices at Threshold 0.5 (Test Set)",
                 fontsize=15, fontweight="bold", y=1.03)
    plt.tight_layout()
    fig.savefig(HERE / "confusion_matrices_combined_corrected.png",
                dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("\nRF confusion matrix (rows=true, columns=predicted; labels 0,1):\n", rf_cm)
    print("XGBoost confusion matrix (rows=true, columns=predicted; labels 0,1):\n", xgb_cm)
else:
    print("\nXGBoost not installed; generated Random Forest confusion matrix only.")
    print("RF confusion matrix (rows=true, columns=predicted; labels 0,1):\n", rf_cm)

print("\nCompleted corrected supplementary analysis.")

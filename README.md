# Explainable Machine Learning for Water Potability Prediction

[![DOI](https://zenodo.org/badge/1418388413.svg)](https://doi.org/10.5281/zenodo.23495833)

This repository accompanies the M.Tech. thesis *An Explainable Machine Learning Approach to Water Potability Prediction* by Anas Umar Aliyu, Mewar University, India, and related manuscript analyses.

The project evaluates Random Forest (RF) and XGBoost classifiers on the public Kaggle Water Potability dataset and examines model explanations using SHAP, permutation importance, and native model importance.

## Dataset

Download `water_potability.csv` from [Kaggle](https://www.kaggle.com/datasets/adityakadiwal/water-potability). The dataset is not redistributed here. It contains 3,276 samples, nine physicochemical predictors, and a binary `Potability` label. The original sampling and collection process is not documented.

## Scripts

| Script | Purpose |
|---|---|
| `water_potability_full_analysis_v2.py` | EDA, leakage-aware preprocessing, RF/XGBoost modelling, cross-validation, test evaluation, and RF SHAP analysis |
| `supplementary_eda_and_importance.py` | Supplementary EDA figures, RF permutation/native importance, and confusion-matrix visualisation |
| `regenerate_themed_plots.py` | Regenerates thesis-style ROC and RF SHAP plots |
| `xgboost_shap_comparison.py` | XGBoost SHAP explanations and comparison with RF feature rankings |
| `paper_performance_analyses.py` | Bootstrap uncertainty, paired comparisons, repeated splits, label-shuffling, tuning, and cost-sensitive analyses |
| `paper_explanation_stability.py` | Explanation-rank stability, cross-model agreement, dependence-curve patterns, and raw-data checks |

The `original_scripts/` directory preserves the versions that were previously present at the repository root. The root scripts are intended to contain the consolidated versions. Audit notes and representative outputs are provided in the accompanying package.

## Main analysis results

In the reported Google Colab run (Python 3.13.18; package versions listed in `requirements.txt`):

| Model | Training CV ROC-AUC (mean) | Test ROC-AUC |
|---|---:|---:|
| Random Forest | 0.6909 | 0.6611 |
| XGBoost | 0.6611 | 0.6347 |

Random Forest was selected for the primary SHAP analysis because it had the higher training cross-validation ROC-AUC. The difference should not be described as definitive model superiority: the paired test-set AUC difference was small and its confidence interval was approximately -0.000 to 0.055.

The test results also show a trade-off in potable-water detection. RF had potable-class recall of approximately 0.293, while XGBoost had approximately 0.477. These results do not establish a sufficiently reliable stand-alone water-safety screening system.

Feature-importance comparisons consistently placed Sulfate and pH among the leading predictors, with Hardness also ranking highly. These are explanations of model behaviour, not evidence of causal effects or proof of drinking-water safety.

## Reproducibility

- Main split: stratified 80/20 train/test split; random seed 42.
- Cross-validation: stratified five-fold CV on the training data.
- Missing values: median imputation within scikit-learn pipelines to avoid fitting preprocessing on the held-out test data.
- Reported environment: Python 3.13.18, pandas 2.2.3, NumPy 2.1.3, scikit-learn 1.6.1, XGBoost 3.4.1, SHAP 0.52.0, matplotlib 3.10.0, seaborn 0.13.2, SciPy 1.16.3, Pillow 11.3.0.
- Use `requirements.txt` to install the pinned dependencies.

### Run in Google Colab

```bash
!git clone https://github.com/Anas-Umar/Water-potability-explainable-ML.git
%cd Water-potability-explainable-ML
!pip install -r requirements.txt
```

Upload `water_potability.csv` into the working directory, then run the required script:

```bash
!python water_potability_full_analysis_v2.py
```

Each supplementary script can be run separately. The paper analysis scripts may take longer than the main analysis.

## Evidence and limitations

Execution results reported by the author from Google Colab were reviewed during preparation of this repository update. This is not an independent third-party reproduction. The full Colab notebook/session and complete generated output directory are not archived here. Results may vary with library versions, platform, or execution settings.

The dataset's undocumented provenance, class imbalance, and modest test performance limit the conclusions that can be drawn. The models should not be used as substitutes for laboratory testing or regulatory water-quality assessment.

## Citation and licence

Aliyu, A. U. (2026). *An Explainable Machine Learning Approach to Water Potability Prediction* [M.Tech. thesis]. Mewar University.

See [LICENSE](LICENSE) for licence terms.

# Environment and execution audit

## Reported execution environment

The author reported successful execution of all six consolidated analysis scripts in Google Colab using:

- Python 3.13.18
- pandas 2.2.3
- NumPy 2.1.3
- scikit-learn 1.6.1
- XGBoost 3.4.1
- SHAP 0.52.0
- matplotlib 3.10.0
- seaborn 0.13.2
- SciPy 1.16.3
- Pillow 11.3.0

The dependency pins are recorded in `requirements.txt`.

## Main model results reported by the author

| Model | Training CV ROC-AUC | Test ROC-AUC |
|---|---:|---:|
| Random Forest | 0.690876 ± 0.011491 | 0.661094 |
| XGBoost | 0.661114 ± 0.018433 | 0.634746 |

Random Forest was selected for the primary SHAP analysis based on its higher training CV ROC-AUC. A paired test-set comparison did not establish clear superiority: the RF-minus-XGBoost AUC difference was approximately 0.026, with a confidence interval approximately -0.000 to 0.055.

## Scope and evidence boundary

The execution outputs and terminal summaries supplied by the author were reviewed. The author reported that all six scripts completed successfully. This audit is not an independent replay of the Colab environment: the complete notebook/session, dataset snapshot, and full output directory have not been archived with this repository.

The repository should therefore be described as having author-reported successful execution with reviewed outputs, not as independently reproduced by a third party.

## Interpretation cautions

- Test performance is modest; potable-class recall is limited for the default RF operating point.
- The dataset has undocumented provenance and should not be treated as a representative regulatory monitoring sample.
- SHAP and feature-importance results describe fitted model behaviour, not causality.
- Repeated random splits are sensitivity analyses, not independent external validation.
- A model prediction must not replace laboratory testing or regulatory assessment.

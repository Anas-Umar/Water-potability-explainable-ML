# Reproducibility status

## Status

The author reports that all six consolidated scripts completed successfully in Google Colab. The reported runtime used Python 3.13.18 and the exact package versions pinned in `requirements.txt`. The terminal outputs supplied by the author were reviewed.

This status is an author-reported execution record with reviewed outputs. It is not an independent reproduction by a separate researcher or machine.

## Main results

- Random Forest: training five-fold CV ROC-AUC = 0.690876 ± 0.011491; test ROC-AUC = 0.661094.
- XGBoost: training five-fold CV ROC-AUC = 0.661114 ± 0.018433; test ROC-AUC = 0.634746.
- RF was selected for the primary SHAP analysis based on the higher training CV ROC-AUC.
- Paired test-set AUC difference (RF minus XGBoost) was approximately 0.026, with a confidence interval approximately -0.000 to 0.055; do not claim conclusive RF superiority.
- Sulfate and pH were consistently among the top-ranked features across the reported importance and explanation analyses.

## What is needed for independent reproduction

For a fully auditable independent replay, archive:
1. The exact dataset file and its provenance/licence details, or a stable source/version identifier.
2. The executed Colab notebook with cell outputs.
3. The complete output directory, including generated figures, tables, and logs.
4. The exact repository commit and dependency environment used for the run.

## Limitations

The public benchmark has undocumented source sampling and class imbalance. Model performance is modest, and the potable-class recall is not sufficient to justify stand-alone safety decisions. Explanations are model-specific associations, not causal findings. Repeated random splits do not constitute external validation.

# COMPAS Benchmark (ProPublica Two-Year Sample)

This benchmark is included as a fairness/evaluation case study; it is not directly comparable to NIJ.

## Baselines

| Model | Calibration | Brier | AUROC | AUPRC | Log Loss | ECE |
|---|---|---:|---:|---:|---:|---:|
| lasso_logistic | platt | 0.18918 | 0.77703 | 0.76688 | 0.56025 | 0.03391 |
| logistic | platt | 0.19077 | 0.77383 | 0.76120 | 0.56495 | 0.03851 |
| naive_demographic | platt | 0.23595 | 0.62689 | 0.58481 | 0.67074 | 0.02893 |
| base_rate | platt | 0.24798 | 0.50000 | 0.72756 | 0.68911 | 0.01039 |

## XGBoost

| Model | Calibration | Brier | AUROC | AUPRC | Log Loss | ECE |
|---|---|---:|---:|---:|---:|---:|
| xgboost | raw | 0.17289 | 0.80775 | 0.80706 | 0.52067 | 0.02527 |
| xgboost | platt | 0.17298 | 0.80775 | 0.80706 | 0.52051 | 0.02258 |
| xgboost | isotonic | 0.17425 | 0.80614 | 0.80691 | 0.53280 | 0.03786 |

XGB importance/SHAP tables are written under `reports/plots/` as JSON.

Calibration plot specs are under `reports/plots/` as JSON (HTML-friendly).

## Takeaways

- On this COMPAS benchmark split, tuned XGBoost improves Brier over simple baseline models.
- Calibration variants are close in rank; raw and Platt are nearly tied on primary error in this run.
- NIJ and COMPAS are different datasets with different populations and label mechanisms — results should not be directly compared.

# NCRP Benchmark (ICPSR 37973 selected variables; term-record-derived reincarceration)

Variant: `terms_mod200_r0`.

- Label family: **reincarceration** (return to prison)
- Timing: **year granularity** (ADMITYR/RELEASEYR only in public-use extract)
- Splits: **grouped by ABT_INMATE_ID** (no person appears in both train/test)

## Baselines

| Horizon | Model | Calibration | Brier | AUROC | AUPRC | Log Loss | ECE |
|---|---|---|---:|---:|---:|---:|---:|
| y1 | lasso_logistic | platt | 0.17441 | 0.74320 | 0.55444 | 0.52554 | 0.01675 |
| y1 | logistic | platt | 0.17445 | 0.74313 | 0.55402 | 0.52562 | 0.01663 |
| y1 | naive_demographic | platt | 0.20719 | 0.53695 | 0.34581 | 0.60455 | 0.00743 |
| y1 | base_rate | platt | 0.20815 | 0.50000 | 0.64766 | 0.60694 | 0.00700 |
| y2 | logistic | platt | 0.08928 | 0.65922 | 0.16570 | 0.31607 | 0.00940 |
| y2 | lasso_logistic | platt | 0.08929 | 0.65984 | 0.16564 | 0.31608 | 0.00948 |
| y2 | naive_demographic | platt | 0.09151 | 0.54073 | 0.28189 | 0.32875 | 0.00548 |
| y2 | base_rate | platt | 0.09174 | 0.50000 | 0.55107 | 0.32992 | 0.00543 |
| y3 | logistic | platt | 0.05132 | 0.65624 | 0.08948 | 0.20632 | 0.00359 |
| y3 | lasso_logistic | platt | 0.05132 | 0.65999 | 0.08970 | 0.20630 | 0.00324 |
| y3 | naive_demographic | platt | 0.05182 | 0.56550 | 0.25403 | 0.21162 | 0.00229 |
| y3 | base_rate | platt | 0.05197 | 0.50000 | 0.52750 | 0.21299 | 0.00237 |

## XGBoost

| Horizon | Model | Calibration | Brier | AUROC | AUPRC | Log Loss | ECE |
|---|---|---|---:|---:|---:|---:|---:|
| y1 | xgboost | raw | 0.16610 | 0.77620 | 0.60022 | 0.50265 | 0.01235 |
| y1 | xgboost | platt | 0.16611 | 0.77620 | 0.60022 | 0.50265 | 0.01159 |
| y1 | xgboost | isotonic | 0.16627 | 0.77542 | 0.60193 | 0.50388 | 0.00845 |
| y2 | xgboost | platt | 0.08780 | 0.70111 | 0.20096 | 0.30348 | 0.00187 |
| y2 | xgboost | raw | 0.08780 | 0.70111 | 0.20096 | 0.30347 | 0.00185 |
| y2 | xgboost | isotonic | 0.08795 | 0.69889 | 0.20272 | 0.30586 | 0.00169 |
| y3 | xgboost | raw | 0.04825 | 0.69426 | 0.10057 | 0.18970 | 0.00500 |
| y3 | xgboost | platt | 0.04825 | 0.69426 | 0.10057 | 0.18976 | 0.00580 |
| y3 | xgboost | isotonic | 0.04834 | 0.69404 | 0.10072 | 0.19239 | 0.00435 |

XGB importance/SHAP tables are written under `reports/plots/` as JSON.

Calibration plot specs are under `reports/plots/` as JSON (HTML-friendly).
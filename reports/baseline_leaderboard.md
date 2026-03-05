# NIJ Static Baseline Leaderboard

Primary sort: Brier score (lower is better).

| Horizon | Model | Brier | AUROC | AUPRC | Log Loss | ECE |
|---|---:|---:|---:|---:|---:|---:|
| y1 | lasso_logistic | 0.18796 | 0.69899 | 0.48022 | 0.55564 | 0.00742 |
| y1 | logistic | 0.18814 | 0.69823 | 0.47941 | 0.55605 | 0.00534 |
| y1 | naive_demographic | 0.20464 | 0.59385 | 0.35943 | 0.59802 | 0.00300 |
| y1 | base_rate | 0.20941 | 0.49209 | 0.36174 | 0.60963 | 0.01132 |
| y2 | lasso_logistic | 0.17821 | 0.67158 | 0.39255 | 0.53497 | 0.00907 |
| y2 | logistic | 0.17838 | 0.67091 | 0.39179 | 0.53540 | 0.00748 |
| y2 | naive_demographic | 0.18823 | 0.58157 | 0.30857 | 0.56247 | 0.00752 |
| y2 | base_rate | 0.19104 | 0.49865 | 0.33208 | 0.57009 | 0.00194 |
| y3 | lasso_logistic | 0.14790 | 0.65264 | 0.28593 | 0.46462 | 0.01439 |
| y3 | logistic | 0.14813 | 0.65073 | 0.28506 | 0.46534 | 0.01729 |
| y3 | naive_demographic | 0.15309 | 0.56598 | 0.22342 | 0.48347 | 0.00642 |
| y3 | base_rate | 0.15435 | 0.49760 | 0.27478 | 0.48735 | 0.00185 |

Calibration specs are saved under `reports/plots/` as JSON (HTML-friendly).
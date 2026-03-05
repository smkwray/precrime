# NIJ XGBoost Leaderboard

Primary sort: Brier score (lower is better).

| Dataset | Horizon | Model | Calibration | Brier | AUROC | AUPRC | Log Loss | ECE |
|---|---|---|---|---:|---:|---:|---:|---:|
| dynamic | y1 | xgboost | platt | 0.18758 | 0.70164 | 0.47910 | 0.55461 | 0.00746 |
| dynamic | y1 | xgboost | raw | 0.18763 | 0.70164 | 0.47910 | 0.55480 | 0.01207 |
| dynamic | y1 | xgboost | isotonic | 0.18829 | 0.69936 | 0.48079 | 0.55942 | 0.01170 |
| dynamic | y2 | xgboost | isotonic | 0.17244 | 0.70475 | 0.42645 | 0.53589 | 0.01715 |
| dynamic | y2 | xgboost | platt | 0.17252 | 0.70888 | 0.42229 | 0.51308 | 0.02131 |
| dynamic | y2 | xgboost | raw | 0.17341 | 0.70888 | 0.42229 | 0.51560 | 0.03297 |
| dynamic | y3 | xgboost | isotonic | 0.14324 | 0.69594 | 0.31555 | 0.47159 | 0.01232 |
| dynamic | y3 | xgboost | raw | 0.14370 | 0.69722 | 0.31346 | 0.44738 | 0.01428 |
| dynamic | y3 | xgboost | platt | 0.14374 | 0.69722 | 0.31346 | 0.44743 | 0.01983 |
| static | y1 | xgboost | platt | 0.18758 | 0.70164 | 0.47910 | 0.55461 | 0.00746 |
| static | y1 | xgboost | raw | 0.18763 | 0.70164 | 0.47910 | 0.55480 | 0.01207 |
| static | y1 | xgboost | isotonic | 0.18829 | 0.69936 | 0.48079 | 0.55942 | 0.01170 |
| static | y2 | xgboost | platt | 0.17971 | 0.66353 | 0.37390 | 0.53802 | 0.01403 |
| static | y2 | xgboost | raw | 0.17975 | 0.66353 | 0.37390 | 0.53809 | 0.01288 |
| static | y2 | xgboost | isotonic | 0.18012 | 0.65903 | 0.37007 | 0.54277 | 0.01546 |
| static | y3 | xgboost | platt | 0.14812 | 0.65381 | 0.28016 | 0.46437 | 0.01329 |
| static | y3 | xgboost | raw | 0.14820 | 0.65381 | 0.28016 | 0.46463 | 0.01906 |
| static | y3 | xgboost | isotonic | 0.14871 | 0.64899 | 0.27950 | 0.47247 | 0.02446 |

Includes raw, Platt, and isotonic calibration variants.
Feature-importance and SHAP summary specs are under `reports/plots/` as JSON.
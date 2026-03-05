# Seed-Ensemble Evaluation (NIJ Best XGBoost Configs)

Generated: 2026-03-02 15:55 EST

Split seed: `42` (fixed). Model seeds ensembled: `42,43,44,45,46`.

Ensembling method: average predicted probabilities across model seeds, then optionally recalibrate once (Platt / isotonic).

| Horizon | Dataset | Variant | Calibration | bs_sex_avg | fairacc_sex_avg | Overall Brier | AUROC | AUPRC | ECE |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| Y1 | static | single_seed42 | platt | 0.16829 | 0.83071 | 0.18737 | 0.70252 | 0.48060 | 0.01218 |
| Y1 | static | seed_ensemble | raw | 0.16850 | 0.82832 | 0.18751 | 0.70234 | 0.47924 | 0.01078 |
| Y1 | static | seed_ensemble | platt | 0.16838 | 0.82831 | 0.18746 | 0.70234 | 0.47924 | 0.01151 |
| Y1 | static | seed_ensemble | isotonic | 0.16914 | 0.82671 | 0.18836 | 0.69878 | 0.47667 | 0.01087 |
| Y2 | dynamic | single_seed42 | isotonic | 0.16173 | 0.82904 | 0.17309 | 0.70130 | 0.41628 | 0.01925 |
| Y2 | dynamic | seed_ensemble | raw | 0.16223 | 0.82694 | 0.17360 | 0.70828 | 0.42237 | 0.03597 |
| Y2 | dynamic | seed_ensemble | platt | 0.16154 | 0.82941 | 0.17276 | 0.70828 | 0.42237 | 0.02630 |
| Y2 | dynamic | seed_ensemble | isotonic | 0.16170 | 0.82896 | 0.17296 | 0.70720 | 0.42298 | 0.02239 |
| Y3 | dynamic | single_seed42 | isotonic | 0.12675 | 0.87194 | 0.14325 | 0.69612 | 0.31812 | 0.01092 |
| Y3 | dynamic | seed_ensemble | raw | 0.12678 | 0.87125 | 0.14347 | 0.69807 | 0.31593 | 0.01115 |
| Y3 | dynamic | seed_ensemble | platt | 0.12676 | 0.87128 | 0.14351 | 0.69807 | 0.31593 | 0.01218 |
| Y3 | dynamic | seed_ensemble | isotonic | 0.12712 | 0.87288 | 0.14346 | 0.69439 | 0.32060 | 0.01272 |

Notes:
- This keeps the data split fixed and varies only model random seeds.
- If you want a more pessimistic robustness check, vary the split seed and ensemble across splits (but then test sets differ).
# NIJ Model Sweep (Dependency-Free)

Generated: 2026-03-02 15:55 EST

This sweep compares several model families on a fixed seeded split (seed 42).
Primary sorting is by sex-average Brier (NIJ-style accuracy view).
A secondary NIJ-style view (“FairAcc”) is included to show how rankings can change if you factor in an explicit parity term.

Key NIJ-style terms:
- `bs_sex_avg`: average of male/female Brier errors (lower is better).
- `fairacc_sex_avg`: average of male/female FairAcc indices (higher is better).

## Y1 — with_race

Top by prediction (`bs_sex_avg`): `logistic_gd (platt)`. Top by FairAcc: `logistic_gd (raw)`.

| Dataset | Model | Calibration | bs_sex_avg | fairacc_sex_avg | Overall Brier | AUROC | AUPRC | ECE |
|---|---|---|---:|---:|---:|---:|---:|---:|
| static | logistic_gd | platt | 0.16754 | 0.83112 | 0.18650 | 0.70487 | 0.49208 | 0.00630 |
| static | logistic_gd | isotonic | 0.16762 | 0.83006 | 0.18685 | 0.70128 | 0.49201 | 0.01296 |
| static | logistic_gd | raw | 0.16765 | 0.83171 | 0.18654 | 0.70487 | 0.49208 | 0.00940 |
| static | xgb_retuned | isotonic | 0.16797 | 0.83059 | 0.18768 | 0.70024 | 0.48132 | 0.01198 |
| static | xgb_retuned | platt | 0.16813 | 0.83141 | 0.18732 | 0.70281 | 0.48119 | 0.01056 |
| static | xgb_retuned | raw | 0.16821 | 0.83106 | 0.18735 | 0.70281 | 0.48119 | 0.01045 |
| static | xgb_best | platt | 0.16829 | 0.83071 | 0.18737 | 0.70252 | 0.48060 | 0.01218 |
| static | hist_gb | platt | 0.17137 | 0.81516 | 0.19031 | 0.68962 | 0.45850 | 0.01091 |
| static | hist_gb | raw | 0.17141 | 0.81534 | 0.19037 | 0.68962 | 0.45850 | 0.01128 |
| static | extra_trees | platt | 0.17177 | 0.82356 | 0.19055 | 0.68913 | 0.45809 | 0.01863 |
| static | extra_trees | raw | 0.17183 | 0.82443 | 0.19056 | 0.68913 | 0.45809 | 0.01626 |
| static | hist_gb | isotonic | 0.17189 | 0.81416 | 0.19156 | 0.68829 | 0.45628 | 0.01028 |
| static | random_forest | platt | 0.17201 | 0.81861 | 0.19025 | 0.69037 | 0.46325 | 0.00806 |
| static | random_forest | raw | 0.17234 | 0.81920 | 0.19039 | 0.69037 | 0.46325 | 0.01508 |
| static | extra_trees | isotonic | 0.17239 | 0.82607 | 0.19129 | 0.68758 | 0.46478 | 0.02692 |
| static | random_forest | isotonic | 0.17245 | 0.81995 | 0.19097 | 0.68820 | 0.46296 | 0.01654 |

### Alternate ranking (FairAcc-first; NIJ-style “fair-and-accurate” view)

This is the same block sorted by `fairacc_sex_avg` (descending).

| Dataset | Model | Calibration | fairacc_sex_avg | bs_sex_avg | Overall Brier | AUROC | AUPRC | ECE |
|---|---|---|---:|---:|---:|---:|---:|---:|
| static | logistic_gd | raw | 0.83171 | 0.16765 | 0.18654 | 0.70487 | 0.49208 | 0.00940 |
| static | xgb_retuned | platt | 0.83141 | 0.16813 | 0.18732 | 0.70281 | 0.48119 | 0.01056 |
| static | logistic_gd | platt | 0.83112 | 0.16754 | 0.18650 | 0.70487 | 0.49208 | 0.00630 |
| static | xgb_retuned | raw | 0.83106 | 0.16821 | 0.18735 | 0.70281 | 0.48119 | 0.01045 |
| static | xgb_best | platt | 0.83071 | 0.16829 | 0.18737 | 0.70252 | 0.48060 | 0.01218 |
| static | xgb_retuned | isotonic | 0.83059 | 0.16797 | 0.18768 | 0.70024 | 0.48132 | 0.01198 |
| static | logistic_gd | isotonic | 0.83006 | 0.16762 | 0.18685 | 0.70128 | 0.49201 | 0.01296 |
| static | extra_trees | isotonic | 0.82607 | 0.17239 | 0.19129 | 0.68758 | 0.46478 | 0.02692 |

## Y2 — with_race

Top by prediction (`bs_sex_avg`): `xgb_best (isotonic)`. Top by FairAcc: `hist_gb (isotonic)`.

| Dataset | Model | Calibration | bs_sex_avg | fairacc_sex_avg | Overall Brier | AUROC | AUPRC | ECE |
|---|---|---|---:|---:|---:|---:|---:|---:|
| dynamic | xgb_best | isotonic | 0.16173 | 0.82904 | 0.17309 | 0.70130 | 0.41628 | 0.01925 |
| dynamic | hist_gb | platt | 0.16242 | 0.82279 | 0.17425 | 0.70304 | 0.40698 | 0.02376 |
| dynamic | hist_gb | isotonic | 0.16255 | 0.83604 | 0.17426 | 0.70118 | 0.41006 | 0.02071 |
| dynamic | random_forest | platt | 0.16330 | 0.83389 | 0.17490 | 0.69479 | 0.39444 | 0.01917 |
| dynamic | random_forest | raw | 0.16373 | 0.83361 | 0.17501 | 0.69479 | 0.39444 | 0.02523 |
| dynamic | random_forest | isotonic | 0.16410 | 0.82694 | 0.17646 | 0.69311 | 0.39846 | 0.02259 |
| dynamic | hist_gb | raw | 0.16448 | 0.82381 | 0.17684 | 0.70304 | 0.40698 | 0.04855 |
| dynamic | logistic_gd | isotonic | 0.16502 | 0.83137 | 0.17618 | 0.68730 | 0.40165 | 0.03039 |
| dynamic | logistic_gd | platt | 0.16534 | 0.82362 | 0.17655 | 0.68709 | 0.40123 | 0.02334 |
| dynamic | logistic_gd | raw | 0.16545 | 0.82675 | 0.17674 | 0.68709 | 0.40123 | 0.02200 |
| dynamic | extra_trees | platt | 0.16572 | 0.83149 | 0.17762 | 0.67636 | 0.37716 | 0.02464 |
| dynamic | extra_trees | raw | 0.16577 | 0.83351 | 0.17747 | 0.67636 | 0.37716 | 0.01832 |
| dynamic | extra_trees | isotonic | 0.16669 | 0.82735 | 0.17923 | 0.67329 | 0.37280 | 0.02614 |

### Alternate ranking (FairAcc-first; NIJ-style “fair-and-accurate” view)

This is the same block sorted by `fairacc_sex_avg` (descending).

| Dataset | Model | Calibration | fairacc_sex_avg | bs_sex_avg | Overall Brier | AUROC | AUPRC | ECE |
|---|---|---|---:|---:|---:|---:|---:|---:|
| dynamic | hist_gb | isotonic | 0.83604 | 0.16255 | 0.17426 | 0.70118 | 0.41006 | 0.02071 |
| dynamic | random_forest | platt | 0.83389 | 0.16330 | 0.17490 | 0.69479 | 0.39444 | 0.01917 |
| dynamic | random_forest | raw | 0.83361 | 0.16373 | 0.17501 | 0.69479 | 0.39444 | 0.02523 |
| dynamic | extra_trees | raw | 0.83351 | 0.16577 | 0.17747 | 0.67636 | 0.37716 | 0.01832 |
| dynamic | extra_trees | platt | 0.83149 | 0.16572 | 0.17762 | 0.67636 | 0.37716 | 0.02464 |
| dynamic | logistic_gd | isotonic | 0.83137 | 0.16502 | 0.17618 | 0.68730 | 0.40165 | 0.03039 |
| dynamic | xgb_best | isotonic | 0.82904 | 0.16173 | 0.17309 | 0.70130 | 0.41628 | 0.01925 |
| dynamic | extra_trees | isotonic | 0.82735 | 0.16669 | 0.17923 | 0.67329 | 0.37280 | 0.02614 |

## Y3 — with_race

Top by prediction (`bs_sex_avg`): `hist_gb (isotonic)`.

| Dataset | Model | Calibration | bs_sex_avg | fairacc_sex_avg | Overall Brier | AUROC | AUPRC | ECE |
|---|---|---|---:|---:|---:|---:|---:|---:|
| dynamic | hist_gb | isotonic | 0.12640 | 0.87360 | 0.14358 | 0.69142 | 0.33370 | 0.00650 |
| dynamic | hist_gb | platt | 0.12664 | 0.86932 | 0.14460 | 0.69124 | 0.31057 | 0.01362 |
| dynamic | xgb_best | isotonic | 0.12675 | 0.87194 | 0.14325 | 0.69612 | 0.31812 | 0.01092 |
| dynamic | hist_gb | raw | 0.12731 | 0.86695 | 0.14679 | 0.69124 | 0.31057 | 0.04593 |
| dynamic | random_forest | platt | 0.12840 | 0.87160 | 0.14480 | 0.67806 | 0.31335 | 0.00659 |
| dynamic | random_forest | isotonic | 0.12868 | 0.87045 | 0.14521 | 0.67376 | 0.28145 | 0.00627 |
| dynamic | random_forest | raw | 0.12873 | 0.87127 | 0.14482 | 0.67806 | 0.31335 | 0.00999 |
| dynamic | logistic_gd | isotonic | 0.12896 | 0.87104 | 0.14666 | 0.65571 | 0.29933 | 0.02184 |
| dynamic | logistic_gd | platt | 0.12952 | 0.86895 | 0.14710 | 0.66522 | 0.29624 | 0.02218 |
| dynamic | logistic_gd | raw | 0.12999 | 0.86998 | 0.14766 | 0.66522 | 0.29624 | 0.02226 |
| dynamic | extra_trees | platt | 0.13074 | 0.86872 | 0.14714 | 0.65571 | 0.28783 | 0.00601 |
| dynamic | extra_trees | raw | 0.13085 | 0.86862 | 0.14717 | 0.65571 | 0.28783 | 0.00939 |
| dynamic | extra_trees | isotonic | 0.13089 | 0.86858 | 0.14680 | 0.65599 | 0.28313 | 0.01230 |

### Alternate ranking (FairAcc-first; NIJ-style “fair-and-accurate” view)

This is the same block sorted by `fairacc_sex_avg` (descending).

| Dataset | Model | Calibration | fairacc_sex_avg | bs_sex_avg | Overall Brier | AUROC | AUPRC | ECE |
|---|---|---|---:|---:|---:|---:|---:|---:|
| dynamic | hist_gb | isotonic | 0.87360 | 0.12640 | 0.14358 | 0.69142 | 0.33370 | 0.00650 |
| dynamic | xgb_best | isotonic | 0.87194 | 0.12675 | 0.14325 | 0.69612 | 0.31812 | 0.01092 |
| dynamic | random_forest | platt | 0.87160 | 0.12840 | 0.14480 | 0.67806 | 0.31335 | 0.00659 |
| dynamic | random_forest | raw | 0.87127 | 0.12873 | 0.14482 | 0.67806 | 0.31335 | 0.00999 |
| dynamic | logistic_gd | isotonic | 0.87104 | 0.12896 | 0.14666 | 0.65571 | 0.29933 | 0.02184 |
| dynamic | random_forest | isotonic | 0.87045 | 0.12868 | 0.14521 | 0.67376 | 0.28145 | 0.00627 |
| dynamic | logistic_gd | raw | 0.86998 | 0.12999 | 0.14766 | 0.66522 | 0.29624 | 0.02226 |
| dynamic | hist_gb | platt | 0.86932 | 0.12664 | 0.14460 | 0.69124 | 0.31057 | 0.01362 |

## Notes

- These are *not* official NIJ leaderboard scores (different test set).
- `xgb_best` uses the params from `reports/xgb_best_models.json`.
- `xgb_retuned` trials are controlled by `--xgb-trials` (default 128) and are run only for selected horizons.

Machine-readable metrics: `model_sweep.json`
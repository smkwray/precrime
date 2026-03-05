# Operational-Style Evaluation (NIJ Best Models)

Generated: 2026-03-02 15:57 EST

This report summarizes prediction quality and subgroup error rates under several thresholding policies.
It is meant to be easier to interpret than a single fixed cutoff; it does **not** choose an operational threshold for you.

## What “prediction success” means (plain language)
- **AUROC**: how well the model ranks higher-risk people above lower-risk people (0.5 = random; 1.0 = perfect ranking).
- **Brier score**: average squared error of predicted probabilities (lower = better probability accuracy).
- **FPR/FNR/PPV** below depend on a thresholding policy; they are not intrinsic model properties.

## Thresholding policies shown
- `t=0.5`: fixed threshold at 0.5 (mainly illustrative).
- `top10%`: flag the top 10% highest predicted risks (threshold derived from the test-set distribution).
- `top20%`: flag the top 20% highest predicted risks (threshold derived from the test-set distribution).
- `FPR<=0.06`: choose the **highest** threshold that achieves overall FPR ≤ 6% on the test set (grid search).

Random seed for split/train/calibration: `42`.

## STATIC Y1 — with race (calibration: platt)

| Metric | Value |
|---|---:|
| Brier | 0.18756 |
| AUROC | 0.70234 |
| AUPRC | 0.47736 |
| Log loss | 0.55453 |
| ECE | 0.01456 |

### Race (error rates by policy)

#### t=0.5 (threshold=0.50000; overall selection=0.08708)

| Group | N | FP | FN | FPR | FNR | PPV | TPR | Selection |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BLACK | 2045 | 75 | 527 | 0.05380 | 0.80952 | 0.62312 | 0.19048 | 0.09731 |
| WHITE | 1561 | 57 | 367 | 0.05018 | 0.86353 | 0.50435 | 0.13647 | 0.07367 |

#### top10% (threshold=0.48890; overall selection=0.10011)

| Group | N | FP | FN | FPR | FNR | PPV | TPR | Selection |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BLACK | 2045 | 90 | 510 | 0.06456 | 0.78341 | 0.61039 | 0.21659 | 0.11296 |
| WHITE | 1561 | 64 | 359 | 0.05634 | 0.84471 | 0.50769 | 0.15529 | 0.08328 |

#### top20% (threshold=0.42519; overall selection=0.20022)

| Group | N | FP | FN | FPR | FNR | PPV | TPR | Selection |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BLACK | 2045 | 206 | 401 | 0.14778 | 0.61598 | 0.54825 | 0.38402 | 0.22298 |
| WHITE | 1561 | 139 | 298 | 0.12236 | 0.70118 | 0.47744 | 0.29882 | 0.17040 |

#### FPR<=0.06 (threshold=0.49000; overall selection=0.09734)

| Group | N | FP | FN | FPR | FNR | PPV | TPR | Selection |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BLACK | 2045 | 88 | 514 | 0.06313 | 0.78955 | 0.60889 | 0.21045 | 0.11002 |
| WHITE | 1561 | 63 | 362 | 0.05546 | 0.85176 | 0.50000 | 0.14824 | 0.08072 |

### Top factors (model explanation snapshots)

- SHAP (top 5): Gang_Affiliated_Yes, _v1_0, Age_at_Release_23-27, Supervision_Risk_Score_First, Age_at_Release_48 or older
- XGB importance (top 5): _v1_0, Gang_Affiliated_Yes, _v1_5 or more, Prior_Arrest_Episodes_Property_0, Prior_Arrest_Episodes_Property_5 or more

## STATIC Y1 — without race (calibration: platt)

| Metric | Value |
|---|---:|
| Brier | 0.18751 |
| AUROC | 0.70252 |
| AUPRC | 0.47797 |
| Log loss | 0.55461 |
| ECE | 0.01447 |

### Race (error rates by policy)

#### t=0.5 (threshold=0.50000; overall selection=0.08569)

| Group | N | FP | FN | FPR | FNR | PPV | TPR | Selection |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BLACK | 2045 | 75 | 533 | 0.05380 | 0.81874 | 0.61140 | 0.18126 | 0.09438 |
| WHITE | 1561 | 55 | 364 | 0.04842 | 0.85647 | 0.52586 | 0.14353 | 0.07431 |

#### top10% (threshold=0.48692; overall selection=0.10011)

| Group | N | FP | FN | FPR | FNR | PPV | TPR | Selection |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BLACK | 2045 | 87 | 513 | 0.06241 | 0.78802 | 0.61333 | 0.21198 | 0.11002 |
| WHITE | 1561 | 67 | 356 | 0.05898 | 0.83765 | 0.50735 | 0.16235 | 0.08712 |

#### top20% (threshold=0.42309; overall selection=0.20022)

| Group | N | FP | FN | FPR | FNR | PPV | TPR | Selection |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BLACK | 2045 | 195 | 405 | 0.13989 | 0.62212 | 0.55782 | 0.37788 | 0.21565 |
| WHITE | 1561 | 149 | 293 | 0.13116 | 0.68941 | 0.46975 | 0.31059 | 0.18001 |

#### FPR<=0.06 (threshold=0.48800; overall selection=0.09845)

| Group | N | FP | FN | FPR | FNR | PPV | TPR | Selection |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BLACK | 2045 | 85 | 515 | 0.06098 | 0.79109 | 0.61538 | 0.20891 | 0.10807 |
| WHITE | 1561 | 66 | 357 | 0.05810 | 0.84000 | 0.50746 | 0.16000 | 0.08584 |

### Top factors (model explanation snapshots)

- SHAP (top 5): Gang_Affiliated_Yes, _v1_0, Age_at_Release_23-27, Supervision_Risk_Score_First, Age_at_Release_48 or older
- XGB importance (top 5): _v1_0, Gang_Affiliated_Yes, _v1_5 or more, Prior_Arrest_Episodes_Property_0, Prior_Arrest_Episodes_Property_5 or more

## DYNAMIC Y2 — with race (calibration: isotonic)

| Metric | Value |
|---|---:|
| Brier | 0.17322 |
| AUROC | 0.70166 |
| AUPRC | 0.41942 |
| Log loss | 0.53254 |
| ECE | 0.02490 |

### Race (error rates by policy)

#### t=0.5 (threshold=0.50000; overall selection=0.07902)

| Group | N | FP | FN | FPR | FNR | PPV | TPR | Selection |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BLACK | 1399 | 54 | 307 | 0.05212 | 0.84573 | 0.50909 | 0.15427 | 0.07863 |
| WHITE | 1132 | 46 | 244 | 0.05450 | 0.84722 | 0.48889 | 0.15278 | 0.07951 |

#### top10% (threshold=0.43820; overall selection=0.17582)

| Group | N | FP | FN | FPR | FNR | PPV | TPR | Selection |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BLACK | 1399 | 132 | 251 | 0.12741 | 0.69146 | 0.45902 | 0.30854 | 0.17441 |
| WHITE | 1132 | 109 | 196 | 0.12915 | 0.68056 | 0.45771 | 0.31944 | 0.17756 |

#### top20% (threshold=0.37391; overall selection=0.22718)

| Group | N | FP | FN | FPR | FNR | PPV | TPR | Selection |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BLACK | 1399 | 183 | 230 | 0.17664 | 0.63361 | 0.42089 | 0.36639 | 0.22588 |
| WHITE | 1132 | 146 | 175 | 0.17299 | 0.60764 | 0.43629 | 0.39236 | 0.22880 |

#### FPR<=0.06 (threshold=0.43900; overall selection=0.07942)

| Group | N | FP | FN | FPR | FNR | PPV | TPR | Selection |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BLACK | 1399 | 54 | 307 | 0.05212 | 0.84573 | 0.50909 | 0.15427 | 0.07863 |
| WHITE | 1132 | 46 | 243 | 0.05450 | 0.84375 | 0.49451 | 0.15625 | 0.08039 |

### Top factors (model explanation snapshots)

- SHAP (top 5): Percent_Days_Employed, Jobs_Per_Year, Jobs_Per_Year__missing, Supervision_Risk_Score_First, Avg_Days_per_DrugTest
- XGB importance (top 5): _v1_0, Jobs_Per_Year__missing, Percent_Days_Employed__missing, Gang_Affiliated_Yes, _v1_5 or more

## DYNAMIC Y2 — without race (calibration: isotonic)

| Metric | Value |
|---|---:|
| Brier | 0.17287 |
| AUROC | 0.70625 |
| AUPRC | 0.42149 |
| Log loss | 0.52217 |
| ECE | 0.03557 |

### Race (error rates by policy)

#### t=0.5 (threshold=0.50000; overall selection=0.02924)

| Group | N | FP | FN | FPR | FNR | PPV | TPR | Selection |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BLACK | 1399 | 20 | 341 | 0.01931 | 0.93939 | 0.52381 | 0.06061 | 0.03002 |
| WHITE | 1132 | 17 | 273 | 0.02014 | 0.94792 | 0.46875 | 0.05208 | 0.02827 |

#### top10% (threshold=0.48276; overall selection=0.11300)

| Group | N | FP | FN | FPR | FNR | PPV | TPR | Selection |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BLACK | 1399 | 80 | 285 | 0.07722 | 0.78512 | 0.49367 | 0.21488 | 0.11294 |
| WHITE | 1132 | 65 | 225 | 0.07701 | 0.78125 | 0.49219 | 0.21875 | 0.11307 |

#### top20% (threshold=0.36486; overall selection=0.22323)

| Group | N | FP | FN | FPR | FNR | PPV | TPR | Selection |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BLACK | 1399 | 186 | 223 | 0.17954 | 0.61433 | 0.42945 | 0.38567 | 0.23302 |
| WHITE | 1132 | 133 | 182 | 0.15758 | 0.63194 | 0.44351 | 0.36806 | 0.21113 |

#### FPR<=0.06 (threshold=0.48300; overall selection=0.07586)

| Group | N | FP | FN | FPR | FNR | PPV | TPR | Selection |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BLACK | 1399 | 53 | 306 | 0.05116 | 0.84298 | 0.51818 | 0.15702 | 0.07863 |
| WHITE | 1132 | 43 | 249 | 0.05095 | 0.86458 | 0.47561 | 0.13542 | 0.07244 |

### Top factors (model explanation snapshots)

- SHAP (top 5): Percent_Days_Employed, Jobs_Per_Year, Jobs_Per_Year__missing, Supervision_Risk_Score_First, Avg_Days_per_DrugTest
- XGB importance (top 5): _v1_0, Jobs_Per_Year__missing, Percent_Days_Employed__missing, Gang_Affiliated_Yes, _v1_5 or more

## DYNAMIC Y3 — with race (calibration: isotonic)

| Metric | Value |
|---|---:|
| Brier | 0.14338 |
| AUROC | 0.69649 |
| AUPRC | 0.32323 |
| Log loss | 0.46649 |
| ECE | 0.00794 |

### Race (error rates by policy)

#### t=0.5 (threshold=0.50000; overall selection=0.00000)

| Group | N | FP | FN | FPR | FNR | PPV | TPR | Selection |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BLACK | 1089 | 0 | 219 | 0.00000 | 1.00000 | nan | 0.00000 | 0.00000 |
| WHITE | 791 | 0 | 139 | 0.00000 | 1.00000 | nan | 0.00000 | 0.00000 |

#### top10% (threshold=0.34400; overall selection=0.12394)

| Group | N | FP | FN | FPR | FNR | PPV | TPR | Selection |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BLACK | 1089 | 102 | 164 | 0.11724 | 0.74886 | 0.35032 | 0.25114 | 0.14417 |
| WHITE | 791 | 49 | 112 | 0.07515 | 0.80576 | 0.35526 | 0.19424 | 0.09608 |

#### top20% (threshold=0.32231; overall selection=0.20798)

| Group | N | FP | FN | FPR | FNR | PPV | TPR | Selection |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BLACK | 1089 | 164 | 125 | 0.18851 | 0.57078 | 0.36434 | 0.42922 | 0.23691 |
| WHITE | 791 | 90 | 96 | 0.13804 | 0.69065 | 0.32331 | 0.30935 | 0.16814 |

#### FPR<=0.06 (threshold=0.34500; overall selection=0.06649)

| Group | N | FP | FN | FPR | FNR | PPV | TPR | Selection |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BLACK | 1089 | 52 | 188 | 0.05977 | 0.85845 | 0.37349 | 0.14155 | 0.07622 |
| WHITE | 791 | 24 | 121 | 0.03681 | 0.87050 | 0.42857 | 0.12950 | 0.05310 |

### Top factors (model explanation snapshots)

- SHAP (top 5): Percent_Days_Employed, Jobs_Per_Year, Jobs_Per_Year__missing, _v1_0, Supervision_Risk_Score_First
- XGB importance (top 5): _v1_0, Percent_Days_Employed__missing, Jobs_Per_Year__missing, DrugTests_Meth_Positive__missing, Gang_Affiliated_Yes

## DYNAMIC Y3 — without race (calibration: isotonic)

| Metric | Value |
|---|---:|
| Brier | 0.14436 |
| AUROC | 0.69070 |
| AUPRC | 0.31175 |
| Log loss | 0.48049 |
| ECE | 0.01385 |

### Race (error rates by policy)

#### t=0.5 (threshold=0.50000; overall selection=0.01170)

| Group | N | FP | FN | FPR | FNR | PPV | TPR | Selection |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BLACK | 1089 | 7 | 212 | 0.00805 | 0.96804 | 0.50000 | 0.03196 | 0.01286 |
| WHITE | 791 | 7 | 138 | 0.01074 | 0.99281 | 0.12500 | 0.00719 | 0.01011 |

#### top10% (threshold=0.36464; overall selection=0.13723)

| Group | N | FP | FN | FPR | FNR | PPV | TPR | Selection |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BLACK | 1089 | 114 | 160 | 0.13103 | 0.73059 | 0.34104 | 0.26941 | 0.15886 |
| WHITE | 791 | 57 | 111 | 0.08742 | 0.79856 | 0.32941 | 0.20144 | 0.10746 |

#### top20% (threshold=0.29703; overall selection=0.21862)

| Group | N | FP | FN | FPR | FNR | PPV | TPR | Selection |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BLACK | 1089 | 185 | 120 | 0.21264 | 0.54795 | 0.34859 | 0.45205 | 0.26079 |
| WHITE | 791 | 89 | 101 | 0.13650 | 0.72662 | 0.29921 | 0.27338 | 0.16056 |

#### FPR<=0.06 (threshold=0.36500; overall selection=0.02713)

| Group | N | FP | FN | FPR | FNR | PPV | TPR | Selection |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BLACK | 1089 | 22 | 206 | 0.02529 | 0.94064 | 0.37143 | 0.05936 | 0.03214 |
| WHITE | 791 | 12 | 135 | 0.01840 | 0.97122 | 0.25000 | 0.02878 | 0.02023 |

### Top factors (model explanation snapshots)

- SHAP (top 5): Percent_Days_Employed, Jobs_Per_Year, Jobs_Per_Year__missing, _v1_0, Supervision_Risk_Score_First
- XGB importance (top 5): _v1_0, Percent_Days_Employed__missing, Jobs_Per_Year__missing, DrugTests_Meth_Positive__missing, Gang_Affiliated_Yes

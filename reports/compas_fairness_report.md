# COMPAS Fairness / Subgroup Audit (XGBoost)

Generated: 2026-03-02 06:53 EST

This report trains a tuned XGBoost model on a seeded train/calibration/test split and reports subgroup metrics.

Two variants are reported:
- **With race**: training features include `race` (if present).
- **Without race**: training features exclude `race`, but evaluation still reports subgroup metrics by `race`.

Important: Several COMPAS race subgroups are very small in this sample (e.g., `Asian` N=7, `Native American` N=1 in this test split), so per-group and “max gap” metrics can be highly unstable. Treat these as exploratory diagnostics, not operational evidence.

## COMPAS — with race (calibration: platt)

| Metric | Value |
|---|---:|
| Brier | 0.17458 |
| Brier (CI) | 0.16410–0.18546 |
| AUROC | 0.80431 |
| AUPRC | 0.80440 |
| Log loss | 0.52374 |
| ECE | 0.02476 |

### Race (gap summary)

- Threshold sweep gaps (0..1 step 0.05): `fpr_gap_max=1.00000`, `fpr_gap_mean=0.22761`, `fnr_gap_max=0.83851`, `fnr_gap_mean=0.39262`
- Bootstrap gaps @0.5 (CI): `fpr_gap=0.22039` (0.18077–0.27196), `fnr_gap=0.57778` (0.19774–0.75000)

### Race

| Group | N | Brier | Brier (CI) | AUROC | AUPRC |
|---|---:|---:|---|---:|---:|
| African-American | 641 | 0.17427 | 0.16142–0.18862 | 0.81636 | 0.84637 |
| Caucasian | 427 | 0.17230 | 0.15670–0.19158 | 0.77615 | 0.72636 |
| Hispanic | 94 | 0.22903 | 0.18195–0.27773 | 0.66871 | 0.71750 |
| Other | 65 | 0.13136 | 0.10019–0.16481 | 0.84515 | 0.75826 |
| Asian | 7 | 0.02684 |  | 1.00000 | 1.00000 |
| Native American | 1 | 0.07677 |  | nan | 0.00000 |

#### Threshold 0.5 (error rates by group)

| Group | N | FP | FN | FPR | FNR | PPV | Selection rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| African-American | 641 | 67 | 103 | 0.22039 | 0.30564 | 0.77741 | 0.46958 |
| Caucasian | 427 | 22 | 74 | 0.08271 | 0.45963 | 0.79817 | 0.25527 |
| Hispanic | 94 | 7 | 26 | 0.14286 | 0.57778 | 0.73077 | 0.27660 |
| Other | 65 | 3 | 9 | 0.06383 | 0.50000 | 0.75000 | 0.18462 |
| Asian | 7 | 0 | 0 | 0.00000 | 0.00000 | 1.00000 | 0.14286 |
| Native American | 1 | 0 | 0 | 0.00000 | nan | nan | 0.00000 |

### Sex (gap summary)

- Threshold sweep gaps (0..1 step 0.05): `fpr_gap_max=0.20671`, `fpr_gap_mean=0.07161`, `fnr_gap_max=0.20096`, `fnr_gap_mean=0.09989`
- Bootstrap gaps @0.5 (CI): `fpr_gap=0.07597` (0.02297–0.13065), `fnr_gap=0.16545` (0.04623–0.27975)

### Sex

| Group | N | Brier | Brier (CI) | AUROC | AUPRC |
|---|---:|---:|---|---:|---:|
| Male | 1002 | 0.17497 | 0.16463–0.18775 | 0.80851 | 0.81706 |
| Female | 233 | 0.17290 | 0.14847–0.19734 | 0.77015 | 0.72012 |

#### Threshold 0.5 (error rates by group)

| Group | N | FP | FN | FPR | FNR | PPV | Selection rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| Male | 1002 | 86 | 168 | 0.16381 | 0.35220 | 0.78228 | 0.39421 |
| Female | 233 | 13 | 44 | 0.08784 | 0.51765 | 0.75926 | 0.23176 |

### Age Group (gap summary)

- Threshold sweep gaps (0..1 step 0.05): `fpr_gap_max=0.40831`, `fpr_gap_mean=0.13536`, `fnr_gap_max=0.21882`, `fnr_gap_mean=0.12089`
- Bootstrap gaps @0.5 (CI): `fpr_gap=0.09659` (0.03142–0.18991), `fnr_gap=0.17241` (0.05600–0.28670)

### Age Group

| Group | N | Brier | Brier (CI) | AUROC | AUPRC |
|---|---:|---:|---|---:|---:|
| 25 - 45 | 661 | 0.17115 | 0.15833–0.18628 | 0.81298 | 0.82775 |
| Less than 25 | 288 | 0.19082 | 0.17281–0.21204 | 0.76901 | 0.82987 |
| Greater than 45 | 286 | 0.16616 | 0.14603–0.18747 | 0.76610 | 0.65361 |

#### Threshold 0.5 (error rates by group)

| Group | N | FP | FN | FPR | FNR | PPV | Selection rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| 25 - 45 | 661 | 46 | 114 | 0.13218 | 0.36422 | 0.81224 | 0.37065 |
| Less than 25 | 288 | 28 | 54 | 0.22222 | 0.33333 | 0.79412 | 0.47222 |
| Greater than 45 | 286 | 25 | 44 | 0.12563 | 0.50575 | 0.63235 | 0.23776 |

## COMPAS — without race (calibration: platt)

| Metric | Value |
|---|---:|
| Brier | 0.17422 |
| Brier (CI) | 0.16370–0.18467 |
| AUROC | 0.80500 |
| AUPRC | 0.80088 |
| Log loss | 0.52378 |
| ECE | 0.02289 |

### Race (gap summary)

- Threshold sweep gaps (0..1 step 0.05): `fpr_gap_max=1.00000`, `fpr_gap_mean=0.22900`, `fnr_gap_max=0.80745`, `fnr_gap_mean=0.38860`
- Bootstrap gaps @0.5 (CI): `fpr_gap=0.22697` (0.18568–0.27717), `fnr_gap=0.53333` (0.18273–0.75000)

### Race

| Group | N | Brier | Brier (CI) | AUROC | AUPRC |
|---|---:|---:|---|---:|---:|
| African-American | 641 | 0.17505 | 0.16204–0.19007 | 0.81481 | 0.84168 |
| Caucasian | 427 | 0.17179 | 0.15463–0.18986 | 0.77516 | 0.72115 |
| Hispanic | 94 | 0.22477 | 0.17825–0.27671 | 0.67778 | 0.71872 |
| Other | 65 | 0.12625 | 0.09593–0.16011 | 0.85816 | 0.78015 |
| Asian | 7 | 0.02669 |  | 1.00000 | 1.00000 |
| Native American | 1 | 0.07613 |  | nan | 0.00000 |

#### Threshold 0.5 (error rates by group)

| Group | N | FP | FN | FPR | FNR | PPV | Selection rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| African-American | 641 | 69 | 105 | 0.22697 | 0.31157 | 0.77076 | 0.46958 |
| Caucasian | 427 | 19 | 79 | 0.07143 | 0.49068 | 0.81188 | 0.23653 |
| Hispanic | 94 | 7 | 24 | 0.14286 | 0.53333 | 0.75000 | 0.29787 |
| Other | 65 | 2 | 9 | 0.04255 | 0.50000 | 0.81818 | 0.16923 |
| Asian | 7 | 0 | 0 | 0.00000 | 0.00000 | 1.00000 | 0.14286 |
| Native American | 1 | 0 | 0 | 0.00000 | nan | nan | 0.00000 |

### Sex (gap summary)

- Threshold sweep gaps (0..1 step 0.05): `fpr_gap_max=0.21138`, `fpr_gap_mean=0.06459`, `fnr_gap_max=0.21564`, `fnr_gap_mean=0.10735`
- Bootstrap gaps @0.5 (CI): `fpr_gap=0.06350` (0.00970–0.11793), `fnr_gap=0.16882` (0.05472–0.27473)

### Sex

| Group | N | Brier | Brier (CI) | AUROC | AUPRC |
|---|---:|---:|---|---:|---:|
| Male | 1002 | 0.17357 | 0.16219–0.18659 | 0.81169 | 0.81500 |
| Female | 233 | 0.17699 | 0.15329–0.20142 | 0.75990 | 0.70787 |

#### Threshold 0.5 (error rates by group)

| Group | N | FP | FN | FPR | FNR | PPV | Selection rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| Male | 1002 | 83 | 172 | 0.15810 | 0.36059 | 0.78608 | 0.38723 |
| Female | 233 | 14 | 45 | 0.09459 | 0.52941 | 0.74074 | 0.23176 |

### Age Group (gap summary)

- Threshold sweep gaps (0..1 step 0.05): `fpr_gap_max=0.49087`, `fpr_gap_mean=0.14531`, `fnr_gap_max=0.25032`, `fnr_gap_mean=0.13074`
- Bootstrap gaps @0.5 (CI): `fpr_gap=0.11669` (0.04381–0.20696), `fnr_gap=0.19540` (0.07699–0.31592)

### Age Group

| Group | N | Brier | Brier (CI) | AUROC | AUPRC |
|---|---:|---:|---|---:|---:|
| 25 - 45 | 661 | 0.17149 | 0.15931–0.18704 | 0.81287 | 0.82317 |
| Less than 25 | 288 | 0.18911 | 0.17275–0.21063 | 0.77104 | 0.82575 |
| Greater than 45 | 286 | 0.16551 | 0.14658–0.18583 | 0.76544 | 0.65398 |

#### Threshold 0.5 (error rates by group)

| Group | N | FP | FN | FPR | FNR | PPV | Selection rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| 25 - 45 | 661 | 48 | 117 | 0.13793 | 0.37380 | 0.80328 | 0.36914 |
| Less than 25 | 288 | 28 | 54 | 0.22222 | 0.33333 | 0.79412 | 0.47222 |
| Greater than 45 | 286 | 21 | 46 | 0.10553 | 0.52874 | 0.66129 | 0.21678 |

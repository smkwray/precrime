# NIJ-Style Scoring (Held-out Split; Not Official Leaderboard Scores)

Generated: 2026-03-02 15:53 EST

This report computes NIJ-style scoring terms on this project's seeded train/calibration/test split (seed 42).
NIJ's official Challenge scoring used a separate held-out test set, so these values are **not directly comparable** to the NIJ leaderboard.

Definitions (NIJ-style):
- `BS`: Brier score **error** (lower is better).
- `FPR_black@0.5` / `FPR_white@0.5`: false positive rates at threshold 0.5 within a sex subgroup.
- `FP = 1 - |FPR_black@0.5 - FPR_white@0.5|` (fairness penalty term).
- `FairAcc = (1 - BS) * FP`.

## Y1 (static; calibration=platt)

### Variant: with_race

| Metric | Value |
|---|---:|
| Overall Brier | 0.18737 |
| AUROC | 0.70252 |
| AUPRC | 0.48060 |
| Log loss | 0.55411 |
| ECE | 0.01218 |

#### NIJ-style terms by sex (threshold=0.5)

| Sex | N | BS (Brier) | FPR_black@0.5 | FPR_white@0.5 | FP | FairAcc |
|---|---:|---:|---:|---:|---:|---:|
| F | 428 | 0.14326 | 0.00000 | 0.00000 | 1.00000 | 0.85674 |
| M | 3178 | 0.19331 | 0.06086 | 0.06335 | 0.99751 | 0.80468 |

- Sex-average BS: `0.16829`
- Sex-average FairAcc: `0.83071`

### Variant: without_race

| Metric | Value |
|---|---:|
| Overall Brier | 0.18750 |
| AUROC | 0.70207 |
| AUPRC | 0.47840 |
| Log loss | 0.55449 |
| ECE | 0.01083 |

#### NIJ-style terms by sex (threshold=0.5)

| Sex | N | BS (Brier) | FPR_black@0.5 | FPR_white@0.5 | FP | FairAcc |
|---|---:|---:|---:|---:|---:|---:|
| F | 428 | 0.14376 | 0.00000 | 0.00000 | 1.00000 | 0.85624 |
| M | 3178 | 0.19339 | 0.05547 | 0.06561 | 0.98986 | 0.79843 |

- Sex-average BS: `0.16858`
- Sex-average FairAcc: `0.82733`

## Y2 (dynamic; calibration=isotonic)

### Variant: with_race

| Metric | Value |
|---|---:|
| Overall Brier | 0.17309 |
| AUROC | 0.70130 |
| AUPRC | 0.41628 |
| Log loss | 0.51786 |
| ECE | 0.01925 |

#### NIJ-style terms by sex (threshold=0.5)

| Sex | N | BS (Brier) | FPR_black@0.5 | FPR_white@0.5 | FP | FairAcc |
|---|---:|---:|---:|---:|---:|---:|
| F | 363 | 0.14579 | 0.01053 | 0.02083 | 0.98969 | 0.84541 |
| M | 2168 | 0.17767 | 0.05420 | 0.06595 | 0.98825 | 0.81267 |

- Sex-average BS: `0.16173`
- Sex-average FairAcc: `0.82904`

### Variant: without_race

| Metric | Value |
|---|---:|
| Overall Brier | 0.17374 |
| AUROC | 0.70268 |
| AUPRC | 0.42032 |
| Log loss | 0.52247 |
| ECE | 0.02537 |

#### NIJ-style terms by sex (threshold=0.5)

| Sex | N | BS (Brier) | FPR_black@0.5 | FPR_white@0.5 | FP | FairAcc |
|---|---:|---:|---:|---:|---:|---:|
| F | 363 | 0.14550 | 0.00000 | 0.00521 | 0.99479 | 0.85005 |
| M | 2168 | 0.17847 | 0.02444 | 0.02914 | 0.99530 | 0.81767 |

- Sex-average BS: `0.16198`
- Sex-average FairAcc: `0.83386`

## Y3 (dynamic; calibration=isotonic)

### Variant: with_race

| Metric | Value |
|---|---:|
| Overall Brier | 0.14325 |
| AUROC | 0.69612 |
| AUPRC | 0.31812 |
| Log loss | 0.47252 |
| ECE | 0.01092 |

#### NIJ-style terms by sex (threshold=0.5)

| Sex | N | BS (Brier) | FPR_black@0.5 | FPR_white@0.5 | FP | FairAcc |
|---|---:|---:|---:|---:|---:|---:|
| F | 281 | 0.10322 | 0.00000 | 0.00000 | 1.00000 | 0.89678 |
| M | 1599 | 0.15028 | 0.00508 | 0.00816 | 0.99692 | 0.84710 |

- Sex-average BS: `0.12675`
- Sex-average FairAcc: `0.87194`

### Variant: without_race

| Metric | Value |
|---|---:|
| Overall Brier | 0.14297 |
| AUROC | 0.69897 |
| AUPRC | 0.32013 |
| Log loss | 0.44918 |
| ECE | 0.01716 |

#### NIJ-style terms by sex (threshold=0.5)

| Sex | N | BS (Brier) | FPR_black@0.5 | FPR_white@0.5 | FP | FairAcc |
|---|---:|---:|---:|---:|---:|---:|
| F | 281 | 0.10375 | 0.00000 | 0.00000 | 1.00000 | 0.89625 |
| M | 1599 | 0.14986 | 0.00000 | 0.00612 | 0.99388 | 0.84493 |

- Sex-average BS: `0.12681`
- Sex-average FairAcc: `0.87059`

# Summary (sex-averaged)

| Horizon | Variant | Dataset | Calibration | Overall Brier | Sex-avg BS | Sex-avg FairAcc |
|---|---|---|---|---:|---:|---:|
| Y1 | with_race | static | platt | 0.18737 | 0.16829 | 0.83071 |
| Y1 | without_race | static | platt | 0.18750 | 0.16858 | 0.82733 |
| Y2 | with_race | dynamic | isotonic | 0.17309 | 0.16173 | 0.82904 |
| Y2 | without_race | dynamic | isotonic | 0.17374 | 0.16198 | 0.83386 |
| Y3 | with_race | dynamic | isotonic | 0.14325 | 0.12675 | 0.87194 |
| Y3 | without_race | dynamic | isotonic | 0.14297 | 0.12681 | 0.87059 |
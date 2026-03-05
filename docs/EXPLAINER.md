# Plain-Language Guide to the Metrics

This project reports **probability predictions** (e.g., “0.30 risk”), not just yes/no classifications. Two things matter:

1. **Ranking:** are higher-risk people *usually* higher on the list than lower-risk people?
2. **Calibration:** when the model says “30%”, does it actually happen about 30% of the time for similar people?

## Core metrics

- **Brier score (lower is better):** average squared error of the predicted probabilities. It improves when probabilities are numerically closer to outcomes (0/1) *and* when ranking improves.
- **AUROC (higher is better):** how well the model ranks positives above negatives, regardless of calibration.
- **AUPRC (higher is better):** like AUROC but more informative when the positive class is not rare.
- **ECE (lower is better):** how far predicted probabilities are from observed rates across probability bins.

## Thresholded metrics (depend on policy)

To compute false positives/negatives, you must pick a **threshold** (or a “top-k% policy”).

- **False positive (FP):** the model flags someone as “high risk” but the outcome does not occur.
- **False negative (FN):** the model does not flag someone but the outcome occurs.
- **FPR (false positive rate):** FP / (FP + TN). “Among people without the outcome, how many were flagged?”
- **FNR (false negative rate):** FN / (FN + TP). “Among people with the outcome, how many were missed?”
- **PPV (precision):** TP / (TP + FP). “Among flagged people, how many had the outcome?”
- **Selection rate:** fraction of people flagged at that policy/threshold.

Important: changing the threshold changes FP/FN, FPR/FNR, PPV, and subgroup gaps — even if the underlying model is unchanged.

## Fairness diagnostics (what the project reports)

This project reports subgroup error rates by race, gender, and age at multiple thresholds.

- **Gap plots (FPR/FNR vs threshold):** show how disparities vary across thresholds. A disparity can look small at one threshold and large at another.
- **With vs without race:** compares training with `Race`/`race` included vs removed. Removing one feature rarely removes disparities because other features can correlate with group membership.

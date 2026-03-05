# Stability Evaluation (Seed Sweep)

Generated: 2026-03-02 06:11 EST

This report reruns the best-per-horizon NIJ model configs across multiple random seeds (split + fit) and summarizes variability.

Seeds: `42, 43, 44`.

Policy for error-rate summary: `top10%` (flag top 10% highest predicted risks on the test set).

## STATIC Y1 (calibration: platt)

| Metric | Mean | Std |
|---|---:|---:|
| Brier | 0.18764 | 0.00111 |
| AUROC | 0.70193 | 0.00363 |
| AUPRC | 0.47959 | 0.00894 |
| Log loss | 0.55474 | 0.00249 |
| ECE | 0.01490 | 0.00214 |
| Race FPR gap @top10% | 0.00715 | 0.00638 |
| Race FNR gap @top10% | 0.03035 | 0.02680 |

## DYNAMIC Y2 (calibration: isotonic)

| Metric | Mean | Std |
|---|---:|---:|
| Brier | 0.17028 | 0.00270 |
| AUROC | 0.71549 | 0.01278 |
| AUPRC | 0.44092 | 0.01862 |
| Log loss | 0.51409 | 0.01616 |
| ECE | 0.02093 | 0.00344 |
| Race FPR gap @top10% | 0.01696 | 0.01752 |
| Race FNR gap @top10% | 0.02130 | 0.01297 |

## DYNAMIC Y3 (calibration: platt)

| Metric | Mean | Std |
|---|---:|---:|
| Brier | 0.14280 | 0.00046 |
| AUROC | 0.70392 | 0.00796 |
| AUPRC | 0.31801 | 0.00115 |
| Log loss | 0.44324 | 0.00239 |
| ECE | 0.01771 | 0.00429 |
| Race FPR gap @top10% | 0.03198 | 0.01003 |
| Race FNR gap @top10% | 0.03217 | 0.02903 |

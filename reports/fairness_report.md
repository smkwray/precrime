# NIJ Fairness / Subgroup Audit (Best XGBoost Models)

Generated: 2026-03-02 05:58 EST

This report computes subgroup metrics on a fixed train/calibration/test split (seeded) for the best-per-horizon XGBoost configuration in `reports/xgb_best_models.json`.

Two variants are reported:
- **With race**: training features include `Race` (if present in the dataset).
- **Without race**: training features exclude `Race`, but evaluation still reports subgroup metrics by `Race`.

## STATIC Y1 — with race (calibration: platt)

| Metric | Value |
|---|---:|
| Brier | 0.18741 |
| Brier (CI) | 0.18349–0.19128 |
| AUROC | 0.70180 |
| AUPRC | 0.48296 |
| Log loss | 0.55433 |
| ECE | 0.01173 |

### Race (gap summary)

- Threshold sweep gaps (0..1 step 0.05): `fpr_gap_max=0.11619`, `fpr_gap_mean=0.02698`, `fnr_gap_max=0.07716`, `fnr_gap_mean=0.01867`
- Bootstrap gaps @0.5 (CI): `fpr_gap=0.00242` (0.00029–0.02084), `fnr_gap=0.06250` (0.01951–0.10673)

### Race

| Group | N | Brier | Brier (CI) | AUROC | AUPRC |
|---|---:|---:|---|---:|---:|
| BLACK | 2045 | 0.19537 | 0.19080–0.19975 | 0.69173 | 0.50682 |
| WHITE | 1561 | 0.17698 | 0.17198–0.18182 | 0.71348 | 0.44753 |

- Equalized-odds gaps @0.5: `fpr_gap=0.00242`, `fnr_gap=0.06250`, `max=0.06250`
- Predictive parity proxy @0.5: `ppv_gap=0.13721`
- Threshold sweep gaps (0..1 step 0.05): `fpr_gap_max=0.11619`, `fpr_gap_mean=0.02698`, `fnr_gap_max=0.07716`, `fnr_gap_mean=0.01867`

#### Threshold 0.5 (error rates by group)

| Group | N | FP | FN | FPR | FNR | PPV | Selection rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| BLACK | 2045 | 77 | 523 | 0.05524 | 0.80338 | 0.62439 | 0.10024 |
| WHITE | 1561 | 60 | 368 | 0.05282 | 0.86588 | 0.48718 | 0.07495 |

### Gender (gap summary)

- Threshold sweep gaps (0..1 step 0.05): `fpr_gap_max=0.24564`, `fpr_gap_mean=0.07554`, `fnr_gap_max=0.42681`, `fnr_gap_mean=0.11096`
- Bootstrap gaps @0.5 (CI): `fpr_gap=0.06279` (0.05275–0.07331), `fnr_gap=0.17224` (0.13438–0.20489)

### Gender

| Group | N | Brier | Brier (CI) | AUROC | AUPRC |
|---|---:|---:|---|---:|---:|
| M | 3178 | 0.19329 | 0.18993–0.19736 | 0.69678 | 0.49216 |
| F | 428 | 0.14378 | 0.13628–0.15061 | 0.66968 | 0.34092 |

- Equalized-odds gaps @0.5: `fpr_gap=0.06279`, `fnr_gap=0.17224`, `max=0.17224`
- Predictive parity proxy @0.5: `ppv_gap=0.42679`
- Threshold sweep gaps (0..1 step 0.05): `fpr_gap_max=0.24564`, `fpr_gap_mean=0.07554`, `fnr_gap_max=0.42681`, `fnr_gap_mean=0.11096`

#### Threshold 0.5 (error rates by group)

| Group | N | FP | FN | FPR | FNR | PPV | Selection rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| M | 3178 | 137 | 812 | 0.06279 | 0.81526 | 0.57321 | 0.10101 |
| F | 428 | 0 | 79 | 0.00000 | 0.98750 | 1.00000 | 0.00234 |

### Age Group (gap summary)

- Threshold sweep gaps (0..1 step 0.05): `fpr_gap_max=0.54021`, `fpr_gap_mean=0.17883`, `fnr_gap_max=0.54766`, `fnr_gap_mean=0.18278`
- Bootstrap gaps @0.5 (CI): `fpr_gap=0.20513` (0.14102–0.26948), `fnr_gap=0.44348` (0.35713–0.54265)

### Age Group

| Group | N | Brier | Brier (CI) | AUROC | AUPRC |
|---|---:|---:|---|---:|---:|
| 23-27 | 744 | 0.20943 | 0.20016–0.21818 | 0.66510 | 0.51488 |
| 28-32 | 719 | 0.19660 | 0.18912–0.20475 | 0.67042 | 0.46444 |
| 33-37 | 539 | 0.19045 | 0.18242–0.19915 | 0.71534 | 0.53142 |
| 48 or older | 531 | 0.13899 | 0.13164–0.14517 | 0.73472 | 0.37875 |
| 38-42 | 405 | 0.17367 | 0.16471–0.18406 | 0.73903 | 0.49211 |
| 43-47 | 397 | 0.17598 | 0.16733–0.18462 | 0.66358 | 0.37015 |
| 18-22 | 271 | 0.22872 | 0.21174–0.24918 | 0.65491 | 0.56219 |

- Equalized-odds gaps @0.5: `fpr_gap=0.20513`, `fnr_gap=0.44348`, `max=0.44348`
- Predictive parity proxy @0.5: `ppv_gap=0.26471`
- Threshold sweep gaps (0..1 step 0.05): `fpr_gap_max=0.54021`, `fpr_gap_mean=0.17883`, `fnr_gap_max=0.54766`, `fnr_gap_mean=0.18278`

#### Threshold 0.5 (error rates by group)

| Group | N | FP | FN | FPR | FNR | PPV | Selection rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| 23-27 | 744 | 53 | 197 | 0.10973 | 0.75479 | 0.54701 | 0.15726 |
| 28-32 | 719 | 34 | 177 | 0.06814 | 0.80455 | 0.55844 | 0.10709 |
| 33-37 | 539 | 4 | 154 | 0.01075 | 0.92216 | 0.76471 | 0.03154 |
| 48 or older | 531 | 0 | 102 | 0.00000 | 1.00000 | nan | 0.00000 |
| 38-42 | 405 | 10 | 102 | 0.03413 | 0.91071 | 0.50000 | 0.04938 |
| 43-47 | 397 | 4 | 95 | 0.01342 | 0.95960 | 0.50000 | 0.02015 |
| 18-22 | 271 | 32 | 64 | 0.20513 | 0.55652 | 0.61446 | 0.30627 |

## STATIC Y1 — without race (calibration: platt)

| Metric | Value |
|---|---:|
| Brier | 0.18732 |
| Brier (CI) | 0.18344–0.19128 |
| AUROC | 0.70275 |
| AUPRC | 0.48262 |
| Log loss | 0.55408 |
| ECE | 0.01156 |

### Race (gap summary)

- Threshold sweep gaps (0..1 step 0.05): `fpr_gap_max=0.10351`, `fpr_gap_mean=0.02206`, `fnr_gap_max=0.07019`, `fnr_gap_mean=0.01499`
- Bootstrap gaps @0.5 (CI): `fpr_gap=0.00832` (0.00058–0.02708), `fnr_gap=0.05319` (0.01064–0.09654)

### Race

| Group | N | Brier | Brier (CI) | AUROC | AUPRC |
|---|---:|---:|---|---:|---:|
| BLACK | 2045 | 0.19498 | 0.19044–0.19943 | 0.69404 | 0.51010 |
| WHITE | 1561 | 0.17728 | 0.17225–0.18230 | 0.71259 | 0.44471 |

- Equalized-odds gaps @0.5: `fpr_gap=0.00832`, `fnr_gap=0.05319`, `max=0.05319`
- Predictive parity proxy @0.5: `ppv_gap=0.09809`
- Threshold sweep gaps (0..1 step 0.05): `fpr_gap_max=0.10351`, `fpr_gap_mean=0.02206`, `fnr_gap_max=0.07019`, `fnr_gap_mean=0.01499`

#### Threshold 0.5 (error rates by group)

| Group | N | FP | FN | FPR | FNR | PPV | Selection rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| BLACK | 2045 | 84 | 526 | 0.06026 | 0.80799 | 0.59809 | 0.10220 |
| WHITE | 1561 | 59 | 366 | 0.05194 | 0.86118 | 0.50000 | 0.07559 |

### Gender (gap summary)

- Threshold sweep gaps (0..1 step 0.05): `fpr_gap_max=0.24909`, `fpr_gap_mean=0.07494`, `fnr_gap_max=0.39428`, `fnr_gap_mean=0.10551`
- Bootstrap gaps @0.5 (CI): `fpr_gap=0.06554` (0.05566–0.07644), `fnr_gap=0.17123` (0.13352–0.20404)

### Gender

| Group | N | Brier | Brier (CI) | AUROC | AUPRC |
|---|---:|---:|---|---:|---:|
| M | 3178 | 0.19327 | 0.18982–0.19735 | 0.69738 | 0.49154 |
| F | 428 | 0.14314 | 0.13555–0.14966 | 0.67313 | 0.34136 |

- Equalized-odds gaps @0.5: `fpr_gap=0.06554`, `fnr_gap=0.17123`, `max=0.17123`
- Predictive parity proxy @0.5: `ppv_gap=0.43865`
- Threshold sweep gaps (0..1 step 0.05): `fpr_gap_max=0.24909`, `fpr_gap_mean=0.07494`, `fnr_gap_max=0.39428`, `fnr_gap_mean=0.10551`

#### Threshold 0.5 (error rates by group)

| Group | N | FP | FN | FPR | FNR | PPV | Selection rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| M | 3178 | 143 | 813 | 0.06554 | 0.81627 | 0.56135 | 0.10258 |
| F | 428 | 0 | 79 | 0.00000 | 0.98750 | 1.00000 | 0.00234 |

### Age Group (gap summary)

- Threshold sweep gaps (0..1 step 0.05): `fpr_gap_max=0.54079`, `fpr_gap_mean=0.18155`, `fnr_gap_max=0.58465`, `fnr_gap_mean=0.18610`
- Bootstrap gaps @0.5 (CI): `fpr_gap=0.19872` (0.13999–0.26279), `fnr_gap=0.44348` (0.35432–0.53846)

### Age Group

| Group | N | Brier | Brier (CI) | AUROC | AUPRC |
|---|---:|---:|---|---:|---:|
| 23-27 | 744 | 0.20976 | 0.19968–0.21830 | 0.66410 | 0.51037 |
| 28-32 | 719 | 0.19634 | 0.18918–0.20432 | 0.67153 | 0.46172 |
| 33-37 | 539 | 0.19059 | 0.18211–0.19918 | 0.71427 | 0.52442 |
| 48 or older | 531 | 0.13884 | 0.13169–0.14546 | 0.73662 | 0.37454 |
| 38-42 | 405 | 0.17344 | 0.16413–0.18366 | 0.74052 | 0.49285 |
| 43-47 | 397 | 0.17603 | 0.16740–0.18447 | 0.66467 | 0.37095 |
| 18-22 | 271 | 0.22751 | 0.21092–0.24702 | 0.66087 | 0.57113 |

- Equalized-odds gaps @0.5: `fpr_gap=0.19872`, `fnr_gap=0.44348`, `max=0.44348`
- Predictive parity proxy @0.5: `ppv_gap=0.36471`
- Threshold sweep gaps (0..1 step 0.05): `fpr_gap_max=0.54079`, `fpr_gap_mean=0.18155`, `fnr_gap_max=0.58465`, `fnr_gap_mean=0.18610`

#### Threshold 0.5 (error rates by group)

| Group | N | FP | FN | FPR | FNR | PPV | Selection rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| 23-27 | 744 | 59 | 199 | 0.12215 | 0.76245 | 0.51240 | 0.16263 |
| 28-32 | 719 | 33 | 175 | 0.06613 | 0.79545 | 0.57692 | 0.10848 |
| 33-37 | 539 | 4 | 154 | 0.01075 | 0.92216 | 0.76471 | 0.03154 |
| 48 or older | 531 | 0 | 102 | 0.00000 | 1.00000 | nan | 0.00000 |
| 38-42 | 405 | 10 | 103 | 0.03413 | 0.91964 | 0.47368 | 0.04691 |
| 43-47 | 397 | 6 | 95 | 0.02013 | 0.95960 | 0.40000 | 0.02519 |
| 18-22 | 271 | 31 | 64 | 0.19872 | 0.55652 | 0.62195 | 0.30258 |

## DYNAMIC Y2 — with race (calibration: isotonic)

| Metric | Value |
|---|---:|
| Brier | 0.17309 |
| Brier (CI) | 0.16886–0.17784 |
| AUROC | 0.70130 |
| AUPRC | 0.41628 |
| Log loss | 0.51786 |
| ECE | 0.01925 |

### Race (gap summary)

- Threshold sweep gaps (0..1 step 0.05): `fpr_gap_max=0.04775`, `fpr_gap_mean=0.00958`, `fnr_gap_max=0.03969`, `fnr_gap_mean=0.01049`
- Bootstrap gaps @0.5 (CI): `fpr_gap=0.00549` (0.00039–0.02677), `fnr_gap=0.00545` (0.00108–0.06471)

### Race

| Group | N | Brier | Brier (CI) | AUROC | AUPRC |
|---|---:|---:|---|---:|---:|
| BLACK | 1399 | 0.17576 | 0.17001–0.18135 | 0.69251 | 0.41340 |
| WHITE | 1132 | 0.16980 | 0.16317–0.17641 | 0.71238 | 0.41959 |

- Equalized-odds gaps @0.5: `fpr_gap=0.00549`, `fnr_gap=0.00545`, `max=0.00549`
- Predictive parity proxy @0.5: `ppv_gap=0.02389`
- Threshold sweep gaps (0..1 step 0.05): `fpr_gap_max=0.04775`, `fpr_gap_mean=0.00958`, `fnr_gap_max=0.03969`, `fnr_gap_mean=0.01049`

#### Threshold 0.5 (error rates by group)

| Group | N | FP | FN | FPR | FNR | PPV | Selection rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| BLACK | 1399 | 52 | 307 | 0.05019 | 0.84573 | 0.51852 | 0.07720 |
| WHITE | 1132 | 47 | 242 | 0.05569 | 0.84028 | 0.49462 | 0.08216 |

### Gender (gap summary)

- Threshold sweep gaps (0..1 step 0.05): `fpr_gap_max=0.16616`, `fpr_gap_mean=0.04439`, `fnr_gap_max=0.13714`, `fnr_gap_mean=0.03120`
- Bootstrap gaps @0.5 (CI): `fpr_gap=0.04159` (0.02157–0.05998), `fnr_gap=0.05822` (0.00397–0.12899)

### Gender

| Group | N | Brier | Brier (CI) | AUROC | AUPRC |
|---|---:|---:|---|---:|---:|
| M | 2168 | 0.17767 | 0.17309–0.18212 | 0.69360 | 0.41925 |
| F | 363 | 0.14579 | 0.13520–0.15616 | 0.74312 | 0.38922 |

- Equalized-odds gaps @0.5: `fpr_gap=0.04159`, `fnr_gap=0.05822`, `max=0.05822`
- Predictive parity proxy @0.5: `ppv_gap=0.11538`
- Threshold sweep gaps (0..1 step 0.05): `fpr_gap_max=0.16616`, `fpr_gap_mean=0.04439`, `fnr_gap_max=0.13714`, `fnr_gap_mean=0.03120`

#### Threshold 0.5 (error rates by group)

| Group | N | FP | FN | FPR | FNR | PPV | Selection rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| M | 2168 | 94 | 481 | 0.05901 | 0.83652 | 0.50000 | 0.08672 |
| F | 363 | 5 | 68 | 0.01742 | 0.89474 | 0.61538 | 0.03581 |

### Age Group (gap summary)

- Threshold sweep gaps (0..1 step 0.05): `fpr_gap_max=0.39924`, `fpr_gap_mean=0.13658`, `fnr_gap_max=0.33972`, `fnr_gap_mean=0.11140`
- Bootstrap gaps @0.5 (CI): `fpr_gap=0.12937` (0.07848–0.20490), `fnr_gap=0.21132` (0.15268–0.31362)

### Age Group

| Group | N | Brier | Brier (CI) | AUROC | AUPRC |
|---|---:|---:|---|---:|---:|
| 23-27 | 473 | 0.19974 | 0.18863–0.21152 | 0.66254 | 0.44178 |
| 28-32 | 457 | 0.20260 | 0.19201–0.21428 | 0.67345 | 0.45000 |
| 48 or older | 442 | 0.11839 | 0.10952–0.12591 | 0.75386 | 0.34203 |
| 33-37 | 408 | 0.16883 | 0.15918–0.18048 | 0.67266 | 0.39083 |
| 38-42 | 298 | 0.15788 | 0.14815–0.16849 | 0.74293 | 0.38839 |
| 43-47 | 284 | 0.16331 | 0.15216–0.17642 | 0.67410 | 0.36142 |
| 18-22 | 169 | 0.21536 | 0.19781–0.23392 | 0.65678 | 0.47359 |

- Equalized-odds gaps @0.5: `fpr_gap=0.12937`, `fnr_gap=0.21132`, `max=0.21132`
- Predictive parity proxy @0.5: `ppv_gap=0.16757`
- Threshold sweep gaps (0..1 step 0.05): `fpr_gap_max=0.39924`, `fpr_gap_mean=0.13658`, `fnr_gap_max=0.33972`, `fnr_gap_mean=0.11140`

#### Threshold 0.5 (error rates by group)

| Group | N | FP | FN | FPR | FNR | PPV | Selection rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| 23-27 | 473 | 32 | 106 | 0.09846 | 0.71622 | 0.56757 | 0.15645 |
| 28-32 | 457 | 13 | 131 | 0.04180 | 0.89726 | 0.53571 | 0.06127 |
| 48 or older | 442 | 6 | 64 | 0.01609 | 0.92754 | 0.45455 | 0.02489 |
| 33-37 | 408 | 14 | 81 | 0.04502 | 0.83505 | 0.53333 | 0.07353 |
| 38-42 | 298 | 12 | 61 | 0.05240 | 0.88406 | 0.40000 | 0.06711 |
| 43-47 | 284 | 6 | 58 | 0.02715 | 0.92063 | 0.45455 | 0.03873 |
| 18-22 | 169 | 16 | 48 | 0.14545 | 0.81356 | 0.40741 | 0.15976 |

## DYNAMIC Y2 — without race (calibration: isotonic)

| Metric | Value |
|---|---:|
| Brier | 0.17374 |
| Brier (CI) | 0.16957–0.17833 |
| AUROC | 0.70268 |
| AUPRC | 0.42032 |
| Log loss | 0.52247 |
| ECE | 0.02537 |

### Race (gap summary)

- Threshold sweep gaps (0..1 step 0.05): `fpr_gap_max=0.07164`, `fpr_gap_mean=0.01733`, `fnr_gap_max=0.04456`, `fnr_gap_mean=0.00926`
- Bootstrap gaps @0.5 (CI): `fpr_gap=0.00150` (0.00028–0.01608), `fnr_gap=0.01045` (0.00090–0.05223)

### Race

| Group | N | Brier | Brier (CI) | AUROC | AUPRC |
|---|---:|---:|---|---:|---:|
| BLACK | 1399 | 0.17637 | 0.17073–0.18239 | 0.69332 | 0.42182 |
| WHITE | 1132 | 0.17048 | 0.16406–0.17792 | 0.71490 | 0.41940 |

- Equalized-odds gaps @0.5: `fpr_gap=0.00150`, `fnr_gap=0.01045`, `max=0.01045`
- Predictive parity proxy @0.5: `ppv_gap=0.05769`
- Threshold sweep gaps (0..1 step 0.05): `fpr_gap_max=0.07164`, `fpr_gap_mean=0.01733`, `fnr_gap_max=0.04456`, `fnr_gap_mean=0.00926`

#### Threshold 0.5 (error rates by group)

| Group | N | FP | FN | FPR | FNR | PPV | Selection rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| BLACK | 1399 | 23 | 334 | 0.02220 | 0.92011 | 0.55769 | 0.03717 |
| WHITE | 1132 | 20 | 268 | 0.02370 | 0.93056 | 0.50000 | 0.03534 |

### Gender (gap summary)

- Threshold sweep gaps (0..1 step 0.05): `fpr_gap_max=0.16651`, `fpr_gap_mean=0.04971`, `fnr_gap_max=0.22449`, `fnr_gap_mean=0.04182`
- Bootstrap gaps @0.5 (CI): `fpr_gap=0.02288` (0.01162–0.03277), `fnr_gap=0.04053` (0.00243–0.08463)

### Gender

| Group | N | Brier | Brier (CI) | AUROC | AUPRC |
|---|---:|---:|---|---:|---:|
| M | 2168 | 0.17847 | 0.17368–0.18296 | 0.69417 | 0.42348 |
| F | 363 | 0.14550 | 0.13625–0.15377 | 0.75394 | 0.40646 |

- Equalized-odds gaps @0.5: `fpr_gap=0.02288`, `fnr_gap=0.04053`, `max=0.04053`
- Predictive parity proxy @0.5: `ppv_gap=0.22727`
- Threshold sweep gaps (0..1 step 0.05): `fpr_gap_max=0.16651`, `fpr_gap_mean=0.04971`, `fnr_gap_max=0.22449`, `fnr_gap_mean=0.04182`

#### Threshold 0.5 (error rates by group)

| Group | N | FP | FN | FPR | FNR | PPV | Selection rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| M | 2168 | 42 | 529 | 0.02637 | 0.92000 | 0.52273 | 0.04059 |
| F | 363 | 1 | 73 | 0.00348 | 0.96053 | 0.75000 | 0.01102 |

### Age Group (gap summary)

- Threshold sweep gaps (0..1 step 0.05): `fpr_gap_max=0.40100`, `fpr_gap_mean=0.14329`, `fnr_gap_max=0.38099`, `fnr_gap_mean=0.11151`
- Bootstrap gaps @0.5 (CI): `fpr_gap=0.08555` (0.04684–0.14706), `fnr_gap=0.12110` (0.08358–0.21819)

### Age Group

| Group | N | Brier | Brier (CI) | AUROC | AUPRC |
|---|---:|---:|---|---:|---:|
| 23-27 | 473 | 0.20121 | 0.18931–0.21323 | 0.66169 | 0.45783 |
| 28-32 | 457 | 0.20118 | 0.19236–0.21177 | 0.68615 | 0.46835 |
| 48 or older | 442 | 0.11679 | 0.10813–0.12515 | 0.76204 | 0.35261 |
| 33-37 | 408 | 0.17307 | 0.16279–0.18350 | 0.67151 | 0.39500 |
| 38-42 | 298 | 0.15889 | 0.14961–0.16849 | 0.73071 | 0.36129 |
| 43-47 | 284 | 0.16391 | 0.15321–0.17615 | 0.66645 | 0.35729 |
| 18-22 | 169 | 0.21588 | 0.19377–0.23503 | 0.66433 | 0.47410 |

- Equalized-odds gaps @0.5: `fpr_gap=0.08555`, `fnr_gap=0.12110`, `max=0.12110`
- Predictive parity proxy @0.5: `ppv_gap=0.46667`
- Threshold sweep gaps (0..1 step 0.05): `fpr_gap_max=0.40100`, `fpr_gap_mean=0.14329`, `fnr_gap_max=0.38099`, `fnr_gap_mean=0.11151`

#### Threshold 0.5 (error rates by group)

| Group | N | FP | FN | FPR | FNR | PPV | Selection rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| 23-27 | 473 | 17 | 129 | 0.05231 | 0.87162 | 0.52778 | 0.07611 |
| 28-32 | 457 | 4 | 139 | 0.01286 | 0.95205 | 0.63636 | 0.02407 |
| 48 or older | 442 | 2 | 66 | 0.00536 | 0.95652 | 0.60000 | 0.01131 |
| 33-37 | 408 | 4 | 89 | 0.01286 | 0.91753 | 0.66667 | 0.02941 |
| 38-42 | 298 | 4 | 68 | 0.01747 | 0.98551 | 0.20000 | 0.01678 |
| 43-47 | 284 | 2 | 60 | 0.00905 | 0.95238 | 0.60000 | 0.01761 |
| 18-22 | 169 | 10 | 51 | 0.09091 | 0.86441 | 0.44444 | 0.10651 |

## DYNAMIC Y3 — with race (calibration: platt)

| Metric | Value |
|---|---:|
| Brier | 0.14284 |
| Brier (CI) | 0.13961–0.14627 |
| AUROC | 0.70279 |
| AUPRC | 0.32160 |
| Log loss | 0.44485 |
| ECE | 0.01402 |

### Race (gap summary)

- Threshold sweep gaps (0..1 step 0.05): `fpr_gap_max=0.09517`, `fpr_gap_mean=0.01860`, `fnr_gap_max=0.10742`, `fnr_gap_mean=0.01598`
- Bootstrap gaps @0.5 (CI): `fpr_gap=0.00076` (0.00012–0.00953), `fnr_gap=0.00388` (0.00052–0.03111)

### Race

| Group | N | Brier | Brier (CI) | AUROC | AUPRC |
|---|---:|---:|---|---:|---:|
| BLACK | 1089 | 0.14890 | 0.14482–0.15315 | 0.69772 | 0.33218 |
| WHITE | 791 | 0.13449 | 0.12936–0.13885 | 0.70739 | 0.30503 |

- Equalized-odds gaps @0.5: `fpr_gap=0.00076`, `fnr_gap=0.00388`, `max=0.00388`
- Predictive parity proxy @0.5: `ppv_gap=0.06667`
- Threshold sweep gaps (0..1 step 0.05): `fpr_gap_max=0.09517`, `fpr_gap_mean=0.01860`, `fnr_gap_max=0.10742`, `fnr_gap_mean=0.01598`

#### Threshold 0.5 (error rates by group)

| Group | N | FP | FN | FPR | FNR | PPV | Selection rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| BLACK | 1089 | 6 | 215 | 0.00690 | 0.98174 | 0.40000 | 0.00918 |
| WHITE | 791 | 4 | 137 | 0.00613 | 0.98561 | 0.33333 | 0.00759 |

### Gender (gap summary)

- Threshold sweep gaps (0..1 step 0.05): `fpr_gap_max=0.19341`, `fpr_gap_mean=0.03789`, `fnr_gap_max=0.19393`, `fnr_gap_mean=0.03760`
- Bootstrap gaps @0.5 (CI): `fpr_gap=0.00783` (0.00314–0.01307), `fnr_gap=0.01863` (0.00613–0.03459)

### Gender

| Group | N | Brier | Brier (CI) | AUROC | AUPRC |
|---|---:|---:|---|---:|---:|
| M | 1599 | 0.14975 | 0.14574–0.15333 | 0.69225 | 0.32552 |
| F | 281 | 0.10347 | 0.09758–0.10951 | 0.73220 | 0.27416 |

- Equalized-odds gaps @0.5: `fpr_gap=0.00783`, `fnr_gap=0.01863`, `max=0.01863`
- Predictive parity proxy @0.5: `ppv_gap=0.00000`
- Threshold sweep gaps (0..1 step 0.05): `fpr_gap_max=0.19341`, `fpr_gap_mean=0.03789`, `fnr_gap_max=0.19393`, `fnr_gap_mean=0.03760`

#### Threshold 0.5 (error rates by group)

| Group | N | FP | FN | FPR | FNR | PPV | Selection rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| M | 1599 | 10 | 316 | 0.00783 | 0.98137 | 0.37500 | 0.01001 |
| F | 281 | 0 | 36 | 0.00000 | 1.00000 | nan | 0.00000 |

### Age Group (gap summary)

- Threshold sweep gaps (0..1 step 0.05): `fpr_gap_max=0.41030`, `fpr_gap_mean=0.09497`, `fnr_gap_max=0.48045`, `fnr_gap_mean=0.10047`
- Bootstrap gaps @0.5 (CI): `fpr_gap=0.01581` (0.00826–0.04348), `fnr_gap=0.04918` (0.01695–0.10773)

### Age Group

| Group | N | Brier | Brier (CI) | AUROC | AUPRC |
|---|---:|---:|---|---:|---:|
| 48 or older | 368 | 0.11568 | 0.11166–0.11991 | 0.69931 | 0.22343 |
| 23-27 | 324 | 0.16524 | 0.15828–0.17307 | 0.64766 | 0.29168 |
| 28-32 | 323 | 0.13819 | 0.13058–0.14563 | 0.72006 | 0.39338 |
| 33-37 | 312 | 0.15957 | 0.15135–0.16722 | 0.72415 | 0.40430 |
| 38-42 | 241 | 0.15287 | 0.14483–0.16107 | 0.75632 | 0.43997 |
| 43-47 | 207 | 0.10858 | 0.10202–0.11501 | 0.68402 | 0.19465 |
| 18-22 | 105 | 0.17792 | 0.16249–0.19302 | 0.58076 | 0.26143 |

- Equalized-odds gaps @0.5: `fpr_gap=0.01581`, `fnr_gap=0.04918`, `max=0.04918`
- Predictive parity proxy @0.5: `ppv_gap=0.75000`
- Threshold sweep gaps (0..1 step 0.05): `fpr_gap_max=0.41030`, `fpr_gap_mean=0.09497`, `fnr_gap_max=0.48045`, `fnr_gap_mean=0.10047`

#### Threshold 0.5 (error rates by group)

| Group | N | FP | FN | FPR | FNR | PPV | Selection rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| 48 or older | 368 | 0 | 52 | 0.00000 | 1.00000 | nan | 0.00000 |
| 23-27 | 324 | 4 | 70 | 0.01581 | 0.98592 | 0.20000 | 0.01543 |
| 28-32 | 323 | 1 | 58 | 0.00382 | 0.95082 | 0.75000 | 0.01238 |
| 33-37 | 312 | 1 | 70 | 0.00415 | 0.98592 | 0.50000 | 0.00641 |
| 38-42 | 241 | 1 | 52 | 0.00532 | 0.98113 | 0.50000 | 0.00830 |
| 43-47 | 207 | 2 | 26 | 0.01105 | 1.00000 | 0.00000 | 0.00966 |
| 18-22 | 105 | 1 | 24 | 0.01235 | 1.00000 | 0.00000 | 0.00952 |

## DYNAMIC Y3 — without race (calibration: platt)

| Metric | Value |
|---|---:|
| Brier | 0.14283 |
| Brier (CI) | 0.13956–0.14624 |
| AUROC | 0.70310 |
| AUPRC | 0.32112 |
| Log loss | 0.44458 |
| ECE | 0.01456 |

### Race (gap summary)

- Threshold sweep gaps (0..1 step 0.05): `fpr_gap_max=0.10306`, `fpr_gap_mean=0.01904`, `fnr_gap_max=0.11590`, `fnr_gap_mean=0.01778`
- Bootstrap gaps @0.5 (CI): `fpr_gap=0.00077` (0.00012–0.01051), `fnr_gap=0.00069` (0.00052–0.03039)

### Race

| Group | N | Brier | Brier (CI) | AUROC | AUPRC |
|---|---:|---:|---|---:|---:|
| BLACK | 1089 | 0.14895 | 0.14516–0.15330 | 0.69785 | 0.33367 |
| WHITE | 791 | 0.13441 | 0.12951–0.13873 | 0.70858 | 0.30648 |

- Equalized-odds gaps @0.5: `fpr_gap=0.00077`, `fnr_gap=0.00069`, `max=0.00077`
- Predictive parity proxy @0.5: `ppv_gap=0.04762`
- Threshold sweep gaps (0..1 step 0.05): `fpr_gap_max=0.10306`, `fpr_gap_mean=0.01904`, `fnr_gap_max=0.11590`, `fnr_gap_mean=0.01778`

#### Threshold 0.5 (error rates by group)

| Group | N | FP | FN | FPR | FNR | PPV | Selection rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| BLACK | 1089 | 6 | 216 | 0.00690 | 0.98630 | 0.33333 | 0.00826 |
| WHITE | 791 | 5 | 137 | 0.00767 | 0.98561 | 0.28571 | 0.00885 |

### Gender (gap summary)

- Threshold sweep gaps (0..1 step 0.05): `fpr_gap_max=0.19233`, `fpr_gap_mean=0.03993`, `fnr_gap_max=0.17857`, `fnr_gap_mean=0.04085`
- Bootstrap gaps @0.5 (CI): `fpr_gap=0.00861` (0.00387–0.01399), `fnr_gap=0.01553` (0.00313–0.03096)

### Gender

| Group | N | Brier | Brier (CI) | AUROC | AUPRC |
|---|---:|---:|---|---:|---:|
| M | 1599 | 0.14973 | 0.14571–0.15318 | 0.69270 | 0.32511 |
| F | 281 | 0.10353 | 0.09725–0.10965 | 0.73175 | 0.27996 |

- Equalized-odds gaps @0.5: `fpr_gap=0.00861`, `fnr_gap=0.01553`, `max=0.01553`
- Predictive parity proxy @0.5: `ppv_gap=0.00000`
- Threshold sweep gaps (0..1 step 0.05): `fpr_gap_max=0.19233`, `fpr_gap_mean=0.03993`, `fnr_gap_max=0.17857`, `fnr_gap_mean=0.04085`

#### Threshold 0.5 (error rates by group)

| Group | N | FP | FN | FPR | FNR | PPV | Selection rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| M | 1599 | 11 | 317 | 0.00861 | 0.98447 | 0.31250 | 0.01001 |
| F | 281 | 0 | 36 | 0.00000 | 1.00000 | nan | 0.00000 |

### Age Group (gap summary)

- Threshold sweep gaps (0..1 step 0.05): `fpr_gap_max=0.39448`, `fpr_gap_mean=0.09470`, `fnr_gap_max=0.48045`, `fnr_gap_mean=0.10375`
- Bootstrap gaps @0.5 (CI): `fpr_gap=0.01976` (0.01111–0.04478), `fnr_gap=0.04918` (0.01639–0.10773)

### Age Group

| Group | N | Brier | Brier (CI) | AUROC | AUPRC |
|---|---:|---:|---|---:|---:|
| 48 or older | 368 | 0.11561 | 0.11177–0.11977 | 0.70344 | 0.22385 |
| 23-27 | 324 | 0.16520 | 0.15829–0.17264 | 0.64844 | 0.29527 |
| 28-32 | 323 | 0.13781 | 0.12959–0.14587 | 0.72137 | 0.39100 |
| 33-37 | 312 | 0.15927 | 0.15103–0.16724 | 0.72608 | 0.40363 |
| 38-42 | 241 | 0.15230 | 0.14406–0.15999 | 0.76325 | 0.44045 |
| 43-47 | 207 | 0.11011 | 0.10352–0.11679 | 0.65108 | 0.17169 |
| 18-22 | 105 | 0.17855 | 0.16424–0.19303 | 0.57253 | 0.24865 |

- Equalized-odds gaps @0.5: `fpr_gap=0.01976`, `fnr_gap=0.04918`, `max=0.04918`
- Predictive parity proxy @0.5: `ppv_gap=1.00000`
- Threshold sweep gaps (0..1 step 0.05): `fpr_gap_max=0.39448`, `fpr_gap_mean=0.09470`, `fnr_gap_max=0.48045`, `fnr_gap_mean=0.10375`

#### Threshold 0.5 (error rates by group)

| Group | N | FP | FN | FPR | FNR | PPV | Selection rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| 48 or older | 368 | 0 | 52 | 0.00000 | 1.00000 | nan | 0.00000 |
| 23-27 | 324 | 5 | 71 | 0.01976 | 1.00000 | 0.00000 | 0.01543 |
| 28-32 | 323 | 1 | 58 | 0.00382 | 0.95082 | 0.75000 | 0.01238 |
| 33-37 | 312 | 2 | 70 | 0.00830 | 0.98592 | 0.33333 | 0.00962 |
| 38-42 | 241 | 0 | 52 | 0.00000 | 0.98113 | 1.00000 | 0.00415 |
| 43-47 | 207 | 2 | 26 | 0.01105 | 1.00000 | 0.00000 | 0.00966 |
| 18-22 | 105 | 1 | 24 | 0.01235 | 1.00000 | 0.00000 | 0.00952 |

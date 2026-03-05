# NCRP Fairness / Subgroup Audit (ICPSR 37973 term-record benchmark)

Generated: 2026-03-03T14:45:22.852234+00:00

Variant: `terms_mod200_r0`.

- Label family: **reincarceration** (return to prison)
- Timing: **year granularity** (ADMITYR/RELEASEYR only in public-use extract)
- Model: tuned **XGBoost** with post-hoc calibration; splits grouped by **ABT_INMATE_ID**

## Y1

- calibration=raw, brier=0.16716, auroc=0.77202, auprc=0.59616, log_loss=0.50606, ece=0.01176

### By race

| Group | N | Brier | Brier (CI) | AUROC | AUPRC |
|---|---:|---:|---|---:|---:|
| White, non-Hispanic | 4399 | 0.16409 | 0.15951–0.16924 | 0.78305 | 0.61517 |
| Black, non-Hispanic | 4298 | 0.17672 | 0.17255–0.18183 | 0.75994 | 0.60535 |
| Hispanic, any race | 2004 | 0.16686 | 0.16034–0.17360 | 0.76271 | 0.54838 |
| Unknown | 1166 | 0.14610 | 0.13996–0.15300 | 0.77511 | 0.51115 |
| Other race(s), non-Hispanic | 237 | 0.15708 | 0.13647–0.18084 | 0.79817 | 0.69933 |

#### Threshold 0.5 (error rates by group)

| Group | N | FP | FN | FPR | FNR | PPV | Selection rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| White, non-Hispanic | 4399 | 279 | 765 | 0.09064 | 0.57911 | 0.66587 | 0.18982 |
| Black, non-Hispanic | 4298 | 248 | 867 | 0.08522 | 0.62464 | 0.67750 | 0.17892 |
| Hispanic, any race | 2004 | 142 | 340 | 0.09889 | 0.59859 | 0.61622 | 0.18463 |
| Unknown | 1166 | 31 | 212 | 0.03456 | 0.78810 | 0.64773 | 0.07547 |
| Other race(s), non-Hispanic | 237 | 13 | 37 | 0.07975 | 0.50000 | 0.74000 | 0.21097 |

- Max FPR gap across thresholds: 0.13689
- Max FNR gap across thresholds: 0.30197

### By sex

| Group | N | Brier | Brier (CI) | AUROC | AUPRC |
|---|---:|---:|---|---:|---:|
| Male | 10719 | 0.16862 | 0.16598–0.17119 | 0.77313 | 0.60289 |
| Female | 1385 | 0.15586 | 0.14818–0.16249 | 0.75636 | 0.52875 |

#### Threshold 0.5 (error rates by group)

| Group | N | FP | FN | FPR | FNR | PPV | Selection rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| Male | 10719 | 659 | 1981 | 0.08852 | 0.60507 | 0.66240 | 0.18211 |
| Female | 1385 | 54 | 240 | 0.05197 | 0.69364 | 0.66250 | 0.11552 |

- Max FPR gap across thresholds: 0.14048
- Max FNR gap across thresholds: 0.16220

### By age group

| Group | N | Brier | Brier (CI) | AUROC | AUPRC |
|---|---:|---:|---|---:|---:|
| <25 | 12104 | 0.16716 | 0.16431–0.17005 | 0.77202 | 0.59616 |

#### Threshold 0.5 (error rates by group)

| Group | N | FP | FN | FPR | FNR | PPV | Selection rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| <25 | 12104 | 713 | 2221 | 0.08404 | 0.61354 | 0.66241 | 0.17449 |

- Max FPR gap across thresholds: 0.00000
- Max FNR gap across thresholds: 0.00000

- FPR gap@0.5 (race): 0.06433
- FNR gap@0.5 (race): 0.28810

## Y2

- calibration=isotonic, brier=0.08765, auroc=0.70191, auprc=0.21328, log_loss=0.30479, ece=0.00085

### By race

| Group | N | Brier | Brier (CI) | AUROC | AUPRC |
|---|---:|---:|---|---:|---:|
| White, non-Hispanic | 3078 | 0.07869 | 0.07766–0.07986 | 0.68979 | 0.17737 |
| Black, non-Hispanic | 2910 | 0.10509 | 0.10341–0.10678 | 0.69495 | 0.25347 |
| Hispanic, any race | 1436 | 0.07855 | 0.07713–0.07958 | 0.69738 | 0.17270 |
| Unknown | 897 | 0.07861 | 0.07678–0.08056 | 0.71490 | 0.19599 |
| Other race(s), non-Hispanic | 163 | 0.07518 | 0.07217–0.07891 | 0.70614 | 0.20320 |

#### Threshold 0.5 (error rates by group)

| Group | N | FP | FN | FPR | FNR | PPV | Selection rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| White, non-Hispanic | 3078 | 0 | 277 | 0.00000 | 1.00000 | nan | 0.00000 |
| Black, non-Hispanic | 2910 | 0 | 371 | 0.00000 | 1.00000 | nan | 0.00000 |
| Hispanic, any race | 1436 | 0 | 129 | 0.00000 | 1.00000 | nan | 0.00000 |
| Unknown | 897 | 0 | 82 | 0.00000 | 1.00000 | nan | 0.00000 |
| Other race(s), non-Hispanic | 163 | 0 | 14 | 0.00000 | 1.00000 | nan | 0.00000 |

- Max FPR gap across thresholds: 0.23725
- Max FNR gap across thresholds: 0.31941

### By sex

| Group | N | Brier | Brier (CI) | AUROC | AUPRC |
|---|---:|---:|---|---:|---:|
| Male | 7445 | 0.09103 | 0.09024–0.09204 | 0.69636 | 0.21653 |
| Female | 1039 | 0.06340 | 0.06212–0.06478 | 0.73418 | 0.18103 |

#### Threshold 0.5 (error rates by group)

| Group | N | FP | FN | FPR | FNR | PPV | Selection rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| Male | 7445 | 0 | 799 | 0.00000 | 1.00000 | nan | 0.00000 |
| Female | 1039 | 0 | 74 | 0.00000 | 1.00000 | nan | 0.00000 |

- Max FPR gap across thresholds: 0.11175
- Max FNR gap across thresholds: 0.06334

### By age group

| Group | N | Brier | Brier (CI) | AUROC | AUPRC |
|---|---:|---:|---|---:|---:|
| <25 | 8484 | 0.08765 | 0.08694–0.08834 | 0.70191 | 0.21328 |

#### Threshold 0.5 (error rates by group)

| Group | N | FP | FN | FPR | FNR | PPV | Selection rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| <25 | 8484 | 0 | 873 | 0.00000 | 1.00000 | nan | 0.00000 |

- Max FPR gap across thresholds: 0.00000
- Max FNR gap across thresholds: 0.00000

- FPR gap@0.5 (race): 0.00000
- FNR gap@0.5 (race): 0.00000

## Y3

- calibration=raw, brier=0.04827, auroc=0.69233, auprc=0.09952, log_loss=0.18971, ece=0.00475

### By race

| Group | N | Brier | Brier (CI) | AUROC | AUPRC |
|---|---:|---:|---|---:|---:|
| White, non-Hispanic | 2801 | 0.04675 | 0.04636–0.04703 | 0.63369 | 0.07732 |
| Black, non-Hispanic | 2539 | 0.06030 | 0.05984–0.06072 | 0.70131 | 0.12001 |
| Hispanic, any race | 1307 | 0.03164 | 0.03139–0.03187 | 0.67094 | 0.05143 |
| Unknown | 815 | 0.04343 | 0.04294–0.04391 | 0.73159 | 0.09829 |
| Other race(s), non-Hispanic | 149 | 0.04405 | 0.04257–0.04539 | 0.67203 | 0.08278 |

#### Threshold 0.5 (error rates by group)

| Group | N | FP | FN | FPR | FNR | PPV | Selection rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| White, non-Hispanic | 2801 | 0 | 140 | 0.00000 | 1.00000 | nan | 0.00000 |
| Black, non-Hispanic | 2539 | 0 | 169 | 0.00000 | 1.00000 | nan | 0.00000 |
| Hispanic, any race | 1307 | 0 | 43 | 0.00000 | 1.00000 | nan | 0.00000 |
| Unknown | 815 | 0 | 38 | 0.00000 | 1.00000 | nan | 0.00000 |
| Other race(s), non-Hispanic | 149 | 0 | 7 | 0.00000 | 1.00000 | nan | 0.00000 |

- Max FPR gap across thresholds: 0.23112
- Max FNR gap across thresholds: 0.33728

### By sex

| Group | N | Brier | Brier (CI) | AUROC | AUPRC |
|---|---:|---:|---|---:|---:|
| Male | 6646 | 0.04978 | 0.04951–0.05005 | 0.68639 | 0.10031 |
| Female | 965 | 0.03786 | 0.03735–0.03828 | 0.73573 | 0.08864 |

#### Threshold 0.5 (error rates by group)

| Group | N | FP | FN | FPR | FNR | PPV | Selection rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| Male | 6646 | 0 | 358 | 0.00000 | 1.00000 | nan | 0.00000 |
| Female | 965 | 0 | 39 | 0.00000 | 1.00000 | nan | 0.00000 |

- Max FPR gap across thresholds: 0.08117
- Max FNR gap across thresholds: 0.09905

### By age group

| Group | N | Brier | Brier (CI) | AUROC | AUPRC |
|---|---:|---:|---|---:|---:|
| <25 | 7611 | 0.04827 | 0.04804–0.04849 | 0.69233 | 0.09952 |

#### Threshold 0.5 (error rates by group)

| Group | N | FP | FN | FPR | FNR | PPV | Selection rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| <25 | 7611 | 0 | 397 | 0.00000 | 1.00000 | nan | 0.00000 |

- Max FPR gap across thresholds: 0.00000
- Max FNR gap across thresholds: 0.00000

- FPR gap@0.5 (race): 0.00000
- FNR gap@0.5 (race): 0.00000

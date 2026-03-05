# NCRP (ICPSR 37973) — Predictive Factors and Simple Associations

This report shows **which features the model uses** (SHAP) and **how reincarceration rates vary** by key variables (unadjusted associations) for the NCRP dataset.

- Label: **return to prison / reincarceration** (not rearrest).
- Y2/Y3 are conditional (only individuals not reincarcerated in prior horizons).
- Event timing: **year granularity** (ICPSR 37973 public-use extract).
- SHAP shows association with predictions, not causality.
- Not directly comparable to NIJ (different label, jurisdiction, features).

## Top Predictive Factors (SHAP, Year 1)

From `reports/plots/xgb_shap_ncrp37973_y1.json` (XGBoost).

| Rank | Feature | Mean \|SHAP\| |
|---:|---|---:|
| 1 | Release year | 0.3078 |
| 2 | Release type: Unconditional release (max-out) | 0.3077 |
| 3 | Admission type: Parole revocation/return | 0.2308 |
| 4 | State: California | 0.2128 |
| 5 | Time served (months) | 0.1682 |
| 6 | Offense category: Property | 0.1121 |
| 7 | Admission year | 0.0875 |
| 8 | Race: Hispanic, any race | 0.0746 |
| 9 | Age at release | 0.0730 |
| 10 | Sex: female | 0.0677 |

## Observed Reincarceration Rates by Key Variables

These are raw subgroup rates (not adjusted for other variables).

### Y1 (N=60,381, base rate=0.295)

#### Race

| Group | N | Reincarceration rate |
|---|---:|---:|
| White, non-Hispanic | 22233 | 0.291 |
| Black, non-Hispanic | 21598 | 0.315 |
| Hispanic, any race | 9985 | 0.299 |
| Unknown | 5439 | 0.222 |
| Other race(s), non-Hispanic | 1126 | 0.335 |

#### Sex

| Group | N | Reincarceration rate |
|---|---:|---:|
| Male | 53436 | 0.303 |
| Female | 6945 | 0.240 |

#### Age at release

| Group | N | Reincarceration rate |
|---|---:|---:|
| 25–34 | 22059 | 0.302 |
| 35–44 | 16543 | 0.309 |
| 18–24 | 10507 | 0.331 |
| 45–54 | 8409 | 0.253 |
| 55+ | 2629 | 0.159 |
| Unknown | 234 | 0.197 |

#### Admission type

| Group | N | Reincarceration rate |
|---|---:|---:|
| New court commitment | 39091 | 0.217 |
| Parole revocation/return | 18998 | 0.462 |
| 9 | 1602 | 0.234 |
| Other/transfer | 690 | 0.299 |

#### Release type

| Group | N | Reincarceration rate |
|---|---:|---:|
| Conditional release (parole) | 41736 | 0.341 |
| Unconditional release (max-out) | 14298 | 0.139 |
| 9 | 3151 | 0.353 |
| Other/transfer | 1196 | 0.426 |

#### Offense category

| Group | N | Reincarceration rate |
|---|---:|---:|
| Property | 17786 | 0.345 |
| Drug | 17519 | 0.294 |
| Violent | 15117 | 0.262 |
| Public order | 9130 | 0.252 |
| Other | 445 | 0.362 |
| Unknown | 384 | 0.310 |

#### Time served

| Group | N | Reincarceration rate |
|---|---:|---:|
| < 6 months | 34134 | 0.352 |
| 6–11 months | 12070 | 0.258 |
| 1–2 years | 9852 | 0.212 |
| 2–5 years | 2985 | 0.160 |
| 5+ years | 1340 | 0.109 |

#### Sentence length

| Group | N | Reincarceration rate |
|---|---:|---:|
| 2–3 years | 24219 | 0.340 |
| 3–5 years | 13171 | 0.293 |
| < 1 year | 8080 | 0.232 |
| 5–10 years | 6969 | 0.221 |
| 1–2 years | 5997 | 0.323 |
| 10–25 years | 1156 | 0.204 |
| 25+ years / life | 414 | 0.121 |
| Unknown | 375 | 0.285 |

### Y2 (N=42,547, base rate=0.102)

#### Race

| Group | N | Reincarceration rate |
|---|---:|---:|
| White, non-Hispanic | 15767 | 0.092 |
| Black, non-Hispanic | 14796 | 0.121 |
| Hispanic, any race | 7001 | 0.091 |
| Unknown | 4234 | 0.092 |
| Other race(s), non-Hispanic | 749 | 0.087 |

#### Sex

| Group | N | Reincarceration rate |
|---|---:|---:|
| Male | 37266 | 0.105 |
| Female | 5281 | 0.082 |

#### Age at release

| Group | N | Reincarceration rate |
|---|---:|---:|
| 25–34 | 15406 | 0.108 |
| 35–44 | 11434 | 0.099 |
| 18–24 | 7026 | 0.140 |
| 45–54 | 6282 | 0.072 |
| 55+ | 2211 | 0.038 |
| Unknown | 188 | 0.154 |

#### Admission type

| Group | N | Reincarceration rate |
|---|---:|---:|
| New court commitment | 30608 | 0.090 |
| Parole revocation/return | 10228 | 0.137 |
| 9 | 1227 | 0.097 |
| Other/transfer | 484 | 0.157 |

#### Release type

| Group | N | Reincarceration rate |
|---|---:|---:|
| Conditional release (parole) | 27514 | 0.110 |
| Unconditional release (max-out) | 12306 | 0.087 |
| 9 | 2040 | 0.107 |
| Other/transfer | 687 | 0.057 |

#### Offense category

| Group | N | Reincarceration rate |
|---|---:|---:|
| Drug | 12374 | 0.101 |
| Property | 11643 | 0.124 |
| Violent | 11153 | 0.091 |
| Public order | 6828 | 0.087 |
| Other | 284 | 0.077 |
| Unknown | 265 | 0.060 |

#### Time served

| Group | N | Reincarceration rate |
|---|---:|---:|
| < 6 months | 22131 | 0.110 |
| 6–11 months | 8951 | 0.105 |
| 1–2 years | 7765 | 0.096 |
| 2–5 years | 2506 | 0.071 |
| 5+ years | 1194 | 0.048 |

#### Sentence length

| Group | N | Reincarceration rate |
|---|---:|---:|
| 2–3 years | 15984 | 0.104 |
| 3–5 years | 9315 | 0.106 |
| < 1 year | 6204 | 0.102 |
| 5–10 years | 5431 | 0.101 |
| 1–2 years | 4061 | 0.104 |
| 10–25 years | 920 | 0.079 |
| 25+ years / life | 364 | 0.041 |
| Unknown | 268 | 0.063 |

### Y3 (N=38,201, base rate=0.055)

#### Race

| Group | N | Reincarceration rate |
|---|---:|---:|
| White, non-Hispanic | 14311 | 0.050 |
| Black, non-Hispanic | 13000 | 0.070 |
| Hispanic, any race | 6361 | 0.039 |
| Unknown | 3845 | 0.052 |
| Other race(s), non-Hispanic | 684 | 0.045 |

#### Sex

| Group | N | Reincarceration rate |
|---|---:|---:|
| Male | 33354 | 0.057 |
| Female | 4847 | 0.039 |

#### Age at release

| Group | N | Reincarceration rate |
|---|---:|---:|
| 25–34 | 13749 | 0.061 |
| 35–44 | 10299 | 0.052 |
| 18–24 | 6041 | 0.074 |
| 45–54 | 5827 | 0.041 |
| 55+ | 2126 | 0.018 |
| Unknown | 159 | 0.075 |

#### Admission type

| Group | N | Reincarceration rate |
|---|---:|---:|
| New court commitment | 27858 | 0.052 |
| Parole revocation/return | 8827 | 0.063 |
| 9 | 1108 | 0.052 |
| Other/transfer | 408 | 0.071 |

#### Release type

| Group | N | Reincarceration rate |
|---|---:|---:|
| Conditional release (parole) | 24497 | 0.055 |
| Unconditional release (max-out) | 11234 | 0.056 |
| 9 | 1822 | 0.057 |
| Other/transfer | 648 | 0.031 |

#### Offense category

| Group | N | Reincarceration rate |
|---|---:|---:|
| Drug | 11119 | 0.060 |
| Property | 10196 | 0.064 |
| Violent | 10143 | 0.044 |
| Public order | 6232 | 0.051 |
| Other | 262 | 0.061 |
| Unknown | 249 | 0.056 |

#### Time served

| Group | N | Reincarceration rate |
|---|---:|---:|
| < 6 months | 19705 | 0.060 |
| 6–11 months | 8007 | 0.052 |
| 1–2 years | 7023 | 0.051 |
| 2–5 years | 2329 | 0.049 |
| 5+ years | 1137 | 0.024 |

#### Sentence length

| Group | N | Reincarceration rate |
|---|---:|---:|
| 2–3 years | 14327 | 0.055 |
| 3–5 years | 8332 | 0.054 |
| < 1 year | 5573 | 0.063 |
| 5–10 years | 4882 | 0.057 |
| 1–2 years | 3640 | 0.050 |
| 10–25 years | 847 | 0.047 |
| 25+ years / life | 349 | 0.020 |
| Unknown | 251 | 0.020 |

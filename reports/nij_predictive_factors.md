# NIJ — Predictive Factors and Simple Associations

This report is intentionally descriptive: it shows **which features the model uses** (SHAP) and **how rearrest rates vary** by a few key variables (unadjusted associations).

Notes:
- The Y2 and Y3 tasks in this repo are **conditional** (only people not rearrested in prior horizons are included).
- SHAP shows association with the model’s predictions, not causality.

## Top Predictive Factors (SHAP, Year 1 Static)

From `reports/plots/xgb_shap_static_y1.json` (XGBoost, static-at-release features).

| Rank | Feature | Mean \|SHAP\| |
|---:|---|---:|
| 1 | Gang_Affiliated_Yes | 0.1627 |
| 2 | _v1_0 | 0.1143 |
| 3 | Age_at_Release_23-27 | 0.1058 |
| 4 | Supervision_Risk_Score_First | 0.1002 |
| 5 | Age_at_Release_48 or older | 0.0919 |
| 6 | Age_at_Release_18-22 | 0.0824 |
| 7 | Prior_Arrest_Episodes_Property_5 or more | 0.0801 |
| 8 | _v1_5 or more | 0.0795 |
| 9 | Prison_Years_Less than 1 year | 0.0686 |
| 10 | Gang_Affiliated__missing | 0.0588 |
| 11 | Age_at_Release_28-32 | 0.0537 |
| 12 | Prior_Arrest_Episodes_Property_0 | 0.0528 |
| 13 | Prior_Arrest_Episodes_Felony_10 or more | 0.0499 |
| 14 | Condition_MH_SA_No | 0.0496 |
| 15 | Prior_Arrest_Episodes_Misd_6 or more | 0.0474 |

## Observed Rearrest Rates by Key Variables (Static-at-release)

These are raw subgroup rates (not adjusted for other variables).

### Y1 (N=18,028, base rate=0.298)

#### Lift (binary variables; unadjusted)

| Variable | N(Yes) | P(Y=1 \| Yes) | N(No) | P(Y=1 \| No) | Δ (pp) | RR |
|---|---:|---:|---:|---:|---:|---:|
| Condition_MH_SA | 11841 | 0.325 | 6187 | 0.247 | 7.8 | 1.315 |

| Variable | N(Yes) | P(Y=1 \| Yes) | N(No) | P(Y=1 \| No) | Δ (pp) | RR |
|---|---:|---:|---:|---:|---:|---:|
| Gang_Affiliated | 2781 | 0.469 | 13030 | 0.278 | 19.1 | 1.690 |

#### Top Unadjusted Risk Lifts (Binary Indicators)

| Variable | N(Yes) | P(Y=1\|Yes) | N(No) | P(Y=1\|No) | Δ (pp) | RR |
|---|---:|---:|---:|---:|---:|---:|
| Gang affiliated | 2781 | 0.469 | 13030 | 0.278 | 19.1 | 1.690 |
| MH/SA condition | 11841 | 0.325 | 6187 | 0.247 | 7.8 | 1.315 |
| Prior parole revocations | 1704 | 0.367 | 16324 | 0.291 | 7.6 | 1.260 |
| Any prior PP-violation-charge conviction episode | 5927 | 0.344 | 12101 | 0.276 | 6.9 | 1.249 |
| Any prior domestic-violence-charge conviction episode | 1450 | 0.359 | 16578 | 0.293 | 6.6 | 1.227 |
| Missing: supervision risk score | 330 | 0.361 | 17698 | 0.297 | 6.4 | 1.214 |
| Any prior domestic-violence-charge arrest episode | 2985 | 0.349 | 15043 | 0.288 | 6.1 | 1.210 |
| Prior probation revocations | 2650 | 0.338 | 15378 | 0.291 | 4.6 | 1.159 |
| Any prior violent conviction episode | 5852 | 0.328 | 12176 | 0.284 | 4.4 | 1.155 |
| Cognitive/education condition | 8020 | 0.321 | 10008 | 0.280 | 4.0 | 1.144 |

#### Age at release

| Group | N | Rearrest rate |
|---|---:|---:|
| 23-27 | 3611 | 0.361 |
| 28-32 | 3449 | 0.324 |
| 33-37 | 2975 | 0.286 |
| 48 or older | 2641 | 0.189 |
| 38-42 | 2040 | 0.262 |
| 43-47 | 1858 | 0.248 |
| 18-22 | 1454 | 0.420 |

#### Gang affiliated

| Group | N | Rearrest rate |
|---|---:|---:|
| No | 13030 | 0.278 |
| Yes | 2781 | 0.469 |
| Unknown | 2217 | 0.206 |

#### MH/SA condition

| Group | N | Rearrest rate |
|---|---:|---:|
| Yes | 11841 | 0.325 |
| No | 6187 | 0.247 |

#### Supervision risk score (binned)

| Group | N | Rearrest rate |
|---|---:|---:|
| 7–10 | 7797 | 0.361 |
| 4–6 | 7095 | 0.268 |
| 1–3 | 2806 | 0.191 |
| Unknown | 330 | 0.361 |

#### Prior property-arrest episodes

| Group | N | Rearrest rate |
|---|---:|---:|
| 0 | 4561 | 0.200 |
| 5 or more | 4088 | 0.390 |
| 1 | 3525 | 0.268 |
| 2 | 2720 | 0.303 |
| 3 | 1852 | 0.343 |
| 4 | 1282 | 0.363 |

#### Prison years

| Group | N | Rearrest rate |
|---|---:|---:|
| 1-2 years | 5629 | 0.313 |
| Less than 1 year | 5622 | 0.348 |
| More than 3 years | 3842 | 0.217 |
| Greater than 2 to 3 years | 2935 | 0.281 |

### Y2 (N=12,651, base rate=0.257)

#### Lift (binary variables; unadjusted)

| Variable | N(Yes) | P(Y=1 \| Yes) | N(No) | P(Y=1 \| No) | Δ (pp) | RR |
|---|---:|---:|---:|---:|---:|---:|
| Condition_MH_SA | 7993 | 0.282 | 4658 | 0.215 | 6.7 | 1.311 |

| Variable | N(Yes) | P(Y=1 \| Yes) | N(No) | P(Y=1 \| No) | Δ (pp) | RR |
|---|---:|---:|---:|---:|---:|---:|
| Gang_Affiliated | 1477 | 0.412 | 9414 | 0.242 | 16.9 | 1.699 |

#### Top Unadjusted Risk Lifts (Binary Indicators)

| Variable | N(Yes) | P(Y=1\|Yes) | N(No) | P(Y=1\|No) | Δ (pp) | RR |
|---|---:|---:|---:|---:|---:|---:|
| Gang affiliated | 1477 | 0.412 | 9414 | 0.242 | 16.9 | 1.699 |
| MH/SA condition | 7993 | 0.282 | 4658 | 0.215 | 6.7 | 1.311 |
| Any prior domestic-violence-charge conviction episode | 929 | 0.326 | 11722 | 0.252 | 7.4 | 1.296 |
| Any prior domestic-violence-charge arrest episode | 1944 | 0.312 | 10707 | 0.247 | 6.5 | 1.261 |
| Prior parole revocations | 1079 | 0.312 | 11572 | 0.252 | 6.0 | 1.239 |
| Any prior PP-violation-charge conviction episode | 3886 | 0.294 | 8765 | 0.241 | 5.3 | 1.219 |
| Missing: supervision level | 820 | 0.294 | 11831 | 0.255 | 3.9 | 1.154 |
| Any prior drug conviction episode | 3120 | 0.265 | 6334 | 0.237 | 2.8 | 1.119 |
| Any prior violent conviction episode | 3933 | 0.276 | 8718 | 0.248 | 2.8 | 1.112 |
| Prior probation revocations | 1755 | 0.280 | 10896 | 0.253 | 2.6 | 1.104 |

#### Age at release

| Group | N | Rearrest rate |
|---|---:|---:|
| 28-32 | 2332 | 0.281 |
| 23-27 | 2306 | 0.321 |
| 48 or older | 2142 | 0.169 |
| 33-37 | 2124 | 0.253 |
| 38-42 | 1506 | 0.235 |
| 43-47 | 1397 | 0.220 |
| 18-22 | 844 | 0.351 |

#### Gang affiliated

| Group | N | Rearrest rate |
|---|---:|---:|
| No | 9414 | 0.242 |
| Unknown | 1760 | 0.207 |
| Yes | 1477 | 0.412 |

#### MH/SA condition

| Group | N | Rearrest rate |
|---|---:|---:|
| Yes | 7993 | 0.282 |
| No | 4658 | 0.215 |

#### Supervision risk score (binned)

| Group | N | Rearrest rate |
|---|---:|---:|
| 4–6 | 5191 | 0.235 |
| 7–10 | 4980 | 0.316 |
| 1–3 | 2269 | 0.177 |
| Unknown | 211 | 0.280 |

#### Prior property-arrest episodes

| Group | N | Rearrest rate |
|---|---:|---:|
| 0 | 3648 | 0.193 |
| 1 | 2580 | 0.252 |
| 5 or more | 2495 | 0.329 |
| 2 | 1895 | 0.260 |
| 3 | 1216 | 0.285 |
| 4 | 817 | 0.291 |

#### Prison years

| Group | N | Rearrest rate |
|---|---:|---:|
| 1-2 years | 3867 | 0.273 |
| Less than 1 year | 3666 | 0.303 |
| More than 3 years | 3008 | 0.193 |
| Greater than 2 to 3 years | 2110 | 0.239 |

### Y3 (N=9,398, base rate=0.191)

#### Lift (binary variables; unadjusted)

| Variable | N(Yes) | P(Y=1 \| Yes) | N(No) | P(Y=1 \| No) | Δ (pp) | RR |
|---|---:|---:|---:|---:|---:|---:|
| Condition_MH_SA | 5741 | 0.209 | 3657 | 0.161 | 4.8 | 1.300 |

| Variable | N(Yes) | P(Y=1 \| Yes) | N(No) | P(Y=1 \| No) | Δ (pp) | RR |
|---|---:|---:|---:|---:|---:|---:|
| Gang_Affiliated | 869 | 0.322 | 7133 | 0.185 | 13.7 | 1.740 |

#### Top Unadjusted Risk Lifts (Binary Indicators)

| Variable | N(Yes) | P(Y=1\|Yes) | N(No) | P(Y=1\|No) | Δ (pp) | RR |
|---|---:|---:|---:|---:|---:|---:|
| Gang affiliated | 869 | 0.322 | 7133 | 0.185 | 13.7 | 1.740 |
| Any prior PP-violation-charge conviction episode | 2745 | 0.236 | 6653 | 0.172 | 6.5 | 1.377 |
| Prior parole revocations | 742 | 0.245 | 8656 | 0.186 | 5.9 | 1.320 |
| MH/SA condition | 5741 | 0.209 | 3657 | 0.161 | 4.8 | 1.300 |
| Any prior domestic-violence-charge conviction episode | 626 | 0.240 | 8772 | 0.187 | 5.3 | 1.281 |
| Any prior gun-charge arrest episode | 2325 | 0.223 | 7073 | 0.180 | 4.3 | 1.241 |
| Any prior drug conviction episode | 2293 | 0.207 | 4833 | 0.167 | 4.0 | 1.240 |
| Any prior domestic-violence-charge arrest episode | 1338 | 0.226 | 8060 | 0.185 | 4.2 | 1.227 |
| Any prior gun-charge conviction episode | 1199 | 0.220 | 8199 | 0.186 | 3.4 | 1.182 |
| Missing: supervision level | 579 | 0.221 | 8819 | 0.189 | 3.3 | 1.172 |

#### Age at release

| Group | N | Rearrest rate |
|---|---:|---:|
| 48 or older | 1781 | 0.129 |
| 28-32 | 1676 | 0.217 |
| 33-37 | 1586 | 0.202 |
| 23-27 | 1566 | 0.231 |
| 38-42 | 1152 | 0.181 |
| 43-47 | 1089 | 0.152 |
| 18-22 | 548 | 0.255 |

#### Gang affiliated

| Group | N | Rearrest rate |
|---|---:|---:|
| No | 7133 | 0.185 |
| Unknown | 1396 | 0.136 |
| Yes | 869 | 0.322 |

#### MH/SA condition

| Group | N | Rearrest rate |
|---|---:|---:|
| Yes | 5741 | 0.209 |
| No | 3657 | 0.161 |

#### Supervision risk score (binned)

| Group | N | Rearrest rate |
|---|---:|---:|
| 4–6 | 3971 | 0.179 |
| 7–10 | 3408 | 0.235 |
| 1–3 | 1867 | 0.132 |
| Unknown | 152 | 0.217 |

#### Prior property-arrest episodes

| Group | N | Rearrest rate |
|---|---:|---:|
| 0 | 2945 | 0.142 |
| 1 | 1929 | 0.172 |
| 5 or more | 1674 | 0.247 |
| 2 | 1402 | 0.214 |
| 3 | 869 | 0.211 |
| 4 | 579 | 0.252 |

#### Prison years

| Group | N | Rearrest rate |
|---|---:|---:|
| 1-2 years | 2811 | 0.210 |
| Less than 1 year | 2554 | 0.212 |
| More than 3 years | 2427 | 0.148 |
| Greater than 2 to 3 years | 1606 | 0.188 |

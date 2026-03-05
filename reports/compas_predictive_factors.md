# COMPAS — Predictive Factors and Simple Associations

This report shows **which features the model uses** (SHAP) and **how rearrest rates vary** by key variables (unadjusted associations) for the COMPAS (ProPublica Broward County) dataset.

- Single horizon: two-year rearrest.
- SHAP shows association with predictions, not causality.
- Not directly comparable to NIJ (different jurisdiction, population, features, label).

## Top Predictive Factors (SHAP)

From `reports/plots/xgb_shap_compas.json` (XGBoost).

| Rank | Feature | Mean \|SHAP\| |
|---:|---|---:|
| 1 | Prior violent recidivism | 0.5964 |
| 2 | priors_count | 0.5420 |
| 3 | age | 0.3802 |
| 4 | decile_score | 0.2720 |
| 5 | c_charge_desc_arrest case no charge | 0.0820 |
| 6 | Sex: female | 0.0787 |
| 7 | v_decile_score | 0.0760 |
| 8 | c_charge_desc_Possession of Cocaine | 0.0731 |
| 9 | days_b_screening_arrest | 0.0707 |
| 10 | Charge degree: felony | 0.0588 |

## Observed Rearrest Rates by Key Variables (N=6,172, base rate=0.455)

These are raw subgroup rates (not adjusted for other variables).

### Top Unadjusted Risk Lifts (Binary Indicators)

| Variable | N(Yes) | P(Y=1\|Yes) | N(No) | P(Y=1\|No) | Δ (pp) | RR |
|---|---:|---:|---:|---:|---:|---:|
| Prior violent recidivism | 692 | 0.942 | 5480 | 0.394 | 54.9 | 2.394 |

### Race

| Group | N | Rearrest rate |
|---|---:|---:|
| African-American | 3175 | 0.523 |
| Caucasian | 2103 | 0.391 |
| Hispanic | 509 | 0.371 |
| Other | 343 | 0.362 |
| Asian | 31 | 0.258 |
| Native American | 11 | 0.455 |

### Sex

| Group | N | Rearrest rate |
|---|---:|---:|
| Male | 4997 | 0.479 |
| Female | 1175 | 0.351 |

### Age category

| Group | N | Rearrest rate |
|---|---:|---:|
| 25 - 45 | 3532 | 0.465 |
| Less than 25 | 1347 | 0.560 |
| Greater than 45 | 1293 | 0.320 |

### COMPAS decile score (binned)

| Group | N | Rearrest rate |
|---|---:|---:|
| 1–3 (low) | 2755 | 0.285 |
| 4–6 (medium) | 1777 | 0.495 |
| 7–10 (high) | 1640 | 0.698 |

### COMPAS violence decile score (binned)

| Group | N | Rearrest rate |
|---|---:|---:|
| 1–3 (low) | 3432 | 0.339 |
| 4–6 (medium) | 1780 | 0.543 |
| 7–10 (high) | 960 | 0.707 |

### Prior offense count (binned)

| Group | N | Rearrest rate |
|---|---:|---:|
| 0 | 2085 | 0.286 |
| 1–2 | 1810 | 0.413 |
| 3–5 | 1056 | 0.561 |
| 6–10 | 729 | 0.684 |
| 11+ | 492 | 0.758 |

### Charge degree

| Group | N | Rearrest rate |
|---|---:|---:|
| F | 3970 | 0.500 |
| M | 2202 | 0.375 |

### COMPAS risk level

| Group | N | Rearrest rate |
|---|---:|---:|
| Low | 3421 | 0.315 |
| Medium | 1607 | 0.551 |
| High | 1144 | 0.740 |

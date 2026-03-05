# Individual Prediction Analysis

Profiles of individuals with the **lowest model-predicted probability** (closest to 0) and individuals where the **model performs worst** (largest prediction errors).

All results are from the held-out test set (20% split, seed 42). Trait profiles show the top-5 most common values for each characteristic within the subset. These are **aggregate summaries** — no individual-level predictions are exported.

> **Terminology:**
> - *Lowest predicted*: individuals the model assigns the smallest rearrest/reincarceration probability.
> - *Worst false negatives*: individuals who **were** rearrested but had the **lowest** predicted probability (model missed them).
> - *Worst false positives*: individuals who were **not** rearrested but had the **highest** predicted probability (model was wrong about them).
> - *Highest individual error*: individuals with the largest (p − y)² regardless of direction.

---

## NIJ Georgia Parole

### Y1 (N_test=3606, base rate=0.298)

**Bottom 5% predicted (p ≤ 0.0812)** — N=181, pred mean=0.0635, pred median=0.0662, pred min=0.0256, pred max=0.0812, actual rearrest rate=0.055

#### Trait profile — lowest predicted

| Trait | Value | % of Group | N |
|---|---|---:|---:|
| Age_at_Release | 48 or older | 51.9% | 94 |
| Age_at_Release | 43-47 | 13.8% | 25 |
| Age_at_Release | 38-42 | 13.3% | 24 |
| Age_at_Release | 33-37 | 9.9% | 18 |
| Age_at_Release | 28-32 | 6.1% | 11 |
| Gender | M | 72.9% | 132 |
| Gender | F | 27.1% | 48 |
| Race | WHITE | 64.6% | 117 |
| Race | BLACK | 35.4% | 64 |
| Gang_Affiliated | No | 100.0% | 181 |
| Supervision_Risk_Score_First | 2 | 26.3% | 47 |
| Supervision_Risk_Score_First | 3 | 25.7% | 46 |
| Supervision_Risk_Score_First | 4 | 15.1% | 27 |
| Supervision_Risk_Score_First | 5 | 14.0% | 25 |
| Supervision_Risk_Score_First | 1 | 8.4% | 15 |
| Prison_Years | More than 3 years | 54.7% | 99 |
| Prison_Years | Greater than 2 to 3 years | 16.6% | 30 |
| Prison_Years | 1-2 years | 15.5% | 28 |
| Prison_Years | Less than 1 year | 13.3% | 24 |
| Prison_Offense | Violent/Non-Sex | 31.4% | 56 |
| Prison_Offense | Violent/Sex | 28.3% | 51 |
| Prison_Offense | Drug | 20.8% | 37 |
| Prison_Offense | Other | 10.1% | 18 |
| Prison_Offense | Property | 9.4% | 17 |
| Prior_Arrest_Episodes_Felony | 1 | 49.7% | 90 |
| Prior_Arrest_Episodes_Felony | 2 | 23.2% | 42 |
| Prior_Arrest_Episodes_Felony | 3 | 8.3% | 15 |
| Prior_Arrest_Episodes_Felony | 4 | 7.7% | 14 |
| Prior_Arrest_Episodes_Felony | 5 | 2.8% | 5 |
| Prior_Arrest_Episodes_Property | 0 | 66.9% | 120 |
| Prior_Arrest_Episodes_Property | 1 | 19.3% | 35 |
| Prior_Arrest_Episodes_Property | 2 | 11.0% | 20 |
| Prior_Arrest_Episodes_Property | 3 | 1.7% | 3 |
| Prior_Arrest_Episodes_Property | 4 | 0.6% | 1 |
| Prior_Arrest_Episodes_Violent | 0 | 55.2% | 100 |
| Prior_Arrest_Episodes_Violent | 1 | 29.3% | 52 |
| Prior_Arrest_Episodes_Violent | 2 | 8.3% | 15 |
| Prior_Arrest_Episodes_Violent | 3 or more | 7.2% | 13 |
| Prior_Arrest_Episodes_Drug | 0 | 68.5% | 124 |
| Prior_Arrest_Episodes_Drug | 1 | 17.1% | 31 |
| Prior_Arrest_Episodes_Drug | 3 | 6.6% | 12 |
| Prior_Arrest_Episodes_Drug | 2 | 6.1% | 11 |
| Prior_Arrest_Episodes_Drug | 5 or more | 1.1% | 2 |
| Condition_MH_SA | No | 63.0% | 113 |
| Condition_MH_SA | Yes | 37.0% | 67 |
| Condition_Cog_Ed | No | 74.6% | 135 |
| Condition_Cog_Ed | Yes | 25.4% | 46 |

**Bottom 1% predicted (p ≤ 0.0536)** — N=37, pred mean=0.0437, pred median=0.0451, pred min=0.0256, pred max=0.0535, actual rearrest rate=0.000

#### Trait profile — bottom 1% predicted

| Trait | Value | % of Group | N |
|---|---|---:|---:|
| Age_at_Release | 48 or older | 59.5% | 22 |
| Age_at_Release | 43-47 | 21.6% | 8 |
| Age_at_Release | 38-42 | 8.1% | 3 |
| Age_at_Release | 33-37 | 5.4% | 2 |
| Age_at_Release | 28-32 | 2.7% | 1 |
| Gender | M | 73.0% | 27 |
| Gender | F | 27.0% | 10 |
| Race | WHITE | 81.1% | 30 |
| Race | BLACK | 18.9% | 7 |
| Gang_Affiliated | No | 100.0% | 37 |
| Supervision_Risk_Score_First | 3 | 32.4% | 12 |
| Supervision_Risk_Score_First | 2 | 27.0% | 10 |
| Supervision_Risk_Score_First | 5 | 21.6% | 8 |
| Supervision_Risk_Score_First | 1 | 8.1% | 3 |
| Supervision_Risk_Score_First | 4 | 8.1% | 3 |
| Prison_Years | More than 3 years | 67.6% | 25 |
| Prison_Years | Greater than 2 to 3 years | 16.2% | 6 |
| Prison_Years | Less than 1 year | 8.1% | 3 |
| Prison_Years | 1-2 years | 8.1% | 3 |
| Prison_Offense | Violent/Sex | 66.7% | 24 |
| Prison_Offense | Violent/Non-Sex | 19.4% | 7 |
| Prison_Offense | Drug | 11.1% | 4 |
| Prison_Offense | Property | 2.8% | 1 |
| Prior_Arrest_Episodes_Felony | 1 | 59.5% | 22 |
| Prior_Arrest_Episodes_Felony | 2 | 29.7% | 11 |
| Prior_Arrest_Episodes_Felony | 9 | 2.7% | 1 |
| Prior_Arrest_Episodes_Felony | 7 | 2.7% | 1 |
| Prior_Arrest_Episodes_Felony | 4 | 2.7% | 1 |
| Prior_Arrest_Episodes_Property | 0 | 78.4% | 29 |
| Prior_Arrest_Episodes_Property | 2 | 10.8% | 4 |
| Prior_Arrest_Episodes_Property | 1 | 10.8% | 4 |
| Prior_Arrest_Episodes_Violent | 0 | 51.4% | 19 |
| Prior_Arrest_Episodes_Violent | 1 | 35.1% | 13 |
| Prior_Arrest_Episodes_Violent | 2 | 8.1% | 3 |
| Prior_Arrest_Episodes_Violent | 3 or more | 5.4% | 2 |
| Prior_Arrest_Episodes_Drug | 0 | 83.8% | 31 |
| Prior_Arrest_Episodes_Drug | 1 | 10.8% | 4 |
| Prior_Arrest_Episodes_Drug | 3 | 2.7% | 1 |
| Prior_Arrest_Episodes_Drug | 2 | 2.7% | 1 |
| Condition_MH_SA | No | 75.7% | 28 |
| Condition_MH_SA | Yes | 24.3% | 9 |
| Condition_Cog_Ed | No | 86.5% | 32 |
| Condition_Cog_Ed | Yes | 13.5% | 5 |

**Worst false negatives (top 5% of positives by lowest prediction)** — N=53, pred mean=0.1058, pred median=0.1051, pred min=0.0562, pred max=0.1426, actual rearrest rate=1.000

#### Trait profile — worst false negatives

| Trait | Value | % of Group | N |
|---|---|---:|---:|
| Age_at_Release | 48 or older | 30.2% | 16 |
| Age_at_Release | 28-32 | 20.8% | 11 |
| Age_at_Release | 33-37 | 18.9% | 10 |
| Age_at_Release | 43-47 | 11.3% | 6 |
| Age_at_Release | 18-22 | 9.4% | 5 |
| Gender | M | 84.9% | 45 |
| Gender | F | 15.1% | 8 |
| Race | BLACK | 60.4% | 32 |
| Race | WHITE | 39.6% | 21 |
| Gang_Affiliated | No | 100.0% | 53 |
| Supervision_Risk_Score_First | 5 | 23.1% | 12 |
| Supervision_Risk_Score_First | 4 | 21.2% | 11 |
| Supervision_Risk_Score_First | 2 | 15.4% | 8 |
| Supervision_Risk_Score_First | 6 | 11.5% | 6 |
| Supervision_Risk_Score_First | 3 | 11.5% | 6 |
| Prison_Years | More than 3 years | 34.0% | 18 |
| Prison_Years | Less than 1 year | 26.4% | 14 |
| Prison_Years | 1-2 years | 26.4% | 14 |
| Prison_Years | Greater than 2 to 3 years | 13.2% | 7 |
| Prison_Offense | Violent/Non-Sex | 35.0% | 18 |
| Prison_Offense | Drug | 30.0% | 15 |
| Prison_Offense | Other | 12.5% | 6 |
| Prison_Offense | Property | 12.5% | 6 |
| Prison_Offense | Violent/Sex | 10.0% | 5 |
| Prior_Arrest_Episodes_Felony | 1 | 34.0% | 18 |
| Prior_Arrest_Episodes_Felony | 2 | 20.8% | 11 |
| Prior_Arrest_Episodes_Felony | 3 | 9.4% | 5 |
| Prior_Arrest_Episodes_Felony | 4 | 9.4% | 5 |
| Prior_Arrest_Episodes_Felony | 0 | 9.4% | 5 |
| Prior_Arrest_Episodes_Property | 0 | 62.3% | 33 |
| Prior_Arrest_Episodes_Property | 1 | 26.4% | 14 |
| Prior_Arrest_Episodes_Property | 2 | 9.4% | 5 |
| Prior_Arrest_Episodes_Property | 5 or more | 1.9% | 1 |
| Prior_Arrest_Episodes_Violent | 0 | 54.7% | 29 |
| Prior_Arrest_Episodes_Violent | 1 | 26.4% | 14 |
| Prior_Arrest_Episodes_Violent | 2 | 11.3% | 6 |
| Prior_Arrest_Episodes_Violent | 3 or more | 7.5% | 4 |
| Prior_Arrest_Episodes_Drug | 0 | 50.9% | 27 |
| Prior_Arrest_Episodes_Drug | 1 | 28.3% | 15 |
| Prior_Arrest_Episodes_Drug | 2 | 7.5% | 4 |
| Prior_Arrest_Episodes_Drug | 5 or more | 7.5% | 4 |
| Prior_Arrest_Episodes_Drug | 3 | 5.7% | 3 |
| Condition_MH_SA | No | 62.3% | 33 |
| Condition_MH_SA | Yes | 37.7% | 20 |
| Condition_Cog_Ed | No | 71.7% | 38 |
| Condition_Cog_Ed | Yes | 28.3% | 15 |

**Worst false positives (top 5% of negatives by highest prediction)** — N=126, pred mean=0.5694, pred median=0.5615, pred min=0.5070, pred max=0.7105, actual rearrest rate=0.000

#### Trait profile — worst false positives

| Trait | Value | % of Group | N |
|---|---|---:|---:|
| Age_at_Release | 23-27 | 35.7% | 45 |
| Age_at_Release | 28-32 | 23.8% | 30 |
| Age_at_Release | 18-22 | 23.8% | 30 |
| Age_at_Release | 38-42 | 7.1% | 9 |
| Age_at_Release | 43-47 | 4.8% | 6 |
| Gender | M | 100.0% | 126 |
| Race | BLACK | 57.1% | 72 |
| Race | WHITE | 42.9% | 54 |
| Gang_Affiliated | Yes | 71.4% | 90 |
| Gang_Affiliated | No | 28.6% | 36 |
| Supervision_Risk_Score_First | 10 | 25.6% | 32 |
| Supervision_Risk_Score_First | 8 | 20.0% | 25 |
| Supervision_Risk_Score_First | 7 | 16.0% | 20 |
| Supervision_Risk_Score_First | 9 | 13.6% | 17 |
| Supervision_Risk_Score_First | 6 | 12.8% | 16 |
| Prison_Years | Less than 1 year | 45.2% | 57 |
| Prison_Years | 1-2 years | 32.5% | 41 |
| Prison_Years | Greater than 2 to 3 years | 15.1% | 19 |
| Prison_Years | More than 3 years | 7.1% | 9 |
| Prison_Offense | Property | 55.5% | 69 |
| Prison_Offense | Other | 17.3% | 21 |
| Prison_Offense | Violent/Non-Sex | 16.4% | 20 |
| Prison_Offense | Drug | 10.9% | 13 |
| Prior_Arrest_Episodes_Felony | 10 or more | 32.5% | 41 |
| Prior_Arrest_Episodes_Felony | 5 | 13.5% | 17 |
| Prior_Arrest_Episodes_Felony | 8 | 12.7% | 16 |
| Prior_Arrest_Episodes_Felony | 4 | 10.3% | 13 |
| Prior_Arrest_Episodes_Felony | 6 | 7.1% | 9 |
| Prior_Arrest_Episodes_Property | 5 or more | 38.9% | 49 |
| Prior_Arrest_Episodes_Property | 1 | 16.7% | 21 |
| Prior_Arrest_Episodes_Property | 2 | 13.5% | 17 |
| Prior_Arrest_Episodes_Property | 4 | 12.7% | 16 |
| Prior_Arrest_Episodes_Property | 3 | 10.3% | 13 |
| Prior_Arrest_Episodes_Violent | 0 | 38.1% | 48 |
| Prior_Arrest_Episodes_Violent | 1 | 34.1% | 43 |
| Prior_Arrest_Episodes_Violent | 3 or more | 15.9% | 20 |
| Prior_Arrest_Episodes_Violent | 2 | 11.9% | 15 |
| Prior_Arrest_Episodes_Drug | 1 | 27.0% | 34 |
| Prior_Arrest_Episodes_Drug | 0 | 23.8% | 30 |
| Prior_Arrest_Episodes_Drug | 2 | 16.7% | 21 |
| Prior_Arrest_Episodes_Drug | 5 or more | 11.9% | 15 |
| Prior_Arrest_Episodes_Drug | 3 | 11.1% | 14 |
| Condition_MH_SA | Yes | 82.5% | 104 |
| Condition_MH_SA | No | 17.5% | 22 |
| Condition_Cog_Ed | No | 50.8% | 64 |
| Condition_Cog_Ed | Yes | 49.2% | 62 |

**Highest individual Brier error (top 5%)** — N=180, pred mean=0.1694, pred median=0.1808, pred min=0.0562, pred max=0.2362, actual rearrest rate=1.000

#### Trait profile — highest individual error

| Trait | Value | % of Group | N |
|---|---|---:|---:|
| Age_at_Release | 48 or older | 22.8% | 41 |
| Age_at_Release | 28-32 | 18.3% | 33 |
| Age_at_Release | 43-47 | 13.9% | 25 |
| Age_at_Release | 33-37 | 13.9% | 25 |
| Age_at_Release | 23-27 | 12.8% | 23 |
| Gender | M | 83.3% | 150 |
| Gender | F | 16.7% | 30 |
| Race | BLACK | 55.6% | 100 |
| Race | WHITE | 44.4% | 80 |
| Gang_Affiliated | No | 98.0% | 176 |
| Gang_Affiliated | Yes | 2.0% | 3 |
| Supervision_Risk_Score_First | 4 | 20.7% | 37 |
| Supervision_Risk_Score_First | 5 | 20.1% | 36 |
| Supervision_Risk_Score_First | 6 | 14.0% | 25 |
| Supervision_Risk_Score_First | 7 | 9.5% | 17 |
| Supervision_Risk_Score_First | 2 | 8.9% | 16 |
| Prison_Years | 1-2 years | 28.9% | 51 |
| Prison_Years | More than 3 years | 28.9% | 51 |
| Prison_Years | Less than 1 year | 25.6% | 46 |
| Prison_Years | Greater than 2 to 3 years | 16.7% | 30 |
| Prison_Offense | Drug | 30.4% | 54 |
| Prison_Offense | Violent/Non-Sex | 29.1% | 52 |
| Prison_Offense | Property | 20.9% | 37 |
| Prison_Offense | Other | 14.2% | 25 |
| Prison_Offense | Violent/Sex | 5.4% | 9 |
| Prior_Arrest_Episodes_Felony | 2 | 18.9% | 34 |
| Prior_Arrest_Episodes_Felony | 1 | 14.4% | 25 |
| Prior_Arrest_Episodes_Felony | 4 | 11.7% | 21 |
| Prior_Arrest_Episodes_Felony | 3 | 11.1% | 20 |
| Prior_Arrest_Episodes_Felony | 10 or more | 10.6% | 19 |
| Prior_Arrest_Episodes_Property | 0 | 40.0% | 72 |
| Prior_Arrest_Episodes_Property | 1 | 25.6% | 46 |
| Prior_Arrest_Episodes_Property | 2 | 15.6% | 28 |
| Prior_Arrest_Episodes_Property | 5 or more | 7.8% | 14 |
| Prior_Arrest_Episodes_Property | 3 | 5.6% | 10 |
| Prior_Arrest_Episodes_Violent | 0 | 46.1% | 83 |
| Prior_Arrest_Episodes_Violent | 1 | 28.3% | 51 |
| Prior_Arrest_Episodes_Violent | 3 or more | 14.4% | 25 |
| Prior_Arrest_Episodes_Violent | 2 | 11.1% | 20 |
| Prior_Arrest_Episodes_Drug | 0 | 42.2% | 76 |
| Prior_Arrest_Episodes_Drug | 1 | 22.2% | 40 |
| Prior_Arrest_Episodes_Drug | 2 | 11.7% | 21 |
| Prior_Arrest_Episodes_Drug | 3 | 8.9% | 16 |
| Prior_Arrest_Episodes_Drug | 5 or more | 8.3% | 15 |
| Condition_MH_SA | Yes | 53.3% | 96 |
| Condition_MH_SA | No | 46.7% | 84 |
| Condition_Cog_Ed | No | 62.2% | 112 |
| Condition_Cog_Ed | Yes | 37.8% | 68 |

### Y2 (N_test=2531, base rate=0.257)

**Bottom 5% predicted (p ≤ 0.0000)** — N=306, pred mean=0.0000, pred median=0.0000, pred min=0.0000, pred max=0.0000, actual rearrest rate=0.016

#### Trait profile — lowest predicted

| Trait | Value | % of Group | N |
|---|---|---:|---:|
| Age_at_Release | 48 or older | 34.6% | 106 |
| Age_at_Release | 43-47 | 15.7% | 48 |
| Age_at_Release | 33-37 | 15.0% | 46 |
| Age_at_Release | 38-42 | 13.7% | 42 |
| Age_at_Release | 28-32 | 12.4% | 38 |
| Gender | M | 82.7% | 252 |
| Gender | F | 17.3% | 53 |
| Race | BLACK | 52.9% | 162 |
| Race | WHITE | 47.1% | 144 |
| Gang_Affiliated | No | 96.4% | 295 |
| Gang_Affiliated | Yes | 3.6% | 10 |
| Supervision_Risk_Score_First | 4 | 19.7% | 60 |
| Supervision_Risk_Score_First | 3 | 17.1% | 52 |
| Supervision_Risk_Score_First | 5 | 16.8% | 51 |
| Supervision_Risk_Score_First | 7 | 10.5% | 32 |
| Supervision_Risk_Score_First | 6 | 9.9% | 30 |
| Prison_Years | More than 3 years | 45.1% | 138 |
| Prison_Years | Greater than 2 to 3 years | 23.5% | 72 |
| Prison_Years | 1-2 years | 22.9% | 70 |
| Prison_Years | Less than 1 year | 8.5% | 26 |
| Prison_Offense | Drug | 32.3% | 98 |
| Prison_Offense | Violent/Non-Sex | 26.6% | 81 |
| Prison_Offense | Property | 22.8% | 69 |
| Prison_Offense | Violent/Sex | 9.5% | 29 |
| Prison_Offense | Other | 8.7% | 26 |
| Prior_Arrest_Episodes_Felony | 1 | 22.2% | 68 |
| Prior_Arrest_Episodes_Felony | 2 | 16.7% | 51 |
| Prior_Arrest_Episodes_Felony | 3 | 12.1% | 37 |
| Prior_Arrest_Episodes_Felony | 10 or more | 10.8% | 33 |
| Prior_Arrest_Episodes_Felony | 4 | 9.8% | 30 |
| Prior_Arrest_Episodes_Property | 0 | 42.2% | 129 |
| Prior_Arrest_Episodes_Property | 1 | 24.2% | 74 |
| Prior_Arrest_Episodes_Property | 2 | 14.4% | 44 |
| Prior_Arrest_Episodes_Property | 5 or more | 7.8% | 24 |
| Prior_Arrest_Episodes_Property | 3 | 6.2% | 19 |
| Prior_Arrest_Episodes_Violent | 0 | 52.0% | 159 |
| Prior_Arrest_Episodes_Violent | 1 | 24.5% | 75 |
| Prior_Arrest_Episodes_Violent | 2 | 12.1% | 37 |
| Prior_Arrest_Episodes_Violent | 3 or more | 11.4% | 35 |
| Prior_Arrest_Episodes_Drug | 0 | 37.6% | 115 |
| Prior_Arrest_Episodes_Drug | 1 | 19.9% | 61 |
| Prior_Arrest_Episodes_Drug | 2 | 17.0% | 52 |
| Prior_Arrest_Episodes_Drug | 3 | 10.5% | 32 |
| Prior_Arrest_Episodes_Drug | 5 or more | 9.8% | 30 |
| Condition_MH_SA | Yes | 50.3% | 154 |
| Condition_MH_SA | No | 49.7% | 152 |
| Condition_Cog_Ed | No | 59.5% | 182 |
| Condition_Cog_Ed | Yes | 40.5% | 124 |

**Bottom 1% predicted (p ≤ 0.0000)** — N=306, pred mean=0.0000, pred median=0.0000, pred min=0.0000, pred max=0.0000, actual rearrest rate=0.016

#### Trait profile — bottom 1% predicted

| Trait | Value | % of Group | N |
|---|---|---:|---:|
| Age_at_Release | 48 or older | 34.6% | 106 |
| Age_at_Release | 43-47 | 15.7% | 48 |
| Age_at_Release | 33-37 | 15.0% | 46 |
| Age_at_Release | 38-42 | 13.7% | 42 |
| Age_at_Release | 28-32 | 12.4% | 38 |
| Gender | M | 82.7% | 252 |
| Gender | F | 17.3% | 53 |
| Race | BLACK | 52.9% | 162 |
| Race | WHITE | 47.1% | 144 |
| Gang_Affiliated | No | 96.4% | 295 |
| Gang_Affiliated | Yes | 3.6% | 10 |
| Supervision_Risk_Score_First | 4 | 19.7% | 60 |
| Supervision_Risk_Score_First | 3 | 17.1% | 52 |
| Supervision_Risk_Score_First | 5 | 16.8% | 51 |
| Supervision_Risk_Score_First | 7 | 10.5% | 32 |
| Supervision_Risk_Score_First | 6 | 9.9% | 30 |
| Prison_Years | More than 3 years | 45.1% | 138 |
| Prison_Years | Greater than 2 to 3 years | 23.5% | 72 |
| Prison_Years | 1-2 years | 22.9% | 70 |
| Prison_Years | Less than 1 year | 8.5% | 26 |
| Prison_Offense | Drug | 32.3% | 98 |
| Prison_Offense | Violent/Non-Sex | 26.6% | 81 |
| Prison_Offense | Property | 22.8% | 69 |
| Prison_Offense | Violent/Sex | 9.5% | 29 |
| Prison_Offense | Other | 8.7% | 26 |
| Prior_Arrest_Episodes_Felony | 1 | 22.2% | 68 |
| Prior_Arrest_Episodes_Felony | 2 | 16.7% | 51 |
| Prior_Arrest_Episodes_Felony | 3 | 12.1% | 37 |
| Prior_Arrest_Episodes_Felony | 10 or more | 10.8% | 33 |
| Prior_Arrest_Episodes_Felony | 4 | 9.8% | 30 |
| Prior_Arrest_Episodes_Property | 0 | 42.2% | 129 |
| Prior_Arrest_Episodes_Property | 1 | 24.2% | 74 |
| Prior_Arrest_Episodes_Property | 2 | 14.4% | 44 |
| Prior_Arrest_Episodes_Property | 5 or more | 7.8% | 24 |
| Prior_Arrest_Episodes_Property | 3 | 6.2% | 19 |
| Prior_Arrest_Episodes_Violent | 0 | 52.0% | 159 |
| Prior_Arrest_Episodes_Violent | 1 | 24.5% | 75 |
| Prior_Arrest_Episodes_Violent | 2 | 12.1% | 37 |
| Prior_Arrest_Episodes_Violent | 3 or more | 11.4% | 35 |
| Prior_Arrest_Episodes_Drug | 0 | 37.6% | 115 |
| Prior_Arrest_Episodes_Drug | 1 | 19.9% | 61 |
| Prior_Arrest_Episodes_Drug | 2 | 17.0% | 52 |
| Prior_Arrest_Episodes_Drug | 3 | 10.5% | 32 |
| Prior_Arrest_Episodes_Drug | 5 or more | 9.8% | 30 |
| Condition_MH_SA | Yes | 50.3% | 154 |
| Condition_MH_SA | No | 49.7% | 152 |
| Condition_Cog_Ed | No | 59.5% | 182 |
| Condition_Cog_Ed | Yes | 40.5% | 124 |

**Worst false negatives (top 5% of positives by lowest prediction)** — N=32, pred mean=0.0770, pred median=0.0961, pred min=0.0000, pred max=0.0961, actual rearrest rate=1.000

#### Trait profile — worst false negatives

| Trait | Value | % of Group | N |
|---|---|---:|---:|
| Age_at_Release | 48 or older | 34.4% | 11 |
| Age_at_Release | 28-32 | 15.6% | 5 |
| Age_at_Release | 23-27 | 15.6% | 5 |
| Age_at_Release | 43-47 | 12.5% | 4 |
| Age_at_Release | 33-37 | 12.5% | 4 |
| Gender | M | 93.8% | 30 |
| Gender | F | 6.2% | 2 |
| Race | BLACK | 62.5% | 20 |
| Race | WHITE | 37.5% | 12 |
| Gang_Affiliated | No | 96.7% | 30 |
| Gang_Affiliated | Yes | 3.3% | 1 |
| Supervision_Risk_Score_First | 3 | 23.3% | 7 |
| Supervision_Risk_Score_First | 5 | 23.3% | 7 |
| Supervision_Risk_Score_First | 6 | 16.7% | 5 |
| Supervision_Risk_Score_First | 7 | 6.7% | 2 |
| Supervision_Risk_Score_First | 9 | 6.7% | 2 |
| Prison_Years | More than 3 years | 40.6% | 13 |
| Prison_Years | 1-2 years | 25.0% | 8 |
| Prison_Years | Greater than 2 to 3 years | 18.8% | 6 |
| Prison_Years | Less than 1 year | 15.6% | 5 |
| Prison_Offense | Violent/Non-Sex | 39.3% | 12 |
| Prison_Offense | Violent/Sex | 25.0% | 8 |
| Prison_Offense | Drug | 14.3% | 4 |
| Prison_Offense | Property | 14.3% | 4 |
| Prison_Offense | Other | 7.1% | 2 |
| Prior_Arrest_Episodes_Felony | 1 | 18.8% | 6 |
| Prior_Arrest_Episodes_Felony | 2 | 18.8% | 6 |
| Prior_Arrest_Episodes_Felony | 3 | 15.6% | 5 |
| Prior_Arrest_Episodes_Felony | 8 | 12.5% | 4 |
| Prior_Arrest_Episodes_Felony | 6 | 9.4% | 3 |
| Prior_Arrest_Episodes_Property | 0 | 46.9% | 15 |
| Prior_Arrest_Episodes_Property | 1 | 21.9% | 7 |
| Prior_Arrest_Episodes_Property | 3 | 15.6% | 5 |
| Prior_Arrest_Episodes_Property | 2 | 6.2% | 2 |
| Prior_Arrest_Episodes_Property | 5 or more | 6.2% | 2 |
| Prior_Arrest_Episodes_Violent | 0 | 50.0% | 16 |
| Prior_Arrest_Episodes_Violent | 1 | 28.1% | 9 |
| Prior_Arrest_Episodes_Violent | 3 or more | 12.5% | 4 |
| Prior_Arrest_Episodes_Violent | 2 | 9.4% | 3 |
| Prior_Arrest_Episodes_Drug | 0 | 50.0% | 16 |
| Prior_Arrest_Episodes_Drug | 1 | 18.8% | 6 |
| Prior_Arrest_Episodes_Drug | 3 | 15.6% | 5 |
| Prior_Arrest_Episodes_Drug | 2 | 9.4% | 3 |
| Prior_Arrest_Episodes_Drug | 5 or more | 3.1% | 1 |
| Condition_MH_SA | No | 68.8% | 22 |
| Condition_MH_SA | Yes | 31.2% | 10 |
| Condition_Cog_Ed | No | 75.0% | 24 |
| Condition_Cog_Ed | Yes | 25.0% | 8 |

**Worst false positives (top 5% of negatives by highest prediction)** — N=94, pred mean=0.5434, pred median=0.5207, pred min=0.5207, pred max=0.7500, actual rearrest rate=0.000

#### Trait profile — worst false positives

| Trait | Value | % of Group | N |
|---|---|---:|---:|
| Age_at_Release | 23-27 | 37.2% | 35 |
| Age_at_Release | 18-22 | 17.0% | 16 |
| Age_at_Release | 33-37 | 13.8% | 13 |
| Age_at_Release | 28-32 | 9.6% | 9 |
| Age_at_Release | 38-42 | 9.6% | 9 |
| Gender | M | 93.6% | 88 |
| Gender | F | 6.4% | 5 |
| Race | BLACK | 56.4% | 53 |
| Race | WHITE | 43.6% | 41 |
| Gang_Affiliated | No | 61.4% | 57 |
| Gang_Affiliated | Yes | 38.6% | 36 |
| Supervision_Risk_Score_First | 6 | 21.7% | 20 |
| Supervision_Risk_Score_First | 7 | 19.6% | 18 |
| Supervision_Risk_Score_First | 9 | 16.3% | 15 |
| Supervision_Risk_Score_First | 8 | 16.3% | 15 |
| Supervision_Risk_Score_First | 10 | 10.9% | 10 |
| Prison_Years | Less than 1 year | 38.3% | 36 |
| Prison_Years | 1-2 years | 28.7% | 27 |
| Prison_Years | Greater than 2 to 3 years | 19.1% | 18 |
| Prison_Years | More than 3 years | 13.8% | 13 |
| Prison_Offense | Property | 48.2% | 45 |
| Prison_Offense | Drug | 21.7% | 20 |
| Prison_Offense | Violent/Non-Sex | 15.7% | 14 |
| Prison_Offense | Other | 13.3% | 12 |
| Prison_Offense | Violent/Sex | 1.2% | 1 |
| Prior_Arrest_Episodes_Felony | 10 or more | 35.1% | 33 |
| Prior_Arrest_Episodes_Felony | 5 | 14.9% | 14 |
| Prior_Arrest_Episodes_Felony | 2 | 8.5% | 8 |
| Prior_Arrest_Episodes_Felony | 3 | 8.5% | 8 |
| Prior_Arrest_Episodes_Felony | 6 | 7.4% | 7 |
| Prior_Arrest_Episodes_Property | 5 or more | 34.0% | 32 |
| Prior_Arrest_Episodes_Property | 1 | 18.1% | 17 |
| Prior_Arrest_Episodes_Property | 0 | 14.9% | 14 |
| Prior_Arrest_Episodes_Property | 2 | 13.8% | 13 |
| Prior_Arrest_Episodes_Property | 3 | 12.8% | 11 |
| Prior_Arrest_Episodes_Violent | 0 | 37.2% | 35 |
| Prior_Arrest_Episodes_Violent | 3 or more | 25.5% | 23 |
| Prior_Arrest_Episodes_Violent | 1 | 23.4% | 22 |
| Prior_Arrest_Episodes_Violent | 2 | 13.8% | 13 |
| Prior_Arrest_Episodes_Drug | 0 | 27.7% | 26 |
| Prior_Arrest_Episodes_Drug | 1 | 19.1% | 18 |
| Prior_Arrest_Episodes_Drug | 5 or more | 18.1% | 17 |
| Prior_Arrest_Episodes_Drug | 2 | 13.8% | 13 |
| Prior_Arrest_Episodes_Drug | 4 | 10.6% | 10 |
| Condition_MH_SA | Yes | 72.3% | 68 |
| Condition_MH_SA | No | 27.7% | 26 |
| Condition_Cog_Ed | No | 53.2% | 50 |
| Condition_Cog_Ed | Yes | 46.8% | 44 |

**Highest individual Brier error (top 5%)** — N=126, pred mean=0.1769, pred median=0.1532, pred min=0.0000, pred max=0.7500, actual rearrest rate=0.944

#### Trait profile — highest individual error

| Trait | Value | % of Group | N |
|---|---|---:|---:|
| Age_at_Release | 28-32 | 24.6% | 31 |
| Age_at_Release | 48 or older | 17.5% | 22 |
| Age_at_Release | 23-27 | 16.7% | 21 |
| Age_at_Release | 33-37 | 15.9% | 20 |
| Age_at_Release | 43-47 | 11.9% | 15 |
| Gender | M | 86.5% | 109 |
| Gender | F | 13.5% | 17 |
| Race | BLACK | 51.6% | 65 |
| Race | WHITE | 48.4% | 61 |
| Gang_Affiliated | No | 92.7% | 116 |
| Gang_Affiliated | Yes | 7.3% | 9 |
| Supervision_Risk_Score_First | 5 | 21.0% | 26 |
| Supervision_Risk_Score_First | 4 | 15.1% | 19 |
| Supervision_Risk_Score_First | 6 | 14.3% | 18 |
| Supervision_Risk_Score_First | 3 | 12.6% | 15 |
| Supervision_Risk_Score_First | 7 | 11.8% | 14 |
| Prison_Years | More than 3 years | 27.8% | 35 |
| Prison_Years | Less than 1 year | 27.8% | 35 |
| Prison_Years | 1-2 years | 24.6% | 31 |
| Prison_Years | Greater than 2 to 3 years | 19.8% | 25 |
| Prison_Offense | Violent/Non-Sex | 36.0% | 45 |
| Prison_Offense | Property | 25.4% | 32 |
| Prison_Offense | Drug | 17.5% | 22 |
| Prison_Offense | Other | 11.4% | 14 |
| Prison_Offense | Violent/Sex | 9.6% | 12 |
| Prior_Arrest_Episodes_Felony | 2 | 17.5% | 22 |
| Prior_Arrest_Episodes_Felony | 1 | 15.1% | 19 |
| Prior_Arrest_Episodes_Felony | 4 | 12.7% | 16 |
| Prior_Arrest_Episodes_Felony | 3 | 11.9% | 15 |
| Prior_Arrest_Episodes_Felony | 10 or more | 10.3% | 13 |
| Prior_Arrest_Episodes_Property | 0 | 34.9% | 44 |
| Prior_Arrest_Episodes_Property | 1 | 25.4% | 32 |
| Prior_Arrest_Episodes_Property | 2 | 11.9% | 15 |
| Prior_Arrest_Episodes_Property | 3 | 11.9% | 15 |
| Prior_Arrest_Episodes_Property | 5 or more | 11.1% | 14 |
| Prior_Arrest_Episodes_Violent | 0 | 50.0% | 63 |
| Prior_Arrest_Episodes_Violent | 1 | 28.6% | 36 |
| Prior_Arrest_Episodes_Violent | 3 or more | 11.9% | 15 |
| Prior_Arrest_Episodes_Violent | 2 | 9.5% | 12 |
| Prior_Arrest_Episodes_Drug | 0 | 50.8% | 64 |
| Prior_Arrest_Episodes_Drug | 1 | 15.9% | 20 |
| Prior_Arrest_Episodes_Drug | 2 | 12.7% | 16 |
| Prior_Arrest_Episodes_Drug | 3 | 7.9% | 10 |
| Prior_Arrest_Episodes_Drug | 5 or more | 7.9% | 10 |
| Condition_MH_SA | Yes | 59.5% | 75 |
| Condition_MH_SA | No | 40.5% | 51 |
| Condition_Cog_Ed | No | 63.5% | 80 |
| Condition_Cog_Ed | Yes | 36.5% | 46 |

### Y3 (N_test=1880, base rate=0.190)

**Bottom 5% predicted (p ≤ 0.0000)** — N=222, pred mean=0.0000, pred median=0.0000, pred min=0.0000, pred max=0.0000, actual rearrest rate=0.018

#### Trait profile — lowest predicted

| Trait | Value | % of Group | N |
|---|---|---:|---:|
| Age_at_Release | 48 or older | 32.0% | 71 |
| Age_at_Release | 33-37 | 20.7% | 46 |
| Age_at_Release | 38-42 | 14.9% | 33 |
| Age_at_Release | 43-47 | 14.0% | 30 |
| Age_at_Release | 28-32 | 10.8% | 24 |
| Gender | M | 80.6% | 179 |
| Gender | F | 19.4% | 43 |
| Race | BLACK | 50.9% | 113 |
| Race | WHITE | 49.1% | 109 |
| Gang_Affiliated | No | 96.1% | 213 |
| Gang_Affiliated | Yes | 3.9% | 8 |
| Supervision_Risk_Score_First | 5 | 19.4% | 43 |
| Supervision_Risk_Score_First | 4 | 16.7% | 37 |
| Supervision_Risk_Score_First | 3 | 15.8% | 35 |
| Supervision_Risk_Score_First | 6 | 12.6% | 27 |
| Supervision_Risk_Score_First | 2 | 10.4% | 23 |
| Prison_Years | More than 3 years | 60.8% | 135 |
| Prison_Years | Greater than 2 to 3 years | 16.2% | 36 |
| Prison_Years | 1-2 years | 11.7% | 26 |
| Prison_Years | Less than 1 year | 11.3% | 25 |
| Prison_Offense | Violent/Non-Sex | 36.4% | 80 |
| Prison_Offense | Drug | 22.7% | 50 |
| Prison_Offense | Property | 21.2% | 47 |
| Prison_Offense | Violent/Sex | 13.6% | 30 |
| Prison_Offense | Other | 6.1% | 13 |
| Prior_Arrest_Episodes_Felony | 1 | 29.7% | 66 |
| Prior_Arrest_Episodes_Felony | 2 | 24.8% | 55 |
| Prior_Arrest_Episodes_Felony | 3 | 13.5% | 30 |
| Prior_Arrest_Episodes_Felony | 4 | 8.6% | 19 |
| Prior_Arrest_Episodes_Felony | 5 | 5.0% | 11 |
| Prior_Arrest_Episodes_Property | 0 | 54.1% | 120 |
| Prior_Arrest_Episodes_Property | 1 | 26.6% | 59 |
| Prior_Arrest_Episodes_Property | 2 | 9.5% | 21 |
| Prior_Arrest_Episodes_Property | 5 or more | 4.1% | 9 |
| Prior_Arrest_Episodes_Property | 4 | 3.6% | 8 |
| Prior_Arrest_Episodes_Violent | 0 | 51.8% | 115 |
| Prior_Arrest_Episodes_Violent | 1 | 27.5% | 61 |
| Prior_Arrest_Episodes_Violent | 2 | 15.3% | 34 |
| Prior_Arrest_Episodes_Violent | 3 or more | 5.4% | 12 |
| Prior_Arrest_Episodes_Drug | 0 | 54.5% | 121 |
| Prior_Arrest_Episodes_Drug | 1 | 18.9% | 42 |
| Prior_Arrest_Episodes_Drug | 2 | 12.2% | 27 |
| Prior_Arrest_Episodes_Drug | 3 | 7.7% | 17 |
| Prior_Arrest_Episodes_Drug | 5 or more | 5.0% | 11 |
| Condition_MH_SA | No | 62.6% | 139 |
| Condition_MH_SA | Yes | 37.4% | 83 |
| Condition_Cog_Ed | No | 67.6% | 150 |
| Condition_Cog_Ed | Yes | 32.4% | 72 |

**Bottom 1% predicted (p ≤ 0.0000)** — N=222, pred mean=0.0000, pred median=0.0000, pred min=0.0000, pred max=0.0000, actual rearrest rate=0.018

#### Trait profile — bottom 1% predicted

| Trait | Value | % of Group | N |
|---|---|---:|---:|
| Age_at_Release | 48 or older | 32.0% | 71 |
| Age_at_Release | 33-37 | 20.7% | 46 |
| Age_at_Release | 38-42 | 14.9% | 33 |
| Age_at_Release | 43-47 | 14.0% | 30 |
| Age_at_Release | 28-32 | 10.8% | 24 |
| Gender | M | 80.6% | 179 |
| Gender | F | 19.4% | 43 |
| Race | BLACK | 50.9% | 113 |
| Race | WHITE | 49.1% | 109 |
| Gang_Affiliated | No | 96.1% | 213 |
| Gang_Affiliated | Yes | 3.9% | 8 |
| Supervision_Risk_Score_First | 5 | 19.4% | 43 |
| Supervision_Risk_Score_First | 4 | 16.7% | 37 |
| Supervision_Risk_Score_First | 3 | 15.8% | 35 |
| Supervision_Risk_Score_First | 6 | 12.6% | 27 |
| Supervision_Risk_Score_First | 2 | 10.4% | 23 |
| Prison_Years | More than 3 years | 60.8% | 135 |
| Prison_Years | Greater than 2 to 3 years | 16.2% | 36 |
| Prison_Years | 1-2 years | 11.7% | 26 |
| Prison_Years | Less than 1 year | 11.3% | 25 |
| Prison_Offense | Violent/Non-Sex | 36.4% | 80 |
| Prison_Offense | Drug | 22.7% | 50 |
| Prison_Offense | Property | 21.2% | 47 |
| Prison_Offense | Violent/Sex | 13.6% | 30 |
| Prison_Offense | Other | 6.1% | 13 |
| Prior_Arrest_Episodes_Felony | 1 | 29.7% | 66 |
| Prior_Arrest_Episodes_Felony | 2 | 24.8% | 55 |
| Prior_Arrest_Episodes_Felony | 3 | 13.5% | 30 |
| Prior_Arrest_Episodes_Felony | 4 | 8.6% | 19 |
| Prior_Arrest_Episodes_Felony | 5 | 5.0% | 11 |
| Prior_Arrest_Episodes_Property | 0 | 54.1% | 120 |
| Prior_Arrest_Episodes_Property | 1 | 26.6% | 59 |
| Prior_Arrest_Episodes_Property | 2 | 9.5% | 21 |
| Prior_Arrest_Episodes_Property | 5 or more | 4.1% | 9 |
| Prior_Arrest_Episodes_Property | 4 | 3.6% | 8 |
| Prior_Arrest_Episodes_Violent | 0 | 51.8% | 115 |
| Prior_Arrest_Episodes_Violent | 1 | 27.5% | 61 |
| Prior_Arrest_Episodes_Violent | 2 | 15.3% | 34 |
| Prior_Arrest_Episodes_Violent | 3 or more | 5.4% | 12 |
| Prior_Arrest_Episodes_Drug | 0 | 54.5% | 121 |
| Prior_Arrest_Episodes_Drug | 1 | 18.9% | 42 |
| Prior_Arrest_Episodes_Drug | 2 | 12.2% | 27 |
| Prior_Arrest_Episodes_Drug | 3 | 7.7% | 17 |
| Prior_Arrest_Episodes_Drug | 5 or more | 5.0% | 11 |
| Condition_MH_SA | No | 62.6% | 139 |
| Condition_MH_SA | Yes | 37.4% | 83 |
| Condition_Cog_Ed | No | 67.6% | 150 |
| Condition_Cog_Ed | Yes | 32.4% | 72 |

**Worst false negatives (top 5% of positives by lowest prediction)** — N=17, pred mean=0.0489, pred median=0.0629, pred min=0.0000, pred max=0.0753, actual rearrest rate=1.000

#### Trait profile — worst false negatives

| Trait | Value | % of Group | N |
|---|---|---:|---:|
| Age_at_Release | 48 or older | 41.2% | 7 |
| Age_at_Release | 28-32 | 17.6% | 3 |
| Age_at_Release | 33-37 | 17.6% | 3 |
| Age_at_Release | 43-47 | 17.6% | 3 |
| Age_at_Release | 23-27 | 5.9% | 1 |
| Gender | M | 82.4% | 14 |
| Gender | F | 17.6% | 3 |
| Race | WHITE | 52.9% | 9 |
| Race | BLACK | 47.1% | 8 |
| Gang_Affiliated | No | 100.0% | 17 |
| Supervision_Risk_Score_First | 5 | 29.4% | 5 |
| Supervision_Risk_Score_First | 3 | 17.6% | 3 |
| Supervision_Risk_Score_First | 2 | 17.6% | 3 |
| Supervision_Risk_Score_First | 1 | 11.8% | 2 |
| Supervision_Risk_Score_First | 6 | 11.8% | 2 |
| Prison_Years | More than 3 years | 47.1% | 8 |
| Prison_Years | Less than 1 year | 29.4% | 5 |
| Prison_Years | Greater than 2 to 3 years | 17.6% | 3 |
| Prison_Years | 1-2 years | 5.9% | 1 |
| Prison_Offense | Violent/Non-Sex | 38.5% | 6 |
| Prison_Offense | Drug | 38.5% | 6 |
| Prison_Offense | Violent/Sex | 15.4% | 2 |
| Prison_Offense | Property | 7.7% | 1 |
| Prior_Arrest_Episodes_Felony | 1 | 29.4% | 5 |
| Prior_Arrest_Episodes_Felony | 4 | 17.6% | 3 |
| Prior_Arrest_Episodes_Felony | 2 | 17.6% | 3 |
| Prior_Arrest_Episodes_Felony | 0 | 11.8% | 2 |
| Prior_Arrest_Episodes_Felony | 5 | 11.8% | 2 |
| Prior_Arrest_Episodes_Property | 0 | 58.8% | 10 |
| Prior_Arrest_Episodes_Property | 1 | 29.4% | 5 |
| Prior_Arrest_Episodes_Property | 3 | 5.9% | 1 |
| Prior_Arrest_Episodes_Property | 2 | 5.9% | 1 |
| Prior_Arrest_Episodes_Violent | 0 | 52.9% | 9 |
| Prior_Arrest_Episodes_Violent | 1 | 23.5% | 4 |
| Prior_Arrest_Episodes_Violent | 2 | 17.6% | 3 |
| Prior_Arrest_Episodes_Violent | 3 or more | 5.9% | 1 |
| Prior_Arrest_Episodes_Drug | 0 | 52.9% | 9 |
| Prior_Arrest_Episodes_Drug | 1 | 17.6% | 3 |
| Prior_Arrest_Episodes_Drug | 2 | 17.6% | 3 |
| Prior_Arrest_Episodes_Drug | 4 | 5.9% | 1 |
| Prior_Arrest_Episodes_Drug | 5 or more | 5.9% | 1 |
| Condition_MH_SA | No | 64.7% | 11 |
| Condition_MH_SA | Yes | 35.3% | 6 |
| Condition_Cog_Ed | No | 76.5% | 13 |
| Condition_Cog_Ed | Yes | 23.5% | 4 |

**Worst false positives (top 5% of negatives by highest prediction)** — N=76, pred mean=0.3790, pred median=0.3714, pred min=0.3636, pred max=0.4615, actual rearrest rate=0.000

#### Trait profile — worst false positives

| Trait | Value | % of Group | N |
|---|---|---:|---:|
| Age_at_Release | 23-27 | 36.8% | 28 |
| Age_at_Release | 33-37 | 17.1% | 13 |
| Age_at_Release | 28-32 | 15.8% | 12 |
| Age_at_Release | 18-22 | 11.8% | 9 |
| Age_at_Release | 43-47 | 7.9% | 6 |
| Gender | M | 97.4% | 74 |
| Gender | F | 2.6% | 2 |
| Race | BLACK | 68.4% | 52 |
| Race | WHITE | 31.6% | 24 |
| Gang_Affiliated | Yes | 54.1% | 41 |
| Gang_Affiliated | No | 45.9% | 34 |
| Supervision_Risk_Score_First | 6 | 20.0% | 15 |
| Supervision_Risk_Score_First | 9 | 18.7% | 14 |
| Supervision_Risk_Score_First | 7 | 16.0% | 12 |
| Supervision_Risk_Score_First | 8 | 16.0% | 12 |
| Supervision_Risk_Score_First | 10 | 14.7% | 11 |
| Prison_Years | 1-2 years | 32.9% | 25 |
| Prison_Years | Less than 1 year | 25.0% | 19 |
| Prison_Years | Greater than 2 to 3 years | 25.0% | 19 |
| Prison_Years | More than 3 years | 17.1% | 13 |
| Prison_Offense | Property | 35.4% | 26 |
| Prison_Offense | Violent/Non-Sex | 33.8% | 25 |
| Prison_Offense | Drug | 20.0% | 15 |
| Prison_Offense | Other | 9.2% | 7 |
| Prison_Offense | Violent/Sex | 1.5% | 1 |
| Prior_Arrest_Episodes_Felony | 10 or more | 27.6% | 21 |
| Prior_Arrest_Episodes_Felony | 4 | 13.2% | 10 |
| Prior_Arrest_Episodes_Felony | 6 | 11.8% | 9 |
| Prior_Arrest_Episodes_Felony | 3 | 9.2% | 7 |
| Prior_Arrest_Episodes_Felony | 8 | 7.9% | 6 |
| Prior_Arrest_Episodes_Property | 2 | 26.3% | 20 |
| Prior_Arrest_Episodes_Property | 0 | 23.7% | 18 |
| Prior_Arrest_Episodes_Property | 5 or more | 19.7% | 15 |
| Prior_Arrest_Episodes_Property | 3 | 10.5% | 8 |
| Prior_Arrest_Episodes_Property | 4 | 10.5% | 8 |
| Prior_Arrest_Episodes_Violent | 0 | 36.8% | 28 |
| Prior_Arrest_Episodes_Violent | 1 | 26.3% | 20 |
| Prior_Arrest_Episodes_Violent | 2 | 25.0% | 19 |
| Prior_Arrest_Episodes_Violent | 3 or more | 11.8% | 9 |
| Prior_Arrest_Episodes_Drug | 0 | 30.3% | 23 |
| Prior_Arrest_Episodes_Drug | 1 | 22.4% | 17 |
| Prior_Arrest_Episodes_Drug | 3 | 22.4% | 17 |
| Prior_Arrest_Episodes_Drug | 2 | 11.8% | 9 |
| Prior_Arrest_Episodes_Drug | 5 or more | 9.2% | 7 |
| Condition_MH_SA | Yes | 73.7% | 56 |
| Condition_MH_SA | No | 26.3% | 20 |
| Condition_Cog_Ed | Yes | 57.9% | 44 |
| Condition_Cog_Ed | No | 42.1% | 32 |

**Highest individual Brier error (top 5%)** — N=94, pred mean=0.1272, pred median=0.1158, pred min=0.0000, pred max=0.2028, actual rearrest rate=1.000

#### Trait profile — highest individual error

| Trait | Value | % of Group | N |
|---|---|---:|---:|
| Age_at_Release | 48 or older | 25.5% | 23 |
| Age_at_Release | 33-37 | 19.1% | 18 |
| Age_at_Release | 28-32 | 13.8% | 13 |
| Age_at_Release | 38-42 | 12.8% | 11 |
| Age_at_Release | 23-27 | 11.7% | 11 |
| Gender | M | 87.2% | 82 |
| Gender | F | 12.8% | 11 |
| Race | BLACK | 55.3% | 52 |
| Race | WHITE | 44.7% | 42 |
| Gang_Affiliated | No | 97.6% | 91 |
| Gang_Affiliated | Yes | 2.4% | 2 |
| Supervision_Risk_Score_First | 6 | 20.4% | 19 |
| Supervision_Risk_Score_First | 5 | 19.4% | 18 |
| Supervision_Risk_Score_First | 4 | 15.1% | 14 |
| Supervision_Risk_Score_First | 3 | 14.0% | 13 |
| Supervision_Risk_Score_First | 2 | 11.8% | 11 |
| Prison_Years | Less than 1 year | 33.0% | 30 |
| Prison_Years | More than 3 years | 26.6% | 25 |
| Prison_Years | 1-2 years | 23.4% | 22 |
| Prison_Years | Greater than 2 to 3 years | 17.0% | 16 |
| Prison_Offense | Drug | 29.6% | 27 |
| Prison_Offense | Property | 29.6% | 27 |
| Prison_Offense | Violent/Non-Sex | 24.7% | 23 |
| Prison_Offense | Other | 13.6% | 12 |
| Prison_Offense | Violent/Sex | 2.5% | 2 |
| Prior_Arrest_Episodes_Felony | 10 or more | 18.1% | 17 |
| Prior_Arrest_Episodes_Felony | 1 | 14.9% | 14 |
| Prior_Arrest_Episodes_Felony | 2 | 13.8% | 13 |
| Prior_Arrest_Episodes_Felony | 3 | 13.8% | 13 |
| Prior_Arrest_Episodes_Felony | 4 | 10.6% | 10 |
| Prior_Arrest_Episodes_Property | 0 | 37.2% | 35 |
| Prior_Arrest_Episodes_Property | 1 | 24.5% | 23 |
| Prior_Arrest_Episodes_Property | 5 or more | 16.0% | 15 |
| Prior_Arrest_Episodes_Property | 2 | 11.7% | 11 |
| Prior_Arrest_Episodes_Property | 3 | 5.3% | 5 |
| Prior_Arrest_Episodes_Violent | 0 | 45.7% | 43 |
| Prior_Arrest_Episodes_Violent | 1 | 27.7% | 26 |
| Prior_Arrest_Episodes_Violent | 3 or more | 14.9% | 14 |
| Prior_Arrest_Episodes_Violent | 2 | 11.7% | 11 |
| Prior_Arrest_Episodes_Drug | 0 | 40.4% | 38 |
| Prior_Arrest_Episodes_Drug | 1 | 18.1% | 17 |
| Prior_Arrest_Episodes_Drug | 2 | 13.8% | 13 |
| Prior_Arrest_Episodes_Drug | 5 or more | 11.7% | 11 |
| Prior_Arrest_Episodes_Drug | 4 | 8.5% | 8 |
| Condition_MH_SA | Yes | 58.5% | 55 |
| Condition_MH_SA | No | 41.5% | 39 |
| Condition_Cog_Ed | No | 57.4% | 54 |
| Condition_Cog_Ed | Yes | 42.6% | 40 |


---

## COMPAS (Broward County, 2-year)

N_test=1235, base rate=0.455

**Bottom 5% predicted (p ≤ 0.1169)** — N=62, pred mean=0.1011, pred median=0.1035, pred min=0.0627, pred max=0.1168, actual rearrest rate=0.129

#### Trait profile — lowest predicted

| Trait | Value | % of Group | N |
|---|---|---:|---:|
| age_cat | Greater than 45 | 58.1% | 36 |
| age_cat | 25 - 45 | 41.9% | 26 |
| sex | Female | 54.8% | 34 |
| sex | Male | 45.2% | 28 |
| race | Caucasian | 54.8% | 34 |
| race | African-American | 17.7% | 11 |
| race | Other | 12.9% | 8 |
| race | Hispanic | 11.3% | 7 |
| race | Asian | 3.2% | 2 |
| c_charge_degree | M | 61.3% | 38 |
| c_charge_degree | F | 38.7% | 24 |
| priors_count | 0 | 95.2% | 59 |
| priors_count | 1 | 4.8% | 3 |
| score_text | Low | 100.0% | 62 |
| v_score_text | Low | 100.0% | 62 |

**Worst false negatives** — N=28, pred mean=0.1292, pred median=0.1249, pred min=0.0869, pred max=0.1669, actual rearrest rate=1.000

#### Trait profile — worst false negatives

| Trait | Value | % of Group | N |
|---|---|---:|---:|
| age_cat | 25 - 45 | 53.6% | 15 |
| age_cat | Greater than 45 | 46.4% | 13 |
| sex | Male | 67.9% | 19 |
| sex | Female | 32.1% | 9 |
| race | Caucasian | 57.1% | 16 |
| race | Hispanic | 28.6% | 8 |
| race | African-American | 14.3% | 4 |
| c_charge_degree | M | 57.1% | 16 |
| c_charge_degree | F | 42.9% | 12 |
| priors_count | 0 | 78.6% | 22 |
| priors_count | 1 | 17.9% | 5 |
| priors_count | 2 | 3.6% | 1 |
| score_text | Low | 100.0% | 28 |
| v_score_text | Low | 100.0% | 28 |

**Worst false positives** — N=33, pred mean=0.7805, pred median=0.7571, pred min=0.6591, pred max=0.9593, actual rearrest rate=0.000

#### Trait profile — worst false positives

| Trait | Value | % of Group | N |
|---|---|---:|---:|
| age_cat | 25 - 45 | 39.4% | 13 |
| age_cat | Less than 25 | 33.3% | 11 |
| age_cat | Greater than 45 | 27.3% | 9 |
| sex | Male | 93.9% | 31 |
| sex | Female | 6.1% | 2 |
| race | African-American | 78.8% | 26 |
| race | Caucasian | 18.2% | 6 |
| race | Hispanic | 3.0% | 1 |
| c_charge_degree | F | 84.8% | 28 |
| c_charge_degree | M | 15.2% | 5 |
| priors_count | 1 | 12.1% | 4 |
| priors_count | 11 | 12.1% | 4 |
| priors_count | 0 | 9.1% | 3 |
| priors_count | 10 | 9.1% | 3 |
| priors_count | 3 | 9.1% | 3 |
| score_text | High | 45.5% | 15 |
| score_text | Medium | 45.5% | 15 |
| score_text | Low | 9.1% | 3 |
| v_score_text | Medium | 36.4% | 12 |
| v_score_text | Low | 36.4% | 12 |
| v_score_text | High | 27.3% | 9 |

**Highest individual Brier error (top 5%)** — N=61, pred mean=0.3316, pred median=0.1786, pred min=0.0869, pred max=0.9593, actual rearrest rate=0.754

#### Trait profile — highest individual error

| Trait | Value | % of Group | N |
|---|---|---:|---:|
| age_cat | 25 - 45 | 52.5% | 32 |
| age_cat | Greater than 45 | 41.0% | 25 |
| age_cat | Less than 25 | 6.6% | 4 |
| sex | Male | 75.4% | 46 |
| sex | Female | 24.6% | 15 |
| race | Caucasian | 45.9% | 28 |
| race | African-American | 29.5% | 18 |
| race | Hispanic | 23.0% | 14 |
| race | Other | 1.6% | 1 |
| c_charge_degree | M | 52.5% | 32 |
| c_charge_degree | F | 47.5% | 29 |
| priors_count | 0 | 54.1% | 33 |
| priors_count | 1 | 21.3% | 13 |
| priors_count | 2 | 6.6% | 4 |
| priors_count | 10 | 4.9% | 3 |
| priors_count | 5 | 3.3% | 2 |
| score_text | Low | 78.7% | 48 |
| score_text | Medium | 11.5% | 7 |
| score_text | High | 9.8% | 6 |
| v_score_text | Low | 83.6% | 51 |
| v_score_text | Medium | 11.5% | 7 |
| v_score_text | High | 4.9% | 3 |


---

## NCRP (ICPSR 37973, reincarceration)

### Y1 (N_test=12104, base rate=0.299)

**Bottom 5% predicted (p ≤ 0.0588)** — N=715, pred mean=0.0306, pred median=0.0429, pred min=0.0000, pred max=0.0588, actual rearrest rate=0.025

#### Trait profile — lowest predicted

| Trait | Value | % of Group | N |
|---|---|---:|---:|
| sex | Male | 82.1% | 587 |
| sex | Female | 17.9% | 128 |
| race | White, non-Hispanic | 38.2% | 273 |
| race | Black, non-Hispanic | 24.6% | 176 |
| race | Hispanic, any race | 18.3% | 131 |
| race | Unknown | 16.9% | 121 |
| race | Other race(s), non-Hispanic | 2.0% | 14 |
| age_group | <25 | 100.0% | 715 |
| ADMTYPE | 1 | 84.8% | 606 |
| ADMTYPE | 2 | 12.9% | 91 |
| ADMTYPE | 9 | 1.3% | 9 |
| ADMTYPE | 3 | 1.1% | 8 |
| RELTYPE | 1 | 52.3% | 374 |
| RELTYPE | 2 | 42.9% | 307 |
| RELTYPE | 3 | 3.1% | 22 |
| RELTYPE | 9 | 1.7% | 12 |
| OFFGENERAL | 1 | 39.7% | 284 |
| OFFGENERAL | 3 | 24.2% | 173 |
| OFFGENERAL | 4 | 19.0% | 136 |
| OFFGENERAL | 2 | 15.9% | 114 |
| OFFGENERAL | 5 | 0.8% | 6 |
| state | 12 | 12.6% | 90 |
| state | 6 | 11.5% | 82 |
| state | 48 | 11.5% | 82 |
| state | 13 | 7.1% | 50 |
| state | 40 | 5.6% | 40 |

**Worst false negatives** — N=181, pred mean=0.1052, pred median=0.1224, pred min=0.0000, pred max=0.1299, actual rearrest rate=1.000

#### Trait profile — worst false negatives

| Trait | Value | % of Group | N |
|---|---|---:|---:|
| sex | Male | 84.5% | 153 |
| sex | Female | 15.5% | 28 |
| race | Black, non-Hispanic | 40.3% | 73 |
| race | White, non-Hispanic | 31.5% | 56 |
| race | Hispanic, any race | 17.1% | 31 |
| race | Unknown | 8.8% | 16 |
| race | Other race(s), non-Hispanic | 2.2% | 4 |
| age_group | <25 | 100.0% | 181 |
| ADMTYPE | 1 | 76.2% | 138 |
| ADMTYPE | 2 | 17.7% | 32 |
| ADMTYPE | 9 | 6.1% | 11 |
| RELTYPE | 2 | 61.9% | 112 |
| RELTYPE | 1 | 35.9% | 65 |
| RELTYPE | 9 | 2.2% | 4 |
| OFFGENERAL | 1 | 30.4% | 55 |
| OFFGENERAL | 3 | 26.5% | 48 |
| OFFGENERAL | 2 | 22.1% | 40 |
| OFFGENERAL | 4 | 20.4% | 37 |
| OFFGENERAL | 5 | 0.6% | 1 |
| state | 12 | 21.5% | 39 |
| state | 48 | 13.3% | 24 |
| state | 13 | 11.0% | 20 |
| state | 6 | 8.3% | 15 |
| state | 39 | 5.5% | 10 |

**Worst false positives** — N=424, pred mean=0.6965, pred median=0.6592, pred min=0.6349, pred max=0.8182, actual rearrest rate=0.000

#### Trait profile — worst false positives

| Trait | Value | % of Group | N |
|---|---|---:|---:|
| sex | Male | 95.3% | 404 |
| sex | Female | 4.7% | 20 |
| race | White, non-Hispanic | 36.8% | 156 |
| race | Black, non-Hispanic | 34.7% | 147 |
| race | Hispanic, any race | 22.6% | 96 |
| race | Unknown | 4.7% | 20 |
| race | Other race(s), non-Hispanic | 1.2% | 5 |
| age_group | <25 | 100.0% | 424 |
| ADMTYPE | 2 | 91.5% | 388 |
| ADMTYPE | 1 | 8.0% | 34 |
| ADMTYPE | 3 | 0.5% | 2 |
| RELTYPE | 1 | 90.3% | 383 |
| RELTYPE | 9 | 5.4% | 23 |
| RELTYPE | 3 | 4.2% | 18 |
| OFFGENERAL | 2 | 38.7% | 164 |
| OFFGENERAL | 3 | 29.5% | 125 |
| OFFGENERAL | 1 | 21.2% | 90 |
| OFFGENERAL | 4 | 9.2% | 39 |
| OFFGENERAL | 9 | 0.7% | 3 |
| state | 6 | 82.3% | 349 |
| state | 42 | 7.5% | 32 |
| state | 29 | 2.1% | 9 |
| state | 8 | 2.1% | 9 |
| state | 47 | 1.2% | 5 |

**Highest individual Brier error (top 5%)** — N=605, pred mean=0.1601, pred median=0.1429, pred min=0.0000, pred max=0.8182, actual rearrest rate=0.979

#### Trait profile — highest individual error

| Trait | Value | % of Group | N |
|---|---|---:|---:|
| sex | Male | 86.0% | 520 |
| sex | Female | 14.0% | 85 |
| race | Black, non-Hispanic | 40.7% | 246 |
| race | White, non-Hispanic | 33.7% | 203 |
| race | Hispanic, any race | 13.6% | 81 |
| race | Unknown | 10.1% | 61 |
| race | Other race(s), non-Hispanic | 2.0% | 12 |
| age_group | <25 | 100.0% | 605 |
| ADMTYPE | 1 | 78.8% | 476 |
| ADMTYPE | 2 | 17.0% | 103 |
| ADMTYPE | 9 | 3.3% | 20 |
| ADMTYPE | 3 | 0.8% | 5 |
| RELTYPE | 1 | 48.3% | 292 |
| RELTYPE | 2 | 47.9% | 290 |
| RELTYPE | 9 | 3.3% | 20 |
| RELTYPE | 3 | 0.5% | 3 |
| OFFGENERAL | 3 | 28.9% | 175 |
| OFFGENERAL | 1 | 28.1% | 170 |
| OFFGENERAL | 2 | 25.0% | 151 |
| OFFGENERAL | 4 | 16.9% | 101 |
| OFFGENERAL | 5 | 0.7% | 4 |
| state | 48 | 12.9% | 78 |
| state | 12 | 11.9% | 72 |
| state | 13 | 9.6% | 58 |
| state | 37 | 8.3% | 50 |
| state | 6 | 7.1% | 43 |


---

*Generated by `src/pipelines/run_individual_analysis.py`. All caveats about observational data, label definitions, and generalizability apply.*
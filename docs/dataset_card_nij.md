# Dataset Card — NIJ Georgia Parole (Recidivism Forecasting Challenge)

## Summary
This dataset is used to build and evaluate models that predict **rearrest outcomes** within **Year 1**, **Year 2**, and **Year 3** after release/supervision start for people in a Georgia parole cohort. It is used strictly for research/prototyping in this repo.

## Provenance
- Source: National Institute of Justice (NIJ) — *Recidivism Forecasting Challenge* (Georgia parole). Downloaded via scripts in `src/data/` into `data/raw/nij/`.
- License/terms: see NIJ challenge page and accompanying materials (record locally if/when provided).

## Unit of analysis
- One row per person (parolee) in the NIJ challenge data.
- Fields: ~53 variables (per NIJ materials), spanning demographics, criminal history, and supervision/release context.

## Labels (what is predicted)
- Targets (binary): `Recidivism_Arrest_Year1`, `Recidivism_Arrest_Year2`, `Recidivism_Arrest_Year3`.
- These are **rearrest outcomes** — observed events in the justice system.

## Official NIJ challenge scoring context (for citation)
- Challenge leaderboard scoring was run on NIJ-held-out test sets, not on this repo's internal splits.
- NIJ scoring used **average Brier by sex**: Brier was computed separately for male and female groups, then averaged.
- NIJ scoring also applied a **fairness penalty at threshold `t=0.5`** (thresholded classification behavior was part of judging criteria).
- NIJ references:
  - Challenge overview + judging criteria: https://nij.ojp.gov/funding/recidivism-forecasting-challenge
  - Official results (archive): https://nij.ojp.gov/funding/recidivism-forecasting-challenge-results
  - Results article: https://nij.ojp.gov/topics/articles/results-national-institute-justice-recidivism-forecasting-challenge

### Horizon conditioning (no leakage rule)
To avoid invalid “conditioning on the future”:
- Year 2 modeling/evaluation is performed **only** on the subset with `Year1 == 0`.
- Year 3 modeling/evaluation is performed **only** on the subset with `Year1 == 0` **and** `Year2 == 0`.

## Sensitive / protected attributes
The dataset includes attributes commonly treated as sensitive (exact field names depend on NIJ schema):
- Race / ethnicity
- Sex / gender
- Age (or age band)

Released NIJ challenge files in this repo use limited subgroup categories:
- `Race`: `BLACK`, `WHITE` (no additional race categories in the released training file)
- `Gender`: `F`, `M`

Policy in this repo:
- These fields may be **excluded from training features** for some model variants.
- They are **retained for evaluation** to compute subgroup metrics (calibration, error rates, etc.).

## Intended use in this repo
- Build reproducible baselines and calibrated probability outputs.
- Perform fairness and subgroup error auditing (reporting disparities, not optimizing them away).
- Produce model cards and dataset cards documenting limitations and risks.

## Out-of-scope / prohibited use
- Any real-world decisioning (sentencing, release, supervision intensity).
- Using these labels as if they measure anything beyond rearrest events.
- Any attempt to identify individuals (the data should be treated as sensitive even if de-identified).

## Preprocessing and feature policy (planned)
Preprocessing is implemented in `src/data/` and `src/features/` and should follow:
- Schema-driven typing and missing-value handling (`schemas/nij.yml`).
- Categorical handling (e.g., one-hot) and numeric normalization as needed.
- Two feature tracks:
  - **Static-at-release**: only information plausibly available at release/supervision start.
  - **Dynamic-supervision**: includes supervision-activity block released later in the NIJ challenge materials.

## Known risks and limitations
- **Measurement bias**: rearrest is influenced by policing/surveillance intensity and other structural factors.
- **Selection bias**: cohort reflects a specific jurisdiction/time period and may not generalize.
- **Label ambiguity**: rearrest ≠ reconviction ≠ actual offending.
- **Fairness harms**: models may amplify existing disparities; fairness metrics are necessary but not sufficient.

## Documentation to be filled after download
Current (local) record counts:
- Train: 18,028 rows (`data/raw/nij/nij-challenge2021_training_dataset.csv`)
- Test releases:
  - Test 1: 7,807 rows (`nij-challenge2021_test_dataset_1.csv`)
  - Test 2: 5,460 rows (`nij-challenge2021_test_dataset_2.csv`)
  - Test 3: 4,146 rows (`nij-challenge2021_test_dataset_3.csv`)

Processed horizon datasets (this repo’s build outputs):
- Static track: `data/processed/nij_static_y{1,2,3}.parquet`
  - Y1: 18,028 rows
  - Y2: 12,651 rows (conditioned on Y1==0)
  - Y3: 9,398 rows (conditioned on Y1==0 & Y2==0)
- Dynamic track: `data/processed/nij_dynamic_y{1,2,3}.parquet`
  - Y1: 18,028 rows (same as static by design)
  - Y2: 12,651 rows
  - Y3: 9,398 rows

Known sensitive attribute column names used for evaluation:
- `Race`, `Gender`, `Age_at_Release` (no blanks in the training CSV for these columns in the local copy)

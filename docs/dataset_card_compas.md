# Dataset Card — COMPAS (ProPublica Broward County)

## Summary
This dataset is used as a *benchmark fairness case study* (not a ground-truth standard) to run the same evaluation harness as the NIJ pipeline and to reproduce/compare common metrics and disparities discussed in prior public analyses.

## Provenance
- Source: ProPublica COMPAS analysis repository (Broward County, FL), downloaded into `data/raw/compas/`.
- Canonical files typically include `compas-scores-two-years.csv` and related artifacts.

## Unit of analysis
- One row per person scored/observed in the ProPublica analysis dataset (exact inclusion criteria defined by that repository’s filtering choices).

## Labels (what is predicted)
Common outcome used in ProPublica’s two-year analysis:
- Two-year recidivism indicator (rearrest outcome; exact column name depends on the CSV used).

As with NIJ, this is a **rearrest outcome** — an observed event in the justice system.
In this repo’s processed output (`data/processed/compas_2yr.parquet`), the modeling label is `y`, renamed from source field `two_year_recid`.

## Sensitive / protected attributes
Common sensitive fields in COMPAS analyses include:
- Race
- Sex
- Age

Policy in this repo:
- Keep sensitive attributes for **subgroup reporting**.
- Compare with/without sensitive features as explicitly documented in the model card for each run.

## Intended use in this repo
- Benchmark: run baselines and boosted models through the same metrics + fairness reporting as NIJ runs.
- Sanity-check evaluation tooling (subgroup metrics, calibration plots, threshold sweeps).
- Document dataset differences and why cross-dataset comparisons are limited.

## Out-of-scope / prohibited use
- Any operational decisioning.
- Any re-identification attempts.
- Any claim that COMPAS data represents ground truth rather than observed outcomes and historical processes.

## Known risks and limitations
- The dataset reflects **historical** criminal legal system practices and may encode systemic bias.
- Filtering choices in the public analysis pipeline can materially change metrics; replication must document these choices.
- Results from this dataset should not be generalized to other jurisdictions/populations.

## Documentation to be filled after download
Current (local) file + row counts:
- Raw file used: `data/raw/compas/compas-scores-two-years.csv` (7,214 rows)
- Filtered “two-year analysis sample” built by this repo: `data/processed/compas_2yr.parquet` (6,172 rows)

Label used in this repo:
- Binary label column: `y` (renamed from the source `two_year_recid`)
- Local base rate: ~0.455 (mean of `y` in `data/processed/compas_2yr.parquet`)

Sensitive attribute columns (for subgroup reporting):
- `race` categories in `data/processed/compas_2yr.parquet`: `African-American`, `Asian`, `Caucasian`, `Hispanic`, `Native American`, `Other`
- `sex` categories in `data/processed/compas_2yr.parquet`: `Female`, `Male`

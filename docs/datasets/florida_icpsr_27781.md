# Florida prison releases (ICPSR 27781) — integration notes

This repo can ingest a local ICPSR download into `data/processed/` and run the same style of **probability + calibration** evaluation used for NIJ/COMPAS.

## What this dataset is (high level)

An adult Florida Department of Corrections release cohort with a 3-year follow-up. Outcomes can be defined as **rearrest** or **reconviction** depending on which event date you choose (this repo defaults to rearrest when configured that way).

## Access and governance

- ICPSR 27781 is often **restricted access** in practice. If you only see “Documentation Only” in ICPSR’s download UI, you’ll need to apply for restricted access (IRB + data use agreement) before you can download the raw data.
- Do **not** commit any row-level files to GitHub.
- Place the extracted ICPSR package under:
  - `data/raw/florida_icpsr_27781/original/`
- Only export aggregate tables/plots via `public_export/` (the export script excludes `data/` by design).

## Ingest steps

1. Download the ICPSR 27781 package from ICPSR/NACJD and extract it locally.
2. Inspect columns:

   ```bash
   make PY=/path/to/your/python florida-27781-inspect
   ```

3. Edit the mapping config:
   - `configs/datasets/florida_icpsr_27781.yaml`
   - Fill in `columns.release_date`, `columns.event_date_rearrest` (or reconviction), and (if available) `columns.person_id`.

4. Process into standard parquets:

   ```bash
   make PY=/path/to/your/python florida-27781-process
   ```

Outputs:
- `data/processed/florida_icpsr_27781_release.parquet` (release cohort + derived `y1/y2/y3`)
- `data/processed/florida_icpsr_27781_y1.parquet`, `_y2.parquet`, `_y3.parquet` (modeling tables with target `y`)
- `data/processed/florida_icpsr_27781_feature_manifest.json`

## Horizon semantics

This repo uses **non-overlapping** yearly windows derived from the first event date:
- `y1`: event in (0, 365] days after release
- `y2`: event in (365, 730] days after release (evaluated only where `y1==0`)
- `y3`: event in (730, 1095] days after release (evaluated only where `y1==0 and y2==0`)

## Notes / pitfalls

- If the source data contains multiple releases per person, keep them, but treat splits as *grouped by person* when you add modeling pipelines (to avoid leakage across episodes).
- Avoid using any post-release summary variables (anything that encodes future arrests, follow-up time, “days to event”, etc.) as features.

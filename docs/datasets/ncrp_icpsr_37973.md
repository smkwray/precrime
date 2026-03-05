# NCRP selected variables (ICPSR 37973) — integration notes

NCRP is a strong public-use benchmark for **return-to-prison / reincarceration** outcomes, but it is not the same label family as rearrest.

## What this dataset is (high level)

NCRP includes prison admissions/releases and related offender/sentence attributes across many states/years (participation varies). In the public-use “selected variables” extract, the most natural outcome for this repo is **reincarceration** (return to prison), not rearrest.

## Access and governance

- Do **not** commit any row-level files to GitHub.
- Place extracted files under:
  - `data/raw/ncrp_icpsr_37973/original/`
- Only export aggregate outputs via `public_export/`.

## Ingest steps (release-cohort table)

1. Download the ICPSR zip (example filename: `ICPSR_37973-V2.zip`).
2. Extract it so the raw TSVs land at paths like:
   - `data/raw/ncrp_icpsr_37973/original/ICPSR_37973/DS0001/37973-0001-Data.tsv`

Example:

```bash
mkdir -p data/raw/ncrp_icpsr_37973/original
unzip /path/to/ICPSR_37973-V2.zip -d data/raw/ncrp_icpsr_37973/original
```

## Recommended approach in this repo (term-record-derived reincarceration)

In the ICPSR 37973 public-use extract, the most reliable way to derive a “return to prison” outcome is from **term records** (DS0001), which include a person identifier (`ABT_INMATE_ID`) and admission/release *years*.

This repo includes a pipeline that:
- deterministically samples a subset of IDs (to keep processing feasible on a laptop)
- computes the next term’s admission year after each term’s release year
- derives `y1/y2/y3` using **year granularity** (not exact dates)

Run:

```bash
make PY=/path/to/your/python ncrp-37973-terms-process
```

You can change sample size by adjusting the modulus:

```bash
make PY=/path/to/your/python ID_MOD=25 ID_REM=0 ncrp-37973-terms-process
```

Smaller `--id-mod` means a larger sample.

## Benchmark steps (baselines + optional XGBoost)

After processing, run:

```bash
make PY=/path/to/your/python ncrp-37973-terms-benchmark
```

This writes `reports/ncrp_37973_terms_benchmark.md` plus calibration plot specs under `reports/plots/`.

## Fairness / subgroup audit (optional)

To generate a subgroup audit report (race/sex/age) using tuned XGBoost with grouped-by-person splits:

```bash
make PY=/path/to/your/python ncrp-37973-terms-fairness
```

This writes `reports/ncrp_37973_fairness_report.md` plus plot specs under `reports/plots/`.

## Generic ingest (optional)

This repo’s generic ingest supports two paths:

### Path A: release-cohort table (simplest)

Provide a **release-cohort table** (one row per release episode) with:
- a release date (`columns.release_date`)
- a first-return/admission date (`columns.event_date_reincarceration`)

### Path B: separate release + admission tables (linkage)

If you have separate tables for releases and admissions, configure the `linkage.*` section in
`configs/datasets/ncrp_icpsr_37973.yaml` and set:
- `linkage.release_globs` / `linkage.admission_globs`
- `linkage.admission_date`

1. Inspect columns:

   ```bash
   python -m src.pipelines.run_release_ingest --dataset ncrp_icpsr_37973 --task inspect
   ```

2. Edit `configs/datasets/ncrp_icpsr_37973.yaml` and fill in required `columns.*`.
3. Process:

   ```bash
   python -m src.pipelines.run_release_ingest --dataset ncrp_icpsr_37973 --task process
   ```

Outputs mirror the Florida integration:
- `data/processed/ncrp_icpsr_37973_release.parquet`
- `data/processed/ncrp_icpsr_37973_y1.parquet`, `_y2.parquet`, `_y3.parquet`
- `data/processed/ncrp_icpsr_37973_feature_manifest.json`

## Interpretation notes

- Treat NCRP results as a **separate benchmark family**: “return-to-prison” is not directly comparable to “rearrest”.
- Participation and data quality vary by state/year; document any cohort selection you use.
- This repo maps `sex` and `race` codes into codebook labels for readability:
  - `sex`: `1=Male`, `2=Female`
  - `race`: `1=White, non-Hispanic`, `2=Black, non-Hispanic`, `3=Hispanic, any race`, `4=Other race(s), non-Hispanic`, `9=Missing`

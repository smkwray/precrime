# Publishing and Data-Governance Guidance

This document describes what can and cannot be published from this repository, and what disclaimers should accompany any public release.

## What Is Included in This Repository

The following materials are safe to share publicly:

- **Source code**: All files under `src/`, `tests/`, `scripts/`, `configs/`, `schemas/`.
- **Documentation**: All files under `docs/`, including dataset cards, this publishing guide, and the report.
- **Aggregate reports**: All Markdown files under `reports/` (leaderboards, fairness report, COMPAS benchmark, final summary). These contain only aggregate metrics, never row-level data.
- **Plot specifications**: All JSON files under `reports/plots/`. These are Vega-lite-compatible chart specs containing aggregate statistics (SHAP means, calibration curves, feature importances).
- **Static figures**: All PNG files under `docs/figures/`. These are rendered from aggregate plot specs and contain no row-level data.
- **Build and environment files**: `Makefile`, `requirements.txt`, `requirements-modeling.txt`, `README.md`, `LICENSE`.

## What Must NOT Be Committed

The following must never be added to this repository:

- **`data/raw/`**: Raw datasets (NIJ CSVs, COMPAS CSVs). These contain row-level individual records.
- **`data/processed/`**: Processed parquet files. These are derived from row-level data and contain individual records.
- **Any row-level exports**: CSVs, parquets, or other files that contain one-row-per-person data, regardless of where they are stored.
- **Internal infrastructure details**: Private SSH hostnames, IP addresses, or machine-specific paths should be replaced with placeholders before committing.

## Preflight Safety Checks

Before pushing any changes, run the automated preflight script:

```bash
make preflight
# or equivalently:
bash scripts/preflight_public_export.sh .
```

This verifies: no `data/` or `do/` directories, no `.csv`/`.parquet` files, no repo-local venvs, no private paths or IP addresses. The CI workflow also runs these checks on every push and pull request.

To customize the username check, set the `PREFLIGHT_USER_PATTERN` environment variable before running:

```bash
PREFLIGHT_USER_PATTERN="yourname@" make preflight
```

## CI

This repository includes a GitHub Actions workflow (`.github/workflows/ci.yml`) that runs unit tests and the preflight safety checks on every push and pull request.

## Suggested Disclaimer Text

The README and REPORT.md both contain a Limitations section. If citing the project elsewhere, include at minimum:

> Research prototype. Prediction targets are rearrest outcomes, not validated for operational use.

## Pre-Publication Checklist

Before making this repository public:

- [ ] **No row-level data**: Confirm `data/` is absent. Search for `.csv` and `.parquet` files.
- [ ] **No private hostnames/IPs**: Run `make preflight` — it checks for IP addresses and private paths.
- [ ] **Disclaimer present**: Confirm the README contains the "not validated for operational use" disclaimer.
- [ ] **Labels described correctly**: Labels should be called "rearrest outcomes" (NIJ, COMPAS) or "reincarceration outcomes" (NCRP) throughout.
- [ ] **Horizon leakage documented**: Verify that the README and report explicitly note that Year 2 and Year 3 tasks condition on prior non-recidivists.
- [ ] **Numbers traceable**: Spot-check that any numbers cited in the README or REPORT.md match the corresponding values in `reports/` files.
- [ ] **Dataset cards included**: Confirm `docs/dataset_card_nij.md`, `docs/dataset_card_compas.md`, and `docs/datasets/ncrp_icpsr_37973.md` are present and describe data provenance and limitations.
- [ ] **COMPAS and NCRP framed as benchmarks**: Confirm that COMPAS and NCRP results are described as benchmarks and not directly compared with NIJ results as if they measured the same thing.
- [ ] **Preflight passes**: Run `make preflight` and confirm it prints `[preflight] OK`.

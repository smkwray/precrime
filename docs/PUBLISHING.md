# Publishing and Data-Governance Guidance

This document describes what can and cannot be published from this repository, how to create a shareable export, and what disclaimers should accompany any public release.

## What Can Be Published

The following materials are safe to share publicly:

- **Source code**: All files under `src/`, `tests/`, `scripts/`, `configs/`, `schemas/`.
- **Documentation**: All files under `docs/`, including dataset cards, this publishing guide, and the report.
- **Aggregate reports**: All Markdown files under `reports/` (leaderboards, fairness report, COMPAS benchmark, final summary). These contain only aggregate metrics, never row-level data.
- **Plot specifications**: All JSON files under `reports/plots/`. These are Vega-lite-compatible chart specs containing aggregate statistics (SHAP means, calibration curves, feature importances).
- **Static figures**: All PNG files under `docs/figures/`. These are rendered from aggregate plot specs and contain no row-level data.
- **Build and environment files**: `Makefile`, `requirements.txt`, `requirements-modeling.txt`, `README.md`, `LICENSE`.

## What Should NOT Be Published

The following must be excluded from any public release:

- **`data/raw/`**: Raw datasets (NIJ CSVs, COMPAS CSVs). These contain row-level individual records.
- **`data/processed/`**: Processed parquet files. These are derived from row-level data and contain individual records.
- **Any row-level exports**: CSVs, parquets, or other files that contain one-row-per-person data, regardless of where they are stored.
- **Internal working notes**: The `do/` directory (or any similar internal logs/checklists) may contain machine-specific details and is not intended for public release. Prefer publishing `public_export/` instead.
- **Internal infrastructure details**: Private SSH hostnames, IP addresses, or machine-specific paths should be replaced with placeholders before publishing. The `remote_heavy_refresh.sh` script uses `REMOTE_HOST` as an environment variable — ensure this defaults to a placeholder (e.g., `your.host`) rather than a real address.

## How to Create a Shareable Package

Run the public export script from the repository root:

```bash
bash scripts/build_public_export.sh
```

This creates a `public_export/` directory containing code, docs, aggregate reports, and plot specs — but **no data**. The script explicitly excludes `data/raw/` and `data/processed/`.

### GitHub policy (important)

Only the contents of `public_export/` should ever be pushed to GitHub (whether the GitHub repo is private or public). Do **not** `git init` / push the parent working directory that contains `data/` and `do/`.

Recommended workflow:

```bash
# From the main repo root
bash scripts/build_public_export.sh
bash scripts/preflight_public_export.sh public_export

# Treat public_export/ as its own repo
cd public_export
git init
git add .
git commit -m "Initial public export"
git remote add origin git@github.com:YOUR_ORG_OR_USER/precrime-public-export.git
git push -u origin main
```

Review the export before sharing:

1. Verify `data/` is absent: `ls public_export/data/` should fail.
2. Grep for private hostnames/IPs: `grep -r "100\.\|192\.168\.\|10\.\|ssh " public_export/` and review any matches.
3. Grep for absolute personal paths: `grep -r "/Users/" public_export/` and ensure none remain in published files.

## CI checks (recommended)

If you host `public_export/` on GitHub, you can add a simple GitHub Actions workflow to enforce:
- unit tests (`make test`)
- export safety checks (`bash scripts/preflight_public_export.sh .`)

This repo already includes a workflow file under `.github/workflows/ci.yml` (copied into `public_export/` during export).

## Suggested Disclaimer Text

The README and REPORT.md both contain a Limitations section. If citing the project elsewhere, include at minimum:

> Research prototype. Prediction targets are rearrest outcomes, not validated for operational use.

## Pre-Publication Checklist

Before sharing the repository or the public export with anyone outside the project:

- [ ] **No row-level data**: Confirm `data/raw/` and `data/processed/` are excluded. Search for `.csv` and `.parquet` files in the export.
- [ ] **No private hostnames/IPs**: Grep for IP addresses and SSH connection strings. Replace any that remain with `REMOTE_HOST=your.host` or similar placeholders.
- [ ] **No absolute personal paths in docs**: Grep for `/Users/` or home-directory paths in all `.md` and `.sh` files.
- [ ] **Disclaimer present**: Confirm the README contains the "not validated for operational use" disclaimer.
- [ ] **Labels described correctly**: Labels should be called "rearrest outcomes" throughout.
- [ ] **Horizon leakage documented**: Verify that the README and report explicitly note that Year 2 and Year 3 tasks condition on prior non-recidivists.
- [ ] **Numbers traceable**: Spot-check that any numbers cited in the README or REPORT.md match the corresponding values in `reports/` files.
- [ ] **Dataset cards included**: Confirm `docs/dataset_card_nij.md` and `docs/dataset_card_compas.md` are present and describe data provenance, limitations, and prohibited uses.
- [ ] **COMPAS framed as benchmark**: Confirm that COMPAS results are described as a "benchmark case study" and not directly compared with NIJ results as if they measured the same thing.
- [ ] **No internal working notes**: Confirm `do/` and any other internal log/checklist directories are absent from the export. (The export script copies only specific directories, so `do/` should be excluded by default.)
- [ ] **Export script tested**: Run `bash scripts/build_public_export.sh` and verify the output directory passes all checks above.

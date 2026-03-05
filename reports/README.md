# Reports index

These files are the “full tables” companion to the README and docs. They contain **aggregate** metrics and plots (no row-level data).

If you read one report, start with: `final_summary.md`.

## Start here
- `final_summary.md` — high-level summary + key numbers + reproduction pointers

## Performance leaderboards and configs
- `baseline_leaderboard.md` — baseline models (base-rate, logistic, lasso-logistic)
- `xgb_leaderboard.md` — tuned XGBoost results
- `xgb_best_models.json` — best-per-task configs (machine-readable)
- `model_sweep.md` — model-family sweep under NIJ-style metrics
- `nij_scoring.md` — NIJ-style scoring terms (sex-disaggregated Brier + “FairAcc” terms)

## Fairness and policy views
- `fairness_report.md` — subgroup metrics + threshold sweeps + bootstrap intervals (where applicable)
- `operational_eval.md` — operational-style policies (top-k%, FPR-capped) with FP/FN by subgroup
- `policy_curves.md` — policy/threshold curves summaries

## Stability and benchmarks
- `stability_eval.md` — limited seed-sweep stability check
- `compas_benchmark.md` — COMPAS benchmark run (tooling comparison; not directly comparable to NIJ)
- `compas_fairness_report.md` — COMPAS subgroup diagnostics

## Plots
- `plots/*.json` — Vega-lite-compatible plot specs (machine-readable; rendered figures are under `docs/figures/`)

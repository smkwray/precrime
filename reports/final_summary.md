# Final Summary

Generated: 2026-03-01

## What this repo does
This project builds a **reproducible research pipeline** that predicts **rearrest outcomes** at multiple horizons on the NIJ Georgia parole data and produces:
- Calibrated probability outputs
- Subgroup performance and fairness diagnostics
- COMPAS (ProPublica) and NCRP (national corrections) benchmark runs using the same evaluation harness

## Core guardrails
- No horizon leakage: Year 2 and Year 3 tasks are conditioned on prior non-recidivists and feature families enforce static vs dynamic policies.
- Labels are rearrest outcomes.
- Default outputs are probabilities/percentiles, not categorical risk labels.

## Main artifacts
### NIJ (Georgia parole)
- Baselines leaderboard: `reports/baseline_leaderboard.md`
- XGBoost leaderboard + best configs: `reports/xgb_leaderboard.md`, `reports/xgb_best_models.json`
- Fairness audit (with-race vs without-race; subgroup gaps + CIs): `reports/fairness_report.md`
- Operational-style thresholds + FP/FN by race: `reports/operational_eval.md`
- Stability across seeds: `reports/stability_eval.md`

### COMPAS (ProPublica benchmark)
- Benchmark report: `reports/compas_benchmark.md`
- Fairness audit: `reports/compas_fairness_report.md`

### NCRP (ICPSR 37973, national corrections)
- Benchmark report: `reports/ncrp_37973_terms_benchmark.md`
- Fairness audit: `reports/ncrp_37973_fairness_report.md`

### Cross-dataset
- Lowest-rearrest trait profiles: `reports/lowest_rearrest_traits.md`
- Individual prediction analysis (lowest predicted, worst errors): `reports/individual_analysis.md`

## Publish-ready interpretation highlights

### What AUROC and Brier mean (plain English)
- `AUROC` measures ranking quality: does the model usually assign higher risk scores to people who are rearrested than to those who are not?
- `Brier` measures probability accuracy: are predicted probabilities numerically close to what actually happens?
- In this project, AUROC is generally around `0.70` on NIJ best models, while Brier improves over base-rate baselines (for example, STATIC Y1 best model `0.18727` vs base-rate around `0.209`), which indicates useful but limited predictive signal.

### Race FP/FN at top10% (STATIC Y1, with race)
From `reports/operational_eval.md` at `top10%` policy:
- BLACK (`N=2,045`): `FP=84`, `FN=509`, `FPR=0.06026`, `FNR=0.78187`, `PPV=0.62832`
- WHITE (`N=1,561`): `FP=68`, `FN=358`, `FPR=0.05986`, `FNR=0.84235`, `PPV=0.49630`

Threshold policy materially changes these outcomes: `t=0.5` flags fewer people and misses more positives; `top20%` flags more people, reducing misses but increasing false positives; `FPR<=0.06` explicitly constrains false positives at the cost of high false-negative rates.

### Stability across seeds (mean±std; seeds 42/43/44)
From `reports/stability_eval.md`:
- STATIC Y1: Brier `0.18764±0.00111`, AUROC `0.70193±0.00363`, race FPR gap@top10% `0.00715±0.00638`
- DYNAMIC Y2: Brier `0.17028±0.00270`, AUROC `0.71549±0.01278`, race FPR gap@top10% `0.01696±0.01752`
- DYNAMIC Y3: Brier `0.14280±0.00046`, AUROC `0.70392±0.00796`, race FPR gap@top10% `0.03198±0.01003`

Interpretation: aggregate performance is fairly stable in this 3-seed check; subgroup disparity estimates are less stable and should be treated as uncertainty-aware diagnostics, not fixed constants.

## How to reproduce
Use the shared venv (never create `.venv*` inside repo):
1. `python -m venv ~/venvs/precrime`
2. `source ~/venvs/precrime/bin/activate`
3. `pip install -r requirements.txt`
4. (Optional modeling deps) `pip install -r requirements-modeling.txt`
5. `python -m unittest -q tests/test_env_policy.py tests/test_metrics.py tests/test_leakage.py tests/test_compas.py`
6. NIJ:
   - `python -m src.pipelines.run_nij --task baselines`
   - `python -m src.pipelines.run_nij --task xgb --xgb-trials 16`
   - `PRECRIME_FAIRNESS_BOOTSTRAP=500 python -m src.pipelines.run_fairness`
7. COMPAS:
   - `python -m src.pipelines.run_compas --task all --xgb-trials 16`

### Convenience entrypoints
- Local rebuild via Make targets: `make test`, `make nij-xgb TRIALS=32`, `make compas TRIALS=32`, `make fairness BOOTSTRAP=2000 BOOTSTRAP_SUBGROUP=200`

## Notes / limitations (non-exhaustive)
- Rearrest is influenced by policing and supervision intensity; subgroup disparities can reflect system dynamics, not “risk.”
- NIJ and COMPAS are not directly comparable; treat COMPAS as a benchmark case study for evaluation tooling.

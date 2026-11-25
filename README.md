# precrime

**[Browse the interactive results site](https://smkwray.github.io/precrime/)**

`precrime` predicts rearrest at one-, two- and three-year horizons with calibrated probabilities and audits the models' errors by race, sex and age. The primary data are the National Institute of Justice (NIJ) Recidivism Forecasting Challenge Georgia parole cohort (18,028 people). ProPublica's COMPAS (Correctional Offender Management Profiling for Alternative Sanctions) data from Broward County, Florida, and the National Corrections Reporting Program (NCRP) extract from ICPSR (Inter-university Consortium for Political and Social Research) study 37973 serve as benchmarks.

On the NIJ cohort, a Platt-calibrated XGBoost model reaches a Year 1 Brier score of 0.188 against 0.209 for always predicting the base rate, and it beats lasso-logistic regression by only a small margin. Subgroup error-rate gaps persist whether or not race is a model input, and they change with the decision threshold. This is a research project, not validated for operational use in any justice-system decision. Row-level data are not included in the repository.

## Headline findings

| NIJ horizon | Features | Calibration | XGBoost Brier | AUROC | ECE | Lasso-logistic Brier | Base-rate Brier |
|---|---|---|---:|---:|---:|---:|---:|
| Year 1 | static | Platt | 0.18758 | 0.70164 | 0.00746 | 0.18796 | 0.20941 |
| Year 2 | dynamic | isotonic | 0.17244 | 0.70475 | 0.01715 | 0.17821 | 0.19104 |
| Year 3 | dynamic | isotonic | 0.14324 | 0.69594 | 0.01232 | 0.14790 | 0.15435 |

Lower Brier score and expected calibration error (ECE) are better; a higher area under the ROC curve (AUROC) is better. The logistic baselines use static features only.

- The gain from XGBoost is small. With the same static features it improves the Year 1 Brier score by less than 0.001 and does not beat lasso-logistic regression in Years 2 and 3; the larger Year 2 and Year 3 gains come with the dynamic supervision features.
- The strongest Year 1 predictors by mean absolute SHAP value are gang affiliation, having no prior violation-arrest episodes, age at release and the supervision risk score.
- At a 0.5 threshold with race as a feature, the Black/White false-positive-rate gap is 0.002 and the false-negative-rate gap is 0.063; the male/female gaps are 0.063 and 0.172. Removing race from training narrows some gaps slightly without closing them.
- The race false-positive-rate gap peaks at thresholds around 0.25–0.30, not at 0.5, so no single threshold summarizes fairness.
- On the benchmarks, XGBoost reaches Brier 0.173 and AUROC 0.808 for COMPAS two-year rearrest, and Brier 0.166 and AUROC 0.776 for NCRP Year 1 reincarceration. Neither is directly comparable to NIJ.

Full results are on the site, in [`docs/REPORT.md`](docs/REPORT.md) and in [`reports/`](reports/README.md).

## Method

Year 1 predicts rearrest within a year for the full cohort. Year 2 predicts rearrest in the second year only for people not rearrested in Year 1 (12,651), and Year 3 only for people not rearrested in either earlier year (9,398), so no model conditions on a future outcome. Each horizon has a static track (features known at release) and a dynamic track that adds supervision-activity features. The main model is Optuna-tuned XGBoost with raw, Platt or isotonic calibration, compared with logistic regressions, other tree ensembles and base-rate and demographics-only models. Brier score is the primary metric. The fairness audit reports calibration and error rates by subgroup, with bootstrap confidence intervals, with and without race as a feature. Splits use seed 42, and the best configurations are rerun on seeds 43 and 44. See [`docs/METHODS.md`](docs/METHODS.md) for details and [`docs/EXPLAINER.md`](docs/EXPLAINER.md) for a plain-language guide to the metrics.

## Data sources

| Dataset | Population | Outcome | How to get it |
|---|---|---|---|
| [NIJ Recidivism Forecasting Challenge](https://nij.ojp.gov/funding/recidivism-forecasting-challenge) | Georgia parolees, 18,028 people | Rearrest in Years 1–3 | `python src/data/download_nij.py` |
| [ProPublica COMPAS analysis](https://github.com/propublica/compas-analysis) | Broward County, Florida, 6,172 people in the two-year sample | Two-year rearrest | `python src/data/download_compas.py` |
| NCRP selected variables (ICPSR 37973) | Prison term records from participating states, sampled by person | Return to prison, by year | Download from ICPSR |

The download scripts write to `data/raw/`. For NCRP, extract the ICPSR package into `data/raw/ncrp_icpsr_37973/original/`. The dataset cards ([NIJ](docs/dataset_card_nij.md), [COMPAS](docs/dataset_card_compas.md), [NCRP](docs/datasets/ncrp_icpsr_37973.md)) describe each source, its labels and its prohibited uses.

## Reproduce

Requires Python 3.10 or later. Create the virtual environment outside the checkout: the test suite fails if a `.venv` directory exists inside the repository.

```bash
python3 -m venv ../precrime-venv
source ../precrime-venv/bin/activate
pip install -r requirements.txt -r requirements-modeling.txt
pip install -r requirements-viz.txt     # optional, for make figures

python src/data/download_nij.py
python src/data/download_compas.py

make test
make nij-baselines
make nij-xgb TRIALS=32
make compas TRIALS=32
make fairness BOOTSTRAP=2000 BOOTSTRAP_SUBGROUP=200
make figures
```

For NCRP, run `make ncrp-37973-terms-process` and then `make ncrp-37973-terms-benchmark`. The `Makefile` lists the other report targets (`operational`, `stability`, `model-sweep`, `nij-scoring`, `policy-curves`). To preview the results site, run `python3 -m http.server -d docs 8000` and open `http://localhost:8000`.

## Outputs and layout

```
reports/          Markdown reports, best-model JSON and plot specs (reports/plots/)
docs/             Results site, report, methods, explainer, dataset cards and figures
src/data/         Download scripts and loaders
src/pipelines/    Pipelines behind the make targets
configs/, schemas/  Dataset configuration and schemas
tests/            pytest suite
```

## Limitations

- Labels are rearrest outcomes (return to prison for NCRP), not a complete measure of behavior.
- Results come from one Georgia parole cohort; COMPAS and NCRP cover different populations and labels.
- Scores come from an internal split of the NIJ data, so they are not comparable to the official Challenge leaderboard, which was scored on a held-out test set.
- Female and some COMPAS race subgroups are small, so their metrics are unstable.
- All error-rate fairness metrics depend on the chosen threshold.

## License

MIT. See [`LICENSE`](LICENSE).

## Citation

```
@misc{precrime2026,
  title   = {precrime: Calibrated Multi-Horizon Rearrest Prediction},
  year    = {2026},
  url     = {https://github.com/smkwray/precrime}
}
```

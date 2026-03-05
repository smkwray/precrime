# Model Card Template (Precrime Research Pipeline)

## Model summary
- Model name:
- Model type (e.g., logistic regression, XGBoost):
- Prediction target (Y1 / Y2 / Y3; NIJ vs COMPAS):
- Output: calibrated probability + percentile rank (no categorical labels by default)

## Intended use
- Research/prototyping only; not for operational decisioning.
- Decision support is explicitly out of scope.

## Training data
- Dataset(s) used:
- Row counts:
- Feature track: NIJ static-at-release vs dynamic-supervision (if NIJ)
- Sensitive features in training: included / excluded (list)
- Train/validation methodology (CV, split strategy, seeds):

## Label definition
- Label column(s):
- Horizon definition:
- Conditioning rules (Y2 only among Y1==0; Y3 only among Y1==0 & Y2==0):

## Evaluation
### Core metrics (overall)
- Brier score (primary)
- AUROC, AUPRC, log loss
- Calibration error (ECE) + calibration curve

### Subgroup / fairness reporting
Report by race, sex/gender, and age group:
- Brier, AUROC, AUPRC
- Calibration curves by group
- Error-rate gaps across thresholds (FPR/FNR, equalized-odds gaps)
- Predictive parity proxies (as applicable)
- Threshold sweep plots + bootstrap CIs for key metrics

## Calibration
- Method(s) used: Platt / isotonic
- Calibrated on: held-out validation folds only (no leakage)
- Calibration diagnostics:

## Explainability (optional, for tabular boosted models)
- Global feature importance:
- SHAP summary (if computed):

## Ethical considerations and limitations
- Rearrest outcomes (often used as a proxy for recidivism in this context); measurement/selection bias risks.
- Known dataset limitations (jurisdiction/time specificity).
- Potential harms: disparity amplification, feedback loops, overconfidence in probabilities.

## Reproducibility
- Code version:
- Config file:
- Random seeds:
- Environment (Python version, dependencies):
- Command(s) run:

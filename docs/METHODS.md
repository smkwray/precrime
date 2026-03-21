# Methods

Every model, algorithm, library, and configuration used in the precrime pipeline.

---

## 1. Models

### 1.1 XGBoost (primary)

Gradient-boosted decision trees via the `xgboost` library (`XGBClassifier`).

**Default parameters** (before tuning):

| Parameter | Value |
|---|---|
| objective | `binary:logistic` |
| eval_metric | `logloss` |
| tree_method | `hist` |
| n_estimators | 500 |
| learning_rate | 0.05 |
| max_depth | 6 |
| min_child_weight | 1.0 |
| subsample | 0.9 |
| colsample_bytree | 0.9 |
| reg_alpha | 0.0 |
| reg_lambda | 1.0 |

Implementation: `src/models/xgb.py`.

### 1.2 Logistic Regression (custom gradient descent)

A from-scratch logistic regression trained with batch gradient descent and L2 regularization.

| Parameter | Value |
|---|---|
| learning_rate | 0.05 |
| n_iter | 400 |
| l2 | 1e-3 |

Weights initialized to zero. Sigmoid clipped to [-35, 35] for numerical stability. Output probabilities clipped to [1e-6, 1 - 1e-6].

Implementation: `src/models/logistic.py` (`LogisticRegressionGD`).

### 1.3 Lasso-Logistic Regression (L1, proximal gradient descent)

L1-regularized logistic regression using proximal gradient descent with a soft-thresholding step.

| Parameter | Value |
|---|---|
| learning_rate | 0.05 |
| n_iter | 500 |
| l1 | 5e-4 |

The L1 penalty is applied via soft-thresholding after each gradient step (`sign(w) * max(|w| - lr*l1, 0)`), encouraging sparsity.

Implementation: `src/models/lasso.py` (`LassoLogisticRegression`).

### 1.4 Model sweep comparisons

The model-sweep evaluation (`reports/model_sweep.md`) also includes scikit-learn models:

- **HistGradientBoostingClassifier** — histogram-based gradient boosting (scikit-learn)
- **RandomForestClassifier** — bagged ensemble of decision trees
- **ExtraTreesClassifier** — extremely randomized trees

These use default scikit-learn hyperparameters and are included for comparative analysis, not as primary models.

### 1.5 Floor baselines

**Base-rate model** — Predicts the training-set positive-class mean for every sample. Implementation: `src/models/baselines.py` (`BaseRateModel`).

**Demographic-naive model** — Predicts group-level base rates by demographic slices (Gender, Race, age_group). Falls back to the global rate for unseen group combinations. Implementation: `src/models/baselines.py` (`DemographicNaiveModel`).

---

## 2. Hyperparameter Tuning

XGBoost hyperparameters are tuned via **Optuna** using the Tree-structured Parzen Estimator (TPE) sampler.

| Setting | Value |
|---|---|
| Sampler | `TPESampler(seed=seed)` |
| Direction | minimize |
| Objective metric | Brier score (validation set) |
| Default trials | 32 (configurable via `TRIALS=`) |

**Tuned parameter ranges:**

| Parameter | Range | Scale |
|---|---|---|
| n_estimators | 200 – 900 | linear |
| learning_rate | 0.01 – 0.2 | log |
| max_depth | 3 – 10 | linear |
| min_child_weight | 0.5 – 8.0 | linear |
| subsample | 0.6 – 1.0 | linear |
| colsample_bytree | 0.6 – 1.0 | linear |
| reg_alpha | 1e-6 – 2.0 | log |
| reg_lambda | 1e-3 – 10.0 | log |

The best trial's parameters are used to retrain a final model on the full training set. Implementation: `src/models/xgb.py` (`tune_xgb`, `_suggest_params`).

---

## 3. Calibration

Three post-hoc calibration strategies are applied to XGBoost predictions. Calibration models are fit on a held-out calibration fold (not the training or test set).

### 3.1 Raw

Uncalibrated XGBoost output probabilities. Included as a comparison baseline.

### 3.2 Platt Scaling

A logistic regression fit on the log-odds of the raw probabilities: `sigmoid(a * logit(p) + b)`. Trained with gradient descent (lr=0.05, 300 iterations). Implementation: `src/models/calibration.py` (`PlattCalibrator`).

### 3.3 Isotonic Regression

Non-parametric monotonic calibration via scikit-learn's `IsotonicRegression` (y_min=0.0, y_max=1.0, out_of_bounds="clip"). Implementation: `src/models/calibration.py` (`IsotonicCalibrator`).

---

## 4. Feature Attribution

**SHAP** (SHapley Additive exPlanations) is used for feature importance analysis.

- Explainer: `shap.TreeExplainer` (exact, tree-path-based)
- Computation: Mean absolute SHAP values across samples
- Subsampling: Up to 2,000 samples (random, seed=42) for efficiency
- Top features reported: 30 per model/horizon
- XGBoost gain-based feature importance is also computed as a secondary measure

Implementation: `src/models/xgb.py` (`shap_summary_table`, `feature_importance_table`).

---

## 5. Evaluation Metrics

| Metric | Role | Definition |
|---|---|---|
| **Brier score** | Primary | Mean squared error of predicted probabilities: `(1/N) * sum((p_i - y_i)^2)` |
| **AUROC** | Secondary | Area under the ROC curve; measures ranking/discrimination |
| **AUPRC** | Secondary | Area under the precision-recall curve; more informative with class imbalance |
| **ECE** | Secondary | Expected calibration error across probability bins |
| **Log loss** | Secondary | Logarithmic scoring rule; penalizes confident wrong predictions |

Brier score is used as the optimization target (Optuna objective) and the primary comparison metric. See `docs/EXPLAINER.md` for plain-language definitions.

---

## 6. Libraries and Versions

All version constraints are from the project's requirements files.

**Core (`requirements.txt`):**

| Library | Version constraint |
|---|---|
| pandas | >= 2.2 |
| numpy | >= 1.26 |
| pyarrow | >= 15 |
| PyYAML | >= 6.0 |
| scikit-learn | >= 1.4 |

**Modeling (`requirements-modeling.txt`):**

| Library | Version constraint |
|---|---|
| xgboost | >= 2.0, < 3 |
| optuna | >= 3.6, < 4 |
| shap | >= 0.44, < 1 |

**Visualization (`requirements-viz.txt`):** matplotlib (for figure generation only).

Exact versions used for the results in this repo are recorded in `requirements-lock.txt`.
To reproduce, install from the lock file: `pip install -r requirements-lock.txt`.

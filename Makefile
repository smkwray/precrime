PY ?= python3
TRIALS ?= 16
BOOTSTRAP ?= 500
BOOTSTRAP_SUBGROUP ?= 200
ID_MOD ?= 200
ID_REM ?= 0
CHUNKSIZE ?= 300000

.PHONY: test nij-baselines nij-xgb nij-factors compas fairness compas-fairness all remote-refresh
.PHONY: operational stability
.PHONY: figures
.PHONY: nij-scoring model-sweep ensemble-eval policy-curves export preflight
.PHONY: model-sweep-tradeoff
.PHONY: florida-27781-inspect florida-27781-process ncrp-37973-inspect ncrp-37973-process
.PHONY: ncrp-37973-terms-process ncrp-37973-terms-benchmark
.PHONY: ncrp-37973-terms-fairness
.PHONY: individual-analysis

test:
		$(PY) -m unittest -q tests/test_env_policy.py tests/test_metrics.py tests/test_leakage.py tests/test_compas.py tests/test_nij_scoring.py tests/test_ncrp_37973_terms.py

nij-baselines:
	$(PY) -m src.pipelines.run_nij --task baselines

nij-xgb:
	$(PY) -m src.pipelines.run_nij --task xgb --xgb-trials $(TRIALS)

nij-factors:
	$(PY) -m src.pipelines.run_nij_factor_report

compas:
	$(PY) -m src.pipelines.run_compas --task all --xgb-trials $(TRIALS)

fairness:
	PRECRIME_FAIRNESS_BOOTSTRAP=$(BOOTSTRAP) PRECRIME_FAIRNESS_BOOTSTRAP_SUBGROUP=$(BOOTSTRAP_SUBGROUP) $(PY) -m src.pipelines.run_fairness

compas-fairness:
	PRECRIME_FAIRNESS_BOOTSTRAP=$(BOOTSTRAP) PRECRIME_FAIRNESS_BOOTSTRAP_SUBGROUP=$(BOOTSTRAP_SUBGROUP) $(PY) -m src.pipelines.run_compas_fairness --xgb-trials $(TRIALS)

operational:
	$(PY) -m src.pipelines.run_operational_eval

stability:
	$(PY) -m src.pipelines.run_stability_eval

figures:
	$(PY) scripts/render_figures.py

nij-scoring:
	$(PY) -m src.pipelines.run_nij_scoring

model-sweep:
	$(PY) -m src.pipelines.run_model_sweep

model-sweep-tradeoff:
	$(PY) -m src.pipelines.run_model_sweep_tradeoff_plot

ensemble-eval:
	$(PY) -m src.pipelines.run_ensemble_eval

policy-curves:
	$(PY) -m src.pipelines.run_policy_curves

export:
	bash scripts/build_public_export.sh

preflight:
	bash scripts/preflight_public_export.sh public_export

all: test nij-baselines nij-xgb compas fairness

florida-27781-inspect:
	$(PY) -m src.pipelines.run_release_ingest --dataset florida_icpsr_27781 --task inspect

florida-27781-process:
	$(PY) -m src.pipelines.run_release_ingest --dataset florida_icpsr_27781 --task process

ncrp-37973-inspect:
	$(PY) -m src.pipelines.run_release_ingest --dataset ncrp_icpsr_37973 --task inspect

ncrp-37973-process:
	$(PY) -m src.pipelines.run_release_ingest --dataset ncrp_icpsr_37973 --task process

ncrp-37973-terms-process:
	$(PY) -m src.pipelines.run_ncrp_37973_terms --id-mod $(ID_MOD) --id-rem $(ID_REM) --chunksize $(CHUNKSIZE)

ncrp-37973-terms-benchmark:
	$(PY) -m src.pipelines.run_ncrp_37973_benchmark --id-mod $(ID_MOD) --id-rem $(ID_REM) --xgb-trials $(TRIALS)

ncrp-37973-terms-fairness:
	$(PY) -m src.pipelines.run_ncrp_37973_fairness --id-mod $(ID_MOD) --id-rem $(ID_REM) --xgb-trials $(TRIALS)

individual-analysis:
	$(PY) -m src.pipelines.run_individual_analysis --id-mod $(ID_MOD) --id-rem $(ID_REM)

remote-refresh:
	TRIALS=$(TRIALS) N_JOBS_PER_JOB=$(N_JOBS_PER_JOB) BOOTSTRAP=$(BOOTSTRAP) BOOTSTRAP_SUBGROUP=$(BOOTSTRAP_SUBGROUP) bash scripts/remote_heavy_refresh.sh

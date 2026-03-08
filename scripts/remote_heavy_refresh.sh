#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

REMOTE_HOST="${REMOTE_HOST:-your.host}"
REMOTE_RUN_DIR="${REMOTE_RUN_DIR:-/path/to/remote/workdir}"

TRIALS="${TRIALS:-32}"
N_JOBS_PER_JOB="${N_JOBS_PER_JOB:-8}"

BOOTSTRAP="${BOOTSTRAP:-2000}"
BOOTSTRAP_SUBGROUP="${BOOTSTRAP_SUBGROUP:-200}"

REMOTE_PY="${REMOTE_PY:-python}"

RUN_NIJ="${RUN_NIJ:-1}"
RUN_COMPAS="${RUN_COMPAS:-1}"
RUN_FAIRNESS="${RUN_FAIRNESS:-1}"

echo "[remote-refresh] syncing repo -> ${REMOTE_HOST}:${REMOTE_RUN_DIR}"
ssh "${REMOTE_HOST}" "mkdir -p '${REMOTE_RUN_DIR}'"

# rsync remote paths can't be shell-quoted safely; escape spaces for the remote-path argument.
REMOTE_RUN_DIR_RSYNC="${REMOTE_RUN_DIR// /\\ }"
rsync -az --delete \
  --exclude '.venv*' \
  --exclude 'data/' \
  --exclude 'do/' \
  --exclude '__pycache__' \
  --exclude '*.pyc' \
  --exclude '.DS_Store' \
  "${ROOT_DIR}/" "${REMOTE_HOST}:${REMOTE_RUN_DIR_RSYNC}/"

echo "[remote-refresh] remote tests"
ssh "${REMOTE_HOST}" "cd '${REMOTE_RUN_DIR}' && '${REMOTE_PY}' -m unittest -q tests/test_env_policy.py tests/test_metrics.py tests/test_leakage.py tests/test_compas.py tests/test_nij_scoring.py"

echo "[remote-refresh] remote heavy runs: trials=${TRIALS} n_jobs_per_job=${N_JOBS_PER_JOB} run_nij=${RUN_NIJ} run_compas=${RUN_COMPAS}"
ssh "${REMOTE_HOST}" "set -u; cd '${REMOTE_RUN_DIR}'; mkdir -p remote_logs; \
  s1=0; s2=0; \
  if [ '${RUN_NIJ}' -eq 1 ] && [ '${RUN_COMPAS}' -eq 1 ]; then \
    : > remote_logs/nij_xgb.log; : > remote_logs/compas_all.log; \
    (PRECRIME_N_JOBS='${N_JOBS_PER_JOB}' '${REMOTE_PY}' -m src.pipelines.run_nij --task xgb --xgb-trials '${TRIALS}' > remote_logs/nij_xgb.log 2>&1) & pid1=\$!; \
    (PRECRIME_N_JOBS='${N_JOBS_PER_JOB}' '${REMOTE_PY}' -m src.pipelines.run_compas --task all --xgb-trials '${TRIALS}' > remote_logs/compas_all.log 2>&1) & pid2=\$!; \
    wait \$pid1 || s1=\$?; wait \$pid2 || s2=\$?; \
  elif [ '${RUN_NIJ}' -eq 1 ]; then \
    : > remote_logs/nij_xgb.log; \
    PRECRIME_N_JOBS='${N_JOBS_PER_JOB}' '${REMOTE_PY}' -m src.pipelines.run_nij --task xgb --xgb-trials '${TRIALS}' > remote_logs/nij_xgb.log 2>&1 || s1=\$?; \
  elif [ '${RUN_COMPAS}' -eq 1 ]; then \
    : > remote_logs/compas_all.log; \
    PRECRIME_N_JOBS='${N_JOBS_PER_JOB}' '${REMOTE_PY}' -m src.pipelines.run_compas --task all --xgb-trials '${TRIALS}' > remote_logs/compas_all.log 2>&1 || s2=\$?; \
  fi; \
  echo \"[remote-refresh] statuses: nij=\$s1 compas=\$s2\"; \
  if [ \$s1 -ne 0 ] || [ \$s2 -ne 0 ]; then exit 1; fi"

if [ "${RUN_FAIRNESS}" -eq 1 ]; then
  echo "[remote-refresh] remote fairness: bootstrap=${BOOTSTRAP} subgroup=${BOOTSTRAP_SUBGROUP}"
  ssh "${REMOTE_HOST}" "cd '${REMOTE_RUN_DIR}' && PRECRIME_FAIRNESS_BOOTSTRAP='${BOOTSTRAP}' PRECRIME_FAIRNESS_BOOTSTRAP_SUBGROUP='${BOOTSTRAP_SUBGROUP}' '${REMOTE_PY}' -m src.pipelines.run_fairness"
else
  echo "[remote-refresh] skipping fairness (RUN_FAIRNESS=${RUN_FAIRNESS})"
fi

echo "[remote-refresh] syncing reports back -> ${ROOT_DIR}/reports"
rsync -az "${REMOTE_HOST}:${REMOTE_RUN_DIR_RSYNC}/reports/" "${ROOT_DIR}/reports/"

echo "[remote-refresh] done"

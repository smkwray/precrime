#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${OUT_DIR:-${ROOT_DIR}/public_export}"

OUT_DIR_HAS_GIT=0
if [ -d "${OUT_DIR}/.git" ]; then
  OUT_DIR_HAS_GIT=1
fi

# Create/clear the export directory.
# - If public_export is itself a git repo (intended for GitHub), preserve `.git/`.
# - Cloud-sync/Finder can race and recreate `.DS_Store`, so treat removal as best-effort.
if [ "${OUT_DIR_HAS_GIT}" -eq 1 ]; then
  mkdir -p "${OUT_DIR}"
else
  rm -rf "${OUT_DIR}" 2>/dev/null || true
  mkdir -p "${OUT_DIR}"
fi

TMP_OUT="$(mktemp -d)"

copy() {
  local src="$1"
  local dst="$2"
  mkdir -p "$(dirname "${dst}")"
  if [ -d "${src}" ]; then
    rsync -a --delete \
      --exclude '.DS_Store' \
      --exclude '__pycache__' \
      --exclude '*.pyc' \
      --exclude '.pytest_cache' \
      --exclude '.mypy_cache' \
      --exclude '.ruff_cache' \
      --exclude '.ipynb_checkpoints' \
      --exclude '.venv*' \
      "${src%/}/" "${dst%/}/"
  else
    rsync -a \
      --exclude '.DS_Store' \
      --exclude '*.pyc' \
      "${src}" "${dst}"
  fi
}

copy "${ROOT_DIR}/README.md" "${TMP_OUT}/README.md"
copy "${ROOT_DIR}/Makefile" "${TMP_OUT}/Makefile"
copy "${ROOT_DIR}/LICENSE" "${TMP_OUT}/LICENSE"
copy "${ROOT_DIR}/requirements.txt" "${TMP_OUT}/requirements.txt"
copy "${ROOT_DIR}/requirements-modeling.txt" "${TMP_OUT}/requirements-modeling.txt"
copy "${ROOT_DIR}/requirements-viz.txt" "${TMP_OUT}/requirements-viz.txt"

copy "${ROOT_DIR}/configs/" "${TMP_OUT}/configs/"
copy "${ROOT_DIR}/docs/" "${TMP_OUT}/docs/"
copy "${ROOT_DIR}/schemas/" "${TMP_OUT}/schemas/"
copy "${ROOT_DIR}/src/" "${TMP_OUT}/src/"
copy "${ROOT_DIR}/tests/" "${TMP_OUT}/tests/"
copy "${ROOT_DIR}/scripts/" "${TMP_OUT}/scripts/"
if [ -d "${ROOT_DIR}/.github" ]; then
  copy "${ROOT_DIR}/.github/" "${TMP_OUT}/.github/"
fi

mkdir -p "${TMP_OUT}/reports"
rsync -a --exclude '.DS_Store' "${ROOT_DIR}/reports/"*.md "${TMP_OUT}/reports/" 2>/dev/null || true
mkdir -p "${TMP_OUT}/reports/plots"
rsync -a --exclude '.DS_Store' "${ROOT_DIR}/reports/plots/"*.json "${TMP_OUT}/reports/plots/" 2>/dev/null || true

copy "${ROOT_DIR}/docs/public_export_template.md" "${TMP_OUT}/README_EXPORT.md"

# Add a conservative gitignore for the export, so it can be initialized as a separate repo safely.
cat > "${TMP_OUT}/.gitignore" <<'EOF'
.DS_Store
__pycache__/
*.pyc
.pytest_cache/
.mypy_cache/
.ruff_cache/
.ipynb_checkpoints/
.venv*/
.venv*

# Never commit row-level datasets
/data/
*.csv
*.parquet

# Logs
remote_logs/
*.log
EOF

# Sync into OUT_DIR (preserving .git if present).
if [ "${OUT_DIR_HAS_GIT}" -eq 1 ]; then
  rsync -a --delete --exclude '.git' "${TMP_OUT%/}/" "${OUT_DIR%/}/"
else
  rsync -a --delete "${TMP_OUT%/}/" "${OUT_DIR%/}/"
fi

rm -rf "${TMP_OUT}" 2>/dev/null || true

echo "[public_export] wrote: ${OUT_DIR}"
echo "[public_export] NOTE: data/ is intentionally excluded"

# Defensive cleanup for macOS Finder artifacts.
find "${OUT_DIR}" -name '.DS_Store' -delete 2>/dev/null || true

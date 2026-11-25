#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${1:-${ROOT_DIR}/public_export}"

if [ ! -d "${OUT_DIR}" ]; then
  echo "[preflight] missing export dir: ${OUT_DIR}"
  echo "[preflight] run: bash scripts/build_public_export.sh"
  exit 2
fi

fail() {
  echo "[preflight] FAIL: $*" >&2
  exit 1
}

echo "[preflight] checking export: ${OUT_DIR}"

# Hard exclusions.
[ ! -d "${OUT_DIR}/data" ] || fail "export contains data/: ${OUT_DIR}/data"
[ ! -d "${OUT_DIR}/do" ] || fail "export contains do/: ${OUT_DIR}/do"

# Defensive: no repo-local venvs or caches inside export.
if find "${OUT_DIR}" -maxdepth 2 -name '.venv*' -type d | rg -q .; then
  fail "export contains .venv* directory"
fi
if find "${OUT_DIR}" -name '__pycache__' -type d | rg -q .; then
  fail "export contains __pycache__ (rebuild export after cleaning)"
fi

# Row-level files should not be present in the export at all.
if find "${OUT_DIR}" -type f \( -name '*.parquet' -o -name '*.csv' \) | rg -q .; then
  fail "export contains .csv/.parquet row-level files"
fi

# Grep for common private-machine leaks.
tmp="$(mktemp)"
terms="$(mktemp)"
trap 'rm -f "${tmp}" "${terms}"' EXIT

rg -n "/Users/|Library/CloudStorage|GoogleDrive-" "${OUT_DIR}" \
  --glob '!**/scripts/preflight_public_export.sh' >"${tmp}" || true
if [ -s "${tmp}" ]; then
  echo "[preflight] found potential private paths:" >&2
  cat "${tmp}" >&2
  fail "remove private paths"
fi

# Flag private identifiers (usernames, host names) listed one per line in a
# local, gitignored .private-denylist. Lines starting with # are comments.
denylist="${ROOT_DIR}/.private-denylist"
if [ -f "${denylist}" ]; then
  grep -v '^[[:space:]]*#' "${denylist}" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//' -e '/^$/d' >"${terms}" || true
fi
if [ -s "${terms}" ]; then
  rg -n -i -w -F -f "${terms}" "${OUT_DIR}" >"${tmp}" || true
  if [ -s "${tmp}" ]; then
    echo "[preflight] found private identifiers from ${denylist}:" >&2
    cut -d: -f1,2 "${tmp}" >&2
    fail "remove private identifiers"
  fi
else
  echo "[preflight] no .private-denylist terms; skipping identifier check"
fi

rg -n "\\b(?:\\d{1,3}\\.){3}\\d{1,3}\\b" "${OUT_DIR}" \
  --glob '!**/scripts/preflight_public_export.sh' >"${tmp}" || true
if [ -s "${tmp}" ]; then
  echo "[preflight] found IPv4-looking strings (replace with placeholders):" >&2
  cat "${tmp}" >&2
  fail "remove IP addresses"
fi

echo "[preflight] OK"
echo "[preflight] next: from ${OUT_DIR} initialize git + push to GitHub (private or public)"

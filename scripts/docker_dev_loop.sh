#!/usr/bin/env bash
set -euo pipefail

ITERATIONS=1
FULL_TESTS=0
CONFIG_DIR="config"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --iterations)
      ITERATIONS="$2"
      shift 2
      ;;
    --full-tests)
      FULL_TESTS=1
      shift
      ;;
    --config-dir)
      CONFIG_DIR="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if ! [[ "$ITERATIONS" =~ ^[0-9]+$ ]] || [[ "$ITERATIONS" -lt 1 ]]; then
  echo "--iterations must be a positive integer" >&2
  exit 2
fi

mkdir -p reports
STAMP="$(date +%Y%m%d-%H%M%S)"
REPORT="reports/docker-dev-loop-${STAMP}.log"

echo "Docker dev loop: iterations=${ITERATIONS}, full_tests=${FULL_TESTS}, config_dir=${CONFIG_DIR}" | tee "$REPORT"

docker compose build autotrader >>"$REPORT" 2>&1

for ((i = 1; i <= ITERATIONS; i++)); do
  {
    echo
    echo "=== ITERATION ${i}/${ITERATIONS} ==="
    docker compose run --rm autotrader sh -lc "
      pip install --quiet -e '[dev]'

      echo '[loop] format/lint'
      ruff format src/ tests/
      ruff check --fix src/ tests/

      echo '[loop] preflight'
      autotrader preflight --config-dir '${CONFIG_DIR}'

      echo '[loop] fast tests'
      pytest -m 'not integration' -q
    "

    if [[ "${FULL_TESTS}" -eq 1 ]]; then
      echo "[loop] full tests"
      docker compose run --rm autotrader sh -lc "pip install --quiet -e '[dev]' && pytest -q"
    fi

    echo "[$(date -Iseconds)] iteration ${i} completed"
  } >>"$REPORT" 2>&1
done

echo "Done. Report: ${REPORT}"

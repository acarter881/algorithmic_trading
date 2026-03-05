#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

MODE="${1:-fast}"

run() {
  echo
  echo "==> $*"
  "$@"
}

need_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 127
  fi
}

usage() {
  echo "Usage: scripts/agent_verify.sh [fast|full|trading]" >&2
  echo "  fast     -> format, lint, type-check, test" >&2
  echo "  full     -> fast + docker dev loop (1 iteration)" >&2
  echo "  trading  -> fast + docker dev loop (3 iterations)" >&2
  exit 2
}

need_cmd ruff
need_cmd mypy
need_cmd pytest

case "${MODE}" in
  fast)
    ;;
  full|trading)
    need_cmd docker
    if [[ ! -f "./scripts/docker_dev_loop.sh" ]]; then
      echo "Missing required script: ./scripts/docker_dev_loop.sh" >&2
      exit 1
    fi
    ;;
  *)
    usage
    ;;
esac

echo "Verification mode: ${MODE}"
echo "Repository root: ${REPO_ROOT}"

run ruff format .
run ruff check --fix .
run mypy src/autotrader/
run pytest -q

case "${MODE}" in
  full)
    run bash ./scripts/docker_dev_loop.sh --iterations 1
    ;;
  trading)
    run bash ./scripts/docker_dev_loop.sh --iterations 3
    ;;
esac

echo
echo "Verification passed (${MODE})."

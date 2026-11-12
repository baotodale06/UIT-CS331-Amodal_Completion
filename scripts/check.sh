#!/usr/bin/env bash
set -Eeuo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd -- "$PROJECT_DIR"
PYTHON_BIN="${PYTHON_BIN:-python3}"
"$PYTHON_BIN" -m unittest discover -s tests -v
for TEST_SCRIPT in tests/test_*.sh; do
    bash "$TEST_SCRIPT"
done
"$PYTHON_BIN" -m amodal_benchmark smoke

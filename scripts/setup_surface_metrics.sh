#!/usr/bin/env bash
# Isolated metric dependencies; never install into the pinned inference env.
set -Eeuo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
CACHE_DIR="$PWD/.amodal-cache"
BASE_ENV="$CACHE_DIR/conda-env"
METRIC_ENV="$CACHE_DIR/surface-metrics-env"
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH="$BASE_ENV/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
if [[ ! -x "$METRIC_ENV/bin/python" ]]; then
    "$BASE_ENV/bin/python" -m venv --system-site-packages "$METRIC_ENV"
fi
"$METRIC_ENV/bin/python" -m pip install --no-deps \
    dreamsim==0.2.1 peft==0.5.0 open-clip-torch==2.20.0 ftfy==6.1.1 \
    openai-clip==1.0.1 sentencepiece==0.1.99 protobuf==3.20.3 wcwidth==0.2.13
"$METRIC_ENV/bin/python" scripts/cache_surface_dreamsim.py --cache-dir "$CACHE_DIR/dreamsim"

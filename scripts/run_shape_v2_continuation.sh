#!/usr/bin/env bash
set -Eeuo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the frozen evaluation source}"
export PYTHONNOUSERSITE=1
exec "$HOME/Vu_Hung/cv_nangcao/.amodal-cache/surface-metrics-env/bin/python" \
    scripts/continue_shape_v2.py "$@"

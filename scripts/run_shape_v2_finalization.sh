#!/usr/bin/env bash
set -Eeuo pipefail
OPERATIONAL_DIR="${SLURM_SUBMIT_DIR:?Submit from the separate operational source}"
STUDY="$(dirname -- "$OPERATIONAL_DIR")"
cd "$OPERATIONAL_DIR"
export PYTHONNOUSERSITE=1 PYTHONPATH="$STUDY/postprocessing-source"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
exec "$HOME/Vu_Hung/cv_nangcao/.amodal-cache/surface-metrics-env/bin/python" \
    scripts/finalize_shape_v2.py "$@"

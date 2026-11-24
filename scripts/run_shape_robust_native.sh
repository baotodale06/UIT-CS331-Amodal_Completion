#!/usr/bin/env bash
set -Eeuo pipefail
PROJECT_DIR="$HOME/Vu_Hung/cv_nangcao"
STUDY="$PROJECT_DIR/output/shape-sam-robustness-20261123"
V2="$PROJECT_DIR/output/shape-v2-slurm-20261123"
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export LD_LIBRARY_PATH="$PROJECT_DIR/.amodal-cache/conda-env/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$STUDY/native-source:$STUDY/training-source:$V2/evaluation-source"
unset PYTHONHOME
cd "$STUDY/native-source"
exec "$PROJECT_DIR/.amodal-cache/surface-metrics-env/bin/python" -u scripts/evaluate_shape_robust_native.py \
    --training "$STUDY/training" --training-source "$STUDY/training-source" --data "$V2/data" \
    --sam-validation "$PROJECT_DIR/output/shape-validation-sam-20261123/result" --output "$STUDY/native-validation"

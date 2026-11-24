#!/usr/bin/env bash
set -Eeuo pipefail
PROJECT_DIR="$HOME/Vu_Hung/cv_nangcao"
STUDY="$PROJECT_DIR/output/shape-validation-sam-20261123"
V2="$PROJECT_DIR/output/shape-v2-slurm-20261123"
CODE_DIR="$V2/evaluation-source"
CACHE_DIR="$PROJECT_DIR/.amodal-cache"
ENV_DIR="$CACHE_DIR/conda-env"
REPO_DIR="$CACHE_DIR/amodal"
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export CUDA_HOME="$ENV_DIR" PATH="$ENV_DIR/bin:$PATH"
export LD_LIBRARY_PATH="$ENV_DIR/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export AMODAL_CACHE_DIR="$CACHE_DIR" AMODAL_ENV_DIR="$ENV_DIR" AMODAL_REPO_DIR="$REPO_DIR"
unset PYTHONHOME PYTHONPATH
cd "$CODE_DIR"
source "$CODE_DIR/run_amodal_slurm.sh"
configure_python_imports
export PYTHONPATH="$STUDY/source:$CODE_DIR:$PYTHONPATH"
exec "$CACHE_DIR/surface-metrics-env/bin/python" -u "$STUDY/source/scripts/audit_shape_sam_validation.py" \
    --data "$V2/data" --checkpoint "$V2/training/best.pt" \
    --sam-checkpoint "$REPO_DIR/Grounded-Segment-Anything/sam_vit_h_4b8939.pth" \
    --original-validation "$V2/validation-native.json" --output "$STUDY/result"

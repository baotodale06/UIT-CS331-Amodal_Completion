#!/usr/bin/env bash
set -Eeuo pipefail
PROJECT_DIR="$HOME/Vu_Hung/cv_nangcao"
STUDY="$PROJECT_DIR/output/shape-sam-robustness-20261123"
V2="$PROJECT_DIR/output/shape-v2-slurm-20261123"
CODE_DIR="$V2/evaluation-source"
CACHE_DIR="$PROJECT_DIR/.amodal-cache"
ENV_DIR="$CACHE_DIR/conda-env"
REPO_DIR="$CACHE_DIR/amodal"
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export CUDA_HOME="$ENV_DIR" PATH="$ENV_DIR/bin:$PATH"
export LD_LIBRARY_PATH="$ENV_DIR/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export AMODAL_CACHE_DIR="$CACHE_DIR" AMODAL_ENV_DIR="$ENV_DIR" AMODAL_REPO_DIR="$REPO_DIR"
unset PYTHONHOME PYTHONPATH
cd "$CODE_DIR"
source "$CODE_DIR/run_amodal_slurm.sh"
configure_python_imports
export PYTHONPATH="$STUDY/source:$CODE_DIR:$PYTHONPATH"
PYTHON_EXEC="$CACHE_DIR/surface-metrics-env/bin/python"
ARGS=(--data "$V2/data" --sam-checkpoint "$REPO_DIR/Grounded-Segment-Anything/sam_vit_h_4b8939.pth" --output "$STUDY/training-masks")
"$PYTHON_EXEC" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=4 \
    "$STUDY/source/scripts/prepare_shape_sam_training.py" "${ARGS[@]}"
"$PYTHON_EXEC" "$STUDY/source/scripts/prepare_shape_sam_training.py" "${ARGS[@]}" --merge

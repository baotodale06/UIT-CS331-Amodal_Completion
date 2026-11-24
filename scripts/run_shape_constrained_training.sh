#!/usr/bin/env bash
set -Eeuo pipefail
PROJECT_DIR="$HOME/Vu_Hung/cv_nangcao"
STUDY="$PROJECT_DIR/output/shape-sam-constrained-v2-20261123"
V2="$PROJECT_DIR/output/shape-v2-slurm-20261123"
CACHE_DIR="$PROJECT_DIR/.amodal-cache"
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export TORCH_HOME="$CACHE_DIR/torch"
export LD_LIBRARY_PATH="$CACHE_DIR/conda-env/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
unset PYTHONHOME
export PYTHONPATH="$STUDY/training-source:$V2/evaluation-source"
cd "$STUDY/training-source"
IFS=',' read -r -a DEVICES <<< "${CUDA_VISIBLE_DEVICES:?Four Slurm GPUs required}"
[[ "${#DEVICES[@]}" == 4 ]]
MODE="${1:?preflight or train}"
shift
case "$MODE" in
    preflight) OUTPUT_DIR="$STUDY/preflight"; EXTRA=(--preflight) ;;
    train) OUTPUT_DIR="$STUDY/training"; EXTRA=() ;;
    *) echo 'Expected preflight or train' >&2; exit 2 ;;
esac
exec "$CACHE_DIR/surface-metrics-env/bin/python" -m torch.distributed.run \
    --standalone --nnodes=1 --nproc_per_node=4 scripts/train_shape_constrained_slurm.py \
    --data "$V2/data" --sam-training "$PROJECT_DIR/output/shape-sam-robustness-20261123/training-masks" \
    --sam-validation "$PROJECT_DIR/output/shape-validation-sam-20261123/result" \
    --initial-checkpoint "$V2/training/best.pt" --output "$OUTPUT_DIR" "${EXTRA[@]}" "$@"

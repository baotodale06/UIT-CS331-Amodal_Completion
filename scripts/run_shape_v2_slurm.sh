#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=80G
set -Eeuo pipefail
cd -- "${SLURM_SUBMIT_DIR:?Submit from the isolated source snapshot}"
PROJECT_DIR="${SHAPE_PROJECT_DIR:?Set the original project root}"
CACHE_DIR="$PROJECT_DIR/.amodal-cache"
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export TORCH_HOME="$CACHE_DIR/torch"
export LD_LIBRARY_PATH="$CACHE_DIR/conda-env/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
unset PYTHONPATH PYTHONHOME
IFS=',' read -r -a DEVICES <<< "${CUDA_VISIBLE_DEVICES:?Slurm GPUs required}"
[[ "${#DEVICES[@]}" == 4 ]] || { echo 'Require four allocated GPUs' >&2; exit 2; }
PYTHON_EXEC="$CACHE_DIR/surface-metrics-env/bin/python"
"$PYTHON_EXEC" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=4 \
    scripts/train_shape_v2_slurm.py "$@"

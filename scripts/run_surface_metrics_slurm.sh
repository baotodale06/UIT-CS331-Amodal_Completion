#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=01:00:00
set -Eeuo pipefail
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)}"
cd -- "$PROJECT_DIR"
[[ "${1:-}" == --benchmark-workers && "${2:-}" == 4 ]] || { echo 'Require --benchmark-workers 4' >&2; exit 2; }
shift 2
CACHE_DIR="$PROJECT_DIR/.amodal-cache"
ENV_DIR="$CACHE_DIR/conda-env"
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=2 MPLBACKEND=Agg
export TORCH_HOME="$CACHE_DIR/torch" PYTHONPATH="$PROJECT_DIR"
export LD_LIBRARY_PATH="$ENV_DIR/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
unset PYTHONHOME
IFS=',' read -r -a DEVICES <<< "${CUDA_VISIBLE_DEVICES:?Slurm GPU allocation required}"
[[ "${#DEVICES[@]}" == 4 ]] || { echo 'Expected four GPUs' >&2; exit 2; }
PIDS=()
for INDEX in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES="${DEVICES[$INDEX]}" "$ENV_DIR/bin/python" -u scripts/measure_surface_perceptual.py \
        --worker-index "$INDEX" --worker-count 4 "$@" \
        > "$PROJECT_DIR/output/surface-consistency-20261108/job-logs/${SLURM_JOB_ID}-metrics-${INDEX}.log" 2>&1 &
    PIDS+=("$!")
done
STATUS=0
for PID in "${PIDS[@]}"; do wait "$PID" || STATUS=1; done
exit "$STATUS"

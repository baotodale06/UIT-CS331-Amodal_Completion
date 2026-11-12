#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=05:00:00
set -Eeuo pipefail
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)}"
cd -- "$PROJECT_DIR"
WORKERS=4
if [[ "${1:-}" == --benchmark-workers ]]; then WORKERS="$2"; shift 2; fi
[[ "$WORKERS" == 4 ]] || { echo 'This protocol requires four GPU workers' >&2; exit 2; }
CACHE_DIR="$PROJECT_DIR/.amodal-cache"
ENV_DIR="$CACHE_DIR/conda-env"
REPO_DIR="$CACHE_DIR/amodal"
export PYTHONNOUSERSITE=1 MPLBACKEND=Agg TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2
export HF_HOME="$CACHE_DIR/huggingface" HUGGINGFACE_HUB_CACHE="$CACHE_DIR/huggingface/hub"
export TRANSFORMERS_CACHE="$CACHE_DIR/huggingface/transformers" TORCH_HOME="$CACHE_DIR/torch"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CUDA_HOME="$ENV_DIR" PATH="$ENV_DIR/bin:$PATH"
export LD_LIBRARY_PATH="$ENV_DIR/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
unset PYTHONHOME PYTHONPATH
source "$PROJECT_DIR/run_amodal_slurm.sh"
configure_python_imports
export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"
IFS=',' read -r -a DEVICES <<< "${CUDA_VISIBLE_DEVICES:?Slurm GPU allocation required}"
[[ "${#DEVICES[@]}" -eq 4 ]] || { echo 'Expected four allocated GPUs' >&2; exit 2; }
mkdir -p "$PROJECT_DIR/output/surface-consistency-20261108/job-logs"
if [[ "${SURFACE_VALIDATE_RUNTIME:-1}" == 1 ]]; then
    bash tests/validate_cuda_extension.sh "$ENV_DIR" "$REPO_DIR"
    bash tests/validate_pipeline_imports.sh "$ENV_DIR" "$REPO_DIR"
fi
PIDS=()
for INDEX in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES="${DEVICES[$INDEX]}" "$ENV_DIR/bin/python" -u scripts/run_surface_candidates.py \
        --repo-dir "$REPO_DIR" --worker-index "$INDEX" --worker-count "$WORKERS" "$@" \
        > "$PROJECT_DIR/output/surface-consistency-20261108/job-logs/${SLURM_JOB_ID}-worker-${INDEX}.log" 2>&1 &
    PIDS+=("$!")
done
STATUS=0
for PID in "${PIDS[@]}"; do wait "$PID" || STATUS=1; done
exit "$STATUS"

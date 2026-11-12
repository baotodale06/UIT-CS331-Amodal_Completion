#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
set -Eeuo pipefail
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)}"
cd -- "$PROJECT_DIR"
[[ "${1:-}" == --benchmark-workers && "${2:-}" == 4 ]] || { echo 'Require four workers' >&2; exit 2; }
shift 2
[[ "${1:-}" == --script ]] || { echo 'Require a worker script' >&2; exit 2; }
WORKER_SCRIPT="$2"; shift 2
[[ "${1:-}" == --log-dir ]] || { echo 'Require isolated log directory' >&2; exit 2; }
LOG_DIR="$2"; shift 2
mkdir -p "$LOG_DIR"
CACHE_DIR="$PROJECT_DIR/.amodal-cache"
ENV_DIR="$CACHE_DIR/conda-env"
REPO_DIR="$CACHE_DIR/amodal"
export PYTHONNOUSERSITE=1 MPLBACKEND=Agg TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2
export HF_HOME="$CACHE_DIR/huggingface" HUGGINGFACE_HUB_CACHE="$CACHE_DIR/huggingface/hub"
export TRANSFORMERS_CACHE="$CACHE_DIR/huggingface/transformers" TORCH_HOME="$CACHE_DIR/torch"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CUDA_HOME="$ENV_DIR" PATH="$ENV_DIR/bin:$PATH"
export LD_LIBRARY_PATH="$ENV_DIR/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
if [[ "${SURFACE_CUFFT_COMPAT:-0}" == 1 ]]; then
    CUFFT_LIBRARY="$CACHE_DIR/surface-cufft-cu118/nvidia/cufft/lib/libcufft.so.10"
    [[ -f "$CUFFT_LIBRARY" ]] || { echo 'Missing isolated cuFFT compatibility library' >&2; exit 2; }
    export LD_PRELOAD="$CUFFT_LIBRARY${LD_PRELOAD:+:$LD_PRELOAD}"
    sha256sum "$CUFFT_LIBRARY" > "$LOG_DIR/${SLURM_JOB_ID}-cufft.sha256"
fi
unset PYTHONHOME PYTHONPATH
source "$PROJECT_DIR/run_amodal_slurm.sh"
configure_python_imports
export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"
PYTHON_EXEC="$ENV_DIR/bin/python"
if [[ "${SURFACE_METRICS_ENV:-0}" == 1 ]]; then
    PYTHON_EXEC="$CACHE_DIR/surface-metrics-env/bin/python"
fi
IFS=',' read -r -a DEVICES <<< "${CUDA_VISIBLE_DEVICES:?Slurm allocation required}"
[[ "${#DEVICES[@]}" == 4 ]] || { echo 'Expected four GPUs' >&2; exit 2; }
nvidia-smi > "$LOG_DIR/${SLURM_JOB_ID}-nvidia-smi.txt"
if [[ "${SURFACE_VALIDATE_RUNTIME:-0}" == 1 ]]; then
    bash tests/validate_cuda_extension.sh "$ENV_DIR" "$REPO_DIR"
    bash tests/validate_pipeline_imports.sh "$ENV_DIR" "$REPO_DIR"
fi
PIDS=()
for INDEX in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES="${DEVICES[$INDEX]}" "$PYTHON_EXEC" -u "$WORKER_SCRIPT" \
        --worker-index "$INDEX" --worker-count 4 "$@" \
        > "$LOG_DIR/${SLURM_JOB_ID}-worker-${INDEX}.log" 2>&1 &
    PIDS+=("$!")
done
STATUS=0
for PID in "${PIDS[@]}"; do wait "$PID" || STATUS=1; done
exit "$STATUS"

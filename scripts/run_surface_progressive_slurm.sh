#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
set -Eeuo pipefail
PROJECT_DIR="${SLURM_SUBMIT_DIR:?Submit from the project root}"
cd -- "$PROJECT_DIR"
[[ "${1:-}" == --benchmark-workers && "${2:-}" == 4 ]] || { echo 'Require four workers' >&2; exit 2; }
shift 2
CACHE_DIR="$PROJECT_DIR/.amodal-cache"
ENV_DIR="$CACHE_DIR/conda-env"
REPO_DIR="$CACHE_DIR/amodal"
export PYTHONNOUSERSITE=1 MPLBACKEND=Agg TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
export HF_HOME="$CACHE_DIR/huggingface" HUGGINGFACE_HUB_CACHE="$CACHE_DIR/huggingface/hub"
export TRANSFORMERS_CACHE="$CACHE_DIR/huggingface/transformers" TORCH_HOME="$CACHE_DIR/torch"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CUDA_HOME="$ENV_DIR" PATH="$ENV_DIR/bin:$PATH"
export LD_LIBRARY_PATH="$ENV_DIR/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
if [[ "${SURFACE_CUFFT_COMPAT:-0}" == 1 ]]; then
    CUFFT_LIBRARY="$CACHE_DIR/surface-cufft-cu118/nvidia/cufft/lib/libcufft.so.10"
    [[ -f "$CUFFT_LIBRARY" ]] || { echo 'Missing isolated cuFFT compatibility library' >&2; exit 2; }
    export LD_PRELOAD="$CUFFT_LIBRARY${LD_PRELOAD:+:$LD_PRELOAD}"
fi
unset PYTHONHOME PYTHONPATH
source "$PROJECT_DIR/run_amodal_slurm.sh"
configure_python_imports
export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"
IFS=',' read -r -a DEVICES <<< "${CUDA_VISIBLE_DEVICES:?Slurm allocation required}"
[[ "${#DEVICES[@]}" == 4 ]] || { echo 'Expected four allocated GPUs' >&2; exit 2; }
SURFACE_OPTIONS=()
for ARGUMENT in "$@"; do
    if [[ "$ARGUMENT" == --surface-freeze ]]; then
        SURFACE_OPTIONS=(--surface-dreamsim-cache "$CACHE_DIR/dreamsim")
        break
    fi
done
exec "$CACHE_DIR/surface-metrics-env/bin/python" -u evaluate_open_world.py \
    --dataset-root "$PROJECT_DIR/datasets/open_world_amodal" --repo-dir "$REPO_DIR" \
    --pipeline-profile released-sam --target-selection label-semantic --boundary-check released \
    --background-sizing auto --detector-vocabulary released --workers 4 \
    "${SURFACE_OPTIONS[@]}" "$@"

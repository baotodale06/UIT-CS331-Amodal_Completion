#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
set -Eeuo pipefail
cd -- "${SLURM_SUBMIT_DIR:?Submit from the project root}"
[[ "${1:-}" == --benchmark-workers && "${2:-}" == 4 ]] || exit 2
shift 2
[[ "${1:-}" == --stage && "$#" == 2 ]] || exit 2
STAGE="$2"
DESIGN=output/surface-annotated-20261112-design
STUDY=output/surface-annotated-20261112
PYTHON="$PWD/.amodal-cache/surface-metrics-env/bin/python"
export SURFACE_CUFFT_COMPAT=1 SURFACE_METRICS_ENV=1 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=8
"$PYTHON" scripts/check_annotated_surface_freeze.py --design "$DESIGN"
if [[ "$STAGE" == baseline-cohort ]]; then
    export LD_PRELOAD="$PWD/.amodal-cache/surface-cufft-cu118/nvidia/cufft/lib/libcufft.so.10"
    bash tests/validate_cuda_extension.sh "$PWD/.amodal-cache/conda-env" "$PWD/.amodal-cache/amodal"
    bash tests/validate_pipeline_imports.sh "$PWD/.amodal-cache/conda-env" "$PWD/.amodal-cache/amodal"
    bash scripts/run_surface_progressive_slurm.sh --benchmark-workers 4 --detector-vocabulary annotated \
        --output-dir "$STUDY/baseline" --target-manifest "$DESIGN/cohort.json"
    "$PYTHON" scripts/snapshot_annotated_surface.py --run-dir "$STUDY/baseline" \
        --output-dir "$STUDY/baseline-cohort" --dataset-root datasets/open_world_amodal \
        --target-manifest "$DESIGN/cohort.json" > "$DESIGN/baseline-cohort-verification.json"
elif [[ "$STAGE" == surface-cohort ]]; then
    bash scripts/run_surface_progressive_slurm.sh --benchmark-workers 4 --detector-vocabulary annotated \
        --surface-freeze "$DESIGN/method-freeze.json" --output-dir "$STUDY/surface" \
        --target-manifest "$DESIGN/cohort.json"
    "$PYTHON" scripts/snapshot_annotated_surface.py --run-dir "$STUDY/surface" \
        --output-dir "$STUDY/surface-cohort" --dataset-root datasets/open_world_amodal \
        --target-manifest "$DESIGN/cohort.json" > "$DESIGN/surface-cohort-verification.json"
    "$PYTHON" scripts/compare_surface_progressive.py --baseline "$STUDY/baseline-cohort" \
        --surface "$STUDY/surface-cohort" --dataset-root datasets/open_world_amodal \
        --target-manifest "$DESIGN/cohort.json" --output-dir "$STUDY/cohort-comparison"
elif [[ "$STAGE" == baseline-full ]]; then
    bash scripts/run_surface_progressive_slurm.sh --benchmark-workers 4 --detector-vocabulary annotated \
        --output-dir "$STUDY/baseline"
    "$PYTHON" -m amodal_benchmark verify --run-dir "$STUDY/baseline" \
        --dataset-root datasets/open_world_amodal --require-full --require-provenance \
        > "$DESIGN/baseline-full-verification.json"
elif [[ "$STAGE" == surface-full ]]; then
    bash scripts/run_surface_progressive_slurm.sh --benchmark-workers 4 --detector-vocabulary annotated \
        --surface-freeze "$DESIGN/method-freeze.json" --output-dir "$STUDY/surface"
    "$PYTHON" -m amodal_benchmark verify --run-dir "$STUDY/surface" \
        --dataset-root datasets/open_world_amodal --require-full --require-provenance \
        > "$DESIGN/surface-full-verification.json"
else
    echo "Unknown stage: $STAGE" >&2
    exit 2
fi

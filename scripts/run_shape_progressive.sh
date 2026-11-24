#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=24:00:00
set -Eeuo pipefail
cd "$SLURM_SUBMIT_DIR"
[[ "$1" == --benchmark-workers && "$2" == 4 ]]
ARM="$3"
STAGE="$4"
STUDY=output/shape-20261121
OUT="$STUDY/progressive-l40s"
DESIGN=output/surface-annotated-20261112-design
BASE="$OUT/baseline"
PYTHON="$PWD/.amodal-cache/surface-metrics-env/bin/python"
ARGS=(--benchmark-workers 4 --detector-vocabulary annotated --output-dir "$OUT/$ARM")
case "$ARM" in
    baseline) ;;
    surface) ARGS+=(--surface-freeze "$DESIGN/method-freeze.json");;
    combined) ARGS+=(--shape-checkpoint "$STUDY/prior.pt" --surface-freeze "$DESIGN/method-freeze.json");;
    shape) ARGS+=(--shape-checkpoint "$STUDY/prior.pt");;
    *) exit 2;;
esac
export SURFACE_CUFFT_COMPAT=1 SURFACE_METRICS_ENV=1 PYTHONNOUSERSITE=1
case "$STAGE" in
    cohort)
        if [[ "$ARM" != baseline ]]; then
            "$PYTHON" scripts/sync_annotated_references.py --source "${BASE}-cohort" --destination "$OUT/$ARM" --receipt "$OUT/$ARM-refs-cohort.json"
        fi
        ARGS+=(--target-manifest "$DESIGN/cohort.json")
        bash scripts/run_surface_progressive_slurm.sh "${ARGS[@]}"
        "$PYTHON" scripts/snapshot_annotated_surface.py --run-dir "$OUT/$ARM" --output-dir "$OUT/$ARM-cohort" --dataset-root datasets/open_world_amodal --target-manifest "$DESIGN/cohort.json" > "$OUT/$ARM-cohort-verification.json"
        ;;
    full)
        [[ "$ARM" == shape || "$ARM" == combined ]] || { echo "Use the registered historical controls for the full study" >&2; exit 2; }
        if [[ "$ARM" == combined ]]; then
            "$PYTHON" scripts/sync_annotated_references.py --source "$OUT/shape" --destination "$OUT/$ARM" --receipt "$OUT/$ARM-refs-full-$SLURM_JOB_ID.json"
        fi
        bash scripts/run_surface_progressive_slurm.sh "${ARGS[@]}"
        "$PYTHON" -m amodal_benchmark verify --run-dir "$OUT/$ARM" --dataset-root datasets/open_world_amodal --require-full --require-provenance > "$OUT/$ARM-full-verification.json"
        "$PYTHON" -m amodal_benchmark register --run-dir "$OUT/$ARM" --dataset-root datasets/open_world_amodal --registry "$OUT/$ARM-registry.csv" --hypothesis "Progressive $ARM geometry-guided selection on annotated vocabulary"
        ;;
    *) exit 2;;
esac


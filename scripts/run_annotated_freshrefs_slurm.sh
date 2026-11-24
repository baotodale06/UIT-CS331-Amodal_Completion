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
[[ "${1:-}" == --stage && "$#" -ge 2 ]] || exit 2
STAGE="$2"
shift 2
DESIGN=output/surface-annotated-20261112-design
STUDY=output/surface-annotated-20261112-freshrefs
PYTHON="$PWD/.amodal-cache/surface-metrics-env/bin/python"
export SURFACE_CUFFT_COMPAT=1 SURFACE_METRICS_ENV=1 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=8
"$PYTHON" scripts/check_annotated_surface_freeze.py --design "$DESIGN"
"$PYTHON" scripts/check_annotated_freshrefs.py
if [[ "$STAGE" == baseline-cohort ]]; then
    bash scripts/run_surface_progressive_slurm.sh --benchmark-workers 4 --detector-vocabulary annotated \
        --output-dir "$STUDY/baseline" --target-manifest "$DESIGN/cohort.json"
    "$PYTHON" scripts/snapshot_annotated_surface.py --run-dir "$STUDY/baseline" \
        --output-dir "$STUDY/baseline-cohort" --dataset-root datasets/open_world_amodal \
        --target-manifest "$DESIGN/cohort.json" > "$DESIGN/freshrefs-baseline-cohort-verification.json"
    "$PYTHON" scripts/sync_annotated_references.py --source "$STUDY/baseline-cohort" \
        --destination "$STUDY/surface" --receipt "$DESIGN/freshrefs-cohort-reference-copy.json"
elif [[ "$STAGE" == surface-cohort ]]; then
    bash scripts/run_surface_progressive_slurm.sh --benchmark-workers 4 --detector-vocabulary annotated \
        --surface-freeze "$DESIGN/method-freeze.json" --output-dir "$STUDY/surface" \
        --target-manifest "$DESIGN/cohort.json"
    "$PYTHON" scripts/snapshot_annotated_surface.py --run-dir "$STUDY/surface" \
        --output-dir "$STUDY/surface-cohort" --dataset-root datasets/open_world_amodal \
        --target-manifest "$DESIGN/cohort.json" > "$DESIGN/freshrefs-surface-cohort-verification.json"
    "$PYTHON" scripts/compare_surface_progressive.py --baseline "$STUDY/baseline-cohort" \
        --surface "$STUDY/surface-cohort" --dataset-root datasets/open_world_amodal \
        --target-manifest "$DESIGN/cohort.json" --output-dir "$STUDY/cohort-comparison"
elif [[ "$STAGE" == baseline-full ]]; then
    bash scripts/run_surface_progressive_slurm.sh --benchmark-workers 4 --detector-vocabulary annotated \
        --output-dir "$STUDY/baseline"
    "$PYTHON" -m amodal_benchmark verify --run-dir "$STUDY/baseline" \
        --dataset-root datasets/open_world_amodal --require-full --require-provenance \
        > "$DESIGN/freshrefs-baseline-full-verification.json"
elif [[ "$STAGE" == surface-full || "$STAGE" == surface-resume ]]; then
    if [[ "$STAGE" == surface-resume ]]; then
        [[ "${1:-}" == --previous-job && "$#" == 2 ]] || exit 2
        PREVIOUS_JOB="$2"
        PREVIOUS_STATE="$(sacct -j "$PREVIOUS_JOB" -X -n -P --format=State | head -1 | cut -d '|' -f 1)"
        if [[ "$PREVIOUS_STATE" == COMPLETED ]]; then
            "$PYTHON" -m amodal_benchmark verify --run-dir "$STUDY/surface" \
                --dataset-root datasets/open_world_amodal --require-full --require-provenance
            exit 0
        fi
        [[ "$PREVIOUS_STATE" == TIMEOUT || "$PREVIOUS_STATE" == PREEMPTED || "$PREVIOUS_STATE" == NODE_FAIL ]] || {
            echo "Inspect failed inference before resuming: $PREVIOUS_STATE" >&2; exit 2;
        }
    fi
    "$PYTHON" scripts/sync_annotated_references.py --source "$STUDY/baseline" \
        --destination "$STUDY/surface" --receipt "$DESIGN/freshrefs-full-reference-copy-${SLURM_JOB_ID}.json"
    bash scripts/run_surface_progressive_slurm.sh --benchmark-workers 4 --detector-vocabulary annotated \
        --surface-freeze "$DESIGN/method-freeze.json" --output-dir "$STUDY/surface"
    "$PYTHON" -m amodal_benchmark verify --run-dir "$STUDY/surface" \
        --dataset-root datasets/open_world_amodal --require-full --require-provenance \
        > "$DESIGN/freshrefs-surface-full-verification.json"
elif [[ "$STAGE" == pool ]]; then
    FREEZE="$DESIGN/pool-freshrefs-freeze.json"
    RESULT="$STUDY/pool-diagnostic"
    "$PYTHON" scripts/measure_annotated_surface_pools.py --freeze "$FREEZE" --output-dir "$RESULT" --preflight
    bash scripts/run_surface_followup_slurm.sh --benchmark-workers 4 \
        --script scripts/measure_annotated_surface_pools.py --log-dir "$DESIGN/freshrefs-pool-worker-logs" \
        --freeze "$FREEZE" --output-dir "$RESULT"
    "$PYTHON" scripts/measure_annotated_surface_pools.py --freeze "$FREEZE" --output-dir "$RESULT" --summarize
else
    echo "Unknown stage: $STAGE" >&2
    exit 2
fi

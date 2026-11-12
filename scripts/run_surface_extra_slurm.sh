#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=03:00:00
set -Eeuo pipefail
cd -- "${SLURM_SUBMIT_DIR:?Submit from project root}"
[[ "${1:-}" == --benchmark-workers && "${2:-}" == 4 ]] || exit 2
DESIGN=output/surface-followup-20261110-design
export SURFACE_CUFFT_COMPAT=1
for COHORT in 96 87 166; do
    case "$COHORT" in
        96) EXPERIMENT=output/surface-consistency-20261108 ;;
        87) EXPERIMENT=output/surface-consistency-replication87-20261108 ;;
        166) EXPERIMENT=output/surface-followup-20261110 ;;
    esac
    EXTRA="$DESIGN/extra-pools-$COHORT"
    bash scripts/run_surface_followup_slurm.sh --benchmark-workers 4 \
        --script scripts/run_surface_candidates.py --log-dir "$DESIGN/extra-$COHORT-generation-logs" \
        --experiment-dir "$EXTRA" --output-dir "$EXTRA/full" --repo-dir .amodal-cache/amodal
    SURFACE_METRICS_ENV=1 bash scripts/run_surface_followup_slurm.sh --benchmark-workers 4 \
        --script scripts/extract_surface_deep_features.py --log-dir "$DESIGN/extra-$COHORT-features-logs" \
        --experiment-dir "$EXPERIMENT" --run-dir "$EXPERIMENT/full" --extra-run-dir "$EXTRA/full" \
        --output-dir "$DESIGN/eight/dev-features-$COHORT" --dreamsim-cache .amodal-cache/dreamsim \
        --candidate-seeds 11 22 33 44 55 66 77 88
    if [[ "$COHORT" != 166 ]]; then
        SURFACE_METRICS_ENV=1 bash scripts/run_surface_followup_slurm.sh --benchmark-workers 4 \
            --script scripts/measure_surface_followup.py --log-dir "$DESIGN/extra-$COHORT-metrics-logs" \
            --experiment-dir "$EXPERIMENT" --run-dir "$EXPERIMENT/full" --extra-run-dir "$EXTRA/full" \
            --freeze "$DESIGN/provisional-development-eight.json" \
            --output-dir "$DESIGN/eight/dev-metrics-$COHORT" --dreamsim-cache .amodal-cache/dreamsim \
            --development-review
    fi
done

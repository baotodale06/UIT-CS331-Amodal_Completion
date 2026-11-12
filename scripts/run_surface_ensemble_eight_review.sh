#!/usr/bin/env bash
set -Eeuo pipefail
cd -- "${SLURM_SUBMIT_DIR:?Run inside the study allocation}"
DESIGN=output/surface-followup-20261110-design
export SURFACE_METRICS_ENV=1
for COHORT in 96 87; do
    case "$COHORT" in
        96) EXPERIMENT=output/surface-consistency-20261108 ;;
        87) EXPERIMENT=output/surface-consistency-replication87-20261108 ;;
    esac
    EXTRA="$DESIGN/extra-pools-$COHORT/full"
    EXPECTED=$((COHORT * 4))
    while [[ "$(find "$EXTRA/candidates" -name 'seed-*.json' 2>/dev/null | wc -l)" -lt "$EXPECTED" ]]; do
        sleep 15
    done
    bash scripts/run_surface_followup_slurm.sh --benchmark-workers 4 \
        --script scripts/measure_surface_ensemble_development.py \
        --log-dir "$DESIGN/ensemble-eight-$COHORT-logs" \
        --experiment-dir "$EXPERIMENT" --run-dir "$EXPERIMENT/full" --extra-run-dir "$EXTRA" \
        --search "$DESIGN/model_search.json" --configs "$DESIGN/ensemble-metric-configs.json" \
        --output-dir "$DESIGN/ensemble-eight-$COHORT" --dreamsim-cache .amodal-cache/dreamsim \
        --candidate-seeds 11 22 33 44 55 66 77 88 --all-candidates
done

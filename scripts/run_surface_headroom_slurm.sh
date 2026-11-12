#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
set -Eeuo pipefail
cd -- "${SLURM_SUBMIT_DIR:?Submit from project root}"
[[ "${1:-}" == --benchmark-workers && "${2:-}" == 4 ]] || exit 2
DESIGN=output/surface-candidate-headroom-20261111-design
RUN=output/surface-candidate-headroom-20261111
bash scripts/run_surface_followup_slurm.sh --benchmark-workers 4 \
    --script scripts/measure_surface_candidate_headroom.py --log-dir "$DESIGN/worker-logs" \
    --freeze "$DESIGN/freeze.json" --output-dir "$RUN"
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=2
.amodal-cache/surface-metrics-env/bin/python scripts/measure_surface_candidate_headroom.py \
    --summarize --freeze "$DESIGN/freeze.json" --output-dir "$RUN"

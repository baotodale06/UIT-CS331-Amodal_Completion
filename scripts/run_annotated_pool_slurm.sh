#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
set -Eeuo pipefail
cd -- "${SLURM_SUBMIT_DIR:?Submit from project root}"
[[ "${1:-}" == --benchmark-workers && "${2:-}" == 4 && "$#" == 2 ]] || exit 2
DESIGN=output/surface-annotated-20261112-design
RESULT=output/surface-annotated-20261112/pool-diagnostic
PYTHON="$PWD/.amodal-cache/surface-metrics-env/bin/python"
export SURFACE_CUFFT_COMPAT=1 SURFACE_METRICS_ENV=1 PYTHONNOUSERSITE=1
"$PYTHON" scripts/measure_annotated_surface_pools.py --freeze "$DESIGN/pool-freeze.json" \
    --output-dir "$RESULT" --preflight
bash scripts/run_surface_followup_slurm.sh --benchmark-workers 4 \
    --script scripts/measure_annotated_surface_pools.py --log-dir "$DESIGN/pool-worker-logs" \
    --freeze "$DESIGN/pool-freeze.json" --output-dir "$RESULT"
"$PYTHON" scripts/measure_annotated_surface_pools.py --freeze "$DESIGN/pool-freeze.json" \
    --output-dir "$RESULT" --summarize

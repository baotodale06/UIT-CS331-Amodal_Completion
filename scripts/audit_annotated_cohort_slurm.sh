#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=02:00:00
set -Eeuo pipefail
cd -- "${SLURM_SUBMIT_DIR:?Submit from the project root}"
[[ "${1:-}" == --phase && "$#" == 2 ]] || exit 2
PHASE="$2"
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4
PYTHON="$PWD/.amodal-cache/surface-metrics-env/bin/python"
"$PYTHON" scripts/check_annotated_surface_freeze.py --design output/surface-annotated-20261112-design
"$PYTHON" scripts/check_annotated_freshrefs.py
if [[ "$PHASE" == cohort ]]; then
    "$PYTHON" scripts/audit_annotated_surface.py
    "$PYTHON" scripts/export_annotated_review_assets.py
elif [[ "$PHASE" == pool ]]; then
    "$PYTHON" scripts/audit_annotated_surface_pools.py
else
    echo "Unknown phase: $PHASE" >&2
    exit 2
fi

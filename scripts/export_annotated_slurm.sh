#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
set -Eeuo pipefail
cd -- "${SLURM_SUBMIT_DIR:?Submit from project root}"
[[ "${1:-}" == --phase && "$#" == 2 ]] || exit 2
PHASE="$2"
.amodal-cache/conda-env/bin/python scripts/export_annotated_surface.py --phase "$PHASE" --metadata-only
if [[ "$PHASE" == cohort ]]; then
    .amodal-cache/conda-env/bin/python scripts/export_annotated_surface.py --phase "$PHASE"
fi

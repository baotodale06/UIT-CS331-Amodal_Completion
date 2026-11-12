#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
set -Eeuo pipefail
cd -- "${SLURM_SUBMIT_DIR:?Submit from the project root}"
export PYTHONNOUSERSITE=1
.amodal-cache/conda-env/bin/python scripts/compare_surface_progressive.py \
    --baseline output/pdmc-matched-control-20261110 \
    --surface output/pdmc-surface-progressive-20261110 \
    --dataset-root datasets/open_world_amodal \
    --output-dir output/surface-followup-20261110/progressive-comparison \
    --registry output/surface-followup-20261110-design/progressive-experiments.csv

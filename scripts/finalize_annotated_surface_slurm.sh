#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=01:00:00
set -Eeuo pipefail
cd -- "${SLURM_SUBMIT_DIR:?Submit from the project root}"
export PYTHONNOUSERSITE=1
PYTHON="$PWD/.amodal-cache/conda-env/bin/python"
STUDY=output/surface-annotated-20261112
DESIGN=output/surface-annotated-20261112-design
"$PYTHON" scripts/check_annotated_surface_freeze.py --design "$DESIGN"
"$PYTHON" scripts/compare_surface_progressive.py --baseline "$STUDY/baseline" \
    --surface "$STUDY/surface" --dataset-root datasets/open_world_amodal \
    --output-dir "$STUDY/full-comparison" --registry "$DESIGN/experiments.csv"

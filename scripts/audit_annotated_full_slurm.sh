#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=01:00:00
set -Eeuo pipefail
cd -- "${SLURM_SUBMIT_DIR:?Submit from the project root}"
export PYTHONNOUSERSITE=1
.amodal-cache/conda-env/bin/python scripts/check_annotated_surface_freeze.py \
    --design output/surface-annotated-20261112-design
.amodal-cache/conda-env/bin/python scripts/check_annotated_freshrefs.py
.amodal-cache/conda-env/bin/python scripts/audit_annotated_full.py

#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=00:20:00
set -Eeuo pipefail
cd -- "${SLURM_SUBMIT_DIR:?Submit from the project directory}"
ENV_DIR="$PWD/.amodal-cache/conda-env"
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=2
export LD_LIBRARY_PATH="$ENV_DIR/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
unset PYTHONHOME PYTHONPATH
PILOT=output/surface-consistency-20261108
REPLICATION=output/surface-consistency-replication87-20261108
"$ENV_DIR/bin/python" scripts/validate_surface_experiment.py \
    --experiment-dir "$PILOT" --run-dir "$PILOT/full" \
    --dataset-root datasets/open_world_amodal \
    --reference-run output/pdmc-baseline-provenance-20261105 \
    --smoke-dir "$PILOT/smoke" --workers 4
"$ENV_DIR/bin/python" scripts/validate_surface_experiment.py \
    --experiment-dir "$REPLICATION" --run-dir "$REPLICATION/full" \
    --dataset-root datasets/open_world_amodal \
    --reference-run output/pdmc-baseline-provenance-20261105 \
    --freeze-from "$PILOT/analysis/development_freeze.json" --workers 4
"$ENV_DIR/bin/python" scripts/explain_surface_results.py \
    --pilot "$PILOT" --replication "$REPLICATION" \
    --output-dir "$PILOT/diagnostics" --include-truth-check

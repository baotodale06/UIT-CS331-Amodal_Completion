#!/usr/bin/env bash
# Match the candidate generator's numerical CPU path for exact scientific replay.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
set -Eeuo pipefail
cd -- "${SLURM_SUBMIT_DIR:?Submit from the project root}"
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=2
unset PYTHONHOME PYTHONPATH
PYTHON="$PWD/.amodal-cache/surface-metrics-env/bin/python"
EXPERIMENT=output/surface-followup-20261110
DESIGN=output/surface-followup-20261110-design
"$PYTHON" scripts/diagnose_surface_replay_runtime.py
"$PYTHON" scripts/validate_surface_followup.py --experiment-dir "$EXPERIMENT" \
    --run-dir "$EXPERIMENT/full" --extra-run-dir "$DESIGN/extra-pools-166/full" \
    --metrics-dir "$EXPERIMENT/metrics-eight" --inference-dir "$EXPERIMENT/inference-eight" \
    --features-dir "$DESIGN/eight/dev-features-166" --freeze "$DESIGN/final-method-freeze.json" \
    --output "$EXPERIMENT/analysis-eight/replay-verification.json" --workers 4

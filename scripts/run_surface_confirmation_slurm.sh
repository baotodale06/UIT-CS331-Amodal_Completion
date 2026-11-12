#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
set -Eeuo pipefail
cd -- "${SLURM_SUBMIT_DIR:?Submit from the project root}"
[[ "${1:-}" == --benchmark-workers && "${2:-}" == 4 && "$#" == 2 ]] || exit 2
EXPERIMENT=output/surface-confirmation128-20261110
DESIGN="${EXPERIMENT}-design"
FREEZE="$DESIGN/selection-only-freeze.json"
PYTHON="$PWD/.amodal-cache/surface-metrics-env/bin/python"
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=2 SURFACE_CUFFT_COMPAT=1
unset PYTHONHOME PYTHONPATH
"$PYTHON" scripts/finalize_surface_confirmation.py --experiment-dir "$EXPERIMENT" --freeze "$FREEZE" --preflight

SURFACE_METRICS_ENV=0 SURFACE_VALIDATE_RUNTIME=1 bash scripts/run_surface_followup_slurm.sh \
    --benchmark-workers 4 --script scripts/run_surface_candidates.py --log-dir "$DESIGN/generation-logs" \
    --experiment-dir "$EXPERIMENT" --output-dir "$EXPERIMENT/full" --repo-dir .amodal-cache/amodal
SURFACE_METRICS_ENV=1 bash scripts/run_surface_followup_slurm.sh \
    --benchmark-workers 4 --script scripts/extract_surface_deep_features.py --log-dir "$DESIGN/features-logs" \
    --experiment-dir "$EXPERIMENT" --run-dir "$EXPERIMENT/full" --output-dir "$EXPERIMENT/features" \
    --dreamsim-cache .amodal-cache/dreamsim --candidate-seeds 11 22 33 44 55 66 77 88
if [[ ! -e "$EXPERIMENT/inference" ]]; then
    "$PYTHON" scripts/select_surface_pool.py --experiment-dir "$EXPERIMENT" --run-dir "$EXPERIMENT/full" \
        --freeze "$FREEZE" --features-dir "$EXPERIMENT/features" --output-dir "$EXPERIMENT/inference"
else
    "$PYTHON" - "$EXPERIMENT/inference/contract.json" "$FREEZE" <<'PY'
import hashlib
import json
from pathlib import Path
import sys
record = json.loads(Path(sys.argv[1]).read_text())
assert record.get('finished_at'), 'Interrupted selection needs repair before metrics'
assert record['freeze_sha256'] == hashlib.sha256(Path(sys.argv[2]).read_bytes()).hexdigest()
PY
fi
SURFACE_METRICS_ENV=1 bash scripts/run_surface_followup_slurm.sh \
    --benchmark-workers 4 --script scripts/measure_surface_followup.py --log-dir "$DESIGN/metric-logs" \
    --experiment-dir "$EXPERIMENT" --run-dir "$EXPERIMENT/full" --freeze "$FREEZE" \
    --features-dir "$EXPERIMENT/features" --output-dir "$EXPERIMENT/metrics" --dreamsim-cache .amodal-cache/dreamsim
"$PYTHON" scripts/analyze_surface_followup.py --experiment-dir "$EXPERIMENT" \
    --metrics-dir "$EXPERIMENT/metrics" --freeze "$FREEZE" --output-dir "$EXPERIMENT/analysis"
# Replay runs on the generation node with its original numerical CPU dispatch.
"$PYTHON" scripts/validate_surface_followup.py --experiment-dir "$EXPERIMENT" \
    --run-dir "$EXPERIMENT/full" --extra-run-dir "$EXPERIMENT/full" \
    --metrics-dir "$EXPERIMENT/metrics" --inference-dir "$EXPERIMENT/inference" \
    --features-dir "$EXPERIMENT/features" --freeze "$FREEZE" \
    --output "$EXPERIMENT/analysis/replay-verification.json" --workers 4
"$PYTHON" scripts/finalize_surface_confirmation.py --experiment-dir "$EXPERIMENT" --freeze "$FREEZE"
tar -czf "$DESIGN/confirmation-evidence.tar.gz" \
    "$EXPERIMENT/analysis" "$EXPERIMENT/metrics" "$EXPERIMENT/inference" \
    "$EXPERIMENT/features" "$EXPERIMENT/selection-only" \
    "$EXPERIMENT/cases.json" "$EXPERIMENT/protocol.json" "$EXPERIMENT/preparation.json" \
    "$DESIGN/selection-only-freeze.json" "$DESIGN/frozen-source" \
    "$DESIGN/generation-logs" "$DESIGN/features-logs" "$DESIGN/metric-logs"
sha256sum "$DESIGN/confirmation-evidence.tar.gz" > "$DESIGN/confirmation-evidence.tar.gz.sha256"
echo 'Independent selection-only confirmation completed and exactly replayed.'

#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
set -Eeuo pipefail
cd -- "${SLURM_SUBMIT_DIR:?Submit from the project root}"
[[ "${1:-}" == --benchmark-workers && "${2:-}" == 4 ]] || exit 2
shift 2
[[ "${1:-}" == --freeze && -f "${2:-}" ]] || exit 2
FREEZE="$2"; shift 2
[[ "$#" == 0 ]] || exit 2
DESIGN=output/surface-followup-20261110-design
EXPERIMENT=output/surface-followup-20261110
PYTHON="$PWD/.amodal-cache/surface-metrics-env/bin/python"
export SURFACE_CUFFT_COMPAT=1 SURFACE_METRICS_ENV=1 OMP_NUM_THREADS=2 PYTHONNOUSERSITE=1

if [[ "${SURFACE_PROGRESSIVE_ONLY:-0}" == 1 ]]; then
    # A completed exact replay on the original numerical runtime permits the
    # progressive stages to resume without rewriting scored evidence/timestamps.
    "$PYTHON" - "$FREEZE" "$EXPERIMENT" <<'PY'
import hashlib
import json
from pathlib import Path
import sys
from amodal_benchmark.surface_policy import verify_frozen_surface_sources
freeze_path, experiment = map(Path, sys.argv[1:])
freeze = json.loads(freeze_path.read_text())
verify_frozen_surface_sources(freeze)
replay = json.loads((experiment / 'analysis-eight/replay-verification.json').read_text())
cases = json.loads((experiment / 'cases.json').read_text())
assert replay['passed'] and replay['images'] == len(cases)
assert replay['freeze_sha256'] == hashlib.sha256(freeze_path.read_bytes()).hexdigest()
assert {case['case_id'] for case in replay['cases']} == {case['case_id'] for case in cases}
PY
else

# Publish all observation-only choices before opening any fresh truth metrics.
if [[ ! -e "$EXPERIMENT/inference-eight" ]]; then
    "$PYTHON" scripts/select_surface_pool.py \
        --experiment-dir "$EXPERIMENT" --run-dir "$EXPERIMENT/full" \
        --extra-run-dir "$DESIGN/extra-pools-166/full" --freeze "$FREEZE" \
        --features-dir "$DESIGN/eight/dev-features-166" \
        --output-dir "$EXPERIMENT/inference-eight"
else
    "$PYTHON" - "$EXPERIMENT/inference-eight/contract.json" "$FREEZE" <<'PY'
import hashlib
import json
from pathlib import Path
import sys
record = json.loads(Path(sys.argv[1]).read_text())
assert record.get('finished_at'), 'Finish interrupted observation-only selection before resuming'
assert record['freeze_sha256'] == hashlib.sha256(Path(sys.argv[2]).read_bytes()).hexdigest()
PY
fi

bash scripts/run_surface_followup_slurm.sh --benchmark-workers 4 \
    --script scripts/measure_surface_followup.py --log-dir "$DESIGN/fresh-eight-metric-logs" \
    --experiment-dir "$EXPERIMENT" --run-dir "$EXPERIMENT/full" \
    --extra-run-dir "$DESIGN/extra-pools-166/full" --freeze "$FREEZE" \
    --features-dir "$DESIGN/eight/dev-features-166" \
    --output-dir "$EXPERIMENT/metrics-eight" --dreamsim-cache .amodal-cache/dreamsim

"$PYTHON" scripts/analyze_surface_followup.py --experiment-dir "$EXPERIMENT" \
    --metrics-dir "$EXPERIMENT/metrics-eight" --freeze "$FREEZE" \
    --output-dir "$EXPERIMENT/analysis-eight"
"$PYTHON" scripts/validate_surface_followup.py --experiment-dir "$EXPERIMENT" \
    --run-dir "$EXPERIMENT/full" --extra-run-dir "$DESIGN/extra-pools-166/full" \
    --metrics-dir "$EXPERIMENT/metrics-eight" --inference-dir "$EXPERIMENT/inference-eight" \
    --features-dir "$DESIGN/eight/dev-features-166" --freeze "$FREEZE" \
    --output "$EXPERIMENT/analysis-eight/replay-verification.json" --workers 4
fi

# The finalized eight-candidate method must pass an actual progressive smoke.
SMOKE=output/pdmc-surface-progressive-final-smoke-20261110
bash scripts/run_surface_progressive_slurm.sh --benchmark-workers 4 \
    --surface-freeze "$FREEZE" --output-dir "$SMOKE" --limit 8
"$PYTHON" -m amodal_benchmark verify --run-dir "$SMOKE" \
    --dataset-root datasets/open_world_amodal --require-provenance \
    > "$DESIGN/final-progressive-smoke-verification.json"
"$PYTHON" - "$SMOKE" <<'PY'
import json
from pathlib import Path
import sys
run = Path(sys.argv[1])
records = [json.loads(path.read_text()) for path in run.glob('targets/*/*/result.json')]
if len(records) != 8 or not any(row['status'] == 'success' for row in records):
    raise RuntimeError('Eight-target smoke must exercise a saved progressive completion')
if not list(run.glob('targets/*/*/surface/*/decision.json')):
    raise RuntimeError('Smoke did not execute Surface-PMC sampling')
PY

if [[ "${SURFACE_CONTROLLED_ONLY:-0}" == 1 ]]; then
    echo 'Controlled evaluation and final progressive smoke complete.'
    exit 0
fi

# A matched-runtime baseline is essential: the accepted historical baseline
# used the original cuFFT. These two new runs use the same allocation,
# library override, metric environment and reference bytes.
BASELINE=output/pdmc-matched-control-20261110
SURFACE=output/pdmc-surface-progressive-20261110
bash scripts/run_surface_progressive_slurm.sh --benchmark-workers 4 \
    --output-dir "$BASELINE"
"$PYTHON" -m amodal_benchmark verify --run-dir "$BASELINE" \
    --dataset-root datasets/open_world_amodal --require-full --require-provenance \
    > "$DESIGN/matched-control-verification.json"
if [[ ! -e "$SURFACE" ]]; then
    mkdir -p "$SURFACE"
    cp -a "$BASELINE/visible_references" "$SURFACE/visible_references"
fi
bash scripts/run_surface_progressive_slurm.sh --benchmark-workers 4 \
    --surface-freeze "$FREEZE" --output-dir "$SURFACE"
"$PYTHON" -m amodal_benchmark verify --run-dir "$SURFACE" \
    --dataset-root datasets/open_world_amodal --require-full --require-provenance \
    > "$DESIGN/full-progressive-verification.json"
echo 'Controlled evaluation, CPU replay, progressive smoke and both full runs complete.'

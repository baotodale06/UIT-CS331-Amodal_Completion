#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
set -Eeuo pipefail
cd "$SLURM_SUBMIT_DIR"
OUT=output/shape-20261121
export SURFACE_CUFFT_COMPAT=1 SURFACE_METRICS_ENV=1 SURFACE_VALIDATE_RUNTIME=1
bash scripts/run_surface_followup_slurm.sh "$@" --script scripts/shape_oracle.py --log-dir "$OUT/oracle-logs"
PYTHON="$PWD/.amodal-cache/surface-metrics-env/bin/python"
"$PYTHON" - <<'PY'
import json
from pathlib import Path
folder = Path('output/shape-20261121')
rows = [r for i in range(4) for r in json.loads((folder/f'dev-oracle-{i}.json').read_text())['cases']]
assert len({r['target'] for r in rows}) == len(rows) == 183
gain = sum(max(r['scores']) - sum(r['scores'])/4 for r in rows) / len(rows)
print('Development oracle IoU gain over uniform K=4', gain)
if gain <= 0:
    raise ValueError('No geometry headroom on development')
PY
export LD_PRELOAD="$PWD/.amodal-cache/surface-cufft-cu118/nvidia/cufft/lib/libcufft.so.10"
export OMP_NUM_THREADS=2 PYTHONNOUSERSITE=1
"$PYTHON" -u scripts/train_shape.py


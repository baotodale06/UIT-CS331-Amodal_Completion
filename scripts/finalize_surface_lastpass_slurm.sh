#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
set -Eeuo pipefail
cd -- "${SLURM_SUBMIT_DIR:?Submit from the project root}"
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=2
.amodal-cache/surface-metrics-env/bin/python scripts/summarize_surface_lastpass_ablation.py \
    --run-dir output/surface-lastpass-ablation-20261111 \
    --freeze output/surface-lastpass-ablation-20261111-design/freeze.json
python3 - <<'PY'
from pathlib import Path
import hashlib
import json
import tarfile
root = Path('.')
run = root/'output/surface-lastpass-ablation-20261111'
design = root/'output/surface-lastpass-ablation-20261111-design'
files = [*run.glob('targets/*/*/ablation.json'), *run.glob('runtime-worker-*.json'),
         *(p for p in (run/'analysis').rglob('*') if p.is_file()),
         design/'freeze.json', design/'preflight.json']
manifest = {p.as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}
archive = design/'evidence.tar.gz'
if archive.exists():
    raise ValueError('Preserve existing diagnostic evidence archive')
with tarfile.open(archive, 'w:gz') as bundle:
    for p in sorted(files):
        bundle.add(p, arcname=p.as_posix(), recursive=False)
(design/'evidence-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
(design/'evidence.tar.gz.sha256').write_text(hashlib.sha256(archive.read_bytes()).hexdigest()+'\n')
PY

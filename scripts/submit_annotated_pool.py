#!/usr/bin/env python3
"""Insert the pre-specified cohort diagnostic before full Surface-PMC continuation."""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from amodal_benchmark.dataset import sha256, write_json

design = ROOT / 'output/surface-annotated-20261112-design'
submission = json.loads((design / 'submission.json').read_text())
jobs = {row['name']: row['job_id'] for row in submission['jobs']}
destination = design / 'pool-submission.json'
if destination.exists():
    raise ValueError('Diagnostic submission already exists')
freeze = json.loads((design / 'pool-freeze.json').read_text())
for name, expected in freeze['files'].items():
    if sha256(ROOT / name) != expected:
        raise ValueError('Diagnostic source drift: ' + name)
pools = list((ROOT / 'output/surface-annotated-20261112/surface').glob('targets/*/*/surface/*/decision.json'))
if pools:
    raise ValueError('Pre-generation diagnostic freeze requires zero generated pools')
command = ['sbatch', '--parsable', '--partition=debug-gpu', '--gres=gpu:4',
           '--job-name=ann-pool-diagnostic', '--dependency=afterok:' + jobs['ann-surface-cohort'],
           '--output=' + str(design / 'ann-pool-diagnostic-%j.log'),
           'scripts/run_annotated_pool_slurm.sh', '--benchmark-workers', '4']
job = subprocess.check_output(command, cwd=ROOT, text=True).strip().split(';')[0]
if not job.isdigit():
    raise ValueError('Unexpected job ID')
update = ['scontrol', 'update', 'JobId=' + jobs['ann-surface-full'],
          'Dependency=afterok:' + jobs['ann-surface-cohort'] + ',afterany:' + job]
record = dict(submitted_at=datetime.now(timezone.utc).isoformat(), job_id=job, command=command,
              freeze_sha256=sha256(design / 'pool-freeze.json'), generated_pools_at_submission=0,
              full_surface_dependency_update=update)
write_json(destination, record)
subprocess.run(update, check=True)
record['dependency_updated'] = True
write_json(destination, record)
print(json.dumps(record, indent=2))

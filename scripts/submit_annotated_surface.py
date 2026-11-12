#!/usr/bin/env python3
"""Submit the frozen annotated experiment and record every exact Slurm command."""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from amodal_benchmark.dataset import sha256, write_json


def main():
    design = ROOT / 'output/surface-annotated-20261112-design'
    receipt = json.loads((design / 'reference-copy-receipt.json').read_text())
    if not receipt['verified']:
        raise ValueError('Reference setup must complete before submission')
    destination = design / 'submission.json'
    if destination.exists():
        raise ValueError('Submission already recorded; inspect existing jobs')
    record = dict(submitted_at=datetime.now(timezone.utc).isoformat(),
        study_freeze_sha256=sha256(design / 'study-freeze.json'), jobs=[],
        operational_sources={p.name: sha256(p) for p in (Path(__file__), ROOT / 'scripts/resume_annotated_surface_slurm.sh')})
    def submit(name, stage=None, dependency=None, script='scripts/run_annotated_surface_slurm.sh', extra=()):
        args = ['sbatch', '--parsable', '--partition=debug-cpu' if stage == 'finalizer' else '--partition=debug-gpu',
                '--job-name=' + name, '--output=' + str(design / (name + '-%j.log'))]
        if stage != 'finalizer':
            args += ['--gres=gpu:4']
        if stage == 'baseline-cohort':
            args += ['--time=03:00:00']
        elif stage == 'surface-cohort':
            args += ['--time=06:00:00']
        if dependency:
            args += ['--dependency=' + dependency]
        args += [script]
        if stage != 'finalizer':
            args += ['--benchmark-workers', '4']
        if stage not in (None, 'finalizer'):
            args += ['--stage', stage]
        args += list(extra)
        result = subprocess.run(args, cwd=ROOT, check=True, text=True, capture_output=True)
        job = result.stdout.strip().split(';')[0]
        if not job.isdigit():
            raise ValueError('Unexpected Slurm job ID: ' + result.stdout)
        record['jobs'].append(dict(name=name, stage=stage, job_id=job, command=args))
        write_json(destination, record)
        print(name, job, flush=True)
        return job
    control = submit('ann-control-cohort', 'baseline-cohort')
    surface = submit('ann-surface-cohort', 'surface-cohort', 'afterok:' + control)
    full_control = submit('ann-control-full', 'baseline-full', 'afterok:' + control)
    full_surface = submit('ann-surface-full', 'surface-full', 'afterok:' + surface)
    resume = submit('ann-surface-resume', dependency='afterany:' + full_surface,
        script='scripts/resume_annotated_surface_slurm.sh', extra=['--previous-job', full_surface])
    submit('ann-finalize', 'finalizer', 'afterok:' + full_control + ':' + resume,
           script='scripts/finalize_annotated_surface_slurm.sh')


if __name__ == '__main__':
    main()

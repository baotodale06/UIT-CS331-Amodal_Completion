#!/usr/bin/env python3
"""Freeze the reference-runtime amendment without altering the original design."""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from amodal_benchmark.dataset import sha256, write_json

design = ROOT / 'output/surface-annotated-20261112-design'
destination = design / 'fresh-reference-amendment.json'
if destination.exists():
    raise ValueError('Preserve the existing amendment')
pool = json.loads((design / 'pool-freeze.json').read_text())
pool.update(created_at=datetime.now(timezone.utc).isoformat(),
    parent_freeze_sha256=sha256(design / 'pool-freeze.json'),
    reference_amendment='docs/SURFACE_ANNOTATED_REFERENCE_AMENDMENT_20261112.md',
    baseline_run='output/surface-annotated-20261112-freshrefs/baseline-cohort',
    surface_run='output/surface-annotated-20261112-freshrefs/surface-cohort')
write_json(design / 'pool-freshrefs-freeze.json', pool)
paths = [ROOT / 'scripts' / name for name in (
    'sync_annotated_references.py', 'run_annotated_freshrefs_slurm.sh',
    'check_annotated_freshrefs.py', 'finalize_annotated_freshrefs_slurm.sh',
    'prepare_annotated_freshrefs.py')]
paths += [ROOT / 'docs/SURFACE_ANNOTATED_REFERENCE_AMENDMENT_20261112.md']
paths += [design / name for name in ('study-freeze.json', 'method-freeze.json', 'cohort.json', 'pool-freshrefs-freeze.json')]
write_json(destination, dict(at=datetime.now(timezone.utc).isoformat(),
    status='frozen_before_surface_generation_and_paired_analysis',
    parent_study_sha256=sha256(design / 'study-freeze.json'),
    study='output/surface-annotated-20261112-freshrefs',
    reason='Repeated historical SAM mask mismatch; one-pixel isolated reproduction drift',
    unchanged=['cohort', 'ranker', 'candidate_seeds', 'refinement', 'ensemble', 'inference', 'metrics', 'analysis'],
    reference_rule='Generate control references, copy exact bytes, require surface regeneration equality',
    files={path.relative_to(ROOT).as_posix(): sha256(path) for path in paths}))
print('Frozen fresh-reference amendment and unchanged secondary pool policies')

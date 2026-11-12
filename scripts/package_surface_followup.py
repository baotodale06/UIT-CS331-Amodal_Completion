#!/usr/bin/env python3
"""Create a verified portable report package without duplicating raw candidates."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from amodal_benchmark.dataset import sha256, write_json


def main():
    experiment = ROOT / 'output/surface-followup-20261110'
    design = ROOT / 'output/surface-followup-20261110-design'
    report = experiment / 'report'
    validation = json.loads((report / 'validation.json').read_text())
    assert validation['passed']
    assert validation['report_sha256'] == sha256(report / 'REPORT.md')
    assert validation['pdf_sha256'] == sha256(report / 'REPORT.pdf')
    freeze_path = design / 'final-method-freeze.json'
    freeze = json.loads(freeze_path.read_text())
    assert validation['freeze_sha256'] == sha256(freeze_path)
    paths = {path for path in report.rglob('*') if path.is_file()}
    for folder in (experiment / 'analysis-eight', experiment / 'inference-eight/decisions',
                   design / 'final-method-freeze-source'):
        paths.update(path for path in folder.rglob('*') if path.is_file() and '__pycache__' not in path.parts)
    paths.update((experiment / 'metrics-eight').glob('*.json'))
    paths.update((experiment / name) for name in ('cases.json', 'protocol.json',
                 'generation_manifest.json', 'inference-eight/contract.json', 'START_HERE.md'))
    paths.update((design / name) for name in ('final-method-freeze.json', 'eight-model-selection.json',
                 'paired_comparison.json', 'jobs.json', 'commands.sh',
                 'final-progressive-smoke-verification.json', 'progressive-status-delivery.json',
                 'fresh-controlled-download-receipt.json', 'fresh-four-download-receipt.json',
                 'fresh-extra-download-receipt.json', 'decoded-split-audit.json'))
    paths.update((ROOT / name) for name in ('docs/SURFACE_FOLLOWUP_20261110.md',
                 'cookbooks/surface-followup.md', 'scripts/validate_surface_followup.py',
                 'scripts/build_surface_followup_report.py', 'scripts/validate_surface_followup_report.py',
                 'scripts/package_surface_followup.py'))
    for name, expected in freeze['source_hashes'].items():
        assert sha256(ROOT / name) == expected
    manifest = {path.relative_to(ROOT).as_posix(): sha256(path) for path in sorted(paths)}
    archive_path = experiment / 'surface-followup-reading-package.zip'
    if archive_path.exists():
        raise ValueError('Preserve the existing delivery package; use a new name for a revision')
    with zipfile.ZipFile(archive_path, 'x', zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(paths):
            archive.write(path, path.relative_to(ROOT).as_posix())
        archive.writestr('MANIFEST.json', json.dumps(manifest, indent=2) + '\n')
        archive.writestr('README.md', '# Surface-PMC follow-up\n\nStart with '
            '[the handoff](output/surface-followup-20261110/START_HERE.md).\n\n'
            'This reading package includes every test-case panel, metrics, decisions, frozen source '
            'and verification evidence. Raw candidates, datasets, checkpoints and environments remain '
            'in the separately preserved project/cluster artifacts.\n')
    with zipfile.ZipFile(archive_path) as archive:
        assert archive.testzip() is None
        for name, expected in manifest.items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == expected
    receipt = dict(passed=True, created_at=datetime.now(timezone.utc).isoformat(),
        sha256=sha256(archive_path), bytes=archive_path.stat().st_size,
        verified_members=len(manifest), report_validation_sha256=sha256(report / 'validation.json'))
    write_json(experiment / 'reading-package-receipt.json', receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()

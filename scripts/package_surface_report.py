#!/usr/bin/env python3
"""Package the Surface-PMC reading evidence without duplicating raw experiments."""
import argparse
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot', type=Path, required=True)
    parser.add_argument('--replication', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Use a new delivery archive path')
    report = args.pilot / 'report'
    validation = json.loads((report / 'validation.json').read_text())
    if validation['status'] != 'passed' or validation['pdf_sha256'] != sha256(report / 'REPORT.pdf'):
        raise ValueError('Validate the final report before packaging')
    if validation['report_sha256'] != sha256(report / 'REPORT.md'):
        raise ValueError('Report text changed after validation')
    freeze = json.loads((args.pilot / 'analysis/development_freeze.json').read_text())
    for relative, digest in freeze['analysis_source_hashes'].items():
        if sha256(ROOT / relative) != digest:
            raise ValueError('Frozen scientific source changed: ' + relative)
    paths = {p for p in report.rglob('*') if p.is_file() and
             not any(part.startswith('layout-review') for part in p.parts)}
    for experiment in (args.pilot, args.replication):
        paths.update(experiment.glob('*.json'))
        for directory in ('analysis', 'diagnostics', 'cluster-diagnostics', 'historical-census',
                          'portability-windows-numpy254', 'counterexamples', 'source', 'perceptual', 'job-logs'):
            paths.update(p for p in (experiment / directory).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    paths.update(args.pilot.glob('START_HERE.md'))
    for pattern in ('docs/SURFACE*.md', 'docs/research/SURFACE*.md',
                    'cookbooks/surface-consistency.md', 'configs/research/surface*.json',
                    'scripts/*surface*.py', 'scripts/*surface*.sh', 'tests/test_surface*.py',
                    'amodal_benchmark/*.py'):
        paths.update(ROOT.glob(pattern))
    normalized = sorted({p.resolve() for p in paths})
    files = {p.relative_to(ROOT).as_posix(): sha256(p) for p in normalized}
    readme = ('# Surface consistency: report and evidence\n\n'
              f"Start with [{args.pilot.name}/START_HERE.md]({args.pilot.as_posix()}/START_HERE.md).\n\n"
              'This reading package contains the report, every held-out comparison panel, tables, '
              'frozen decisions, diagnostics, validation evidence and source snapshots. '
              'It excludes models, original datasets, environments and full-resolution raw experiment '
              'inputs/outputs. Those are preserved in the project and the separately verified cluster '
              'archives. The included source documents how the study was run; a complete raw replay '
              'requires those original artifacts.\n\n'
              'The primary full-surface result is inconclusive. The local variant has an exploratory '
              'pooled benefit, without established superiority to the other observable selectors.\n')
    manifest = dict(created_at=datetime.now(timezone.utc).isoformat(),
                    purpose='Portable reading evidence; full raw experiment archives are separate',
                    files=files, pdf_sha256=validation['pdf_sha256'],
                    freeze_sha256=sha256(args.pilot / 'analysis/development_freeze.json'))
    with zipfile.ZipFile(args.output, 'x', zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in normalized:
            archive.write(path, path.relative_to(ROOT).as_posix())
        archive.writestr('README.md', readme)
        archive.writestr('MANIFEST.json', json.dumps(manifest, indent=2) + '\n')
    with zipfile.ZipFile(args.output) as archive:
        if archive.testzip() is not None:
            raise ValueError('Archive CRC failure')
        for name, digest in files.items():
            if hashlib.sha256(archive.read(name)).hexdigest() != digest:
                raise ValueError('Archive member changed: ' + name)
        members = len(archive.namelist())
    receipt = dict(status='passed', created_at=datetime.now(timezone.utc).isoformat(),
                   archive=str(args.output), bytes=args.output.stat().st_size,
                   sha256=sha256(args.output), members=members, member_hash_checks=len(files),
                   crc_verified=True, pdf_sha256=validation['pdf_sha256'])
    write_json(args.output.with_suffix('.manifest.json'), manifest)
    write_json(args.output.with_suffix('.receipt.json'), receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()

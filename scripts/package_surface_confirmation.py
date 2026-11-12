#!/usr/bin/env python3
"""Create and verify a portable reading package for the completed confirmation."""
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
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Use a new archive path; preserve earlier deliveries')
    experiment = ROOT / 'output/surface-confirmation128-20261110'
    design = ROOT / 'output/surface-confirmation128-20261110-design'
    recovery = ROOT / 'output/surface-job-recovery-20261110'
    recovered = recovery / 'confirmation/output/surface-confirmation128-20261110'
    frozen = recovery / 'confirmation/output/surface-confirmation128-20261110-design'
    report = experiment / 'report'
    audit = json.loads((report / 'audit.json').read_text())
    if not audit['passed'] or audit['images'] != 128:
        raise ValueError('Complete successful audit required')
    report_hashes = json.loads((report / 'report-manifest.json').read_text())
    for relative, expected in report_hashes.items():
        if sha256(report / relative) != expected:
            raise ValueError('Report changed after audit: ' + relative)
    if sha256(frozen / 'selection-only-freeze.json') != audit['freeze_sha256']:
        raise ValueError('Confirmation policy differs from audited policy')
    files = {}

    def add(source, destination):
        if destination in files or not source.is_file():
            raise ValueError('Missing or duplicate package entry: ' + str(source))
        files[destination] = source

    def tree(source, destination):
        for path in sorted(source.rglob('*')):
            if path.is_file() and '__pycache__' not in path.parts:
                add(path, destination + '/' + path.relative_to(source).as_posix())

    tree(report, 'report')
    tree(recovered / 'analysis', 'evidence/analysis')
    tree(recovered / 'inference/decisions', 'evidence/inference/decisions')
    add(recovered / 'inference/contract.json', 'evidence/inference/contract.json')
    tree(frozen / 'frozen-source', 'reproducibility/frozen-source')
    for path in sorted((recovered / 'metrics').glob('*.json')):
        add(path, 'evidence/metrics/' + path.name)
    for name in ('cases.json', 'generation_manifest.json', 'protocol.json', 'preparation.json'):
        add(experiment / name, 'evidence/cohort/' + name)
    for name in ('selection-only-freeze.json', 'split-audit.json', 'submission.json'):
        add(design / name, 'reproducibility/' + name)
    for name in ('confirmation-receipt.json', 'confirmation-support-receipt.json'):
        add(recovery / name, 'evidence/' + name)
    add(ROOT / 'docs/SURFACE_CONFIRMATION_20261110.md', 'reproducibility/PROTOCOL_AND_RESULT.md')
    add(ROOT / 'scripts/audit_surface_confirmation_result.py', 'reproducibility/audit_surface_confirmation_result.py')
    add(Path(__file__).resolve(), 'reproducibility/package_surface_confirmation.py')
    readme = (
        '# Independent Surface-PMC confirmation\n\n'
        'Start with [the one-page PDF](report/REPORT.pdf), '
        '[full report](report/REPORT.md), or [all 128 cases](report/gallery.html). '
        'Extract the ZIP before opening the gallery.\n\n'
        'The frozen selector improves hidden RGB MAE 7.644%, object LPIPS 11.328%, '
        'DreamSim 16.766%, and CLIP cosine 4.068% versus expected uniform selection '
        'from the same eight candidates. All individual nominal 95% lower bounds '
        'exceed 3%. These are not simultaneous bounds. SSIM improves only 1.668%.\n\n'
        'This is independent one-step artificial-occlusion confirmation on 128 '
        'VG/COCOA images, with zero failures among 1,024 candidates. It is not '
        'the full progressive benchmark or original-paper benchmark. The full '
        'corrected progressive run remains separate. The original five-hour '
        'deadline was missed. No confirmation outcomes were used to refit the policy.\n\n'
        'The package includes every gallery case, metric records, comparisons, '
        'decisions, frozen policy/source, native replay receipt and local audit. '
        'The manifest hashes every payload file. Full-resolution inputs and generated '
        'outputs, model weights and environments remain in the project and verified '
        'recovery archives; they are required for raw replay. Original protocol '
        'source links retain their repository-relative paths.\n'
    ).encode('utf-8')
    hashes = {name: sha256(path) for name, path in sorted(files.items())}
    hashes['START_HERE.md'] = hashlib.sha256(readme).hexdigest()
    manifest = dict(created_at=datetime.now(timezone.utc).isoformat(),
        purpose='Portable reading evidence; complete raw replay needs separate native artifacts',
        freeze_sha256=audit['freeze_sha256'], files=hashes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, 'x', zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name, path in sorted(files.items()):
            archive.write(path, name)
        archive.writestr('START_HERE.md', readme)
        archive.writestr('MANIFEST.json', json.dumps(manifest, indent=2) + '\n')
    with zipfile.ZipFile(args.output) as archive:
        if archive.testzip() is not None:
            raise ValueError('Archive CRC failure')
        for name, expected in hashes.items():
            if hashlib.sha256(archive.read(name)).hexdigest() != expected:
                raise ValueError('Archive payload differs: ' + name)
        members = len(archive.namelist())
    receipt = dict(passed=True, created_at=datetime.now(timezone.utc).isoformat(),
        archive=str(args.output), sha256=sha256(args.output), bytes=args.output.stat().st_size,
        members=members, member_hash_checks=len(hashes), crc_verified=True,
        freeze_sha256=audit['freeze_sha256'])
    write_json(args.output.with_suffix('.receipt.json'), receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()

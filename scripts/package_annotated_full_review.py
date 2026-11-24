#!/usr/bin/env python3
"""Package the audited full results, original metadata archive and analysis source."""
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from amodal_benchmark.dataset import sha256, write_json
from review_annotated_full import OUTPUT, RECOVERY, DESIGN, read, require


def main():
    validation = read(OUTPUT/'validation.json')
    visual = read(OUTPUT/'visual-review.json')
    require(validation['passed'] and visual['passed'], 'Delivery reviews must pass')
    require(validation['report_sha256'] == sha256(OUTPUT/'REPORT.md'), 'Stale report validation')
    require(validation['pdf_sha256'] == sha256(OUTPUT/'REPORT.pdf') == visual['pdf_sha256'], 'Stale PDF review')
    archive = OUTPUT/'annotated-full-results.zip'
    files = {path.relative_to(OUTPUT).as_posix():path for path in OUTPUT.rglob('*') if path.is_file()
             and path.name not in ('annotated-full-results.zip','package-manifest.json','delivery-receipt.json')
             and 'pdf-review' not in path.relative_to(OUTPUT).parts}
    files['pdf-review/contact-sheet.png'] = OUTPUT/'pdf-review/contact-sheet.png'
    files['evidence/full-metadata.tar.gz'] = RECOVERY/'evidence.tar.gz'
    for name in ('recover_annotated_full_followup.py','review_annotated_full.py',
                 'build_annotated_full_report.py','validate_annotated_full_report.py',
                 'package_annotated_full_review.py','audit_surface_progressive_result.py',
                 'recover_annotated_surface.py','collect_surface_job_evidence.py'):
        files['analysis_source/scripts/'+name] = ROOT/'scripts'/name
    for path in (ROOT/'amodal_benchmark').glob('*.py'):
        files['analysis_source/amodal_benchmark/'+path.name] = path
    for name in ('active-study.json','freshrefs-submission.json','freshrefs-followup-submission.json',
                 'freshrefs-export-submission.json','fresh-reference-amendment.json','cohort.json'):
        files['evidence/design/'+name] = DESIGN/name
    manifest = {name:sha256(path) for name,path in sorted(files.items())}
    write_json(OUTPUT/'package-manifest.json',manifest)
    files['package-manifest.json'] = OUTPUT/'package-manifest.json'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as bundle:
        for name,path in sorted(files.items()):
            bundle.write(path,name)
    with zipfile.ZipFile(archive) as bundle:
        require(bundle.testzip() is None, 'Archive CRC failure')
        require(set(bundle.namelist()) == set(files), 'Archive membership differs')
        for name,expected in manifest.items():
            require(hashlib.sha256(bundle.read(name)).hexdigest() == expected, 'Archive payload differs: '+name)
    write_json(OUTPUT/'delivery-receipt.json',dict(verified=True,at=datetime.now(timezone.utc).isoformat(),
        archive=archive.name,bytes=archive.stat().st_size,sha256=sha256(archive),
        payload_files=len(manifest),manifest_sha256=sha256(OUTPUT/'package-manifest.json'),
        source_sha256=sha256(Path(__file__)),
        scope='Report, figure, all-target CSV, audits, registry/accounting, frozen design, analysis source and original full metadata archive. Full native PNGs remain on cluster.'))
    print(archive)
    print(sha256(archive))


if __name__ == '__main__':
    main()

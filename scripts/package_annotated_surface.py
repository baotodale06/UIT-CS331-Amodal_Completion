#!/usr/bin/env python3
"""Package audited annotated results with portable figures and provenance."""
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import shutil
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from amodal_benchmark.dataset import sha256, write_json


def main():
    design = ROOT/'output/surface-annotated-20261112-design'
    active = json.loads((design/'active-study.json').read_text())
    study = ROOT/active['study']
    report = study/'report'
    validation = json.loads((report/'validation.json').read_text())
    if not validation['passed']:
        raise ValueError('Require a validated report')
    evidence = report/'evidence'
    for name in ('freshrefs-reference-audit.json','freshrefs-smoke-submission.json',
                 'freshrefs-scheduling-amendment.json','freshrefs-export-submission.json',
                 'freshrefs-followup-submission.json','freshrefs-cohort-reference-copy.json','freshrefs-smoke-verification.json',
                 'freshrefs-baseline-cohort-verification.json','freshrefs-surface-cohort-verification.json',
                 'reference-drift-diagnostic.json','remote-audit-submission.json',
                 'parallel-transfer-check.json','eight-stream-transfer-check.json',
                 'transfer-decision.json','report-preparation.json','report-interpretation-amendment.json'):
        if (design/name).is_file():
            shutil.copy2(design/name,evidence/name)
    recovery = ROOT/'output/surface-annotated-20261112-recovery'
    cluster_audits = recovery/'cluster-audits'
    if cluster_audits.is_dir():
        for original in sorted(cluster_audits.rglob('*.json')):
            copied = evidence/'cluster-audits'/original.relative_to(cluster_audits)
            copied.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(original, copied)
    for tag in ('cohort-metadata','cohort-native','pool-metadata'):
        recovered = recovery/(study.name+'-'+tag)
        receipt_path = recovered/'recovery-receipt.json'
        if not receipt_path.is_file():
            continue
        receipt = json.loads(receipt_path.read_text())
        if not receipt.get('verified'):
            raise ValueError('Unverified recovery receipt: '+tag)
        destination = evidence/'recovery'/tag
        destination.mkdir(parents=True,exist_ok=True)
        for name in ('recovery-receipt.json','remote-receipt.json','remote-manifest.json'):
            shutil.copy2(recovered/name,destination/name)
    records = {}
    for side in ('baseline','surface'):
        run = study/(side+'-cohort')
        destination = evidence/side
        destination.mkdir(exist_ok=True)
        for name in ('run_config.json','metric_config.json','execution.json','snapshot-receipt.json','summary.json','per_target.csv'):
            shutil.copy2(run/name,destination/name)
        execution = json.loads((run/'execution.json').read_text())
        provenance = run/'provenance'/execution['id']
        shutil.copy2(provenance/'environment.json',destination/'environment.json')
        source_destination = destination/'provenance'
        for original in provenance.rglob('*'):
            if original.is_file():
                copied = source_destination/original.relative_to(provenance)
                copied.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(original,copied)
        records[side] = [json.loads(path.read_text()) for path in sorted(run.glob('targets/*/*/result.json'))]
    write_json(evidence/'cohort-target-records.json',records)
    decision_count = 0
    with gzip.open(evidence/'surface-decisions.jsonl.gz','wt',encoding='utf-8') as stream:
        for path in sorted((study/'surface-cohort').glob('targets/*/*/surface/*/decision.json')):
            stream.write(json.dumps(dict(path=path.relative_to(study/'surface-cohort').as_posix(),
                sha256=sha256(path),decision=json.loads(path.read_text())))+'\n')
            decision_count += 1
    pools = [json.loads(path.read_text()) for path in sorted((study/'pool-diagnostic').glob('targets/*/*/pool.json'))]
    if pools:
        write_json(evidence/'pool-metric-records.json',pools)
    sources = report/'analysis-source'
    sources.mkdir(exist_ok=True)
    for name in ('audit_annotated_surface.py','audit_annotated_surface_pools.py','audit_annotated_references.py',
                 'audit_annotated_full.py','audit_surface_progressive_result.py','build_annotated_surface_report.py',
                 'validate_annotated_surface_report.py','package_annotated_surface.py',
                 'finish_annotated_surface_report.py'):
        shutil.copy2(ROOT/'scripts'/name,sources/name)
    (report/'PACKAGE.md').write_text(
        '# Annotated Surface-PMC evidence package\n\n'
        'Open REPORT.pdf or gallery.html. The gallery includes every frozen target; '
        'thumbnail images preserve their native source identities in gallery-manifest.json.\n\n'
        'The package includes exact paired metrics, target metadata, compressed selector decisions, '
        'pool metric records, frozen designs/parameters, source snapshots, numeric audits, '
        'job lineage and report validation. Native full-resolution candidate pools and model '
        'weights remain in the project/cluster and are not duplicated in this report ZIP. '
        'Report thumbnails cannot reproduce pixel-level or neural-metric audits. '
        'Use the separately checksum-verified cohort native export for those checks.\n',encoding='utf-8')
    payload = {path.relative_to(report).as_posix():sha256(path) for path in sorted(report.rglob('*'))
               if path.is_file() and path.name not in ('package-manifest.json','delivery-receipt.json')}
    write_json(report/'package-manifest.json',payload)
    payload['package-manifest.json'] = sha256(report/'package-manifest.json')
    archive = study/'annotated-surface-results.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as bundle:
        for relative in sorted(payload):
            bundle.write(report/relative,relative)
    import hashlib
    with zipfile.ZipFile(archive) as bundle:
        if bundle.testzip() is not None or set(bundle.namelist()) != set(payload):
            raise ValueError('Package archive failed CRC/inventory checks')
        for relative,expected in payload.items():
            if hashlib.sha256(bundle.read(relative)).hexdigest() != expected:
                raise ValueError('Package member hash differs')
    receipt = dict(verified=True,at=datetime.now(timezone.utc).isoformat(),archive=str(archive),
        bytes=archive.stat().st_size,sha256=sha256(archive),members=len(payload),
        decision_records=decision_count,pool_metric_records=len(pools),
        source_sha256=sha256(Path(__file__)))
    write_json(report/'delivery-receipt.json',receipt)
    print(json.dumps(receipt,indent=2))


if __name__ == '__main__':
    main()

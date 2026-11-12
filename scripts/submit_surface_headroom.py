#!/usr/bin/env python3
"""Freeze and submit the saved-pool diagnostic once, after native serialization checks."""
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from amodal_benchmark.dataset import sha256, write_json
from collect_surface_job_evidence import HOST, PROJECT, remote_json, ssh


def main():
    design = ROOT/'output/surface-candidate-headroom-20261111-design'
    design.mkdir(exist_ok=True)
    if (design/'submission.json').exists():
        raise ValueError('Preserve existing headroom submission; monitor its job')
    old = json.loads((ROOT/'output/surface-lastpass-ablation-20261111-design/freeze.json').read_text())
    recovery = ROOT/'output/surface-lastpass-ablation-20261111-recovery'
    receipt = json.loads((recovery/'receipt.json').read_text())
    audit = json.loads((ROOT/'output/surface-lastpass-review-20261111/audit.json').read_text())
    if not audit['passed'] or audit['targets'] != 2565:
        raise ValueError('Completed independent parent audit required')
    names = ['scripts/measure_surface_candidate_headroom.py', 'scripts/run_surface_headroom_slurm.sh',
             'scripts/audit_surface_lastpass_result.py']
    source_names = [*names, 'scripts/run_surface_followup_slurm.sh',
        'amodal_benchmark/metrics.py', 'amodal_benchmark/protocol.py',
        'amodal_benchmark/references.py', 'amodal_benchmark/dataset.py', 'amodal_benchmark/provenance.py']
    freeze = dict(schema=1, created_at=datetime.now(timezone.utc).isoformat(),
        status='diagnostic_frozen_before_pool_metric_scoring',
        scope='Exploratory fixed-mask final-pool RGB headroom on existing progressive trajectories. Oracle choices use evaluation references and are NOT deployable results, an independent test, or end-to-end progressive gains.',
        decision_gate='If any individual raw-pool metric ceiling is below 3%, final raw RGB reranking alone cannot meet all four full targets on these fixed outputs. Higher ceilings do not establish a jointly attainable or observable selector.',
        target_ids=old['target_ids'], dataset_root='datasets/open_world_amodal',
        surface_run='output/pdmc-surface-progressive-20261110',
        ablation_run='output/surface-lastpass-ablation-20261111',
        metric_config_sha256=old['runs']['surface']['metric_config.json'],
        parent_archive_sha256=receipt['sha256'], selected_metric_replay_tolerance=1e-5,
        candidate_indices=list(range(8)), mask_policy='Retain corrected final alpha for all eight RGB candidates',
        controls=['frozen_raw', 'uniform_raw', 'semantic_only', 'corrected'],
        oracle_policy='Separate unrestricted per-metric best candidate, with its full metric tuple reported',
        no_fitting_or_diffusion=True, unchanged_failure_penalties=True,
        source_sha256={name:sha256(ROOT/name) for name in source_names},
        ablation_record_sha256={identity:receipt['member_sha256'][
            'output/surface-lastpass-ablation-20261111/targets/'+identity+'/ablation.json'] for identity in old['target_ids']})
    path = design/'freeze.json'
    if not path.exists():
        write_json(path, freeze)
    else:
        existing = json.loads(path.read_text())
        freeze['created_at'] = existing['created_at']
        if freeze != existing:
            raise ValueError('Existing freeze differs; preserve it')
    relative_design = design.relative_to(ROOT).as_posix()
    remote_json(HOST, 'from pathlib import Path\nimport json\n'+
        f'p=Path.home()/{PROJECT!r}/{relative_design!r}\np.mkdir(exist_ok=True)\nprint(json.dumps(dict(directory=str(p))))')
    subprocess.run(['scp', *[str(ROOT/name) for name in names], HOST+':'+PROJECT+'/scripts/'], check=True)
    subprocess.run(['scp', str(path), HOST+':'+PROJECT+'/'+relative_design+'/freeze.json'], check=True)
    program = f'''import json,sys,tempfile
from pathlib import Path
root=Path.cwd()
sys.path.insert(0,str(root/'scripts'))
from measure_surface_candidate_headroom import derive_candidate
from amodal_benchmark.dataset import sha256
freeze_path=root/{(relative_design+'/freeze.json')!r}
freeze=json.loads(freeze_path.read_text())
assert sha256(freeze_path)=={sha256(path)!r}
for name,expected in freeze['source_sha256'].items():
    assert sha256(root/name)==expected,name
for identity,expected in freeze['ablation_record_sha256'].items():
    assert sha256(root/freeze['ablation_run']/'targets'/identity/'ablation.json')==expected,identity
identity='VG/43_television'
source=root/freeze['surface_run']/'targets'/identity
record=json.loads((source/'result.json').read_text())
trace=record['diagnostics']['surface_iterations'][-1]
decision=json.loads((source/'surface'/('iteration-%02d'%trace['iteration'])/'decision.json').read_text())
prior=json.loads((root/freeze['ablation_run']/'targets'/identity/'ablation.json').read_text())
with tempfile.TemporaryDirectory(prefix='surface-headroom-preflight-') as temp:
    derived,detail=derive_candidate(source,Path(temp),record,decision,decision['index'])
    assert sha256(derived)==prior['derivation']['derived_sha256']
print(json.dumps(dict(passed=True,frozen_sources=len(freeze['source_sha256']),parent_record_checks=len(freeze['ablation_record_sha256']),native_selected_rgb_replay=True,freeze_sha256=sha256(freeze_path))))
'''
    preflight = ssh(HOST, 'cd ~/'+PROJECT+' && PYTHONNOUSERSITE=1 .amodal-cache/surface-metrics-env/bin/python -',
                    input=program.encode(), capture_output=True)
    write_json(design/'preflight.json', json.loads(preflight.stdout))
    command = ['sbatch', '--parsable', '--partition=debug-gpu', '--gres=gpu:4', '--time=02:00:00',
        '--job-name=surface-pool-ceiling',
        '--export=ALL,SURFACE_CUFFT_COMPAT=1,SURFACE_METRICS_ENV=1,SURFACE_VALIDATE_RUNTIME=1',
        '--output='+relative_design+'/job-%j.log', 'scripts/run_surface_headroom_slurm.sh', '--benchmark-workers', '4']
    submitted = ssh(HOST, 'cd ~/'+PROJECT+' && '+shlex.join(command), capture_output=True)
    job = int(submitted.stdout.decode().strip().split(';')[0])
    result = dict(created_at=datetime.now(timezone.utc).isoformat(), job_id=job, command=command,
        freeze_sha256=sha256(path), source_sha256=freeze['source_sha256'])
    write_json(design/'submission.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

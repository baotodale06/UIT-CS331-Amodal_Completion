"""Recover the fixed-prefix diagnostic and independently replay its statistics."""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from amodal_benchmark.dataset import sha256, write_json
from scripts.assess_shape_v2 import assess, table
from scripts.audit_shape_v2 import paired_estimate
from scripts.verify_shape_v2_metadata import equal

STUDY = ROOT/'output/shape-v2-slurm-20261123'
HOST = 'daoqm@slurm.csse.rose-hulman.edu'
REMOTE = '/home/daoqm/Vu_Hung/cv_nangcao/output/shape-v2-slurm-20261123'
PROGRAM = r'''
import hashlib,json,os,subprocess
from pathlib import Path
study=Path('/home/daoqm/Vu_Hung/cv_nangcao/output/shape-v2-slurm-20261123')
job=json.loads((study/'snapshot-quality-job.json').read_bytes())['job_id']
os.environ['TZ']='UTC'
raw=subprocess.check_output(['sacct','-X','-n','-P','-j',job,'--format=JobIDRaw,State%32,ExitCode,ElapsedRaw,AllocTRES%120,Start,End'],text=True)
fields=raw.strip().split('|')
assert fields[:3]==[job,'COMPLETED','0:0'],raw
names=['snapshot-quality-45665/result.json','snapshot-quality-45665/contract.json',
    'score_shape_v2_snapshot.py','run_snapshot_quality.sh','snapshot-quality-'+job+'.log']
files={name:dict(sha256=hashlib.sha256((study/name).read_bytes()).hexdigest(),bytes=(study/name).stat().st_size) for name in names}
print(json.dumps(dict(job_id=job,accounting=fields,files=files)))
'''


def main():
    response=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=20',HOST,'python3 -'],
        input=PROGRAM,text=True,capture_output=True,check=True,timeout=60)
    receipt=json.loads(response.stdout)
    output=STUDY/'snapshot-quality-recovery'
    output.mkdir(exist_ok=True)
    for name,item in receipt['files'].items():
        target=output/name
        target.parent.mkdir(exist_ok=True,parents=True)
        if not target.exists():
            with TemporaryDirectory(dir=output) as work:
                temporary=Path(work)/target.name
                subprocess.run(['scp','-q','-o','BatchMode=yes',HOST+':'+REMOTE+'/'+name,str(temporary)],check=True,timeout=120)
                if sha256(temporary)!=item['sha256'] or temporary.stat().st_size!=item['bytes']:
                    raise ValueError('Transfer changed: '+name)
                temporary.rename(target)
        equal(sha256(target),item['sha256'],'recovered artifact')
    result=json.loads((output/'snapshot-quality-45665/result.json').read_bytes())
    contract=json.loads((output/'snapshot-quality-45665/contract.json').read_bytes())
    equal(result['contract_sha256'],sha256(output/'snapshot-quality-45665/contract.json'),'contract')
    equal(contract['source_sha256'],sha256(output/'score_shape_v2_snapshot.py'),'scoring implementation')
    equal(contract['audit_sha256'],sha256(STUDY/'partial-audit-45665.json'),'existing fixed snapshot')
    equal(contract['job_id'],receipt['job_id'],'job')
    control_path=ROOT/'output/surface-annotated-20261112-freshrefs/baseline/per_target.csv'
    equal(contract['control_csv_sha256'],sha256(control_path),'registered original control')
    control=table(control_path)
    audit=json.loads((STUDY/'partial-audit-45665.json').read_bytes())
    rows=result['rows']
    if len(rows)!=193 or len({r['id'] for r in rows})!=193 or set(audit['result_hashes'])!={r['id'] for r in rows}:
        raise ValueError('Snapshot inventory changed')
    for row in rows:
        equal(row['method_record_sha256'],audit['result_hashes'][row['id']],'audited inference record')
        equal(row['control'],{key:control[row['id']][key] for key in row['control']},'registered control scores')
    recomputed=assess({r['id']:r['control'] for r in rows},{r['id']:r['method'] for r in rows},[r['id'] for r in rows])
    equal(recomputed['metrics'],result['metrics'],'failure-inclusive snapshot arithmetic')
    for metric in result['metrics']:
        equal(paired_estimate(rows,metric),result['paired_intervals'][metric],'snapshot bootstrap')
    if any(result[key] is not False for key in ('benchmark_success','full_benchmark_verified','policy_changed')):
        raise ValueError('Partial diagnostic scope changed')
    replay=dict(at=datetime.now(timezone.utc).isoformat(),verified=True,targets=193,
        result_sha256=sha256(output/'snapshot-quality-45665/result.json'),
        replay_source_sha256=sha256(Path(__file__)),benchmark_success=False,
        allocation_gpu_hours=int(receipt['accounting'][3])/3600,
        scope='Independent metadata/statistical replay. Native RGB remains on the cluster.')
    write_json(output/'recovery.json',receipt)
    write_json(output/'replay.json',replay)
    lines=[]
    for metric,value in result['metrics'].items():
        lo,hi=result['paired_intervals'][metric]['relative_ci_percent']
        lines.append(f"| {metric} | {value['control']:.6f} | {value['method']:.6f} | {value['relative_gain_percent']:+.3f}% | [{lo:+.3f}%, {hi:+.3f}%] | {value['control_n']} / {value['method_n']} |")
    text=f"""# Frozen v2: partial quality diagnostic

This is the existing **193-target runtime-completed VG snapshot**, fixed by
engineering audit 45665 before quality scoring. It is not a random sample, a
full benchmark or evidence that the >=3% goal is achieved. Frozen v2 continues
unchanged. The diagnostic has no policy tuning or early-stopping rule.

| Metric | Registered control | Frozen v2 | Relative improvement | Paired 95% interval | Control / method n |
| --- | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(lines)}

Every terminal failure is retained with the released appearance penalties;
CLIP uses its actual scored denominator. Method statuses: `{result['method_statuses']}`.
Control statuses: `{result['control_statuses']}`. CLIP coverage is not reduced:
`{result['clip_coverage_not_reduced']}`. Paired intervals resample image clusters
within this prefix; they do not address its nonrandom selection.

The registered control was also rescored on the same Quadro and unchanged
metric implementation. Maximum absolute differences: `{result['registered_control_max_rescore_difference']}`.
The primary table above retains the original registered control scores.

Job {receipt['job_id']} completed with exit 0:0 in {receipt['accounting'][3]} seconds,
adding {replay['allocation_gpu_hours']:.6f} allocated GPU-hours. It reads inference
outputs and writes only its separate diagnostic directory. This is an additional
diagnostic cost, not part of the frozen inference allocation.

Input/reference/completion hashes were checked on the cluster. Local recovery
verified the fixed inventory, registered control and all arithmetic/bootstrap
results. Native RGB remains on the cluster. See `recovery.json`, `replay.json`,
`snapshot-quality-45665/contract.json` and `snapshot-quality-45665/result.json`.
"""
    (output/'REPORT.md').write_text(text,encoding='utf8')
    print(json.dumps(dict(replay=replay,metrics=result['metrics']),indent=2))


if __name__=='__main__':
    main()

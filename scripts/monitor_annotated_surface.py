#!/usr/bin/env python3
"""Read Slurm state and artifact counts without inspecting unfinished outcomes."""
from datetime import datetime, timezone
import argparse
import json
from pathlib import Path
import subprocess

PROGRAM = r'''
from pathlib import Path
from datetime import datetime, timezone
import json, subprocess
root=Path.home()/'Vu_Hung/cv_nangcao'
design=root/'output/surface-annotated-20261112-design'
active=json.loads((design/'active-study.json').read_text()) if (design/'active-study.json').exists() else {}
submission=json.loads((design/active.get('submission','submission.json')).read_text())
ids=[row['job_id'] for row in submission['jobs']]
for name in ('freshrefs-export-submission.json', 'freshrefs-followup-submission.json', 'remote-audit-submission.json'):
    if (design/name).exists():
        ids.extend(row['job_id'] for row in json.loads((design/name).read_text())['jobs'])
if (design/'freshrefs-smoke-submission.json').exists():
    ids.append(json.loads((design/'freshrefs-smoke-submission.json').read_text())['job_id'])
if not active and (design/'pool-submission.json').exists():
    ids.append(json.loads((design/'pool-submission.json').read_text())['job_id'])
if not active:
    for path in design.glob('*retry-submission.json'):
        ids.append(json.loads(path.read_text())['job_id'])
raw=subprocess.check_output(['sacct','-j',','.join(ids),'-X','-n','-P',
    '--format=JobID,JobName,State,Elapsed,ExitCode,NodeList'],text=True)
jobs={}
for line in raw.splitlines():
    fields=line.strip().split('|')
    if len(fields)>=6:
        jobs[fields[0]]=dict(name=fields[1],state=fields[2],elapsed=fields[3],exit_code=fields[4],nodes=fields[5])
study=root/active.get('study','output/surface-annotated-20261112')
runs={}
for name in ('baseline','surface','baseline-cohort','surface-cohort'):
    run=study/name
    logs=sorted(run.glob('workers/*/worker-*.log'))
    errors=[]
    for log in logs:
        with log.open('rb') as stream:
            stream.seek(max(0,log.stat().st_size-8000))
            tail=stream.read().decode(errors='replace')
        if 'Traceback (most recent call last)' in tail or 'ERROR:' in tail:
            errors.append(dict(log=str(log.relative_to(root)),tail=tail[-3500:]))
    runs[name]=dict(target_records=sum(1 for _ in run.glob('targets/*/*/result.json')),
        surface_decisions=sum(1 for _ in run.glob('targets/*/*/surface/*/decision.json')),
        worker_tracebacks=errors)
print(json.dumps(dict(as_of_utc=datetime.now(timezone.utc).isoformat(),jobs=jobs,runs=runs,
    cohort_comparison_ready=(study/'cohort-comparison/comparison.json').is_file(),
    pool_diagnostic_ready=(study/'pool-diagnostic/analysis/summary.json').is_file(),
    full_comparison_ready=(study/'full-comparison/comparison.json').is_file())))
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compact', action='store_true')
    args = parser.parse_args()
    result = subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15',
        'daoqm@slurm.csse.rose-hulman.edu','python3 -'],input=PROGRAM,text=True,
        capture_output=True,check=True,timeout=90)
    record=json.loads(result.stdout)
    destination=Path(__file__).resolve().parents[1]/'output/surface-annotated-20261112-design/status'
    destination.mkdir(parents=True,exist_ok=True)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    (destination/(stamp+'.json')).write_text(json.dumps(record,indent=2)+'\n')
    if args.compact:
        compact = dict(at=record['as_of_utc'],
            jobs={job: value['state'] + ' ' + value['elapsed'] for job,value in record['jobs'].items()},
            runs={name:dict(records=value['target_records'], pools=value['surface_decisions'],
                           errors=value['worker_tracebacks']) for name,value in record['runs'].items()},
            cohort_ready=record['cohort_comparison_ready'],pool_ready=record['pool_diagnostic_ready'],
            full_ready=record['full_comparison_ready'])
        print(json.dumps(compact))
    else:
        print(json.dumps(record,indent=2))


if __name__=='__main__':
    main()

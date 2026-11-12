#!/usr/bin/env python3
"""Read Slurm state and artifact counts without inspecting unfinished outcomes."""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

PROGRAM = r'''
from pathlib import Path
from datetime import datetime, timezone
import json, subprocess
root=Path.home()/'Vu_Hung/cv_nangcao'
design=root/'output/surface-annotated-20261112-design'
submission=json.loads((design/'submission.json').read_text())
ids=[row['job_id'] for row in submission['jobs']]
if (design/'pool-submission.json').exists():
    ids.append(json.loads((design/'pool-submission.json').read_text())['job_id'])
for path in design.glob('*retry-submission.json'):
    ids.append(json.loads(path.read_text())['job_id'])
raw=subprocess.check_output(['sacct','-j',','.join(ids),'-X','-n','-P',
    '--format=JobID,JobName,State,Elapsed,ExitCode,NodeList'],text=True)
jobs={}
for line in raw.splitlines():
    fields=line.strip().split('|')
    if len(fields)>=6:
        jobs[fields[0]]=dict(name=fields[1],state=fields[2],elapsed=fields[3],exit_code=fields[4],nodes=fields[5])
study=root/'output/surface-annotated-20261112'
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
    result = subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15',
        'daoqm@slurm.csse.rose-hulman.edu','python3 -'],input=PROGRAM,text=True,
        capture_output=True,check=True,timeout=90)
    record=json.loads(result.stdout)
    destination=Path(__file__).resolve().parents[1]/'output/surface-annotated-20261112-design/status'
    destination.mkdir(parents=True,exist_ok=True)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    (destination/(stamp+'.json')).write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2))


if __name__=='__main__':
    main()

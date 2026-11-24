#!/usr/bin/env python3
"""Deploy CPU-only evidence helpers and record dependent native audit jobs."""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from amodal_benchmark.dataset import sha256,write_json


def main():
    host='daoqm@slurm.csse.rose-hulman.edu'
    design=ROOT/'output/surface-annotated-20261112-design'
    receipt=design/'remote-audit-submission.json'
    if receipt.exists():
        raise ValueError('Submissions already recorded; inspect them before acting')
    names=('audit_annotated_surface.py','audit_annotated_surface_pools.py',
           'audit_surface_progressive_result.py','audit_annotated_cohort_slurm.sh',
           'export_annotated_review_assets.py')
    shell=ROOT/'scripts/audit_annotated_cohort_slurm.sh'
    shell.write_bytes(shell.read_bytes().replace(b'\r\n',b'\n'))
    expected={name:sha256(ROOT/'scripts'/name) for name in names}
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    prefix=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15',host,'python3 -']
    setup="""
from pathlib import Path
import hashlib,json,shutil
root=Path.home()/'Vu_Hung/cv_nangcao'
expected=EXPECTED
backup=root/'.deployment-backups'/STAMP
for name,digest in expected.items():
    path=root/'scripts'/name
    if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
        backup.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,backup/name)
if (root/'output/surface-annotated-20261112-design/remote-audit-submission.json').exists():
    raise ValueError('Remote submissions already exist')
print('Reporting helper preflight complete')
""".replace('EXPECTED',repr(expected)).replace('STAMP',repr('annotated-reporting-'+stamp))
    subprocess.run(prefix,input=setup,text=True,check=True)
    subprocess.run(['scp','-q',*[str(ROOT/'scripts'/name) for name in names],
                    host+':Vu_Hung/cv_nangcao/scripts/'],check=True)
    program="""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,subprocess
root=Path.home()/'Vu_Hung/cv_nangcao'
design=root/'output/surface-annotated-20261112-design'
expected=EXPECTED
for name,digest in expected.items():
    if hashlib.sha256((root/'scripts'/name).read_bytes()).hexdigest()!=digest:
        raise ValueError('Deployment checksum mismatch: '+name)
runtime=subprocess.check_output([str(root/'.amodal-cache/surface-metrics-env/bin/python'),'-c',
    'import numpy,PIL,scipy; print(numpy.__version__,PIL.__version__,scipy.__version__)'],
    env=dict(os.environ,PYTHONNOUSERSITE='1'),text=True).strip()
if runtime!='1.23.5 9.5.0 1.10.1':
    raise ValueError('Wrong CPU native replay versions: '+runtime)
receipt=dict(at=datetime.now(timezone.utc).isoformat(),sources=expected,runtime=runtime,jobs=[],
    reason='Slow native transfer; run the same independent CPU audits adjacent to immutable artifacts and export compact all-target review images. No inference or scientific settings change.')
for phase,dependency in [('cohort','39487'),('pool','39489')]:
    if phase=='pool':
        dependency+=':'+receipt['jobs'][0]['job_id']
    command=['sbatch','--parsable','--partition=debug-cpu','--dependency=afterok:'+dependency,
        '--job-name=ann-native-audit-'+phase,'--output='+str(design/('ann-native-audit-'+phase+'-%j.log')),
        'scripts/audit_annotated_cohort_slurm.sh','--phase',phase]
    identity=subprocess.check_output(command,cwd=root,text=True).strip().split(';')[0]
    if not identity.isdigit():
        raise ValueError('Invalid Slurm response')
    receipt['jobs'].append(dict(phase=phase,job_id=identity,command=command))
    (design/'remote-audit-submission.json').write_text(json.dumps(receipt,indent=2)+'\\n')
print(json.dumps(receipt))
""".replace('EXPECTED',repr(expected))
    response=subprocess.run(prefix,input=program,text=True,capture_output=True)
    if response.returncode:
        print(response.stdout)
        print(response.stderr,file=sys.stderr)
        response.check_returncode()
    record=json.loads(response.stdout)
    write_json(receipt,record)
    print(json.dumps(record,indent=2))


if __name__=='__main__':
    main()

"""Observe isolated constrained training and recover/replay completed native evidence."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile
from tempfile import TemporaryDirectory
import time
import zipfile

import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from amodal_benchmark.dataset import sha256,write_json
from amodal_benchmark.parallel import output_lock
from scripts.collect_surface_job_evidence import recover_archive
from scripts.recover_shape_sam_validation import geometry,mask,paired_summary,check_ids
from scripts.verify_shape_v2_metadata import equal
from scripts.recover_shape_v2 import allocation_cost

STUDY=ROOT/"output/shape-sam-constrained-v2-20261123"
HOST="daoqm@slurm.csse.rose-hulman.edu"
REMOTE="/home/daoqm/Vu_Hung/cv_nangcao/output/shape-sam-constrained-v2-20261123"
TERMINAL={"COMPLETED","FAILED","TIMEOUT","CANCELLED","NODE_FAIL","OUT_OF_MEMORY","PREEMPTED"}
PROBE=r'''
import json,os,subprocess
from pathlib import Path
study=Path('/home/daoqm/Vu_Hung/cv_nangcao/output/shape-sam-constrained-v2-20261123')
os.environ['TZ']='UTC'
training=json.loads((study/'training-jobs.json').read_bytes())
roles={training['preflight_job_id']:'preflight',training['training_job_id']:'training',json.loads((study/'native-job.json').read_bytes())['job_id']:'native_validation'}
raw=subprocess.check_output(['sacct','-X','-j',','.join(roles),'-P','--format=JobIDRaw,State%32,ExitCode,ElapsedRaw,AllocTRES%120,NodeList,Start,End'],text=True)
jobs={line.split('|')[0]:line.split('|')[1:3] for line in raw.splitlines()[1:] if line.strip()}
assert set(jobs)==set(roles),'Incomplete accounting'
receipt=json.loads((study/'training/training.json').read_bytes()) if (study/'training/training.json').exists() else None
progress={}
if receipt:
    history=receipt.get('history',[])
    progress=dict(epochs_completed=receipt.get('epochs_completed',0),complete=receipt['complete'],latest=history[-1] if history else None)
native=json.loads((study/'native-validation/progress.json').read_bytes()) if (study/'native-validation/progress.json').exists() else None
print(json.dumps(dict(jobs=jobs,roles=roles,accounting=raw,training=progress,native=native)))
'''
EXPORT=r'''
import hashlib,io,json,os,tarfile
from pathlib import Path
from tempfile import TemporaryDirectory
study=Path('/home/daoqm/Vu_Hung/cv_nangcao/output/shape-sam-constrained-v2-20261123')
training=json.loads((study/'training/training.json').read_bytes())
result=json.loads((study/'native-validation/result.json').read_bytes())
assert training['complete'] and not training['preflight'] and training['epochs_completed']==20
assert result['complete'] and result['objects']==1264 and result['images']==217
archive,receipt=study/'native-evidence.tar.gz',study/'native-export.json'
digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
if receipt.exists():
    data=json.loads(receipt.read_bytes())
    assert digest(archive)==data['sha256']
    assert digest(study/'training/training.json')==data['training_sha256']
    assert digest(study/'native-validation/result.json')==data['result_sha256']
else:
    assert not archive.exists(),'Preserve unreceipted archive'
    names=['training-jobs.json','training-source-manifest.json','training-source.zip',
        'native-job.json','native-source-manifest.json','native-source.zip','training-tests.log',
        'training/training.json','training/best-validation-native.json','preflight/preflight.json','preflight/training.json']
    paths=[study/name for name in names]
    for directory in ('native-validation',):
        paths.extend(p for p in (study/directory).rglob('*') if p.is_file())
    paths.extend(study.glob('*-*.log'))
    paths=sorted(set(paths))
    assert all(not p.is_symlink() for p in paths)
    files={p.relative_to(study).as_posix():digest(p) for p in paths}
    with TemporaryDirectory(dir=study) as work:
        staged=Path(work)/'native.tar.gz'
        with tarfile.open(staged,'w:gz',compresslevel=1) as bundle:
            for name in files:
                bundle.add(study/name,arcname=name,recursive=False)
            payload=json.dumps(files,sort_keys=True).encode()
            info=tarfile.TarInfo('member-sha256.json')
            info.size=len(payload)
            bundle.addfile(info,io.BytesIO(payload))
        data=dict(sha256=digest(staged),bytes=staged.stat().st_size,files=len(files),
            training_sha256=files['training/training.json'],result_sha256=files['native-validation/result.json'])
        os.link(staged,archive)
    receipt.write_text(json.dumps(data,indent=2)+'\n')
assert digest(study/'training/best.pt')==training['best_sha256']
data['checkpoint']=dict(sha256=training['best_sha256'],bytes=(study/'training/best.pt').stat().st_size)
print(json.dumps(data))
'''


def remote(program,timeout=120):
    response=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=20',HOST,'python3 -'],
        input=program,text=True,capture_output=True,check=True,timeout=timeout)
    return json.loads(response.stdout)


def download(name,path,checksum,size):
    if not path.exists():
        path.parent.mkdir(parents=True,exist_ok=True)
        with TemporaryDirectory(dir=STUDY) as work:
            staged=Path(work)/path.name
            subprocess.run(['scp','-q','-o','BatchMode=yes','-o','ConnectTimeout=20',HOST+':'+REMOTE+'/'+name,str(staged)],check=True,timeout=1800)
            if sha256(staged)!=checksum or staged.stat().st_size!=size:
                raise ValueError('Downloaded evidence changed')
            staged.rename(path)
    if sha256(path)!=checksum or path.stat().st_size!=size:
        raise ValueError('Preserve differing local evidence')


def replay(evidence):
    read=lambda p:json.loads(p.read_bytes())
    training=read(evidence/'training/training.json')
    result=read(evidence/'native-validation/result.json')
    contract=read(evidence/'native-validation/contract.json')
    baseline=training['initial_validation']
    candidates=[dict(epoch=0,validation=baseline)]+[row for row in training['history']
        if row['validation']['full_iou']>=baseline['full_iou']]
    chosen=max(candidates,key=lambda row:(row['validation']['hidden_iou'],row['validation']['full_iou']))
    equal(training['new_checkpoint_eligible'],chosen['epoch']>0,'constrained eligibility')
    equal(training['selected_epoch'],chosen['epoch'],'selected checkpoint')
    best=read(evidence/'training/best-validation-native.json')
    equal(best['aggregate'],chosen['validation'],'selected checkpoint geometry')
    equal([best['epoch'],result['selected_epoch'],contract['selected_epoch']],[chosen['epoch']]*3,'selected epoch')
    equal(contract['training_receipt_sha256'],sha256(evidence/'training/training.json'),'training binding')
    equal(contract['checkpoint_sha256'],sha256(STUDY/'training/best.pt'),'checkpoint binding')
    equal(training['best_sha256'],contract['checkpoint_sha256'],'selected weights')
    equal(result['contract_sha256'],sha256(evidence/'native-validation/contract.json'),'native contract')
    equal(training['best_validation_sha256'],sha256(evidence/'training/best-validation-native.json'),'recorded native validation')
    if not training['complete'] or training['preflight'] or training['epochs_completed']!=20 or result['benchmark_success'] is not False:
        raise ValueError('Training/native scope changed')
    if any(contract[key] is not False for key in ('holdout_access','benchmark_access')):
        raise ValueError('Evaluation scope changed')
    with zipfile.ZipFile(ROOT/'output/shape-v2-20261123/colab/shape-v2-training-data.zip') as bundle:
        payload=bundle.read('data/manifest.json')
    equal(hashlib.sha256(payload).hexdigest(),contract['data_manifest_sha256'],'data manifest')
    manifest=json.loads(payload)
    selected={row['id']:row for row in manifest['rows'] if row['split']=='validation'}
    if len(selected)!=1264 or len({row['image_id'] for row in selected.values()})!=217:
        raise ValueError('Validation split changed')
    baseline_root=ROOT/'output/shape-validation-sam-20261123/native/result'
    equal(sha256(baseline_root/'result.json'),contract['baseline_result_sha256'],'baseline diagnostic')
    baseline={row['id']:row for row in read(baseline_root/'result.json')['rows']}
    check_ids(result['rows'],selected)
    check_ids(best['rows'],selected)
    best_rows={row['id']:row for row in best['rows']}
    native=evidence/'native-validation'
    data=ROOT/'output/shape-v2-20261123/data'
    pairs=[]
    for row in result['rows']:
        source=selected[row['id']]
        equal({key:row[key] for key in ('image_id','image','mask')},
              {key:source[key] for key in ('image_id','image','mask')},'object identity')
        equal(sha256(data/source['mask']),manifest['artifacts'][source['mask']],'native truth')
        with Image.open(data/source['mask']) as image:
            encoded=np.array(image)
        truth,visible=(encoded&1)>0,(encoded&2)>0
        directory=native/'objects'/row['id']
        equal(read(directory/'result.json'),row,'object record')
        equal(sha256(directory/'prediction.png'),row['prediction_sha256'],'prediction bytes')
        equal(sha256(baseline_root/'objects'/row['id']/'sam-visible.png'),row['sam_sha256'],'SAM support')
        computed=geometry(mask(directory/'prediction.png',truth.shape),truth,visible)
        equal(computed,row['constrained_sam'],'native mask geometry')
        equal(computed,{key:best_rows[row['id']][key] for key in computed},'selected training geometry')
        equal(row['v2_sam'],baseline[row['id']]['sam_input'],'frozen v2 SAM geometry')
        pairs.append(dict(image_id=row['image_id'],annotated_input=row['v2_sam'],sam_input=computed))
    replayed=paired_summary(pairs)
    for metric,values in replayed.items():
        mapped=dict(images=values['images'],objects=values['objects'],v2_sam=values['annotated_input'],
            constrained_sam=values['sam_input'],absolute_change=values['sam_minus_annotated'],absolute_ci=values['absolute_ci'])
        equal(mapped,result['geometry'][metric],'independent image/bootstrap replay')
    equal(result['max_selected_validation_difference'],0.,'GPU checkpoint replay')
    equal(sha256(ROOT/'output/shape-sam-robustness-20261123/recovered/training-masks/manifest.json'),training['contract']['sam_training_manifest_sha256'],'training SAM inputs')
    return dict(verified=True,objects=1264,images=217,selected_epoch=chosen['epoch'],new_checkpoint_eligible=training['new_checkpoint_eligible'],geometry=result['geometry'],
        training_sha256=sha256(evidence/'training/training.json'),checkpoint_sha256=sha256(STUDY/'training/best.pt'),
        result_sha256=sha256(native/'result.json'),replay_source_sha256=sha256(Path(__file__)),benchmark_success=False,
        limitation='Validation-selected development with annotated-visible box prompts; no untouched confirmation or benchmark gain')


def recover():
    receipt=remote(EXPORT,300)
    archive=STUDY/'native-evidence.tar.gz'
    download(archive.name,archive,receipt['sha256'],receipt['bytes'])
    download('training/best.pt',STUDY/'training/best.pt',receipt['checkpoint']['sha256'],receipt['checkpoint']['bytes'])
    with tarfile.open(archive,'r:gz') as bundle:
        payload=bundle.extractfile('member-sha256.json').read()
    members=json.loads(payload)
    if len(members)!=receipt['files'] or members['training/training.json']!=receipt['training_sha256'] or members['native-validation/result.json']!=receipt['result_sha256']:
        raise ValueError('Native archive inventory differs')
    members['member-sha256.json']=hashlib.sha256(payload).hexdigest()
    evidence=STUDY/'recovered'
    prefixes=['native-validation','training','preflight']+[name for name in members if '/' not in name]
    recover_archive(archive,evidence,prefixes,members)
    result=replay(evidence)
    write_json(STUDY/'native-export.json',receipt)
    write_json(STUDY/'native-replay.json',result)
    print(json.dumps(result),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--watch',action='store_true')
    args=parser.parse_args()
    lock=STUDY/'collection-control'
    lock.mkdir(parents=True,exist_ok=True)
    with output_lock(lock):
        while True:
            try:
                status=remote(PROBE)
            except (subprocess.CalledProcessError,subprocess.TimeoutExpired) as error:
                print(json.dumps(dict(at=datetime.now(timezone.utc).isoformat(),observation_error=str(error))),flush=True)
                if not args.watch:
                    raise
                time.sleep(60)
                continue
            status['at']=datetime.now(timezone.utc).isoformat()
            states=[state.split()[0].rstrip('+') for state,code in status['jobs'].values()]
            terminal=all(state in TERMINAL for state in states)
            status['disposition']='ready' if terminal and all(value==['COMPLETED','0:0'] for value in status['jobs'].values()) else 'needs_review' if terminal else 'waiting'
            write_json(STUDY/'collection-status.json',status)
            write_json(STUDY/'allocation-cost.json',allocation_cost(status['accounting'],status['roles']))
            print(json.dumps({key:value for key,value in status.items() if key not in ('roles','accounting')}),flush=True)
            if status['disposition']=='ready':
                recover()
                return
            if status['disposition']=='needs_review':
                raise RuntimeError('Training/native chain did not complete successfully; inspect logs')
            if not args.watch:
                return
            time.sleep(60)


if __name__=='__main__':
    main()

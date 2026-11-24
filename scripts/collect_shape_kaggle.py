"""Collect the private Kaggle run and independently replay native geometry."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import zipfile

import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from amodal_benchmark.dataset import sha256,write_json
from amodal_benchmark.parallel import output_lock
from scripts.recover_shape_sam_validation import geometry,mask,paired_summary,check_ids
from scripts.verify_shape_v2_metadata import equal

STUDY=ROOT/'output/shape-kaggle-20261123'
KERNEL='ez4kacchan/shape-only-constrained-training-20261123/2'
CLI=Path.home()/'.cache/shape-kaggle-cli/Scripts/kaggle.exe'

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
    equal(contract['checkpoint_sha256'],sha256(evidence/'training/best.pt'),'checkpoint binding')
    equal(training['best_sha256'],contract['checkpoint_sha256'],'selected weights')
    equal(result['contract_sha256'],sha256(evidence/'native-validation/contract.json'),'native contract')
    equal(training['best_validation_sha256'],sha256(evidence/'training/best-validation-native.json'),'recorded native validation')
    if training.get('schema') != 5 or training.get('platform') != 'Kaggle' or not training['complete'] or training['preflight'] or training['epochs_completed']!=20 or result['benchmark_success'] is not False:
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
        training_sha256=sha256(evidence/'training/training.json'),checkpoint_sha256=sha256(evidence/'training/best.pt'),
        result_sha256=sha256(native/'result.json'),replay_source_sha256=sha256(Path(__file__)),benchmark_success=False,
        limitation='Validation-selected development with annotated-visible box prompts; no untouched confirmation or benchmark gain')


def parse_status(output):
    matches = re.findall(r'KernelWorkerStatus\.([A-Z_]+)', output)
    if len(matches) != 1 or matches[0] not in {'QUEUED','RUNNING','COMPLETE','ERROR','CANCELLED','CANCELED'}:
        raise ValueError('Unknown Kaggle state; do not infer termination')
    return matches[0]


def verify_sources(evidence):
    expected = STUDY/'dataset/bundle.json'
    equal(sha256(evidence/'bundle.json'), sha256(expected), 'private input bundle')
    bundle = json.loads(expected.read_bytes())
    equal(sha256(evidence/'source.bin'), bundle['payloads']['source.bin']['sha256'], 'frozen sources')
    with zipfile.ZipFile(evidence/'source.bin') as archive:
        equal(set(archive.namelist()), set(bundle['sources']), 'source inventory')
        for name, checksum in bundle['sources'].items():
            equal(hashlib.sha256(archive.read(name)).hexdigest(), checksum, 'source bytes')
    training = json.loads((evidence/'training/training.json').read_bytes())
    equal(training['contract']['config_sha256'], bundle['sources']['configs/research/shape-sam-kaggle-20261123.json'], 'training config')
    for name, checksum in training['contract']['sources'].items():
        equal(checksum, bundle['sources'][name], 'executed training source')


def recover(kernel, evidence, expected_run, state):
    command = [str(CLI.with_name('python.exe')), str(ROOT/'scripts/download_shape_kaggle.py'),
               '--kernel', kernel, '--output', str(evidence)]
    subprocess.run(command, check=True, env={**os.environ, 'PYTHONIOENCODING':'utf-8'})
    inventory = {p.relative_to(evidence).as_posix(): sha256(p) for p in evidence.rglob('*') if p.is_file()}
    write_json(evidence.parent/(evidence.name+'-inventory.json'), dict(kernel=kernel, state=state, files=inventory))
    if state != 'COMPLETE':
        return dict(kernel=kernel, state=state, disposition='needs_review', recovered=True,
                    verified=False, benchmark_success=False)
    completion = json.loads((evidence/'completion.json').read_bytes())
    equal(completion['run'], expected_run, 'Kaggle logical run')
    equal(completion['complete'], True, 'completed training')
    equal(completion['benchmark_success'], False, 'validation-only scope')
    equal(completion['training_receipt_sha256'], sha256(evidence/'training/training.json'), 'completion training binding')
    equal(completion['selected_checkpoint_sha256'], sha256(evidence/'training/best.pt'), 'completion checkpoint binding')
    equal(completion['native_result_sha256'], sha256(evidence/'native-validation/result.json'), 'completion native binding')
    verify_sources(evidence)
    result = replay(evidence)
    equal(completion['selected_epoch'], result['selected_epoch'], 'completion epoch')
    equal(completion['new_checkpoint_eligible'], result['new_checkpoint_eligible'], 'completion eligibility')
    return dict(**result, kernel=kernel, state=state, recovered=True, disposition='verified')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--watch', action='store_true')
    parser.add_argument('--kernel', default=KERNEL)
    parser.add_argument('--expected-run', default='ez4kacchan/shape-only-constrained-training-20261123:v2')
    parser.add_argument('--evidence', type=Path, default=STUDY/'attempt-2')
    args = parser.parse_args()
    if not re.fullmatch(r'ez4kacchan/[a-z0-9-]+/[1-9][0-9]*', args.kernel):
        raise ValueError('An explicit private account/kernel/version is required')
    control = STUDY/('collection-control-'+args.kernel.rsplit('/',1)[-1])
    control.mkdir(exist_ok=True)
    with output_lock(control):
        while True:
            try:
                observed = subprocess.run([str(CLI), 'kernels', 'status', args.kernel],
                    capture_output=True, text=True, check=True, timeout=60,
                    env={**os.environ,'PYTHONIOENCODING':'utf-8'})
                state = parse_status(observed.stdout)
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
                print(json.dumps(dict(at=datetime.now(timezone.utc).isoformat(), observation_error=type(error).__name__)), flush=True)
                if not args.watch:
                    raise
                time.sleep(60)
                continue
            status = dict(at=datetime.now(timezone.utc).isoformat(), kernel=args.kernel, state=state,
                          disposition='waiting', benchmark_success=False)
            if state in {'COMPLETE','ERROR','CANCELLED','CANCELED'}:
                status.update(recover(args.kernel, args.evidence, args.expected_run, state))
                write_json(STUDY/('attempt-'+args.kernel.rsplit('/',1)[-1]+'-replay.json'), status)
            write_json(control/'status.json', status)
            print(json.dumps(status), flush=True)
            if status['disposition'] != 'waiting' or not args.watch:
                return
            time.sleep(60)


if __name__ == '__main__':
    main()


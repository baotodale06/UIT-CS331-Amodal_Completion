"""Isolated local probability-loss ablation with exact native replay."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from amodal_benchmark.dataset import sha256, write_json
from amodal_benchmark.shape_v2 import make_network
from amodal_benchmark.shape_v2_runtime import training_receipt
from scripts.train_shape_v2 import atomic_save, verify_data
from scripts.train_shape_kaggle import verify_sam_inputs, DATA_SHA256
from scripts.shape_constrained_data import freeze_encoder
from scripts.shape_robust_data import RobustData, aggregate
from scripts.shape_local_data import eligible, evaluate_native, selection_floor
from scripts.shape_probability_loss import probability_loss

CONFIG = 'configs/research/shape-probability-workers4-20261123.json'
SOURCES = ('scripts/train_shape_probability_workers4.py','scripts/shape_probability_loss.py','scripts/shape_local_data.py','scripts/shape_robust_data.py',
           'scripts/shape_constrained_data.py','scripts/train_shape_kaggle.py','scripts/train_shape_v2.py',
           'scripts/train_shape_v2_slurm.py','amodal_benchmark/__init__.py','amodal_benchmark/dataset.py',
           'amodal_benchmark/shape_v2.py','amodal_benchmark/shape_v2_runtime.py',
           'configs/research/shape-v2-20261123.json','configs/research/shape-v2-slurm-20261123.json')


def read(path):
    return json.loads(Path(path).read_bytes())


def validate_contract(config):
    settings = config['training']
    if (config.get('schema') != 8 or config.get('training_platform') != 'LocalWSL'
            or settings.get('probability_loss_version') != 1
            or settings.get('world_size') != 1 or settings.get('batch_size') != 64
            or settings.get('micro_batch_size') != 8 or settings.get('epochs') != 20
            or settings.get('memory_fraction') != .75
            or settings.get('workers_per_rank') != 4):
        raise ValueError('Separate frozen schema8 local probability-loss contract required')
    return settings


def encoder_state(model):
    return {key:value.detach().cpu().clone() for key,value in model.state_dict().items()
            if key.startswith(('stem.','layers.'))}


def check_encoder(model, expected):
    actual = encoder_state(model)
    if actual.keys() != expected.keys() or any(not torch.equal(actual[key],value) for key,value in expected.items()):
        raise ValueError('Frozen encoder parameters or BatchNorm buffers changed')


def train_epoch(model, train, optimizer, scaler, settings, seed, epoch, maximum_steps=None):
    train.epoch = epoch
    sampler = DistributedSampler(train,num_replicas=1,rank=0,shuffle=True,seed=seed)
    sampler.set_epoch(epoch)
    loader = DataLoader(train,batch_size=settings['batch_size'],sampler=sampler,
        num_workers=settings['workers_per_rank'],pin_memory=True)
    model.train()
    freeze_encoder(model)
    total_loss,objects,steps,skipped = 0.,0,0,0
    started = time.monotonic()
    for step,(inputs,truth,visible) in enumerate(loader):
        count = len(inputs)
        optimizer.zero_grad(set_to_none=True)
        batch_loss = 0.
        for start in range(0,count,settings['micro_batch_size']):
            end = min(count,start+settings['micro_batch_size'])
            x,y = (item[start:end].cuda(non_blocking=True) for item in (inputs,truth))
            with torch.autocast('cuda',dtype=torch.float16):
                logits = model(x)
            loss = probability_loss(logits.float(),y,x[:,3:4],total_objects=count)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite local shape loss')
            scaler.scale(loss).backward()
            batch_loss += loss.detach().item()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        previous_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        skipped += int(scaler.get_scale()<previous_scale)
        total_loss += batch_loss*count
        objects += count
        steps += 1
        if step%10 == 0:
            print(json.dumps(dict(epoch=epoch+1,step=step,steps=len(loader),loss=batch_loss,
                seconds=time.monotonic()-started,allocated_gpu_mib=torch.cuda.max_memory_allocated()/1024**2)),flush=True)
        if maximum_steps and steps>=maximum_steps:
            break
    return dict(loss=total_loss/objects,sampled_training_objects=objects,optimizer_steps=steps,
                skipped_optimizer_steps=skipped,training_seconds=time.monotonic()-started)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('calibrate','preflight','train','native'),required=True)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--sam-training',type=Path,required=True)
    parser.add_argument('--sam-validation',type=Path,required=True)
    parser.add_argument('--initial-checkpoint',type=Path,required=True)
    parser.add_argument('--study',type=Path,required=True)
    parser.add_argument('--resume',action='store_true')
    args = parser.parse_args()
    started = time.monotonic()
    config = read(ROOT/CONFIG)
    settings = validate_contract(config)
    if config['training_platform']!='LocalWSL' or 'microsoft' not in Path('/proc/sys/kernel/osrelease').read_text().lower():
        raise RuntimeError('Explicit isolated WSL platform required')
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1 or 'RTX 3060' not in torch.cuda.get_device_name():
        raise RuntimeError('One local RTX3060 GPU required')
    if str(torch.__version__)!='1.13.1' or torch.version.cuda!='11.7' or np.__version__!='1.23.5':
        raise RuntimeError('Exact pinned cluster runtime required')
    if settings['world_size']!=1 or settings['batch_size']!=64 or settings['micro_batch_size']!=8:
        raise ValueError('Frozen local batch contract changed')
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(settings['memory_fraction'])
    probe = torch.randn(32,32,device='cuda')
    if not torch.allclose(probe,torch.fft.irfft2(torch.fft.rfft2(probe),s=probe.shape),atol=1e-5):
        raise RuntimeError('Local CUDA FFT failed')
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed_all(config['seed'])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    manifest = read(args.data/'manifest.json')
    if (sha256(args.data/'manifest.json')!=DATA_SHA256 or manifest['holdout_packaged'] is not False
            or set(row['split'] for row in manifest['rows'])!={'train','validation'}):
        raise ValueError('Only the frozen isolated training/validation split is allowed')
    verify_data(args.data,manifest)
    inherited = training_receipt(args.initial_checkpoint)
    if sha256(args.initial_checkpoint)!=config['initial_checkpoint_sha256']:
        raise ValueError('Original v2 initialization changed')
    sam_training,sam_validation = read(args.sam_training/'manifest.json'),read(args.sam_validation/'result.json')
    verify_sam_inputs(args,manifest,sam_training,sam_validation,config)
    contract = dict(config_sha256=sha256(ROOT/CONFIG),sources={name:sha256(ROOT/name) for name in SOURCES},
        data_sha256=DATA_SHA256,initial_checkpoint_sha256=sha256(args.initial_checkpoint),
        inherited_training_contract=inherited['contract'],sam_training_manifest_sha256=sha256(args.sam_training/'manifest.json'),
        validation_sam_result_sha256=sha256(args.sam_validation/'result.json'))
    selected = [row for row in manifest['rows'] if row['split']=='validation']
    if len(selected)!=1264 or len({row['image_id'] for row in selected})!=217:
        raise ValueError('Validation inventory changed')
    args.study.mkdir(parents=True,exist_ok=True)
    model = make_network(pretrained=False).cuda()
    initial = torch.load(args.initial_checkpoint,map_location='cpu',weights_only=False)
    if initial['contract']!=inherited['contract']:
        raise ValueError('Warm-start checkpoint payload changed')
    model.load_state_dict(initial['model'],strict=True)
    del initial
    freeze_encoder(model)
    encoder = encoder_state(model)
    calibration_path = args.study/'calibration/result.json'
    runtime = dict(gpu=torch.cuda.get_device_name(),torch=str(torch.__version__),cuda=torch.version.cuda,
        cudnn=torch.backends.cudnn.version(),cuda_fft_round_trip=True,memory_fraction=settings['memory_fraction'])
    if args.mode=='calibrate':
        folder = args.study/'calibration'
        folder.mkdir(exist_ok=False)
        first = evaluate_native(model,args.data,selected,args.sam_validation,'cuda',folder/'initial-masks.zip')
        repeat = evaluate_native(model,args.data,selected,args.sam_validation,'cuda')
        if first!=repeat:
            raise ValueError('Original checkpoint must replay every native pixel on the local GPU')
        original_rows = {row['id']:row['sam_input'] for row in sam_validation['rows']}
        original = aggregate([dict(id=row['id'],image_id=row['image_id'],**original_rows[row['id']]) for row in selected])
        baseline = aggregate(first)
        different = [row['id'] for row in first if any(row[key]!=original_rows[row['id']][key] for key in ('full_iou','hidden_iou'))]
        write_json(calibration_path,dict(schema=1,complete=True,contract=contract,runtime=runtime,rows=first,
            local=baseline,original_quadro=original,selection_floor=selection_floor(baseline,original),
            original_quadro_different_geometry_objects=different,exact_repeat=True,
            original_masks_sha256=sha256(folder/'initial-masks.zip'),training_performed=False,benchmark_success=False))
        print(json.dumps(dict(calibration_complete=True,local=baseline,original=original,different_objects=len(different))),flush=True)
        return
    calibration = read(calibration_path)
    if calibration['contract']!=contract or calibration['runtime']!=runtime or not calibration['complete'] or not calibration['exact_repeat']:
        raise ValueError('Completed exact original-checkpoint calibration required')
    calibration_rows = {row['id']:row for row in calibration['rows']}
    contract['calibration_sha256'] = sha256(calibration_path)
    folder = args.study/('preflight' if args.mode=='preflight' else 'native' if args.mode=='native' else 'training')
    if not args.resume:
        folder.mkdir(exist_ok=False)
    elif args.mode!='train':
        raise ValueError('Only full training supports resume')
    training_path = args.study/'training/training.json'
    if args.mode=='native':
        receipt = read(training_path)
        if receipt['contract']!=contract or receipt.get('complete') is not True or receipt['epochs_completed']!=20:
            raise ValueError('Completed local training required')
        checkpoint = args.study/'training/best.pt'
        if sha256(checkpoint)!=receipt['best_sha256']:
            raise ValueError('Selected checkpoint bytes changed')
        best = read(args.study/'training/best-validation-native.json')
        payload = torch.load(checkpoint,map_location='cpu',weights_only=False)
        if payload['contract']!=contract or payload['epoch']!=best['epoch'] or payload['validation']!=best['aggregate']:
            raise ValueError('Selected checkpoint payload changed')
        model.load_state_dict(payload['model'],strict=True)
        check_encoder(model,encoder)
        rows = evaluate_native(model,args.data,selected,args.sam_validation,'cuda',folder/'selected-masks.zip')
        if rows!=best['rows']:
            raise ValueError('Selected checkpoint must replay every native pixel exactly')
        write_json(folder/'result.json',dict(schema=1,complete=True,contract=contract,runtime=runtime,rows=rows,
            aggregate=aggregate(rows),selected_epoch=best['epoch'],checkpoint_sha256=sha256(checkpoint),
            training_sha256=sha256(training_path),best_validation_sha256=sha256(args.study/'training/best-validation-native.json'),
            selected_masks_sha256=sha256(folder/'selected-masks.zip'),exact_pixel_replay=True,benchmark_success=False))
        print(json.dumps(dict(native_complete=True,selected_epoch=best['epoch'],aggregate=aggregate(rows))),flush=True)
        return
    if args.mode=='preflight':
        ids = set(sorted({row['image_id'] for row in selected})[:3])
        selected = [row for row in selected if row['image_id'] in ids]
    initial_rows = evaluate_native(model,args.data,selected,args.sam_validation,'cuda')
    if initial_rows!=[calibration_rows[row['id']] for row in initial_rows]:
        raise ValueError('Original checkpoint failed exact local calibration replay')
    train = RobustData(args.data,[row for row in manifest['rows'] if row['split']=='train'],
        args.sam_training,sam_training['rows'],config['seed'],settings['size'])
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
        lr=settings['learning_rate'],weight_decay=settings['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,settings['epochs'])
    scaler = torch.cuda.amp.GradScaler(init_scale=1024.)
    if args.mode=='preflight':
        trained = train_epoch(model,train,optimizer,scaler,settings,config['seed'],0,2)
        check_encoder(model,encoder)
        if not optimizer.state or trained['skipped_optimizer_steps'] or trained['optimizer_steps']!=2:
            raise RuntimeError('Preflight must perform two actual optimizer updates')
        rows = evaluate_native(model,args.data,selected,args.sam_validation,'cuda')
        atomic_save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),contract=contract),folder/'preflight.pt')
        saved = torch.load(folder/'preflight.pt',map_location='cpu',weights_only=False)
        if saved['contract']!=contract:
            raise ValueError('Preflight checkpoint roundtrip failed')
        restored = make_network(pretrained=False).cuda()
        restored.load_state_dict(saved['model'],strict=True)
        check_encoder(restored,encoder)
        if evaluate_native(restored,args.data,selected,args.sam_validation,'cuda')!=rows:
            raise ValueError('Preflight checkpoint native pixels changed after reloading')
        write_json(folder/'result.json',dict(passed=True,contract=contract,runtime=runtime,training=trained,
            native=aggregate(rows),checkpoint_pixel_replay=True,
            max_reserved_gpu_mib=torch.cuda.max_memory_reserved()/1024**2,benchmark_success=False))
        print('Local preflight passed',flush=True)
        return
    preflight = read(args.study/'preflight/result.json')
    if not preflight['passed'] or preflight['contract']!=contract:
        raise ValueError('Matching real optimizer preflight required')
    floor,baseline = calibration['selection_floor'],calibration['local']
    history,start,best_score = [],0,(floor['hidden_iou'],floor['full_iou'])
    receipt = dict(schema=8,method='shape-probability-workers4',platform='LocalWSL',contract=contract,runtime=runtime,
        initial_validation=baseline,original_quadro_validation=calibration['original_quadro'],selection_floor=floor,
        complete=False,holdout_used=False,benchmark_used=False,runs=[])
    if args.resume:
        receipt = read(training_path)
        saved = torch.load(folder/'last.pt',map_location='cpu',weights_only=False)
        if receipt['contract']!=contract or saved['contract']!=contract or receipt['history']!=saved['history']:
            raise ValueError('Resume contract or history changed')
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler'])
        scaler.load_state_dict(saved['scaler'])
        torch.set_rng_state(saved['rng']['torch'])
        torch.cuda.set_rng_state(saved['rng']['cuda'])
        history,start,best_score = saved['history'],saved['epoch'],tuple(saved['best_score'])
    else:
        atomic_save(dict(model=model.state_dict(),contract=contract,epoch=0,validation=baseline),folder/'best.pt')
        write_json(folder/'best-validation-native.json',dict(epoch=0,aggregate=baseline,rows=initial_rows,contract=contract))
    receipt['runs'].append(dict(started=datetime.now(timezone.utc).isoformat(),resumed_epoch=start))
    write_json(training_path,receipt)
    for epoch in range(start,settings['epochs']):
        trained = train_epoch(model,train,optimizer,scaler,settings,config['seed'],epoch)
        if trained['sampled_training_objects']!=9494:
            raise ValueError('Local epoch lost or duplicated training objects')
        check_encoder(model,encoder)
        rows = evaluate_native(model,args.data,selected,args.sam_validation,'cuda')
        result = aggregate(rows)
        history.append(dict(epoch=epoch+1,validation=result,**trained))
        if eligible(result,floor,best_score):
            best_score = result['hidden_iou'],result['full_iou']
            atomic_save(dict(model=model.state_dict(),contract=contract,epoch=epoch+1,validation=result),folder/'best.pt')
            write_json(folder/'best-validation-native.json',dict(epoch=epoch+1,aggregate=result,rows=rows,contract=contract))
        scheduler.step()
        atomic_save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
            scaler=scaler.state_dict(),epoch=epoch+1,history=history,best_score=best_score,contract=contract,
            rng=dict(torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state())),folder/'last.pt')
        selected_epoch = read(folder/'best-validation-native.json')['epoch']
        receipt.update(epochs_completed=epoch+1,history=history,selected_epoch=selected_epoch,new_checkpoint_eligible=selected_epoch>0,
            best_sha256=sha256(folder/'best.pt'),best_validation_sha256=sha256(folder/'best-validation-native.json'),
            last_sha256=sha256(folder/'last.pt'),complete=epoch+1==settings['epochs'])
        receipt['runs'][-1].update(elapsed_seconds=time.monotonic()-started,last_updated=datetime.now(timezone.utc).isoformat())
        write_json(training_path,receipt)
        print(json.dumps(history[-1]),flush=True)


if __name__=='__main__':
    main()

"""Independently replay local shape calibration, native masks and model selection."""
import argparse
from collections import defaultdict
import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from amodal_benchmark.dataset import sha256, write_json
from scripts.recover_shape_sam_validation import check_ids, geometry, mask, paired_summary
from scripts.verify_shape_v2_metadata import equal

STUDY = ROOT/'output/shape-local-20261123'
BASELINE = ROOT/'output/shape-validation-sam-20261123/native/result'
DATA = ROOT/'output/shape-v2-20261123/data'
CONFIG = 'configs/research/shape-local-constrained-20261123.json'
SOURCE_SHA256 = 'dd445099f7171e6ed853bffaf35ab8bd5adffe8ba1a4ad6d337961aa7100e2be'
DATA_SHA256 = 'ab2f4d1112c52bcd1b675e0f213ed8ee387fbc2601a3596f8f2b5e933ea7d13e'
SOURCE_NAMES = ('scripts/train_shape_local.py','scripts/shape_local_data.py','scripts/shape_robust_data.py',
    'scripts/shape_constrained_data.py','scripts/train_shape_kaggle.py','scripts/train_shape_v2.py',
    'scripts/train_shape_v2_slurm.py','amodal_benchmark/__init__.py','amodal_benchmark/dataset.py',
    'amodal_benchmark/shape_v2.py','amodal_benchmark/shape_v2_runtime.py',
    'configs/research/shape-v2-20261123.json','configs/research/shape-v2-slurm-20261123.json')
read = lambda path: json.loads(path.read_bytes())


def aggregate(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row['image_id']].append(row)
    result = dict(images=len(groups),objects=len(rows))
    for metric in ('full_iou','hidden_iou'):
        values = [[row[metric] for row in group if row[metric] is not None] for group in groups.values()]
        values = [float(np.mean(group)) for group in values if group]
        result[metric] = float(np.mean(values))
        result[metric+'_images'] = len(values)
    return result


def select_epoch(training):
    """Recompute both baseline floors and stable eligibility without trainer code."""
    local,original = training['initial_validation'],training['original_quadro_validation']
    floor = {key:max(local[key],original[key]) for key in ('full_iou','hidden_iou')}
    equal(training['selection_floor'],floor,'selection floor')
    chosen = dict(epoch=0,validation=local)
    best = floor['hidden_iou'],floor['full_iou']
    for row in [dict(validation=local),dict(validation=original),*training['history']]:
        for metric in floor:
            value = row['validation'][metric]
            if type(value) not in (int,float) or not np.isfinite(value) or not 0<=value<=1:
                raise ValueError('Nonfinite or invalid native geometry')
    for row in training['history']:
        values = row['validation']
        score = values['hidden_iou'],values['full_iou']
        if all(values[key]>=floor[key] for key in floor) and score>best:
            best,chosen = score,row
    return chosen


def binary_payload(payload, shape):
    with Image.open(io.BytesIO(payload)) as image:
        array = np.array(image)
    if array.shape!=tuple(shape) or not np.isin(array,[0,255]).all():
        raise ValueError('Invalid native binary mask or dimensions')
    return array>0


def source_contract():
    source = STUDY/'training-source-v2.zip'
    manifest = read(STUDY/'training-source-v2-manifest.json')
    equal(sha256(source),SOURCE_SHA256,'frozen local archive')
    equal(manifest['zip_sha256'],SOURCE_SHA256,'frozen source manifest')
    with zipfile.ZipFile(source) as bundle:
        equal(sorted(bundle.namelist()),sorted(manifest['files']),'source archive inventory')
        for name,expected in manifest['files'].items():
            equal(hashlib.sha256(bundle.read(name)).hexdigest(),expected,'frozen source '+name)
        config = json.loads(bundle.read(CONFIG))
    initial = read(ROOT/'output/shape-v2-slurm-20261123/training/training.json')
    equal(initial['best_sha256'],config['initial_checkpoint_sha256'],'original initialization')
    contract = dict(config_sha256=manifest['files'][CONFIG],sources={name:manifest['files'][name] for name in SOURCE_NAMES},
        data_sha256=DATA_SHA256,initial_checkpoint_sha256=config['initial_checkpoint_sha256'],
        inherited_training_contract=initial['contract'],
        sam_training_manifest_sha256=sha256(ROOT/'output/shape-sam-robustness-20261123/recovered/training-masks/manifest.json'),
        validation_sam_result_sha256=config['validation_diagnostic_result_sha256'])
    equal(sha256(BASELINE/'result.json'),contract['validation_sam_result_sha256'],'original SAM diagnostic')
    return config,contract


def replay_masks(archive_path, rows, selected, manifest, baseline, compare_original=False):
    check_ids(rows,selected)
    pairs,differences = [],[]
    with zipfile.ZipFile(archive_path) as archive:
        equal(sorted(archive.namelist()),sorted(key+'.png' for key in selected),'native ZIP inventory')
        for row in rows:
            source = selected[row['id']]
            equal(row['image_id'],source['image_id'],'native image identity')
            truth_path = DATA/source['mask']
            equal(sha256(truth_path),manifest['artifacts'][source['mask']],'annotated truth bytes')
            with Image.open(truth_path) as image:
                encoded = np.array(image)
            truth,visible = (encoded&1)>0,(encoded&2)>0
            equal(row['shape'],list(truth.shape),'native frame')
            prediction = binary_payload(archive.read(row['id']+'.png'),truth.shape)
            equal(hashlib.sha256(prediction.astype(np.uint8).tobytes()).hexdigest(),row['prediction_sha256'],'native pixels')
            computed = geometry(prediction,truth,visible)
            equal(computed,{key:row[key] for key in computed},'independent geometry')
            original = baseline[row['id']]
            support_path = BASELINE/'objects'/row['id']/'sam-visible.png'
            equal(sha256(support_path),original['artifacts']['sam-visible.png'],'SAM support bytes')
            support = mask(support_path,truth.shape)
            if np.any(support & ~prediction) or (not support.any() and prediction.any()):
                raise ValueError('Native support/empty-mask policy differs')
            if support.any():
                yy,xx = np.nonzero(support)
                side = int(np.ceil(max(xx.max()-xx.min()+1,yy.max()-yy.min()+1)*2.))
                x0,y0 = int(np.floor((xx.min()+xx.max()+1-side)/2)),int(np.floor((yy.min()+yy.max()+1-side)/2))
                equal(row['box'],[x0,y0,x0+side,y0+side],'native crop')
            elif row['box'] is not None:
                raise ValueError('Empty support has a crop')
            pairs.append(dict(image_id=row['image_id'],annotated_input=original['sam_input'],sam_input=computed))
            if compare_original:
                path = BASELINE/'objects'/row['id']/'sam-prediction.png'
                equal(sha256(path),original['artifacts']['sam-prediction.png'],'original native mask')
                original_prediction = mask(path,truth.shape)
                equal(geometry(original_prediction,truth,visible),original['sam_input'],'original native geometry')
                count = int(np.count_nonzero(prediction!=original_prediction))
                if count:
                    differences.append(dict(id=row['id'],different_pixels=count))
    return pairs,differences


def replay(run, calibration_only=False):
    run = Path(run)
    config,contract = source_contract()
    calibration = read(run/'calibration/result.json')
    equal(calibration['contract'],contract,'calibration contract')
    for key in ('complete','exact_repeat'):
        equal(calibration[key],True,'completed original replay '+key)
    equal(calibration['training_performed'],False,'no calibration optimization')
    equal(calibration['benchmark_success'],False,'calibration scope')
    equal(calibration['original_masks_sha256'],sha256(run/'calibration/initial-masks.zip'),'original masks archive')
    runtime = calibration['runtime']
    if (runtime['torch']!='1.13.1' or runtime['cuda']!='11.7' or runtime['cudnn']!=8500
            or 'RTX 3060' not in runtime['gpu'] or runtime['cuda_fft_round_trip'] is not True or runtime['memory_fraction']!=.75):
        raise ValueError('Pinned local GPU runtime changed')
    with zipfile.ZipFile(ROOT/'output/shape-v2-20261123/colab/shape-v2-training-data.zip') as archive:
        packed = archive.read('data/manifest.json')
    equal(hashlib.sha256(packed).hexdigest(),DATA_SHA256,'isolated input manifest')
    manifest = json.loads(packed)
    selected = {row['id']:row for row in manifest['rows'] if row['split']=='validation'}
    equal([len(selected),len({row['image_id'] for row in selected.values()})],[1264,217],'validation inventory')
    baseline = {row['id']:row for row in read(BASELINE/'result.json')['rows']}
    pairs,differences = replay_masks(run/'calibration/initial-masks.zip',calibration['rows'],selected,manifest,baseline,True)
    equal(aggregate(calibration['rows']),calibration['local'],'local baseline aggregation')
    original_rows = [dict(id=key,image_id=value['image_id'],**baseline[key]['sam_input']) for key,value in selected.items()]
    equal(aggregate(original_rows),calibration['original_quadro'],'original baseline aggregation')
    geometry_differences = [row['id'] for row in calibration['rows']
        if any(row[key]!=baseline[row['id']]['sam_input'][key] for key in ('full_iou','hidden_iou'))]
    equal(geometry_differences,calibration['original_quadro_different_geometry_objects'],'cross-GPU geometry differences')
    floor = {key:max(calibration['local'][key],calibration['original_quadro'][key]) for key in ('full_iou','hidden_iou')}
    equal(calibration['selection_floor'],floor,'both-platform retention floors')
    result = dict(verified=True,objects=1264,images=217,calibration_sha256=sha256(run/'calibration/result.json'),
        original_quadro=calibration['original_quadro'],local_baseline=calibration['local'],selection_floor=floor,
        original_quadro_different_mask_objects=len(differences),original_quadro_different_pixels=sum(row['different_pixels'] for row in differences),
        original_quadro_pixel_differences=differences,calibration_geometry=paired_summary(pairs),
        source_archive_sha256=SOURCE_SHA256,replay_source_sha256=sha256(Path(__file__)),benchmark_success=False)
    if calibration_only:
        result.update(scope='calibration only; no training success or model admission')
        return result
    contract['calibration_sha256'] = result['calibration_sha256']
    preflight = read(run/'preflight/result.json')
    equal(preflight['contract'],contract,'preflight contract')
    equal(preflight['runtime'],runtime,'preflight runtime')
    equal([preflight['passed'],preflight['checkpoint_pixel_replay']],[True,True],'real checkpoint preflight')
    equal([preflight['training']['optimizer_steps'],preflight['training']['skipped_optimizer_steps']],[2,0],'actual preflight steps')
    training = read(run/'training/training.json')
    equal(training['contract'],contract,'training contract')
    equal(training['runtime'],runtime,'training runtime')
    equal([training['schema'],training['method'],training['platform'],training['complete'],training['epochs_completed'],
        training['holdout_used'],training['benchmark_used']],[6,'shape-local-constrained','LocalWSL',True,20,False,False],'completed isolated training')
    equal([row['epoch'] for row in training['history']],list(range(1,21)),'complete epoch history')
    equal(training['initial_validation'],calibration['local'],'initial validation')
    equal(training['original_quadro_validation'],calibration['original_quadro'],'original Quadro validation')
    for row in training['history']:
        equal([row['sampled_training_objects'],row['optimizer_steps']],[9494,149],'all training objects')
        if not 0<=row['skipped_optimizer_steps']<row['optimizer_steps'] or not np.isfinite(row['loss']):
            raise ValueError('Invalid optimizer evidence')
    chosen = select_epoch(training)
    equal(training['selected_epoch'],chosen['epoch'],'selected local epoch')
    equal(training['new_checkpoint_eligible'],chosen['epoch']>0,'nonzero epoch eligibility')
    equal(sha256(run/'training/best.pt'),training['best_sha256'],'selected checkpoint')
    equal(sha256(run/'training/last.pt'),training['last_sha256'],'last checkpoint')
    equal(sha256(run/'training/best-validation-native.json'),training['best_validation_sha256'],'selected validation binding')
    best = read(run/'training/best-validation-native.json')
    equal(best['epoch'],chosen['epoch'],'selected native epoch')
    equal(best['aggregate'],chosen['validation'],'selected native aggregate')
    equal(best['contract'],contract,'selected validation contract')
    native = read(run/'native/result.json')
    required = dict(schema=1,complete=True,contract=contract,runtime=runtime,selected_epoch=chosen['epoch'],
        checkpoint_sha256=training['best_sha256'],training_sha256=sha256(run/'training/training.json'),
        best_validation_sha256=training['best_validation_sha256'],selected_masks_sha256=sha256(run/'native/selected-masks.zip'),
        exact_pixel_replay=True,benchmark_success=False)
    for key,value in required.items():
        equal(native[key],value,'native binding '+key)
    if native['rows']!=best['rows']:
        raise ValueError('Native checkpoint replay must match exact pixel hashes and geometry')
    pairs,_ = replay_masks(run/'native/selected-masks.zip',native['rows'],selected,manifest,baseline)
    equal(aggregate(native['rows']),native['aggregate'],'selected aggregation')
    equal(native['aggregate'],chosen['validation'],'selected training geometry')
    initial_rows = {row['id']:row for row in calibration['rows']}
    local_pairs = [dict(image_id=row['image_id'],annotated_input={key:initial_rows[row['id']][key] for key in floor},
        sam_input={key:row[key] for key in floor}) for row in native['rows']]
    result.update(scope='completed native validation; no benchmark admission or benchmark result',selected_epoch=chosen['epoch'],
        new_checkpoint_eligible=chosen['epoch']>0,selected_geometry=native['aggregate'],geometry_vs_quadro=paired_summary(pairs),
        geometry_vs_local=paired_summary(local_pairs),training_sha256=required['training_sha256'],
        checkpoint_sha256=training['best_sha256'],native_sha256=sha256(run/'native/result.json'))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=STUDY/'run-v2')
    parser.add_argument('--calibration-only',action='store_true')
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve existing replay evidence')
    result = replay(args.run,args.calibration_only)
    write_json(args.output,result)
    print(json.dumps({key:value for key,value in result.items() if key not in ('original_quadro_pixel_differences','geometry_vs_local','geometry_vs_quadro','calibration_geometry')}))


if __name__=='__main__':
    main()

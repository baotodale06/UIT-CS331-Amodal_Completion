"""Fixed-bias shape-probability diagnosis, isolated from live training."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import sys
import time
import zipfile

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from amodal_benchmark.dataset import sha256, write_json
from amodal_benchmark.shape_v2 import make_network, observation_tensor, visible_box
from scripts.shape_robust_data import aggregate, geometry, native_canvas

CONFIG = 'configs/research/shape-probability-diagnostic-20261123.json'


def corrected_canvas(logits, box, support, bias):
    """Retain the original FP32 Torch sigmoid arithmetic and visible support."""
    import torch
    if not np.isfinite(bias) or not torch.isfinite(logits).all():
        raise ValueError('Finite logits and bias required')
    probability = (logits + bias).sigmoid().cpu().numpy()
    canvas = native_canvas(probability, box, support.shape)
    canvas[support] = 1
    return canvas


def probability_scores(probability, truth, support):
    if probability.shape != truth.shape or support.shape != truth.shape:
        raise ValueError('Native probability frames differ')
    if not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
        raise ValueError('Finite bounded probabilities required')
    eligible = ~support
    if not eligible.any():
        return dict(brier=None, nll=None)
    p, y = probability[eligible].astype(np.float64), truth[eligible].astype(np.float64)
    clipped = np.clip(p, 1e-7, 1-1e-7)
    return dict(brier=float(np.mean((p-y)**2)),
                nll=float(np.mean(-y*np.log(clipped)-(1-y)*np.log1p(-clipped))))


def summarize(rows):
    result = aggregate(rows)
    groups = defaultdict(list)
    for row in rows:
        groups[row['image_id']].append(row)
    for key in ('brier', 'nll'):
        means = [np.mean([row[key] for row in group if row[key] is not None])
                 for group in groups.values() if any(row[key] is not None for row in group)]
        result[key] = float(np.mean(means)) if means else None
        result[key+'_images'] = len(means)
    return result


def read(path):
    return json.loads(Path(path).read_bytes())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--sam-validation', type=Path, required=True)
    parser.add_argument('--initial-checkpoint', type=Path, required=True)
    parser.add_argument('--adapted-checkpoint', type=Path, required=True)
    parser.add_argument('--adapted-native', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--preflight-objects', type=int, default=0)
    args = parser.parse_args()
    import torch
    import PIL
    if str(torch.__version__) != '1.13.1' or np.__version__ != '1.23.5' or PIL.__version__ != '9.5.0':
        raise RuntimeError('Pinned numerical environment required')
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    if not torch.cuda.is_available() or 'RTX 3060' not in torch.cuda.get_device_name():
        raise RuntimeError('The separately validated local GPU is required')
    torch.cuda.set_per_process_memory_fraction(.20)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    config = read(ROOT/CONFIG)
    manifest = read(args.data/'manifest.json')
    if (sha256(args.data/'manifest.json') != config['data_manifest_sha256']
            or manifest['holdout_packaged'] is not False
            or {row['split'] for row in manifest['rows']} != {'train', 'validation'}):
        raise ValueError('Isolated data contract changed')
    selected = sorted((row for row in manifest['rows'] if row['split']=='validation'), key=lambda row:row['id'])
    if len(selected)!=1264 or len({row['image_id'] for row in selected})!=217:
        raise ValueError('Validation inventory changed')
    if args.preflight_objects:
        if args.preflight_objects < 1:
            raise ValueError('Positive preflight count required')
        selected = selected[:args.preflight_objects]
    baseline = read(args.sam_validation/'result.json')
    original_rows = {row['id']:row for row in baseline['rows']}
    adapted = read(args.adapted_native/'result.json')
    adapted_rows = {row['id']:row for row in adapted['rows']}
    native_contract = read(args.adapted_native/'contract.json')
    if (sha256(args.sam_validation/'result.json') != native_contract['baseline_result_sha256']
            or native_contract['checkpoint_sha256'] != config['adapted_checkpoint_sha256']):
        raise ValueError('Completed native references differ')
    sources = [CONFIG, 'scripts/diagnose_shape_probability.py', 'amodal_benchmark/shape_v2.py',
               'scripts/shape_robust_data.py', 'amodal_benchmark/dataset.py']
    contract = dict(config=config, sources={name:sha256(ROOT/name) for name in sources},
                    baseline_sha256=sha256(args.sam_validation/'result.json'),
                    adapted_native_sha256=sha256(args.adapted_native/'result.json'),
                    started=datetime.now(timezone.utc).isoformat(), preflight=bool(args.preflight_objects),
                    runtime=dict(device='cuda', gpu=torch.cuda.get_device_name(), torch=str(torch.__version__),
                        numpy=np.__version__, pillow=PIL.__version__, threads=2, memory_fraction=.20,
                        deterministic=dict(initial=False, adapted=True), tf32=False))
    models = {}
    for name, path in [('initial', args.initial_checkpoint), ('adapted', args.adapted_checkpoint)]:
        if sha256(path)!=config[name+'_checkpoint_sha256']:
            raise ValueError('Checkpoint bytes differ')
        saved = torch.load(path, map_location='cpu', weights_only=False)
        if name=='adapted' and saved['epoch']!=config['adapted_epoch']:
            raise ValueError('Selected epoch differs')
        model = make_network(pretrained=False).cuda().eval()
        model.load_state_dict(saved['model'], strict=True)
        models[name] = model
        del saved
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output/'contract.json', contract)
    records, verified = [], set()
    started = time.monotonic()
    with torch.inference_mode(), zipfile.ZipFile(args.output/'masks.zip', 'x', zipfile.ZIP_DEFLATED) as archive:
        for index, row in enumerate(selected, 1):
            for name in (row['image'], row['mask']):
                asset = (args.data/name).resolve()
                if not asset.is_relative_to(args.data.resolve()):
                    raise ValueError('Data path escapes root')
                if name not in verified and sha256(asset)!=manifest['artifacts'][name]:
                    raise ValueError('Validation asset bytes changed')
                verified.add(name)
            original = original_rows[row['id']]
            learned = adapted_rows[row['id']]
            for reference in (original, learned):
                if any(reference[key]!=row[key] for key in ('image_id', 'image', 'mask')):
                    raise ValueError('Reference object identity changed')
            folder = args.sam_validation/'objects'/row['id']
            support_path = folder/'sam-visible.png'
            if sha256(support_path)!=original['artifacts']['sam-visible.png'] or sha256(support_path)!=learned['sam_sha256']:
                raise ValueError('SAM input bytes changed')
            with Image.open(args.data/row['image']) as image:
                rgb = np.array(image.convert('RGB'))
            with Image.open(args.data/row['mask']) as image:
                encoded = np.array(image)
            with Image.open(support_path) as image:
                support = np.array(image)>0
            truth, visible = (encoded&1)>0, (encoded&2)>0
            if support.shape!=truth.shape or rgb.shape[:2]!=truth.shape:
                raise ValueError('Input frames differ')
            record = dict(id=row['id'], image_id=row['image_id'], variants={}, native_reference_differences={})
            box = visible_box(support, 2.) if support.any() else None
            if box:
                cropped = np.array(Image.fromarray(support).crop(box))
                observed = np.array(Image.fromarray(rgb).crop(box))
                inputs = torch.from_numpy(observation_tensor(observed, cropped)).unsqueeze(0).cuda()
            for name, model in models.items():
                torch.backends.cudnn.deterministic = name=='adapted'
                reference_path = folder/'sam-prediction.png' if name=='initial' else args.adapted_native/'objects'/row['id']/'prediction.png'
                expected = original['artifacts']['sam-prediction.png'] if name=='initial' else learned['prediction_sha256']
                if sha256(reference_path)!=expected:
                    raise ValueError('Native reference mask changed')
                with Image.open(reference_path) as image:
                    reference = (np.array(image)>0)|support
                logits = torch.nn.functional.interpolate(model(inputs), cropped.shape, mode='bilinear', align_corners=False)[0,0] if box else None
                biases = [0.] if name=='initial' else config['logit_biases']
                for bias in biases:
                    key = name if name=='initial' else 'adapted_bias_'+format(bias, 'g')
                    probability = corrected_canvas(logits, box, support, bias) if box else np.zeros(truth.shape, np.float32)
                    prediction = probability>=.5
                    stream = io.BytesIO()
                    Image.fromarray(prediction.astype(np.uint8)*255).save(stream, format='PNG')
                    archive.writestr(key+'/'+row['id']+'.png', stream.getvalue())
                    record['variants'][key] = dict(**geometry(prediction, truth, visible),
                        **probability_scores(probability, truth, support),
                        prediction_sha256=hashlib.sha256(prediction.astype(np.uint8).tobytes()).hexdigest())
                    if bias==0:
                        record['native_reference_differences'][name] = int(np.count_nonzero(prediction!=reference))
            records.append(record)
            if index%25==0 or index==len(selected):
                print(json.dumps(dict(objects=index, total=len(selected), seconds=time.monotonic()-started)), flush=True)
    if contract['sources']!={name:sha256(ROOT/name) for name in sources}:
        raise ValueError('Diagnostic source changed during execution')
    keys = records[0]['variants']
    summaries = {key:summarize([dict(id=row['id'], image_id=row['image_id'], **row['variants'][key]) for row in records]) for key in keys}
    differences = {key:dict(objects=sum(row['native_reference_differences'][key]>0 for row in records),
        pixels=sum(row['native_reference_differences'][key] for row in records)) for key in models}
    result = dict(complete=True, contract_sha256=sha256(args.output/'contract.json'),
        mask_archive_sha256=sha256(args.output/'masks.zip'), preflight=bool(args.preflight_objects),
        summaries=summaries, native_reference_differences=differences, rows=records,
        elapsed_seconds=time.monotonic()-started, peak_allocated_gpu_mib=torch.cuda.max_memory_allocated()/1024**2,
        benchmark_success=False, promoted=False)
    write_json(args.output/'result.json', result)
    print(json.dumps(dict(summaries=summaries, native_reference_differences=differences, benchmark_success=False)), flush=True)


if __name__=='__main__':
    main()

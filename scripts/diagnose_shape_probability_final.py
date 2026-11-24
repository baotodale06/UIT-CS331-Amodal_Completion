"""Evaluate prespecified positive offsets on the completed final epoch only."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import io
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
from scripts.diagnose_shape_probability import corrected_canvas, probability_scores, summarize
from scripts.shape_local_data import evaluate_native, native_arithmetic
from scripts.shape_robust_data import aggregate, geometry
from scripts.shape_probability_final_contract import (
    CONFIG, load_parent, pixel_hash, read, retained_offsets,
    validate_checkpoint, variant_name, verify_sources,
)


def run(args):
    import PIL
    import torch
    policy = read(ROOT / CONFIG)
    source_hash = verify_sources(args.source_manifest)
    training, bindings = load_parent(args.parent, policy)
    expected = policy['runtime']
    if (str(torch.__version__) != expected['torch'] or torch.version.cuda != expected['cuda']
            or torch.backends.cudnn.version() != expected['cudnn']
            or np.__version__ != expected['numpy'] or PIL.__version__ != expected['pillow']
            or not torch.cuda.is_available()
            or expected['gpu_contains'] not in torch.cuda.get_device_name()):
        raise RuntimeError('The pinned local GPU runtime is required')
    torch.set_num_threads(expected['threads'])
    torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.set_per_process_memory_fraction(expected['memory_fraction'])
    manifest = read(args.data / 'manifest.json')
    if (sha256(args.data / 'manifest.json') != policy['data_manifest_sha256']
            or manifest['holdout_packaged'] is not False
            or {row['split'] for row in manifest['rows']} != {'train', 'validation'}):
        raise ValueError('Isolated training/validation data required')
    selected = sorted((row for row in manifest['rows'] if row['split'] == 'validation'), key=lambda row: row['id'])
    if [len(selected), len({row['image_id'] for row in selected})] != [1264, 217]:
        raise ValueError('Validation inventory changed')
    if args.preflight_objects:
        if args.preflight_objects != 3:
            raise ValueError('The preflight uses exactly three objects')
        selected = selected[:3]
    baseline_path = args.sam_validation / 'result.json'
    if sha256(baseline_path) != training['contract']['validation_sam_result_sha256']:
        raise ValueError('Original native SAM references changed')
    baseline = {row['id']: row for row in read(baseline_path)['rows']}
    verified = set()
    for row in selected:
        for name in (row['image'], row['mask']):
            path = (args.data / name).resolve()
            if not path.is_relative_to(args.data.resolve()):
                raise ValueError('Input path escapes isolated data')
            if name not in verified and sha256(path) != manifest['artifacts'][name]:
                raise ValueError('Input bytes changed: ' + name)
            verified.add(name)
        original = baseline[row['id']]
        if any(original[key] != row[key] for key in ('image_id', 'image', 'mask')):
            raise ValueError('Original object identity changed')
        folder = args.sam_validation / 'objects' / row['id']
        for name in ('sam-visible.png', 'sam-prediction.png'):
            if sha256(folder / name) != original['artifacts'][name]:
                raise ValueError('Original native artifact changed')
    if sha256(args.initial_checkpoint) != policy['initial_checkpoint_sha256']:
        raise ValueError('Original checkpoint changed')
    models = {}
    for name, path in [('initial', args.initial_checkpoint), ('final', args.parent / 'run/training/last.pt')]:
        saved = torch.load(path, map_location='cpu', weights_only=False)
        if name == 'final':
            validate_checkpoint(saved, training)
        model = make_network(pretrained=False).cuda().eval()
        model.load_state_dict(saved['model'], strict=True)
        models[name] = model
        del saved
    args.output.mkdir(parents=True, exist_ok=False)
    contract = dict(schema=1, policy=policy, source_manifest_sha256=source_hash,
                    policy_sha256=sha256(ROOT / CONFIG), **bindings,
                    baseline_result_sha256=sha256(baseline_path),
                    initial_checkpoint_sha256=sha256(args.initial_checkpoint),
                    preflight=bool(args.preflight_objects), fixed_epoch=20,
                    started=datetime.now(timezone.utc).isoformat(), runtime=expected,
                    holdout_access=False, benchmark_access=False)
    write_json(args.output / 'contract.json', contract)
    started = time.monotonic()
    # This pass is independent of the offset loop. Its exported masks establish
    # exact repeatability of the final checkpoint, whose training stored aggregates.
    raw = evaluate_native(models['final'], args.data, selected, args.sam_validation,
                          'cuda', args.output / 'final-zero-reference.zip')
    raw_aggregate = aggregate(raw)
    if not args.preflight_objects and raw_aggregate != training['history'][-1]['validation']:
        raise ValueError('Unshifted final geometry differs from the recorded epoch20')
    write_json(args.output / 'final-zero-reference.json', dict(rows=raw, aggregate=raw_aggregate))
    raw = {row['id']: row for row in raw}
    groups = defaultdict(list)
    for row in selected:
        groups[row['image_id']].append(row)
    records = []
    with native_arithmetic(), torch.inference_mode(), zipfile.ZipFile(args.output / 'masks.zip', 'x', zipfile.ZIP_DEFLATED) as archive:
        for image_id, objects in sorted(groups.items()):
            with Image.open(args.data / objects[0]['image']) as image:
                rgb = np.array(image.convert('RGB'))
            for row in objects:
                with Image.open(args.data / row['mask']) as image:
                    encoded = np.array(image)
                folder = args.sam_validation / 'objects' / row['id']
                with Image.open(folder / 'sam-visible.png') as image:
                    support = np.array(image) > 0
                with Image.open(folder / 'sam-prediction.png') as image:
                    original = np.array(image) > 0
                truth, visible = (encoded & 1) > 0, (encoded & 2) > 0
                if support.shape != truth.shape or rgb.shape[:2] != truth.shape:
                    raise ValueError('Native frames differ')
                box = visible_box(support, 2.) if support.any() else None
                if box:
                    cropped = np.array(Image.fromarray(support).crop(box))
                    observed = np.array(Image.fromarray(rgb).crop(box))
                    inputs = torch.from_numpy(observation_tensor(observed, cropped)).unsqueeze(0).cuda()
                record = dict(id=row['id'], image_id=image_id, image=row['image'], mask=row['mask'],
                              shape=list(truth.shape), variants={})
                for name, model in models.items():
                    logits = torch.nn.functional.interpolate(model(inputs), cropped.shape,
                        mode='bilinear', align_corners=False)[0, 0] if box else None
                    biases = [0.] if name == 'initial' else policy['logit_biases']
                    for bias in biases:
                        key = 'initial' if name == 'initial' else variant_name(bias)
                        probability = corrected_canvas(logits, box, support, bias) if box else np.zeros(truth.shape, np.float32)
                        prediction = probability >= .5
                        if name == 'initial' and not np.array_equal(prediction, original):
                            raise ValueError('Initial native pixels differ from original Quadro: ' + row['id'])
                        if name == 'final' and bias == 0 and pixel_hash(prediction) != raw[row['id']]['prediction_sha256']:
                            raise ValueError('Final zero-offset native pixels did not repeat: ' + row['id'])
                        record['variants'][key] = dict(**geometry(prediction, truth, visible),
                            **probability_scores(probability, truth, support), prediction_sha256=pixel_hash(prediction))
                        stream = io.BytesIO()
                        Image.fromarray(prediction.astype(np.uint8) * 255).save(stream, format='PNG')
                        archive.writestr(key + '/' + row['id'] + '.png', stream.getvalue())
                records.append(record)
                if len(records) % 25 == 0 or len(records) == len(selected):
                    print(dict(objects=len(records), total=len(selected), seconds=time.monotonic() - started), flush=True)
    records.sort(key=lambda row: row['id'])
    if verify_sources(args.source_manifest) != source_hash or load_parent(args.parent, policy)[1] != bindings:
        raise ValueError('Source or completed parent changed during diagnosis')
    keys = ['initial'] + [variant_name(bias) for bias in policy['logit_biases']]
    summaries = {key: summarize([dict(id=row['id'], image_id=row['image_id'], **row['variants'][key]) for row in records]) for key in keys}
    result = dict(complete=True, preflight=bool(args.preflight_objects), fixed_epoch=20,
        contract_sha256=sha256(args.output / 'contract.json'),
        mask_archive_sha256=sha256(args.output / 'masks.zip'),
        raw_reference_archive_sha256=sha256(args.output / 'final-zero-reference.zip'),
        raw_reference_json_sha256=sha256(args.output / 'final-zero-reference.json'),
        summaries=summaries, rows=records, exact_initial_pixels=True, exact_final_repeat=True,
        retained_offsets=retained_offsets(summaries, training['selection_floor'], policy['logit_biases']) if not args.preflight_objects else None,
        elapsed_seconds=time.monotonic() - started, benchmark_success=False, promoted=False)
    write_json(args.output / 'result.json', result)
    print(dict(complete=True, objects=len(records), retained_offsets=result['retained_offsets'], benchmark_success=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('parent', 'data', 'sam-validation', 'initial-checkpoint', 'source-manifest', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--preflight-objects', type=int, default=0)
    run(parser.parse_args())


if __name__ == '__main__':
    main()

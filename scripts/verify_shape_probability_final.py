"""Replay every exported diagnostic mask without importing the GPU evaluator."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import io
from pathlib import Path
import sys
import zipfile

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from amodal_benchmark.dataset import sha256, write_json
from scripts.shape_probability_final_contract import (
    CONFIG, load_parent, read, variant_name, verify_sources,
)


def equal(actual, expected, label):
    if actual != expected:
        raise ValueError(label + ' differs')


def binary(payload, shape):
    with Image.open(io.BytesIO(payload)) as source:
        values = np.array(source)
    if values.shape != tuple(shape) or not np.isin(values, [0, 255]).all():
        raise ValueError('Binary mask or native dimensions differ')
    return values > 0


def geometry(prediction, truth, visible):
    hidden = truth & ~visible
    predicted_hidden = prediction & ~visible
    return dict(full_iou=float(np.count_nonzero(prediction & truth) /
                              max(1, np.count_nonzero(prediction | truth))),
                hidden_iou=float(np.count_nonzero(predicted_hidden & hidden) /
                                 max(1, np.count_nonzero(predicted_hidden | hidden)))
                if hidden.any() else None)


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row['image_id']].append(row)
    result = dict(images=len(groups), objects=len(rows))
    for key in ('full_iou', 'hidden_iou', 'brier', 'nll'):
        values = [[row[key] for row in group if row[key] is not None]
                  for group in groups.values()]
        means = [float(np.mean(group)) for group in values if group]
        result[key] = float(np.mean(means)) if means else None
        result[key + '_images'] = len(means)
    return result


def crop_box(support):
    if not support.any():
        return None
    yy, xx = np.nonzero(support)
    side = int(np.ceil(2 * max(xx.max() - xx.min() + 1, yy.max() - yy.min() + 1)))
    x0 = int(np.floor((xx.min() + xx.max() + 1 - side) / 2))
    y0 = int(np.floor((yy.min() + yy.max() + 1 - side) / 2))
    return [x0, y0, x0 + side, y0 + side]


def check_prediction(prediction, truth, visible, support, expected):
    equal(hashlib.sha256(prediction.astype(np.uint8).tobytes()).hexdigest(),
          expected['prediction_sha256'], 'Native prediction pixels')
    equal(geometry(prediction, truth, visible),
          {key: expected[key] for key in ('full_iou', 'hidden_iou')}, 'Independent geometry')
    if np.any(support & ~prediction) or (not support.any() and prediction.any()):
        raise ValueError('Observable support or empty-support behavior differs')
    box = crop_box(support)
    if box:
        height, width = support.shape
        outside = prediction.copy()
        outside[max(0, box[1]):min(height, box[3]), max(0, box[0]):min(width, box[2])] = False
        if outside.any():
            raise ValueError('Prediction leaves the observable native crop')


def check_probability_scores(row, support):
    for key, maximum in (('brier', 1.), ('nll', -np.log(1e-7) + 1e-8)):
        value = row[key]
        if support.all():
            if value is not None:
                raise ValueError('Probability score must exclude fully observed objects')
        elif type(value) not in (int, float) or not np.isfinite(value) or not 0 <= value <= maximum:
            raise ValueError('Probability score is not finite and bounded')


def retained(summaries, floor, biases):
    return [bias for bias in biases if
            summaries[variant_name(bias)]['full_iou'] >= floor['full_iou'] and
            summaries[variant_name(bias)]['hidden_iou'] >= floor['hidden_iou']]


def replay_masks(folder, data, sam, selected, manifest, baseline, result, raw, biases):
    """Use a small synthetic inventory in tests; production inventory is checked below."""
    ids = [row['id'] for row in selected]
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate validation objects')
    equal([row['id'] for row in result['rows']], ids, 'Diagnostic object order')
    equal([row['id'] for row in raw['rows']], ids, 'Separate native pass order')
    keys = ['initial'] + [variant_name(bias) for bias in biases]
    recomputed = {key: [] for key in keys}
    raw_computed, checked = [], set()
    with zipfile.ZipFile(folder / 'masks.zip') as bundle, zipfile.ZipFile(folder / 'final-zero-reference.zip') as reference:
        equal(sorted(bundle.namelist()), sorted(key + '/' + name + '.png' for key in keys for name in ids),
              'Variant archive inventory')
        equal(sorted(reference.namelist()), sorted(name + '.png' for name in ids), 'Reference archive inventory')
        for source, row, raw_row in zip(selected, result['rows'], raw['rows']):
            for name in (source['image'], source['mask']):
                path = (data / name).resolve()
                if not path.is_relative_to(data.resolve()):
                    raise ValueError('Data path escapes isolated root')
                if name not in checked:
                    equal(sha256(path), manifest['artifacts'][name], 'Validation input bytes')
                    checked.add(name)
            original = baseline[source['id']]
            for key in ('id', 'image_id', 'image', 'mask'):
                equal(row[key], source[key], 'Diagnostic identity')
                equal(original[key], source[key], 'Original reference identity')
            equal(raw_row['image_id'], source['image_id'], 'Separate-pass image identity')
            with Image.open(data / source['mask']) as image:
                encoded = np.array(image)
            truth, visible = (encoded & 1) > 0, (encoded & 2) > 0
            with Image.open(data / source['image']) as image:
                equal(image.size, (truth.shape[1], truth.shape[0]), 'RGB native frame')
            equal(row['shape'], list(truth.shape), 'Diagnostic frame')
            equal(raw_row['shape'], list(truth.shape), 'Separate-pass frame')
            original_masks = {}
            for name in ('sam-visible.png', 'sam-prediction.png'):
                payload = (sam / 'objects' / source['id'] / name).read_bytes()
                equal(hashlib.sha256(payload).hexdigest(), original['artifacts'][name], 'Original mask bytes')
                original_masks[name] = binary(payload, truth.shape)
            support = original_masks['sam-visible.png']
            equal(raw_row['box'], crop_box(support), 'Separate-pass native crop')
            raw_prediction = binary(reference.read(source['id'] + '.png'), truth.shape)
            check_prediction(raw_prediction, truth, visible, support, raw_row)
            raw_computed.append(dict(image_id=source['image_id'], **geometry(raw_prediction, truth, visible),
                                     brier=None, nll=None))
            equal(sorted(row['variants']), sorted(keys), 'Fixed variant grid')
            previous = None
            for key in keys:
                values = row['variants'][key]
                prediction = binary(bundle.read(key + '/' + source['id'] + '.png'), truth.shape)
                check_prediction(prediction, truth, visible, support, values)
                check_probability_scores(values, support)
                if key == 'initial' and not np.array_equal(prediction, original_masks['sam-prediction.png']):
                    raise ValueError('Original Quadro pixels differ')
                if key == variant_name(0) and not np.array_equal(prediction, raw_prediction):
                    raise ValueError('Final zero-offset native pixels differ between passes')
                if key != 'initial':
                    if previous is not None and np.any(previous & ~prediction):
                        raise ValueError('Increasing the fixed offset removed pixels')
                    previous = prediction
                recomputed[key].append(dict(image_id=source['image_id'], **geometry(prediction, truth, visible),
                                            brier=values['brier'], nll=values['nll']))
            equal(geometry(original_masks['sam-prediction.png'], truth, visible), original['sam_input'],
                  'Original reference geometry')
    summaries = {key: summarize(rows) for key, rows in recomputed.items()}
    raw_summary = summarize(raw_computed)
    raw_geometry = {key: value for key, value in raw_summary.items() if not key.startswith(('brier', 'nll'))}
    equal(summaries, result['summaries'], 'Independent variant summaries')
    equal(raw_geometry, raw['aggregate'], 'Independent separate-pass summary')
    return summaries, raw_geometry


def replay(args):
    source_hash = verify_sources(args.source_manifest)
    policy = read(ROOT / CONFIG)
    training, bindings = load_parent(args.parent, policy)
    folder = args.result
    contract, result, raw = [read(folder / name) for name in
                            ('contract.json', 'result.json', 'final-zero-reference.json')]
    for key, value in dict(policy=policy, policy_sha256=sha256(ROOT / CONFIG),
            source_manifest_sha256=source_hash, **bindings, runtime=policy['runtime'],
            initial_checkpoint_sha256=policy['initial_checkpoint_sha256'],
            fixed_epoch=20, holdout_access=False, benchmark_access=False,
            preflight=args.preflight).items():
        equal(contract[key], value, 'Diagnostic contract ' + key)
    for key, value in dict(complete=True, preflight=args.preflight, fixed_epoch=20,
            exact_initial_pixels=True, exact_final_repeat=True, promoted=False, benchmark_success=False).items():
        equal(result[key], value, 'Result ' + key)
    for key, name in [('contract_sha256', 'contract.json'), ('mask_archive_sha256', 'masks.zip'),
                      ('raw_reference_archive_sha256', 'final-zero-reference.zip'),
                      ('raw_reference_json_sha256', 'final-zero-reference.json')]:
        equal(result[key], sha256(folder / name), 'Diagnostic artifact ' + name)
    equal(sha256(args.data / 'manifest.json'), policy['data_manifest_sha256'], 'Isolated data manifest')
    manifest = read(args.data / 'manifest.json')
    equal(manifest['holdout_packaged'], False, 'Holdout exclusion')
    equal({row['split'] for row in manifest['rows']}, {'train', 'validation'}, 'Isolated splits')
    selected = sorted((row for row in manifest['rows'] if row['split'] == 'validation'), key=lambda row: row['id'])
    equal([len(selected), len({row['image_id'] for row in selected})], [1264, 217], 'Full validation inventory')
    if args.preflight:
        selected = selected[:3]
    baseline_path = args.sam_validation / 'result.json'
    baseline_hash = sha256(baseline_path)
    equal(baseline_hash, training['contract']['validation_sam_result_sha256'], 'Original SAM result')
    equal(contract['baseline_result_sha256'], baseline_hash, 'Contract original SAM result')
    baseline_rows = read(baseline_path)['rows']
    baseline = {row['id']: row for row in baseline_rows}
    if len(baseline) != len(baseline_rows):
        raise ValueError('Duplicate original reference objects')
    summaries, raw_geometry = replay_masks(folder, args.data, args.sam_validation, selected,
                                           manifest, baseline, result, raw, policy['logit_biases'])
    retained_offsets = None
    if not args.preflight:
        equal(raw_geometry, training['history'][-1]['validation'], 'Recorded epoch20 native geometry')
        equal({key: summaries['initial'][key] for key in raw_geometry},
              training['original_quadro_validation'], 'Original Quadro aggregate')
        floor = {key: max(training['initial_validation'][key], training['original_quadro_validation'][key])
                 for key in ('full_iou', 'hidden_iou')}
        equal(floor, training['selection_floor'], 'Unchanged geometry floors')
        retained_offsets = retained(summaries, floor, policy['logit_biases'])
    equal(result['retained_offsets'], retained_offsets, 'Exploratory retained offsets')
    equal(load_parent(args.parent, policy)[1], bindings, 'Parent evidence after replay')
    equal(verify_sources(args.source_manifest), source_hash, 'Source after replay')
    return dict(verified=True, at=datetime.now(timezone.utc).isoformat(), preflight=args.preflight,
                fixed_epoch=20, objects=len(selected), images=len({row['image_id'] for row in selected}),
                result_sha256=sha256(folder / 'result.json'), source_manifest_sha256=source_hash,
                **bindings, summaries=summaries, retained_offsets=retained_offsets,
                probability_audit='Bounds and aggregate arithmetic only; probabilities were not exported',
                native_pixel_replay=True, holdout_access=False, benchmark_access=False,
                benchmark_success=False, promoted=False)


def report(receipt):
    lines = ['# Fixed final-epoch shape-probability diagnostic', '',
             'Validation-only diagnosis on reused development data. No benchmark admission or success.', '',
             'All exported masks were independently checked against native annotations. Initial masks match the',
             'original Quadro masks; final zero-offset masks match the separate native pass and epoch-20 geometry.', '',
             '| Variant | Full IoU | Hidden IoU | Brier | NLL |',
             '| --- | ---: | ---: | ---: | ---: |']
    for key, values in receipt['summaries'].items():
        cells = [f'{values[name]:.9f}' if values[name] is not None else 'n/a'
                 for name in ('full_iou', 'hidden_iou', 'brier', 'nll')]
        lines.append('| ' + ' | '.join([key] + cells) + ' |')
    lines += ['', 'Offsets meeting both unchanged geometry floors: ' + str(receipt['retained_offsets']) + '.', '',
              'Probability scores were generated by the GPU evaluator. This replay checks their bounds and',
              'aggregation, not probability pixels. All five prespecified offsets are reported.', '',
              'The epoch and grid were fixed after observing three early training epochs. Validation, reserved',
              'holdout and benchmark prefixes had been seen in earlier research. This is not untouched confirmation.', '',
              'The full 2,565-target four-metric 3% improvement goal remains unachieved by this diagnostic.', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('parent', 'data', 'sam-validation', 'source-manifest', 'result', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    if args.output.exists() or (not args.preflight and args.output.with_suffix('.md').exists()):
        raise FileExistsError('Preserve existing diagnostic replay')
    receipt = replay(args)
    write_json(args.output, receipt)
    if not args.preflight:
        args.output.with_suffix('.md').write_text(report(receipt), encoding='utf8')
    print(dict(verified=True, objects=receipt['objects'], retained_offsets=receipt['retained_offsets'],
               benchmark_success=False), flush=True)


if __name__ == '__main__':
    main()

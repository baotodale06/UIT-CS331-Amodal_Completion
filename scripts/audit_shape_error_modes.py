"""Decompose the completed shape fine-tune's native validation errors."""
import argparse
from collections import defaultdict
import hashlib
import itertools
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from amodal_benchmark.dataset import sha256,write_json
from scripts.recover_shape_sam_validation import check_ids,geometry,mask,paired_summary
from scripts.verify_shape_v2_metadata import equal


def error_counts(prediction, truth, visible, support):
    if not (prediction.shape==truth.shape==visible.shape==support.shape) or not truth.any():
        raise ValueError('Nonempty truth and one native coordinate frame are required')
    false_positive = prediction & ~truth
    false_negative = truth & ~prediction
    return dict(truth_pixels=int(truth.sum()),hidden_pixels=int((truth & ~visible).sum()),
        false_positive=int(false_positive.sum()),false_negative=int(false_negative.sum()),
        false_positive_in_support=int((false_positive & support).sum()),
        false_positive_outside_support=int((false_positive & ~support).sum()),
        false_negative_visible=int((false_negative & visible).sum()),
        false_negative_hidden=int((false_negative & ~visible).sum()))


def iou_components(before, after):
    """Exact Shapley attribution of IoU change to FP and FN count changes."""
    area = before['truth_pixels']
    if area<=0 or area!=after['truth_pixels']:
        raise ValueError('The same nonempty truth is required')
    keys = ('false_positive','false_negative_visible','false_negative_hidden')
    def value(counts):
        return (area-counts['false_negative_visible']-counts['false_negative_hidden'])/(area+counts['false_positive'])
    result = dict.fromkeys(keys,0.)
    for order in itertools.permutations(keys):
        state = {key:before[key] for key in keys}
        previous = value(state)
        for key in order:
            state[key] = after[key]
            current = value(state)
            result[key] += (current-previous)/6
            previous = current
    result['total'] = value(after)-value(before)
    if not np.isclose(sum(result[key] for key in keys),result['total'],rtol=0,atol=1e-14):
        raise ValueError('Error attribution does not reconstruct the IoU change')
    return result


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row['image_id']].append(row)
    def image_mean(function):
        return float(np.mean([np.mean([function(row) for row in group]) for group in groups.values()]))
    stats = dict(objects=len(rows),images=len(groups),
        baseline_full_iou=image_mean(lambda row:row['baseline_geometry']['full_iou']),
        adapted_full_iou=image_mean(lambda row:row['adapted_geometry']['full_iou']),
        attribution={key:image_mean(lambda row:row['attribution'][key]) for key in rows[0]['attribution']})
    for condition in ('baseline','adapted'):
        stats[condition+'_errors_per_truth_pixel'] = {key:image_mean(lambda row:row[condition][key]/row[condition]['truth_pixels'])
            for key in ('false_positive','false_positive_outside_support','false_negative_visible','false_negative_hidden')}
    equal(stats['adapted_full_iou']-stats['baseline_full_iou'],stats['attribution']['total'],'aggregate attribution')
    return stats


def audit(output):
    if output.exists():
        raise FileExistsError('Preserve earlier error-mode audits')
    study = ROOT/'output/shape-sam-robustness-20261123'
    native = study/'recovered/native-validation'
    replay = json.loads((study/'native-replay.json').read_bytes())
    if replay['verified'] is not True or replay['selected_epoch']!=7:
        raise ValueError('The completed fixed epoch7 native replay is required')
    equal(sha256(native/'result.json'),replay['result_sha256'],'completed native evidence')
    result = json.loads((native/'result.json').read_bytes())
    baseline_root = ROOT/'output/shape-validation-sam-20261123/native/result'
    baseline_result = json.loads((baseline_root/'result.json').read_bytes())
    native_contract = json.loads((native/'contract.json').read_bytes())
    equal(sha256(baseline_root/'result.json'),native_contract['baseline_result_sha256'],'original diagnostic')
    baseline = {row['id']:row for row in baseline_result['rows']}
    with zipfile.ZipFile(ROOT/'output/shape-v2-20261123/colab/shape-v2-training-data.zip') as bundle:
        payload = bundle.read('data/manifest.json')
    equal(hashlib.sha256(payload).hexdigest(),native_contract['data_manifest_sha256'],'isolated data')
    manifest = json.loads(payload)
    if manifest['holdout_packaged'] is not False:
        raise ValueError('No holdout data allowed')
    selected = {row['id']:row for row in manifest['rows'] if row['split']=='validation'}
    check_ids(result['rows'],selected)
    equal([len(selected),len({row['image_id'] for row in selected.values()})],[1264,217],'full validation inventory')
    rows = []
    for source_row in result['rows']:
        source = selected[source_row['id']]
        original = baseline[source_row['id']]
        equal({key:source_row[key] for key in ('image_id','image','mask')},
            {key:source[key] for key in ('image_id','image','mask')},'object identity')
        truth_path = ROOT/'output/shape-v2-20261123/data'/source['mask']
        equal(sha256(truth_path),manifest['artifacts'][source['mask']],'truth bytes')
        with Image.open(truth_path) as image:
            encoded = np.array(image)
        truth,visible = (encoded&1)>0,(encoded&2)>0
        folder = baseline_root/'objects'/source_row['id']
        original_path,support_path = folder/'sam-prediction.png',folder/'sam-visible.png'
        adapted_path = native/'objects'/source_row['id']/'prediction.png'
        equal(sha256(original_path),original['artifacts']['sam-prediction.png'],'original prediction')
        equal(sha256(support_path),source_row['sam_sha256'],'SAM support')
        equal(sha256(adapted_path),source_row['prediction_sha256'],'adapted prediction')
        original_mask,support,raw = (mask(path,truth.shape) for path in (original_path,support_path,adapted_path))
        equal(geometry(raw,truth,visible),source_row['robust_sam'],'raw adapted geometry')
        adapted = raw|support
        first,second = geometry(original_mask,truth,visible),geometry(adapted,truth,visible)
        equal(first,original['sam_input'],'original geometry')
        if np.any(support & ~original_mask):
            raise ValueError('Original prior no longer preserves input support')
        before,after = (error_counts(value,truth,visible,support) for value in (original_mask,adapted))
        equal(before['false_positive_in_support'],after['false_positive_in_support'],'fixed support false positives')
        fraction = before['hidden_pixels']/before['truth_pixels']
        stratum = 'fully_visible' if fraction==0 else 'occluded_0_to_10pct' if fraction<=.1 else 'occluded_10_to_40pct' if fraction<=.4 else 'occluded_above_40pct'
        rows.append(dict(id=source_row['id'],image_id=source_row['image_id'],stratum=stratum,
            baseline=before,adapted=after,baseline_geometry=first,adapted_geometry=second,
            attribution=iou_components(before,after),
            source_sha256=dict(truth=sha256(truth_path),baseline=sha256(original_path),adapted=sha256(adapted_path),support=sha256(support_path))))
    summary = dict(all=summarize(rows),fully_visible=summarize([row for row in rows if row['stratum']=='fully_visible']),
        occluded=summarize([row for row in rows if row['stratum']!='fully_visible']))
    for name in ('occluded_0_to_10pct','occluded_10_to_40pct','occluded_above_40pct'):
        subset = [row for row in rows if row['stratum']==name]
        if subset:
            summary[name] = summarize(subset)
    pairs = [dict(image_id=row['image_id'],annotated_input=row['baseline_geometry'],sam_input=row['adapted_geometry']) for row in rows]
    stats = paired_summary(pairs)
    clamp = json.loads((study/'support-clamp-ablation/result.json').read_bytes())
    equal(stats,clamp['vs_frozen_v2'],'existing fixed-checkpoint support ablation')
    report = dict(complete=True,objects=1264,images=217,selected_epoch=7,summary=summary,geometry=stats,rows=rows,
        checkpoint_sha256=replay['checkpoint_sha256'],native_replay_sha256=sha256(study/'native-replay.json'),
        baseline_result_sha256=sha256(baseline_root/'result.json'),source_sha256=sha256(Path(__file__)),
        method='Exact three-factor Shapley decomposition of native full-IoU change; fixed SAM support in both arms',
        interpretation='Validation-only diagnosis of one completed fixed checkpoint; no retraining, checkpoint reselection, benchmark or holdout access',
        benchmark_success=False,holdout_access=False,checkpoint_reselected=False)
    output.mkdir(parents=True)
    write_json(output/'result.json',report)
    table = []
    for name,values in summary.items():
        attribution = values['attribution']
        table.append(f"| {name} | {values['objects']} | {values['images']} | {100*attribution['total']:+.3f} | "
            f"{100*attribution['false_positive']:+.3f} | {100*attribution['false_negative_visible']:+.3f} | "
            f"{100*attribution['false_negative_hidden']:+.3f} |")
    text = '''# Native shape error diagnosis

The completed epoch7 shape-only fine-tune and original v2 prior are compared on
all 1,264 validation objects / 217 images. Both preserve identical SAM input
support. Native masks and truth are independently rehashed and rescored. This
exactly reproduces the prior fixed-checkpoint support-clamp ablation.

| Cohort | Objects | Images | Total full-IoU change | False-positive contribution | Missed visible contribution | Missed hidden contribution |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
'''+ '\n'.join(table)+'''

Values are IoU percentage points, averaged per object within image, then per
image. Contributions use all six orders of an exact three-factor Shapley
decomposition of IoU=(truth area-visible FN-hidden FN)/(truth area+FP). Their sum
equals the full-IoU change for every object and every reported cohort. A positive
contribution improves IoU; a negative contribution reduces it. This is arithmetic
attribution, not a causal experiment. Cohort rows have their own image eligibility
and must not be summed to reconstruct the all-object mean.

The false positives inside input SAM support are identical in both conditions.
Any false-positive change therefore occurs outside that support. Visibility and
occlusion bins are fixed from annotated native geometry for diagnosis only; they
do not become inference inputs. No RGB correction, Surface ranker or combined
method was used. No checkpoint, threshold or policy was reselected in this audit.
The >=3% all-four full-benchmark target remains unachieved.
'''
    (output/'REPORT.md').write_text(text,encoding='utf8')
    print(json.dumps({key:value for key,value in report.items() if key!='rows'},indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'output/shape-error-modes-20261123')
    args = parser.parse_args()
    audit(args.output)


if __name__=='__main__':
    main()

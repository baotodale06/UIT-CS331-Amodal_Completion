"""Independently replay fixed-offset masks; retain exploratory scope."""
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
from scripts.diagnose_shape_probability import summarize
from scripts.shape_robust_data import geometry


def main():
    folder = ROOT/'output/shape-probability-20261123/gpu-full'
    result = json.loads((folder/'result.json').read_bytes())
    contract = json.loads((folder/'contract.json').read_bytes())
    if (not result['complete'] or result['preflight'] or result['promoted']
            or result['benchmark_success'] or sha256(folder/'contract.json')!=result['contract_sha256']
            or sha256(folder/'masks.zip')!=result['mask_archive_sha256']):
        raise ValueError('Incomplete or inconsistent diagnostic')
    data = ROOT/'output/shape-v2-20261123/data'
    with zipfile.ZipFile(ROOT/'output/shape-v2-20261123/colab/shape-v2-training-data.zip') as bundle:
        raw = bundle.read('data/manifest.json')
    if hashlib.sha256(raw).hexdigest()!=contract['config']['data_manifest_sha256']:
        raise ValueError('Data manifest changed')
    manifest = json.loads(raw)
    selected = {row['id']:row for row in manifest['rows'] if row['split']=='validation'}
    if len(selected)!=1264 or {row['id'] for row in result['rows']}!=set(selected) or len(result['rows'])!=1264:
        raise ValueError('Incomplete validation inventory')
    keys = ['initial']+['adapted_bias_'+format(bias,'g') for bias in contract['config']['logit_biases']]
    replays = {key:[] for key in keys}
    differences = {key:dict(objects=0,pixels=0) for key in ('initial','adapted')}
    sam_root = ROOT/'output/shape-validation-sam-20261123/native/result'
    adapted_root = ROOT/'output/shape-sam-robustness-20261123/recovered/native-validation'
    baseline = json.loads((sam_root/'result.json').read_bytes())
    adapted = json.loads((adapted_root/'result.json').read_bytes())
    references = {name:{row['id']:row for row in source['rows']} for name,source in [('initial',baseline),('adapted',adapted)]}
    with zipfile.ZipFile(folder/'masks.zip') as bundle:
        if len(bundle.namelist())!=len(set(bundle.namelist())) or set(bundle.namelist())!={key+'/'+name+'.png' for key in keys for name in selected}:
            raise ValueError('Mask inventory differs')
        for row in result['rows']:
            source = selected[row['id']]
            if row['image_id']!=source['image_id'] or set(row['variants'])!=set(keys):
                raise ValueError('Object identity changed')
            path = data/source['mask']
            if sha256(path)!=manifest['artifacts'][source['mask']]:
                raise ValueError('Truth mask changed')
            with Image.open(path) as image:
                encoded = np.array(image)
            truth, visible = (encoded&1)>0, (encoded&2)>0
            support_path = sam_root/'objects'/row['id']/'sam-visible.png'
            if sha256(support_path)!=references['initial'][row['id']]['artifacts']['sam-visible.png']:
                raise ValueError('SAM mask changed')
            with Image.open(support_path) as image:
                support = np.array(image)>0
            for key in keys:
                with Image.open(io.BytesIO(bundle.read(key+'/'+row['id']+'.png'))) as image:
                    raw = np.array(image)
                if raw.shape!=truth.shape or not np.isin(raw,[0,255]).all():
                    raise ValueError('Mask is not native binary geometry')
                prediction = raw>0
                if np.any(support & ~prediction):
                    raise ValueError('Observable support was removed')
                stated = row['variants'][key]
                actual = geometry(prediction,truth,visible)
                if any(actual[name]!=stated[name] for name in actual) or hashlib.sha256(prediction.astype(np.uint8).tobytes()).hexdigest()!=stated['prediction_sha256']:
                    raise ValueError('Native mask replay failed')
                replays[key].append(dict(id=row['id'],image_id=row['image_id'],**stated))
                if key in ('initial','adapted_bias_0'):
                    name = 'initial' if key=='initial' else 'adapted'
                    path = (sam_root/'objects'/row['id']/'sam-prediction.png' if name=='initial'
                            else adapted_root/'objects'/row['id']/'prediction.png')
                    expected = references[name][row['id']]['artifacts']['sam-prediction.png'] if name=='initial' else references[name][row['id']]['prediction_sha256']
                    if sha256(path)!=expected:
                        raise ValueError('Native reference changed')
                    with Image.open(path) as image:
                        reference = (np.array(image)>0)|support
                    count = int(np.count_nonzero(prediction!=reference))
                    if count!=row['native_reference_differences'][name]:
                        raise ValueError('Cross-GPU differences do not replay')
                    differences[name]['objects'] += count>0
                    differences[name]['pixels'] += count
    summaries = {key:summarize(rows) for key,rows in replays.items()}
    if summaries!=result['summaries'] or differences!=result['native_reference_differences']:
        raise ValueError('Aggregate replay differs')
    original = summaries['initial']
    eligible = [key for key,value in summaries.items() if key!='initial' and all(value[metric]>=original[metric] for metric in ('full_iou','hidden_iou'))]
    replay = dict(verified=True,objects=1264,images=217,masks=1264*len(keys),
        result_sha256=sha256(folder/'result.json'),source_sha256=sha256(Path(__file__)),
        native_reference_differences=differences,geometry_floor_candidates=eligible,
        probability_scores_independently_recomputed=False,
        probability_score_note='Brier/NLL aggregate arithmetic replayed from generated rows; native probabilities were not exported',
        benchmark_success=False,promoted=False)
    write_json(folder.parent/'replay.json',replay)
    lines = ['# Fixed-offset shape probability diagnostic', '',
        'All 1,264 validation objects / 217 images; frozen original v2 and robust epoch 7.',
        'Offsets were fixed before execution. No checkpoint or policy is promoted.', '',
        '| Variant | Full IoU | Hidden IoU | Brier outside support | NLL outside support |',
        '| --- | ---: | ---: | ---: | ---: |']
    for key,value in summaries.items():
        lines.append(f"| {key} | {value['full_iou']:.9f} | {value['hidden_iou']:.9f} | {value['brier']:.9f} | {value['nll']:.9f} |")
    lines += ['', 'Full and hidden geometry floors retained by: '+(', '.join(eligible) or 'none')+'.', '',
        'Every native prediction mask was independently replayed from exported PNGs. '
        'Brier/NLL aggregation was replayed from per-object generated values; probabilities were not exported.',
        'Cross-GPU zero-offset mask differences: '+json.dumps(differences)+'.', '',
        'This is exploratory development on reused validation, not untouched confirmation. '
        'No holdout or benchmark assets entered this diagnostic. The 3% improvement target '
        'on all four full-benchmark metrics remains unachieved.']
    (folder.parent/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(replay))


if __name__=='__main__':
    main()

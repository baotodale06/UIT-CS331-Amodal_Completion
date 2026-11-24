"""Validation-only counterfactual: retain SAM support in the completed robust prior."""
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from amodal_benchmark.dataset import sha256,write_json
from scripts.recover_shape_sam_validation import geometry,mask,paired_summary,check_ids
from scripts.verify_shape_v2_metadata import equal


def main():
    study=ROOT/'output/shape-sam-robustness-20261123'
    replay=json.loads((study/'native-replay.json').read_bytes())
    if replay.get('verified') is not True:
        raise ValueError('Complete independent native replay required')
    native=study/'recovered/native-validation'
    result=json.loads((native/'result.json').read_bytes())
    equal(sha256(native/'result.json'),replay['result_sha256'],'completed native replay')
    baseline_root=ROOT/'output/shape-validation-sam-20261123/native/result'
    baseline=json.loads((baseline_root/'result.json').read_bytes())
    baseline_rows={row['id']:row for row in baseline['rows']}
    with zipfile.ZipFile(ROOT/'output/shape-v2-20261123/colab/shape-v2-training-data.zip') as archive:
        manifest=json.loads(archive.read('data/manifest.json'))
    selected={row['id']:row for row in manifest['rows'] if row['split']=='validation'}
    check_ids(result['rows'],selected)
    data=ROOT/'output/shape-v2-20261123/data'
    output=study/'support-clamp-ablation'
    if output.exists():
        raise FileExistsError('Preserve previous counterfactual evidence')
    output.mkdir()
    rows,pairs,raw_pairs=[],[],[]
    for row in result['rows']:
        source=selected[row['id']]
        truth_path=data/source['mask']
        equal(sha256(truth_path),manifest['artifacts'][source['mask']],'amodal truth')
        with Image.open(truth_path) as image:
            encoded=np.array(image)
        truth,visible=(encoded&1)>0,(encoded&2)>0
        raw_path=native/'objects'/row['id']/'prediction.png'
        sam_path=baseline_root/'objects'/row['id']/'sam-visible.png'
        equal(sha256(raw_path),row['prediction_sha256'],'raw prediction')
        equal(sha256(sam_path),row['sam_sha256'],'SAM support')
        raw,sam=mask(raw_path,truth.shape),mask(sam_path,truth.shape)
        # At threshold .5, replacing probability on SAM support by one is
        # exactly this union. No annotated-visible pixels enter prediction.
        predicted=raw|sam
        raw_scores=geometry(raw,truth,visible)
        scores=geometry(predicted,truth,visible)
        equal(raw_scores,row['robust_sam'],'raw geometry')
        destination=output/'objects'/row['id']/'prediction.png'
        destination.parent.mkdir(parents=True)
        Image.fromarray(predicted.astype(np.uint8)*255).save(destination)
        record=dict(id=row['id'],image_id=row['image_id'],raw_sha256=sha256(raw_path),sam_sha256=sha256(sam_path),
            truth_sha256=sha256(truth_path),prediction_sha256=sha256(destination),
            raw_geometry=raw_scores,clamped_geometry=scores,v2_sam=baseline_rows[row['id']]['sam_input'])
        write_json(destination.with_name('result.json'),record)
        rows.append(record)
        pairs.append(dict(image_id=row['image_id'],annotated_input=record['v2_sam'],sam_input=scores))
        raw_pairs.append(dict(image_id=row['image_id'],annotated_input=raw_scores,sam_input=scores))
    report=dict(complete=True,objects=1264,images=217,selected_epoch=replay['selected_epoch'],
        checkpoint_sha256=replay['checkpoint_sha256'],native_replay_sha256=sha256(study/'native-replay.json'),
        source_sha256=sha256(Path(__file__)),condition='Counterfactual SAM support clamp on fixed validation-selected checkpoint',
        derivation='prediction = recovered raw native prediction OR unchanged SAM input support; no RGB modification',
        vs_frozen_v2=paired_summary(pairs),vs_unclamped_robust=paired_summary(raw_pairs),rows=rows,
        benchmark_success=False,holdout_access=False,checkpoint_reselected=False,
        limitation='Exploratory validation-only policy ablation. Binary geometry does not establish candidate ranking, image-quality gains or an untouched confirmation.')
    write_json(output/'result.json',report)
    print(json.dumps({key:value for key,value in report.items() if key!='rows'},indent=2))


if __name__=='__main__':
    main()

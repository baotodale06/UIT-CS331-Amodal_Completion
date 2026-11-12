#!/usr/bin/env python3
"""Apply a frozen surface policy using only the observation manifest and candidate pools."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from amodal_benchmark.dataset import sha256, write_json
from amodal_benchmark.surface_policy import apply_frozen_surface, verify_frozen_surface_sources


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment-dir',type=Path,required=True)
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--extra-run-dir',type=Path)
    p.add_argument('--freeze',type=Path,required=True)
    p.add_argument('--features-dir',type=Path)
    p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args()
    freeze=json.loads(a.freeze.read_text())
    if freeze['status'] != 'frozen_before_fresh_test':
        raise ValueError('Use a finalized frozen method')
    verify_frozen_surface_sources(freeze)
    if a.output_dir.exists():
        raise ValueError('Use a new inference directory')
    manifest=a.experiment_dir/'generation_manifest.json'
    cases=json.loads(manifest.read_text())
    contract=dict(started_at=datetime.now(timezone.utc).isoformat(),freeze_sha256=sha256(a.freeze),
        observation_manifest_sha256=sha256(manifest),evaluation_data_access=False,
        policy_sha256=sha256(ROOT/'amodal_benchmark/surface_policy.py'),source_sha256=sha256(Path(__file__)))
    write_json(a.output_dir/'contract.json',contract)
    for case in cases:
        inputs=a.experiment_dir/'inputs'/case['case_id']
        for name,expected in case['input_hashes'].items():
            if sha256(inputs/name) != expected: raise ValueError('Observation checksum mismatch')
        observed=np.array(Image.open(inputs/'observed.png').convert('RGB'))
        visible=np.array(Image.open(inputs/'visible.png').convert('L'))>0
        occlusion=np.array(Image.open(inputs/'occlusion.png').convert('L'))>0
        images,masks,scores,records,hashes=[],[],[],[],{}
        for seed in freeze['candidate_seeds']:
            folder=a.run_dir/'candidates'/case['case_id']
            if not (folder/f'seed-{seed}.json').exists() and a.extra_run_dir:
                folder=a.extra_run_dir/'candidates'/case['case_id']
            path=folder/f'seed-{seed}.json'
            record=json.loads(path.read_text());records.append(record)
            if record['input_hashes'] != case['input_hashes'] or record['seed'] != seed:
                raise ValueError('Candidate observation or seed mismatch')
            hashes[path.name]=sha256(path)
            for name,expected in record['artifacts'].items():
                if sha256(folder/name) != expected: raise ValueError('Candidate checksum mismatch')
            rgb,mask=folder/f'seed-{seed}.png',folder/f'seed-{seed}-mask.png'
            images.append(np.array(Image.open(rgb).convert('RGB')) if rgb.exists() else np.zeros_like(observed))
            masks.append(np.array(Image.open(mask).convert('L'))>0 if mask.exists() else np.zeros_like(visible))
            scores.append(record.get('surface',{}))
        features=None
        if freeze.get('pretrained_visible_features'):
            if not a.features_dir: raise ValueError('Missing pretrained feature directory')
            cache=json.loads((a.features_dir/(case['case_id']+'.json')).read_text())
            expected=cache['contract']
            if (cache['candidate_record_sha256'] != hashes or expected['truth_access'] is not False or
                expected['observation_manifest_sha256'] != sha256(manifest) or
                expected['features_sha256'] != sha256(ROOT/'amodal_benchmark/surface_features.py')):
                raise ValueError('Feature provenance mismatch')
            features=cache['values']
        rgb,mask,details=apply_frozen_surface(observed,visible,occlusion,images,masks,freeze,scores,features)
        folder=a.output_dir/'selected'/case['case_id']
        selected_record=records[details['index']]
        generated=f"seed-{details['seed']}.png" in selected_record['artifacts']
        artifacts={}
        if generated:
            folder.mkdir(parents=True)
            Image.fromarray(rgb).save(folder/'surface.png')
            Image.fromarray(mask.astype(np.uint8)*255).save(folder/'mask.png')
            artifacts={name:sha256(folder/name) for name in ('surface.png','mask.png')}
        write_json(a.output_dir/'decisions'/(case['case_id']+'.json'),dict(
            case_id=case['case_id'],status='generated' if generated else 'missing_generation',
            artifacts=artifacts,candidate_record_sha256=hashes,**details))
        print('SELECTED',case['case_id'],details['seed'],flush=True)
    contract['finished_at']=datetime.now(timezone.utc).isoformat()
    write_json(a.output_dir/'contract.json',contract)


if __name__=='__main__':
    main()

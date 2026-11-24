#!/usr/bin/env python3
"""Enumerate historical/current reference changes without accessing quality scores."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from amodal_benchmark.cohort import select_manifest_rows
from amodal_benchmark.dataset import read_official, sha256, write_json
from amodal_benchmark.references import reference_paths, cached_reference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-pair', type=Path, nargs=3,
                        metavar=('HISTORICAL', 'CURRENT', 'OUTPUT_JSON'),
                        help='Audit these two reference directories on the frozen 204-target cohort')
    args = parser.parse_args()
    design = ROOT/'output/surface-annotated-20261112-design'
    if args.reference_pair:
        historical, current, output = args.reference_pair
    else:
        study = ROOT/json.loads((design/'active-study.json').read_text())['study']
        current = study/'baseline-cohort/visible_references'
        historical = ROOT/'output/pdmc-open-world-sam-annotated/visible_references'
        if not historical.is_dir():
            historical = ROOT/'output/artifact-recovery-2026-11-05/downloads/pdmc-open-world-sam-annotated/visible_references'
        output = design/'freshrefs-reference-audit.json'
    rows = select_manifest_rows(read_official(ROOT/'datasets/open_world_amodal')[0],design/'cohort.json')
    records = []
    for row in rows:
        source_sha256 = sha256(ROOT/'datasets/open_world_amodal'/row['image'])
        old,new = [cached_reference(folder,row,source_sha256) for folder in (historical,current)]
        if old is None or new is None:
            raise ValueError('Missing reference in the paired audit')
        first,second = [reference_paths(folder,row) for folder in (historical,current)]
        a,b = [np.asarray(Image.open(paths['mask']))>0 for paths in (first,second)]
        if a.shape != b.shape:
            raise ValueError('Reference dimensions changed')
        alpha = [np.asarray(Image.open(paths['full']))[...,3] for paths in (first,second)]
        records.append(dict(id=row['id'],source=row['source'],input_sha256=source_sha256,
            mask_xor_pixels=int((a!=b).sum()),mask_pixels=int(a.size),
            mask_iou=float((a&b).sum()/(a|b).sum()) if (a|b).any() else 1.,
            historical_area=int(a.sum()),current_area=int(b.sum()),
            eroded_alpha_xor_pixels=int((alpha[0]!=alpha[1]).sum()),
            geometry_equal=old['geometry']==new['geometry'],
            historical_artifacts=old['artifacts'],current_artifacts=new['artifacts'],
            artifacts_byte_equal=old['artifacts']==new['artifacts']))
    result=dict(passed=True,at=datetime.now(timezone.utc).isoformat(),targets=len(records),
        changed_masks=sum(row['mask_xor_pixels']>0 for row in records),
        changed_reference_artifacts=sum(not row['artifacts_byte_equal'] for row in records),
        total_changed_mask_pixels=sum(row['mask_xor_pixels'] for row in records),
        max_changed_mask_pixels=max(row['mask_xor_pixels'] for row in records),
        changed_geometry=sum(not row['geometry_equal'] for row in records),
        quality_metric_access=False,records=records,source_sha256=sha256(Path(__file__)))
    result.update(historical=str(historical), current=str(current), cohort_sha256=sha256(design/'cohort.json'))
    write_json(output,result)
    print(json.dumps({key:value for key,value in result.items() if key!='records'},indent=2))


if __name__ == '__main__':
    main()

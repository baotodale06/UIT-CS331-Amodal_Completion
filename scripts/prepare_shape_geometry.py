"""Export the audited, unambiguous natural COCOA mask joins for post-run scoring."""
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image
from pycocotools import mask as coco_mask

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from amodal_benchmark.dataset import sha256, write_json


def main():
    source = ROOT/"output/appearance-failures-20261107"
    out = ROOT/"output/shape-20261121/evaluation_only/cocoa"
    out.mkdir(parents=True,exist_ok=True)
    manifest = json.loads((source/"analysis_manifest.json").read_text())
    annotations = {}
    for name, expected in manifest["external_annotation_hashes"].items():
        file = source/"external"/name
        assert sha256(file)==expected
        annotations[name] = {r["id"]:r for r in json.loads(file.read_text())["annotations"]}
    rows = []
    for row in json.loads((source/"per_target_diagnostics.json").read_text()):
        if row["gt_occlusion_ratio"] is None:
            continue
        ann = annotations[row["gt_annotation_file"]][row["gt_annotation_id"]]
        reg = next(r for r in ann["regions"] if r["order"]==row["gt_region_order"])
        h,w = row["gt_image_height"],row["gt_image_width"]
        amodal = coco_mask.decode(coco_mask.frPyObjects([reg["segmentation"]],h,w)).any(axis=2)
        visible = coco_mask.decode(reg["visible_mask"]).astype(bool) if "visible_mask" in reg else amodal
        hidden = coco_mask.decode(reg["invisible_mask"]).astype(bool) if "invisible_mask" in reg else np.zeros_like(visible)
        ratio = hidden.sum()/max(1,(visible|hidden).sum())
        assert abs(ratio-row["gt_occlusion_ratio"])<1.e-6
        file = out/(row["id"].split("/")[1]+".png")
        Image.fromarray(amodal.astype(np.uint8)+2*visible.astype(np.uint8)).save(file)
        rows.append(dict(id=row["id"], case=file.name, sha256=sha256(file), ratio=ratio,
                         bin=row["gt_bin"], annotation=ann["id"], region=reg["order"]))
    assert len(rows)==321
    write_json(out/"manifest.json",dict(sources=manifest["external_annotation_hashes"],rows=rows))


if __name__=="__main__":
    main()


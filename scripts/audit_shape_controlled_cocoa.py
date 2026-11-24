"""Score frozen controlled selections against independent COCOA polygons in the source frame."""
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from amodal_benchmark.dataset import sha256, write_json
from scripts.analyze_surface_followup import paired_interval
from scripts.audit_shape_controlled import iou


def read(path):
    return json.loads(path.read_bytes())


def check_file(path, digest):
    if sha256(path) != digest:
        raise ValueError(f"File differs from its archived receipt: {path}")


def native_mask(mask, box, size):
    """Undo the square crop. PIL clips padded and out-of-frame pixels during paste."""
    left, top, right, bottom = box
    canvas = Image.new("L", size, 0)
    canvas.paste(mask.resize((right-left, bottom-top), Image.Resampling.NEAREST), (left, top))
    return np.asarray(canvas) > 0


def main():
    study = ROOT/"output/shape-20261121"
    source = ROOT/"output/surface-confirmation128-20261110"
    geo = study/"evaluation_only/cocoa"
    check_file(source/"cases.json", read(source/"preparation.json")["cases_sha256"])
    check_file(geo/"manifest.json", read(study/"cocoa-historical-controls.json")["manifest_sha256"])
    cases = read(source/"cases.json")
    annotations = {row["id"]: row for row in read(geo/"manifest.json")["rows"]}
    rows = []
    for case in cases:
        tid = case["target_id"]
        if tid not in annotations:
            continue
        item = annotations[tid]
        cid = case["case_id"]
        crop = case["crop_box"]
        annotation = geo/item["case"]
        check_file(annotation, item["sha256"])
        with Image.open(annotation) as im:
            size = im.size
            truth = (np.asarray(im) & 1) > 0
        original = ROOT/"datasets/open_world_amodal"/case["source_image"]
        check_file(original, case["source_sha256"])
        with Image.open(original) as im:
            if im.size != size:
                raise ValueError(f"Annotation and source frames differ for {tid}")
        frozen = study/"controlled"/cid/"result.json"
        result = read(frozen)
        if result["target_id"] != tid or result["case_id"] != cid:
            raise ValueError(f"Frozen selection belongs to a different target: {cid}")
        target = source/"evaluation_only"/cid/"full.png"
        check_file(target, case["evaluation_hashes"]["full.png"])
        with Image.open(target) as im:
            sam = np.asarray(im.convert("L")) > 0
            reference_iou = iou(native_mask(im, crop, size), truth)
        decision_path = source/"inference/decisions"/(cid+".json")
        decision = read(decision_path)
        directory = source/"full/candidates"/cid
        scores = []
        digests = []
        for index, seed in enumerate(range(11, 89, 11)):
            name = f"seed-{seed}"
            record = directory/(name+".json")
            check_file(record, decision["candidate_record_sha256"][record.name])
            record_data = read(record)
            if (record_data["case_id"], record_data["seed"], record_data["input_hashes"]) != (
                    cid, seed, case["input_hashes"]):
                raise ValueError(f"Candidate provenance differs for {tid} / {name}")
            path = directory/(name+"-mask.png")
            digest = record_data["artifacts"][path.name]
            check_file(path, digest)
            with Image.open(path) as im:
                mask = im.convert("L")
                sam_score = iou(np.asarray(mask) > 0, sam)
                if abs(sam_score - result["candidate_metrics"][index]["iou"]) > 1e-12:
                    raise ValueError(f"Controlled SAM score differs for {tid} / {name}")
                scores.append(float(iou(native_mask(mask, crop, size), truth)))
            digests.append(digest)
        selected = {method: scores[result["indices"][method]] for method in
                    ("first", "shape4", "shape8", "surface8", "combined8")}
        selected.update(uniform4=float(np.mean(scores[:4])), uniform8=float(np.mean(scores)))
        rows.append(dict(id=tid, case_id=cid, image=case["image_identity"], crop_box=crop,
                         original_occlusion=item["ratio"], pre_occlusion_sam_iou=float(reference_iou),
                         amodal_iou=selected, candidate_amodal_iou=scores, candidate_mask_sha256=digests,
                         controlled_result_sha256=sha256(frozen), decision_sha256=sha256(decision_path)))
    if len({row["image"] for row in rows}) != len(rows) or not rows:
        raise ValueError("Expected one controlled observation per matched COCOA image")
    comparisons = {}
    for method in ("shape4", "shape8", "surface8", "combined8"):
        uniform = "uniform"+method[-1]
        comparisons[method] = paired_interval([r["amodal_iou"][uniform] for r in rows],
                                              [r["amodal_iou"][method] for r in rows],
                                              ["COCOA"]*len(rows), higher=True)
    write_json(study/"controlled-cocoa.json", dict(
        controlled_cases=len(cases), matched=len(rows),
        previously_occluded=sum(r["original_occlusion"] > 0 for r in rows),
        cases_sha256=sha256(source/"cases.json"), geometry_manifest_sha256=sha256(geo/"manifest.json"),
        script_sha256=sha256(Path(__file__)), comparisons=comparisons, rows=rows))


if __name__ == "__main__":
    main()

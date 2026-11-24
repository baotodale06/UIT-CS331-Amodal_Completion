"""Validation-only diagnostic of a frozen prior's sensitivity to SAM support."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if (ROOT/"amodal_benchmark").is_dir():
    sys.path.insert(0, str(ROOT))
from amodal_benchmark.dataset import sha256, write_json
from amodal_benchmark.shape_v2_runtime import TRAINING_DATA_SHA256, load_prior, original_prior, shape_v2_contract


def geometry(prediction, truth, annotated_visible):
    """Always use annotated visibility for scoring, never to repair prediction."""
    prediction, truth, annotated_visible = (np.asarray(value, bool) for value in
                                            (prediction, truth, annotated_visible))
    if prediction.shape != truth.shape or annotated_visible.shape != truth.shape:
        raise ValueError("Native frames differ")
    hidden = truth & ~annotated_visible
    predicted_hidden = prediction & ~annotated_visible
    def iou(a, b):
        return float((a & b).sum()/max(1, (a | b).sum()))
    return dict(full_iou=iou(prediction, truth),
                hidden_iou=iou(predicted_hidden, hidden) if hidden.any() else None)


def paired_images(rows):
    result = {}
    for metric in ("full_iou", "hidden_iou"):
        groups = defaultdict(list)
        for row in rows:
            a, b = row["annotated_input"][metric], row["sam_input"][metric]
            if (a is None) != (b is None):
                raise ValueError("Scoring denominators must match between input conditions")
            if a is not None:
                groups[row["image_id"]].append((a, b))
        means = np.array([np.mean(groups[key], axis=0) for key in sorted(groups)])
        if not len(means):
            raise ValueError("No eligible validation images")
        delta = means[:, 1]-means[:, 0]
        rng = np.random.default_rng(20261123)
        sampled = delta[rng.integers(len(delta), size=(5000, len(delta)))].mean(axis=1)
        result[metric] = dict(images=len(means), objects=sum(map(len, groups.values())),
            annotated_input=float(means[:, 0].mean()), sam_input=float(means[:, 1].mean()),
            sam_minus_annotated=float(delta.mean()), absolute_ci=np.percentile(sampled, [2.5, 97.5]).tolist())
    return result


def save_mask(path, mask):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(mask, np.uint8)*255).save(path)
    return sha256(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--sam-checkpoint", type=Path, required=True)
    parser.add_argument("--original-validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest_path = args.data/"manifest.json"
    if sha256(manifest_path) != TRAINING_DATA_SHA256:
        raise ValueError("Only the original training/validation package is permitted")
    manifest = json.loads(manifest_path.read_bytes())
    if {row["split"] for row in manifest["rows"]} != {"train", "validation"}:
        raise ValueError("Holdout must not be present in this diagnostic's data package")
    selected = [row for row in manifest["rows"] if row["split"] == "validation"]
    grouped = defaultdict(list)
    for row in selected:
        grouped[row["image_id"]].append(row)
    if len(selected) != 1264 or len(grouped) != 217:
        raise ValueError("The complete pre-existing validation split is required")
    if args.output.exists():
        raise FileExistsError("Use a fresh diagnostic output; preserve previous attempts")
    policy = shape_v2_contract(args.checkpoint)
    original = json.loads(args.original_validation.read_bytes())
    if original["shape_policy"] != policy or original["split"] != "validation":
        raise ValueError("Original development geometry does not match this frozen checkpoint")
    import torch
    from segment_anything import SamPredictor, sam_model_registry
    if torch.cuda.device_count() != 1 or "QUADRO" not in torch.cuda.get_device_name(0).upper():
        raise RuntimeError("One allocated Quadro is required for this diagnostic")
    fft_input = torch.randn(32, 32, device="cuda")
    if not torch.allclose(fft_input, torch.fft.irfft2(torch.fft.rfft2(fft_input), s=fft_input.shape), atol=1e-5):
        raise RuntimeError("CUDA FFT round trip failed")
    args.output.mkdir(parents=True)
    contract = dict(schema=1, scope="Validation-only input-mask diagnostic; no checkpoint/policy tuning or benchmark scoring",
        at=datetime.now(timezone.utc).isoformat(), selected_split="validation", targets=1264, images=217,
        data_manifest_sha256=sha256(manifest_path), shape_policy=policy,
        sam_checkpoint_sha256=sha256(args.sam_checkpoint), source_sha256=sha256(Path(__file__)),
        original_validation_sha256=sha256(args.original_validation), job_id=os.environ.get("SLURM_JOB_ID"),
        gpu=torch.cuda.get_device_name(0), torch=torch.__version__, cuda=torch.version.cuda,
        cuda_fft_round_trip=True, threshold=.5,
        sam_prompt="Tight XYXY box from annotated VISIBLE support only, x/y maximum plus one; single SAM mask",
        hidden_metric="Same true hidden pixels in both conditions: amodal truth minus annotated visible support",
        limitation="Box-prompted SAM approximates mask-boundary errors; it does not reproduce GroundingDINO selection errors",
        holdout_access=False, benchmark_access=False, checkpoint_modified=False)
    write_json(args.output/"contract.json", contract)
    prior = load_prior(args.checkpoint)
    sam = sam_model_registry["vit_h"](checkpoint=str(args.sam_checkpoint)).cuda().eval()
    predictor = SamPredictor(sam)
    records, verified_assets = [], set()
    for image_index, (image_id, objects) in enumerate(sorted(grouped.items()), 1):
        if len({row["image"] for row in objects}) != 1:
            raise ValueError("Image identity maps to multiple images")
        for row in objects:
            for name in (row["image"], row["mask"]):
                path = (args.data/name).resolve()
                if not path.is_relative_to(args.data.resolve()):
                    raise ValueError("Validation asset escaped the data package")
                if name not in verified_assets:
                    if sha256(path) != manifest["artifacts"][name]:
                        raise ValueError("Validation asset changed: "+name)
                    verified_assets.add(name)
        with Image.open(args.data/objects[0]["image"]) as image:
            rgb = np.array(image.convert("RGB"))
        masks, boxes = [], []
        for row in objects:
            with Image.open(args.data/row["mask"]) as image:
                encoded = np.array(image)
            truth, visible = (encoded & 1) > 0, (encoded & 2) > 0
            yy, xx = np.where(visible)
            if not len(xx):
                raise ValueError("Original validation contains empty visible support")
            masks.append((truth, visible))
            boxes.append([int(xx.min()), int(yy.min()), int(xx.max()+1), int(yy.max()+1)])
        predictor.set_image(rgb)
        sam_masks = []
        for start in range(0, len(boxes), 16):
            batch = torch.tensor(boxes[start:start+16], device="cuda", dtype=torch.float32)
            transformed = predictor.transform.apply_boxes_torch(batch, rgb.shape[:2])
            with torch.inference_mode():
                predictions, _, _ = predictor.predict_torch(point_coords=None, point_labels=None,
                    boxes=transformed, multimask_output=False, return_logits=False)
            sam_masks.extend(predictions[:, 0].cpu().numpy().astype(bool))
        for row, (truth, visible), prompt, sam_visible in zip(objects, masks, boxes, sam_masks):
            probability, annotated_box, _, _ = original_prior(prior, rgb, visible)
            annotated_prediction = probability >= .5
            if sam_visible.any():
                probability, sam_box, _, _ = original_prior(prior, rgb, sam_visible)
                sam_prediction = probability >= .5
            else:
                sam_box, sam_prediction = None, np.zeros_like(visible)
            directory = args.output/"objects"/row["id"]
            artifacts = {name:save_mask(directory/name, mask) for name, mask in (
                ("sam-visible.png", sam_visible), ("annotated-prediction.png", annotated_prediction),
                ("sam-prediction.png", sam_prediction))}
            record = dict(id=row["id"], image_id=image_id, image=row["image"], mask=row["mask"],
                sam_box_prompt=prompt, annotated_prior_box=annotated_box, sam_prior_box=sam_box,
                empty_sam_mask=not bool(sam_visible.any()),
                visible_mask_iou=float((sam_visible & visible).sum()/max(1, (sam_visible | visible).sum())),
                annotated_input=geometry(annotated_prediction, truth, visible),
                sam_input=geometry(sam_prediction, truth, visible), artifacts=artifacts)
            write_json(directory/"result.json", record)
            records.append(record)
        write_json(args.output/"progress.json", dict(images=image_index, objects=len(records), total_images=217,
            status="running", at=datetime.now(timezone.utc).isoformat()))
        print(f"validation: {image_index}/217 images, {len(records)}/1264 objects", flush=True)
    by_id = {row["id"]:row for row in original["rows"]}
    differences = [abs(row["annotated_input"][metric]-by_id[row["id"]][metric]) for row in records
        for metric in ("full_iou", "hidden_iou") if row["annotated_input"][metric] is not None]
    result = dict(complete=True, contract_sha256=sha256(args.output/"contract.json"), objects=len(records),
        images=217, empty_sam_masks=sum(row["empty_sam_mask"] for row in records),
        max_original_validation_difference=max(differences), geometry=paired_images(records), rows=records,
        benchmark_success=False, interpretation="Development diagnostic only; neither a benchmark gain nor a reserved-holdout result")
    write_json(args.output/"result.json", result)
    write_json(args.output/"progress.json", dict(images=217, objects=1264, status="complete"))
    print(json.dumps({key:value for key,value in result.items() if key != "rows"}, indent=2), flush=True)


if __name__ == "__main__":
    main()

"""Prepare SAM input masks for a separate shape-only training study, train split only."""
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

DATA_SHA256 = "ab2f4d1112c52bcd1b675e0f213ed8ee387fbc2601a3596f8f2b5e933ea7d13e"


def training_groups(manifest):
    if (manifest.get("holdout_packaged") is not False
            or {row["split"] for row in manifest["rows"]} != {"train", "validation"}):
        raise ValueError("Require original holdout-free training/validation package")
    splits, groups = defaultdict(set), defaultdict(list)
    for row in manifest["rows"]:
        splits[row["image_id"]].add(row["split"])
        if row["split"] == "train":
            groups[row["image_id"]].append(row)
    if any(len(value) != 1 for value in splits.values()):
        raise ValueError("Image crosses data splits")
    return sorted(groups.items())


def shard_groups(groups, rank, world):
    if world != 4 or rank not in range(world):
        raise ValueError("Four distinct training-mask shards required")
    return groups[rank::world]


def visible_prompt(visible):
    yy, xx = np.where(visible)
    if not len(xx):
        raise ValueError("Empty annotated visible support")
    return [int(xx.min()), int(yy.min()), int(xx.max()+1), int(yy.max()+1)]


def merge(output, manifest):
    shards = [json.loads((output/f"shard-{rank}.json").read_bytes()) for rank in range(4)]
    contract = shards[0]["contract"]
    if any(shard["contract"] != contract or shard["rank"] != rank or shard["complete"] is not True
           for rank, shard in enumerate(shards)):
        raise ValueError("Training mask shard contracts differ")
    rows = [row for shard in shards for row in shard["rows"]]
    expected = {row["id"]:row for _, group in training_groups(manifest) for row in group}
    if len(rows) != 9494 or len(rows) != len(expected) or {row["id"] for row in rows} != set(expected):
        raise ValueError("Incomplete or duplicate training mask inventory")
    for rank, shard in enumerate(shards):
        images = {image for image, _ in shard_groups(training_groups(manifest), rank, 4)}
        for row in shard["rows"]:
            source = expected[row["id"]]
            if row["image_id"] not in images or any(row[key] != source[key] for key in ("image_id", "image", "mask")):
                raise ValueError("Training object identity or shard changed")
            path = (output/row["sam_mask"]).resolve()
            if not path.is_relative_to(output.resolve()) or sha256(path) != row["sam_sha256"]:
                raise ValueError("Training SAM mask changed")
    receipt = dict(schema=1, complete=True, contract=contract, objects=len(rows), images=1554,
        split="train", validation_objects=0, holdout_objects=0, benchmark_objects=0,
        empty_sam_masks=sum(row["empty_sam_mask"] for row in rows),
        shards={f"shard-{rank}.json":sha256(output/f"shard-{rank}.json") for rank in range(4)},
        rows=sorted(rows, key=lambda row:row["id"]))
    destination = output/"manifest.json"
    if destination.exists() and json.loads(destination.read_bytes()) != receipt:
        raise ValueError("Preserve existing different mask manifest")
    write_json(destination, receipt)
    print(json.dumps({key:value for key,value in receipt.items() if key not in ("rows", "contract")}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--sam-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--merge", action="store_true")
    args = parser.parse_args()
    manifest_path = args.data/"manifest.json"
    if sha256(manifest_path) != DATA_SHA256:
        raise ValueError("Original training/validation data required")
    manifest = json.loads(manifest_path.read_bytes())
    groups = training_groups(manifest)
    if len(groups) != 1554 or sum(len(group) for _,group in groups) != 9494:
        raise ValueError("Complete original training split required")
    if args.merge:
        merge(args.output, manifest)
        return
    import torch
    from segment_anything import SamPredictor, sam_model_registry
    rank, world, local = (int(os.environ[name]) for name in ("RANK", "WORLD_SIZE", "LOCAL_RANK"))
    selected = shard_groups(groups, rank, world)
    torch.cuda.set_device(local)
    if torch.cuda.device_count() != 4 or "QUADRO" not in torch.cuda.get_device_name(local).upper():
        raise RuntimeError("Four allocated Quadro GPUs are required")
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Slurm allocation required")
    torch.manual_seed(20261123)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    sample = torch.randn(32, 32, device="cuda")
    if not torch.allclose(sample, torch.fft.irfft2(torch.fft.rfft2(sample), s=sample.shape), atol=1e-5):
        raise RuntimeError("CUDA FFT failed")
    args.output.mkdir(parents=True, exist_ok=True)
    destination = args.output/f"shard-{rank}.json"
    mask_root = args.output/f"masks-{rank}"
    if destination.exists() or mask_root.exists():
        raise FileExistsError("Preserve earlier mask attempt; use a fresh output")
    mask_root.mkdir()
    contract = dict(scope="Train-only SAM inputs for isolated shape-only robustness development; frozen v2 untouched",
        data_manifest_sha256=DATA_SHA256, sam_checkpoint_sha256=sha256(args.sam_checkpoint),
        source_sha256=sha256(Path(__file__)), job_id=os.environ["SLURM_JOB_ID"],
        world_size=4, seed=20261123, prompt="Tight annotated VISIBLE support XYXY box, maxima plus one; single SAM mask",
        split="train", holdout_access=False, benchmark_access=False, torch=torch.__version__, cuda=torch.version.cuda)
    sam = sam_model_registry["vit_h"](checkpoint=str(args.sam_checkpoint)).cuda().eval()
    predictor = SamPredictor(sam)
    records, verified = [], set()
    for index, (image_id, objects) in enumerate(selected, 1):
        if len({row["image"] for row in objects}) != 1:
            raise ValueError("Training image identity changed")
        for row in objects:
            for name in (row["image"], row["mask"]):
                path = (args.data/name).resolve()
                if not path.is_relative_to(args.data.resolve()):
                    raise ValueError("Training asset escapes package")
                if name not in verified:
                    if sha256(path) != manifest["artifacts"][name]:
                        raise ValueError("Training asset hash changed")
                    verified.add(name)
        with Image.open(args.data/objects[0]["image"]) as source:
            rgb = np.array(source.convert("RGB"))
        supports, boxes = [], []
        for row in objects:
            with Image.open(args.data/row["mask"]) as source:
                visible = (np.array(source) & 2) > 0
            supports.append(visible)
            boxes.append(visible_prompt(visible))
        predictor.set_image(rgb)
        predicted = []
        for start in range(0, len(boxes), 16):
            batch = torch.tensor(boxes[start:start+16], device="cuda", dtype=torch.float32)
            transformed = predictor.transform.apply_boxes_torch(batch, rgb.shape[:2])
            with torch.inference_mode():
                masks, _, _ = predictor.predict_torch(point_coords=None, point_labels=None, boxes=transformed,
                    multimask_output=False, return_logits=False)
            predicted.extend(masks[:, 0].cpu().numpy().astype(bool))
        if len(predicted) != len(objects):
            raise ValueError("SAM returned an incomplete training batch")
        for row, visible, prompt, mask in zip(objects, supports, boxes, predicted):
            path = mask_root/(row["id"]+".png")
            Image.fromarray(mask.astype(np.uint8)*255).save(path)
            records.append(dict(id=row["id"], image_id=image_id, image=row["image"], mask=row["mask"],
                sam_mask=path.relative_to(args.output).as_posix(), sam_sha256=sha256(path),
                prompt=prompt, empty_sam_mask=not bool(mask.any()),
                visible_mask_iou=float((mask & visible).sum()/max(1, (mask | visible).sum()))))
        write_json(args.output/f"progress-{rank}.json", dict(rank=rank, images=index, total_images=len(selected),
            objects=len(records), at=datetime.now(timezone.utc).isoformat()))
        if index % 25 == 0:
            print(f"rank {rank}: {index}/{len(selected)} training images, {len(records)} masks", flush=True)
    write_json(destination, dict(complete=True, rank=rank, gpu=torch.cuda.get_device_name(local),
        cuda_fft_round_trip=True, contract=contract, rows=records))


if __name__ == "__main__":
    main()

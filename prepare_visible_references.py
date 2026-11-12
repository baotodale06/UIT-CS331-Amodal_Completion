#!/usr/bin/env python3
"""Create visible-object references using the open-world authors' LISA procedure."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image

from amodal_benchmark.dataset import SOURCES, audit_files, read_official, sha256, write_json
from amodal_benchmark.protocol import ensure_run_config
from amodal_benchmark.references import (LISA_MODEL, LISA_REVISION, LISA_REPO_COMMIT,
                                         VISION_MODEL, VISION_REVISION, LisaSegmenter,
                                         cached_reference, save_reference)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path)
    parser.add_argument("--lisa-repo", type=Path)
    parser.add_argument("--lisa-model", default=LISA_MODEL)
    parser.add_argument("--precision", choices=("fp16", "bf16"), default="fp16")
    parser.add_argument("--lisa-device-map", choices=("auto", "single"), default="auto",
                        help="Auto balances unquantized LISA across visible GPUs; single uses GPU 0")
    parser.add_argument("--prompt-template", default="{label}", help="Released app receives the raw annotated class label")
    parser.add_argument("--sources", default=",".join(SOURCES))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--check-only", action="store_true", help="Check source inputs without importing or loading LISA")
    args = parser.parse_args(argv)
    if args.seed < 0 or args.limit is not None and args.limit < 1:
        parser.error("Require non-negative seed and positive limit")
    try:
        args.prompt_template.format(label="cat")
    except (ValueError, KeyError) as error:
        parser.error(f"Invalid --prompt-template: {error}")
    if "{label}" not in args.prompt_template:
        parser.error("--prompt-template must contain {label}")
    root = args.dataset_root.resolve()
    references = (args.reference_dir or root / "visible_references").resolve()
    rows, _ = read_official(root)
    sources = args.sources.split(",")
    if not sources or len(set(sources)) != len(sources) or set(sources) - set(SOURCES):
        parser.error("--sources must list unique VG,COCOA,free_images,laion")
    rows = [r for r in rows if r["source"] in sources]
    if args.limit:
        rows = rows[:args.limit]
    audit = audit_files(root, rows, decode=True)
    if audit["corrupt_images"] or audit["missing_images"] and not args.allow_incomplete:
        raise ValueError(f"Reference preparation needs source images: {len(audit['missing_images'])} missing, "
                         f"{len(audit['corrupt_images'])} corrupt; run prepare_open_world.py first")
    if args.check_only:
        print(f"Reference-preparation input preflight passed for {len(rows)} targets; no models loaded.")
        return 0
    if args.lisa_repo is None:
        parser.error("--lisa-repo is required; the shared Slurm runner provisions the pinned checkout automatically")
    repo = args.lisa_repo.resolve()
    commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    if commit != LISA_REPO_COMMIT:
        raise ValueError(f"LISA checkout must be pinned to {LISA_REPO_COMMIT}, got {commit}")
    config = {"schema": 1, "source": "saraao/amodal@14d59e09/LISA/app.py + main.py",
              "lisa_repo_commit": commit, "lisa_model": args.lisa_model,
              "lisa_revision": LISA_REVISION if args.lisa_model == LISA_MODEL else None,
              "vision_model": VISION_MODEL, "vision_revision": VISION_REVISION,
              "precision": args.precision, "quantized": False, "inference": "direct_pytorch",
              "prompt_template": args.prompt_template, "conversation": "llava_v1",
              "seed": args.seed, "sam_long_side": 1024, "max_new_tokens": 512,
              "mask_threshold": 0., "erosion_kernel": [5, 5], "erosion_iterations": 1,
              "crop_buffer": 60, "boundary_padding": 150, "boundary_gap": 10}
    if Path(args.lisa_model).is_dir():
        config["local_model_files"] = {str(p.relative_to(args.lisa_model)): sha256(p)
                                       for p in Path(args.lisa_model).rglob("*") if p.is_file()}
    ensure_run_config(references / "reference_config.json", config)
    model = None
    execution_path = None
    statistics = Counter()
    for index, row in enumerate(rows, 1):
        image_path = root / row["image"]
        if not image_path.exists():
            statistics["missing_image"] += 1
            continue
        input_hash = sha256(image_path)
        record = cached_reference(references, row, input_hash)
        if record is None:
            from evaluate_open_world import seed_for, set_seed
            if model is None:
                model = LisaSegmenter(repo, args.lisa_model, args.precision, args.lisa_device_map)
                execution_path = f"executions/{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{os.getpid()}.json"
                write_json(references / execution_path, model.execution_info)
            target_seed = seed_for(args.seed, row["id"])
            set_seed(target_seed)
            query = args.prompt_template.format(label=row["label"])
            mask = model.predict(image_path, query)
            with Image.open(image_path) as image:
                rgb = np.array(image.convert("RGB"))
            record = save_reference(references, row, rgb, mask, input_hash,
                                    {"model_config": "reference_config.json", "seed": target_seed,
                                     "query": query, "execution": execution_path})
        statistics[record["status"]] += 1
        print(f"[{index}/{len(rows)}] {row['id']}: {record['status']} ({record['mask_pixels']} visible pixels)", flush=True)
    write_json(references / "reference_summary.json", {"selected_targets": len(rows), "counts": dict(statistics),
               "generated_references": True, "human_ground_truth": False,
               "method": "LISA-13B-llama2-v1 + released visible crop and erosion"})
    print(f"Generated-reference report: {references / 'reference_summary.json'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)

"""Check geometric selection headroom on the 183 historical development images only."""
import argparse
import os
from pathlib import Path
import sys
import numpy as np
from PIL import Image
import torch
from train_shape import DEV, OUT, read_json
from amodal_benchmark.dataset import write_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--worker-index", type=int, required=True)
    p.add_argument("--worker-count", type=int, required=True)
    args = p.parse_args()
    records = []
    rows = [(d, r) for d in DEV for r in read_json(d/"cases.json")]
    for folder, row in rows[args.worker_index::args.worker_count]:
        cid = row["case_id"]
        truth = torch.from_numpy(np.array(Image.open(folder/"evaluation_only"/cid/"full.png")) > 0).cuda()
        masks = []
        for seed in (11,22,33,44):
            masks.append(np.array(Image.open(folder/"full"/"candidates"/cid/f"seed-{seed}-mask.png")) > 0)
        masks = torch.from_numpy(np.stack(masks)).cuda()
        iou = ((masks & truth).sum((1,2)) / (masks | truth).sum((1,2)).clamp(min=1)).cpu().tolist()
        records.append(dict(case=cid, target=row["target_id"], scores=iou))
    write_json(OUT/f"dev-oracle-{args.worker_index}.json", dict(device=torch.cuda.get_device_name(),
               job=os.environ["SLURM_JOB_ID"], cases=records))


if __name__ == "__main__":
    main()


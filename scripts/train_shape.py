"""Fit the five-channel auxiliary predictor to the original 183 Surface development objects."""
import json
import os
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from amodal_benchmark.dataset import sha256, write_json
from amodal_benchmark.shape import make_network, resized

DEV = [Path("output/surface-consistency-20261108"), Path("output/surface-consistency-replication87-20261108")]
TEST = Path("output/surface-confirmation128-20261110")
OUT = Path("output/shape-20261121")


def read_json(path):
    return json.loads(path.read_text())


def train():
    torch.manual_seed(20261121)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    rng = np.random.default_rng(20261121)
    rows = [(folder, row) for folder in DEV for row in read_json(folder / "cases.json")]
    test = read_json(TEST / "cases.json")
    for key in ("image_identity", "source_sha256"):
        assert len({r[key] for _, r in rows}) == 183
        assert not {r[key] for _, r in rows}.intersection(r[key] for r in test)
    examples = []
    for folder, row in rows:
        path = folder / "evaluation_only" / row["case_id"]
        for name in ("truth.png", "full.png"):
            assert sha256(path / name) == row["evaluation_hashes"][name]
        rgb = resized(np.array(Image.open(path / "truth.png").convert("RGB")), (256, 256), True)
        mask = resized(np.array(Image.open(path / "full.png")) > 0, (256, 256)).astype(bool)
        examples.append((rgb, mask))
    model = make_network().cuda()
    network = torch.nn.DataParallel(model)
    optim = torch.optim.AdamW(model.parameters(), lr=0.001)
    yy, xx = np.indices((256, 256))
    losses = []
    for epoch in range(40):
        loss_sum = 0.
        order = rng.permutation(len(rows))
        for start in range(0, len(rows), 16):
            inputs, targets = [], []
            for i in order[start:start + 16]:
                rgb, full = examples[i]
                turns = int(rng.integers(4))
                rgb, full = np.rot90(rgb, turns), np.rot90(full, turns)
                if rng.random() < .5:
                    rgb, full = rgb[:, ::-1], full[:, ::-1]
                angle = rng.choice([0., np.pi / 2, np.pi, 1.5*np.pi]) if rng.random() < .5 else rng.uniform(0, 2*np.pi)
                field = xx*np.cos(angle) + yy*np.sin(angle)
                occlusion = field >= np.quantile(field[full], rng.uniform(.20, .80))
                visible = full & ~occlusion
                obs = rgb.copy()
                obs[occlusion] = 127
                inputs.append(np.dstack((obs / 255., visible, occlusion)))
                targets.append(full.copy())
            x = torch.tensor(np.stack(inputs), dtype=torch.float32, device="cuda").permute(0,3,1,2)
            y = torch.tensor(np.stack(targets), dtype=torch.float32, device="cuda")[:,None]
            logits = network(x)
            prob = logits.sigmoid()
            dice = 1 - ((2*(prob*y).sum((2,3))+1) / ((prob+y).sum((2,3))+1)).mean()
            visible = x[:,3:4]
            loss = F.binary_cross_entropy_with_logits(logits, y) + dice + .2*((1-prob)*visible).sum()/visible.sum()
            optim.zero_grad()
            loss.backward()
            optim.step()
            loss_sum += loss.item()*len(inputs)
        losses.append(loss_sum/len(rows))
        print(epoch+1, losses[-1], flush=True)
    temp = OUT / "prior.tmp"
    torch.save(model.state_dict(), temp)
    temp.replace(OUT / "prior.pt")
    write_json(OUT / "training.json", dict(job=os.environ["SLURM_JOB_ID"], devices=[torch.cuda.get_device_name(i)
               for i in range(torch.cuda.device_count())], epochs=40, losses=losses, seed=20261121,
               checkpoint_sha256=sha256(OUT/"prior.pt"), sources={str(p):sha256(p/"cases.json") for p in DEV},
               implementations={p:sha256(ROOT/p) for p in ("scripts/train_shape.py","amodal_benchmark/shape.py")},
               identities=[r["image_identity"] for _,r in rows], confirmation_sha256=sha256(TEST/"cases.json")))


if __name__ == "__main__":
    train()


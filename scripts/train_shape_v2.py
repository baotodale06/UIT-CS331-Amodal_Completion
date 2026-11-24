"""Train the shape-only amodal model on an explicitly verified Colab GPU."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from amodal_benchmark.dataset import sha256, write_json
from amodal_benchmark.shape_v2 import make_network, observation_tensor, shape_loss, visible_box


class AmodalData(Dataset):
    def __init__(self, root, rows, training, seed=20261123, size=256):
        self.root, self.rows, self.training = root, rows, training
        self.seed, self.size, self.epoch = seed, size, 0

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(self.root / row["image"]) as image:
            rgb = np.array(image.convert("RGB"))
        with Image.open(self.root / row["mask"]) as image:
            encoded = np.array(image)
        truth, visible = (encoded & 1) > 0, (encoded & 2) > 0
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, index]))
        if self.training and rng.random() < .5:
            rgb, truth, visible = rgb[:, ::-1].copy(), truth[:, ::-1].copy(), visible[:, ::-1].copy()
        if self.training and rng.random() < .5:
            yy, xx = np.indices(visible.shape)
            vy, vx = np.where(visible)
            kind = rng.integers(3)
            if kind == 0:
                angle = rng.uniform(0, 2*np.pi)
                field = xx*np.cos(angle) + yy*np.sin(angle)
                extra = field >= np.quantile(field[visible], rng.uniform(.2, .8))
            else:
                center = rng.integers(len(vx))
                rx = max(4, np.ptp(vx)*rng.uniform(.2, .65))
                ry = max(4, np.ptp(vy)*rng.uniform(.2, .65))
                dx, dy = (xx-vx[center])/rx, (yy-vy[center])/ry
                extra = (dx*dx + dy*dy <= 1) if kind == 1 else ((np.abs(dx) <= 1) & (np.abs(dy) <= 1))
            remainder = visible & ~extra
            if remainder.sum() >= max(64, .1*visible.sum()):
                visible = remainder
        box = visible_box(visible, rng.uniform(1.5, 3.) if self.training else 2.)
        rgb_crop = np.array(Image.fromarray(rgb).crop(box))
        v = np.array(Image.fromarray(visible).crop(box))
        target = np.array(Image.fromarray(truth).crop(box).resize((self.size, self.size), Image.Resampling.NEAREST))
        inputs = observation_tensor(rgb_crop, v, size=self.size)
        return torch.from_numpy(inputs), torch.from_numpy(target.astype(np.float32))[None], row["image_id"]


def evaluate(model, loader, device):
    model.eval()
    groups = defaultdict(lambda: defaultdict(list))
    with torch.inference_mode():
        for x, y, identities in loader:
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(x)
            visible = x[:, 3:4] > .5
            prediction = (logits >= 0) | visible
            truth = y > .5
            for i, identity in enumerate(identities.tolist()):
                for name, pred, target in (("full_iou", prediction[i], truth[i]),
                                            ("hidden_iou", prediction[i] & ~visible[i], truth[i] & ~visible[i])):
                    if name == "hidden_iou" and not target.any():
                        continue
                    union = (pred | target).sum().item()
                    groups[identity][name].append((pred & target).sum().item() / max(1, union))
    means = {name: float(np.mean([np.mean(g[name]) for g in groups.values() if g[name]]))
             for name in ("full_iou", "hidden_iou")}
    return dict(**means, images=len(groups), hidden_images=sum(bool(g["hidden_iou"]) for g in groups.values()))


def atomic_save(value, path):
    temporary = path.with_suffix(".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def verify_data(root, manifest):
    if manifest.get("status") != "complete":
        raise ValueError("Data exclusion audit is not complete")
    for name, expected in manifest["artifacts"].items():
        file = (root / name).resolve()
        if not file.is_relative_to(root.resolve()) or sha256(file) != expected:
            raise ValueError(f"Data hash mismatch: {name}")
    groups = defaultdict(set)
    for row in manifest["rows"]:
        groups[row["image_id"]].add(row["split"])
    if any(len(splits) != 1 for splits in groups.values()):
        raise ValueError("Image crosses training/validation/holdout splits")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/research/shape-v2-20261123.json")
    parser.add_argument("--expected-gpu", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    started = time.monotonic()
    if not torch.cuda.is_available():
        raise RuntimeError("Training requires the authorized Google Colab GPU, not CPU")
    if not (os.environ.get("COLAB_RELEASE_TAG") or Path("/var/colab").exists()):
        raise RuntimeError("Google Colab environment marker absent; do not train on the local machine or Slurm")
    gpu = torch.cuda.get_device_name(0)
    if args.expected_gpu.upper() not in gpu.upper():
        raise RuntimeError(f"Requested {args.expected_gpu}, received {gpu}; no silent fallback")
    probe = torch.arange(4096, device="cuda", dtype=torch.float32).reshape(64, 64) / 4096
    restored = torch.fft.irfft2(torch.fft.rfft2(probe), s=probe.shape)
    if not torch.allclose(probe, restored, atol=1e-5, rtol=1e-5):
        raise RuntimeError("CUDA FFT round-trip failed before training")
    config = json.loads(args.config.read_bytes())
    settings = config["training"]
    manifest = json.loads((args.data / "manifest.json").read_bytes())
    verify_data(args.data, manifest)
    args.output.mkdir(parents=True, exist_ok=True)
    source_names = ("scripts/train_shape_v2.py", "amodal_benchmark/shape_v2.py")
    contract = dict(config_sha256=sha256(args.config), data_sha256=sha256(args.data / "manifest.json"),
                    sources={name: sha256(ROOT / name) for name in source_names})
    last = args.output / "last.pt"
    if not args.resume and any(args.output.iterdir()):
        raise ValueError("Use a new output directory or --resume with the identical contract")
    if args.resume and not last.exists():
        raise ValueError("Resume checkpoint is missing")
    seed = config["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    model = make_network(pretrained=not args.resume).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, settings["epochs"])
    history, start, best = [], 0, (-1., -1.)
    if args.resume:
        saved = torch.load(last, map_location="cpu", weights_only=False)
        if saved["contract"] != contract:
            raise ValueError("Resume data, config, or implementation changed")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        history, start, best = saved["history"], saved["epoch"], tuple(saved["best"])
        torch.set_rng_state(saved["torch_rng"])
        torch.cuda.set_rng_state_all(saved["cuda_rng"])
    receipt_path = args.output / "training.json"
    receipt = json.loads(receipt_path.read_bytes()) if args.resume else dict(
        schema=2, architecture="resnet34_unet_5channel", platform="Google Colab", contract=contract,
        seed=seed, selection=settings["selection"], holdout_used=False, runs=[])
    receipt["runs"].append(dict(started=datetime.now(timezone.utc).isoformat(), gpu=gpu,
        expected_gpu=args.expected_gpu, torch=torch.__version__, cuda=torch.version.cuda,
        nvidia_smi=subprocess.check_output(["nvidia-smi"], text=True), resumed_epoch=start,
        cuda_fft_round_trip=True))
    if not args.resume:
        weight_files = list((Path(torch.hub.get_dir()) / "checkpoints").glob("resnet34-*.pth"))
        receipt["encoder_weights"] = {p.name:sha256(p) for p in weight_files}
    write_json(receipt_path, receipt)
    train = AmodalData(args.data, [r for r in manifest["rows"] if r["split"] == "train"], True, seed, settings["size"])
    validation = AmodalData(args.data, [r for r in manifest["rows"] if r["split"] == "validation"], False, seed, settings["size"])
    if not len(train) or not len(validation):
        raise ValueError("Training and validation must be nonempty")
    val_loader = DataLoader(validation, batch_size=settings["batch_size"], num_workers=settings["workers"], pin_memory=True)
    for epoch in range(start, settings["epochs"]):
        train.epoch = epoch
        loader = DataLoader(train, batch_size=settings["batch_size"], shuffle=True,
            generator=torch.Generator().manual_seed(seed+epoch), num_workers=settings["workers"], pin_memory=True)
        model.train()
        total = 0.
        for step, (x, y, _) in enumerate(loader):
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(x)
                loss = shape_loss(logits.float(), y, x[:, 3:4])
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            total += loss.item()*len(x)
            if step % 25 == 0:
                print(f"epoch {epoch+1} step {step}/{len(loader)} loss {loss.item():.5f}", flush=True)
        result = evaluate(model, val_loader, "cuda")
        rank = result["hidden_iou"], result["full_iou"]
        if not all(np.isfinite(rank)):
            raise ValueError("Validation geometry is not finite")
        history.append(dict(epoch=epoch+1, loss=total/len(train), validation=result))
        improved = rank > best
        if improved:
            best = rank
        scheduler.step()
        saved = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                     epoch=epoch+1, best=best, history=history, contract=contract,
                     torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all())
        atomic_save(saved, last)
        if improved:
            atomic_save(dict(model=model.state_dict(), contract=contract, epoch=epoch+1, validation=result), args.output / "best.pt")
        receipt.update(epochs_completed=epoch+1, history=history, best_sha256=sha256(args.output / "best.pt"),
                       last_sha256=sha256(last), complete=epoch+1 == settings["epochs"])
        receipt["runs"][-1].update(last_updated=datetime.now(timezone.utc).isoformat(),
                                   elapsed_seconds=time.monotonic()-started)
        write_json(receipt_path, receipt)
        print(json.dumps(history[-1]), flush=True)


if __name__ == "__main__":
    main()

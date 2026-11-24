"""Distributed shape-only training on Slurm; preserves the original Colab package."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from amodal_benchmark.dataset import sha256, write_json
from amodal_benchmark.shape_v2 import make_network, shape_loss
from scripts.train_shape_v2 import AmodalData, atomic_save, verify_data

CONFIG = "configs/research/shape-v2-slurm-20261123.json"
SOURCES = ("scripts/train_shape_v2_slurm.py", "scripts/train_shape_v2.py",
           "amodal_benchmark/shape_v2.py")
DATA_SHA256 = "ab2f4d1112c52bcd1b675e0f213ed8ee387fbc2601a3596f8f2b5e933ea7d13e"


def validate_contract(config, manifest, world_size):
    settings = config["training"]
    if config.get("training_platform") != "Slurm":
        raise ValueError("An explicit Slurm training contract is required")
    if world_size != settings["world_size"] or settings["batch_size"] % world_size:
        raise ValueError("Allocation must match frozen world size and global batch")
    if (manifest.get("holdout_packaged") is not False
            or set(row["split"] for row in manifest["rows"]) != {"train", "validation"}):
        raise ValueError("Only physically isolated training and validation data are allowed")


def evaluate(model, loader, device, maximum_batches=None):
    from collections import defaultdict
    groups = defaultdict(lambda: defaultdict(list))
    model.eval()
    with torch.inference_mode():
        for batch, (x, y, identities) in enumerate(loader):
            x, y = x.to(device), y.to(device)
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(x)
            visible, truth = x[:, 3:4] > .5, y > .5
            prediction = (logits >= 0) | visible
            for name, pred, target in (("full_iou", prediction, truth),
                    ("hidden_iou", prediction & ~visible, truth & ~visible)):
                intersection = (pred & target).flatten(1).sum(1).cpu().tolist()
                union = (pred | target).flatten(1).sum(1).cpu().tolist()
                area = target.flatten(1).sum(1).cpu().tolist()
                for identity, common, total, pixels in zip(identities.tolist(), intersection, union, area):
                    if name != "hidden_iou" or pixels:
                        groups[identity][name].append(common / max(1, total))
            if maximum_batches is not None and batch + 1 >= maximum_batches:
                break
    values = {name: float(np.mean([np.mean(group[name]) for group in groups.values() if group[name]]))
              for name in ("hidden_iou", "full_iou")}
    if not all(np.isfinite(value) for value in values.values()):
        raise ValueError("Validation geometry is not finite")
    return dict(**values, images=len(groups), hidden_images=sum(bool(g["hidden_iou"]) for g in groups.values()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / CONFIG)
    parser.add_argument("--expected-gpu", default="Quadro RTX 6000")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise RuntimeError("A real Slurm CUDA allocation is required")
    rank, world, local_rank = (int(os.environ[name]) for name in ("RANK", "WORLD_SIZE", "LOCAL_RANK"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    config = json.loads(args.config.read_bytes())
    settings = config["training"]
    manifest_path = args.data / "manifest.json"
    if sha256(manifest_path) != DATA_SHA256:
        raise ValueError("Training manifest differs from the verified holdout-free bundle")
    manifest = json.loads(manifest_path.read_bytes())
    validate_contract(config, manifest, world)
    gpu = torch.cuda.get_device_name(local_rank)
    if args.expected_gpu.upper() not in gpu.upper():
        raise RuntimeError(f"Expected {args.expected_gpu}, received {gpu}")
    probe = torch.arange(4096, device=device, dtype=torch.float32).reshape(64, 64) / 4096
    if not torch.allclose(probe, torch.fft.irfft2(torch.fft.rfft2(probe), s=probe.shape), atol=1e-5, rtol=1e-5):
        raise RuntimeError("CUDA FFT round trip failed")
    dist.init_process_group("nccl")
    started = time.monotonic()
    contract = dict(config_sha256=sha256(args.config), data_sha256=sha256(manifest_path),
                    sources={name: sha256(ROOT / name) for name in SOURCES})
    if rank == 0:
        verify_data(args.data, manifest)
        if not args.resume and args.output.exists() and any(args.output.iterdir()):
            raise ValueError("New training requires an empty output directory")
        args.output.mkdir(parents=True, exist_ok=True)
    dist.barrier()
    seed = config["seed"]
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(2)
    # Only rank zero downloads pretrained weights; other ranks use the shared cache.
    if rank == 0:
        model = make_network(pretrained=not args.resume).to(device)
    dist.barrier()
    if rank != 0:
        model = make_network(pretrained=not args.resume).to(device)
    model = DistributedDataParallel(model, device_ids=[local_rank])
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, settings["epochs"])
    scaler = torch.cuda.amp.GradScaler(init_scale=1024.)
    history, start, best = [], 0, (-1., -1.)
    receipt_path, last = args.output / "training.json", args.output / "last.pt"
    if args.resume:
        saved = torch.load(last, map_location="cpu", weights_only=False)
        if saved["contract"] != contract:
            raise ValueError("Resume data, source or configuration changed")
        model.module.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        scaler.load_state_dict(saved["scaler"])
        history, start, best = saved["history"], saved["epoch"], tuple(saved["best"])
        torch.set_rng_state(saved["rng"][rank]["torch"])
        torch.cuda.set_rng_state(saved["rng"][rank]["cuda"], local_rank)
    receipt = json.loads(receipt_path.read_bytes()) if args.resume else dict(
        schema=2, architecture="resnet34_unet_5channel", platform="Slurm", contract=contract,
        seed=seed, selection=settings["selection"], holdout_used=False, runs=[], complete=False)
    devices = [None] * world
    dist.all_gather_object(devices, dict(rank=rank, gpu=gpu, cuda_fft_round_trip=True))
    receipt["runs"].append(dict(started=datetime.now(timezone.utc).isoformat(), gpu=gpu,
        expected_gpu=args.expected_gpu, torch=str(torch.__version__), cuda=torch.version.cuda,
        job_id=os.environ["SLURM_JOB_ID"], devices=devices, world_size=world,
        cuda_fft_round_trip=True, resumed_epoch=start, precision="float16 with GradScaler"))
    if rank == 0:
        receipt["encoder_weights"] = {p.name: sha256(p) for p in (Path(torch.hub.get_dir()) / "checkpoints").glob("resnet34-*.pth")}
        write_json(receipt_path, receipt)
    train = AmodalData(args.data, [r for r in manifest["rows"] if r["split"] == "train"], True, seed, settings["size"])
    validation = AmodalData(args.data, [r for r in manifest["rows"] if r["split"] == "validation"], False, seed, settings["size"])
    batch_size = settings["batch_size"] // world
    val_loader = DataLoader(validation, batch_size=batch_size, num_workers=settings["workers_per_rank"], pin_memory=True)
    sampler = DistributedSampler(train, num_replicas=world, rank=rank, shuffle=True, seed=seed)
    for epoch in range(start, settings["epochs"]):
        epoch_start = time.monotonic()
        train.epoch = epoch
        sampler.set_epoch(epoch)
        loader = DataLoader(train, batch_size=batch_size, sampler=sampler,
                          num_workers=settings["workers_per_rank"], pin_memory=True)
        model.train()
        total = torch.zeros(2, device=device, dtype=torch.float64)
        for step, (x, y, _) in enumerate(loader):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(x)
                loss = shape_loss(logits.float(), y, x[:, 3:4])
            finite = torch.tensor(int(torch.isfinite(loss)), device=device)
            dist.all_reduce(finite, op=dist.ReduceOp.MIN)
            if not finite.item():
                raise RuntimeError("Nonfinite training loss on a worker")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            scaler.step(optimizer)
            scaler.update()
            total[0] += loss.detach().double() * len(x)
            total[1] += len(x)
            if rank == 0 and step % 25 == 0:
                print(f"epoch {epoch+1} step {step}/{len(loader)} loss {loss.item():.5f} seconds {time.monotonic()-epoch_start:.1f}", flush=True)
            if args.preflight and step == 1:
                break
        dist.all_reduce(total)
        dist.barrier()
        result = evaluate(model.module, val_loader, device, 2 if args.preflight else None) if rank == 0 else None
        payload = [result]
        dist.broadcast_object_list(payload, src=0)
        result = payload[0]
        rank_score = result["hidden_iou"], result["full_iou"]
        history.append(dict(epoch=epoch+1, loss=(total[0]/total[1]).item(), validation=result,
                            seconds=time.monotonic()-epoch_start, sampled_training_objects=int(total[1].item())))
        improved = rank_score > best
        if improved:
            best = rank_score
        scheduler.step()
        rng = [None] * world
        dist.all_gather_object(rng, dict(torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state(local_rank)))
        if rank == 0:
            if args.preflight and not optimizer.state:
                raise RuntimeError("Preflight did not perform an optimizer update")
            saved = dict(model=model.module.state_dict(), optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                         scaler=scaler.state_dict(), epoch=epoch+1, best=best, history=history, contract=contract, rng=rng)
            atomic_save(saved, last)
            if improved:
                atomic_save(dict(model=model.module.state_dict(), contract=contract, epoch=epoch+1, validation=result), args.output / "best.pt")
            receipt.update(epochs_completed=epoch+1, history=history, best_sha256=sha256(args.output / "best.pt"),
                last_sha256=sha256(last), complete=epoch+1 == settings["epochs"] and not args.preflight,
                preflight=args.preflight)
            receipt["runs"][-1].update(last_updated=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.monotonic()-started)
            write_json(receipt_path, receipt)
            print(json.dumps(history[-1]), flush=True)
            if args.preflight:
                replay = torch.load(last, map_location="cpu", weights_only=False)
                if replay["contract"] != contract or len(replay["rng"]) != world or "scaler" not in replay:
                    raise ValueError("Checkpoint preflight failed")
                write_json(args.output / "preflight.json", dict(passed=True, devices=devices, contract=contract,
                    checkpoint_roundtrip=True, real_training_steps=2, benchmark_used=False, holdout_used=False))
        dist.barrier()
        if args.preflight:
            break
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

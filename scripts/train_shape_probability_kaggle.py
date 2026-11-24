"""Separate proper-probability shape loss; preserve the exact native retention gate."""
import argparse
import hashlib
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
from amodal_benchmark.shape_v2 import make_network
from scripts.shape_probability_loss import probability_loss
from scripts.train_shape_v2 import AmodalData, atomic_save, verify_data
from amodal_benchmark.shape_v2_runtime import training_receipt
from scripts.shape_robust_data import RobustData, aggregate
from scripts.shape_constrained_data import freeze_encoder, eligible_improvement
from scripts.shape_probability_validation import evaluate_native

CONFIG = "configs/research/shape-proper-loss-kaggle-20261123.json"
SOURCES = ("scripts/train_shape_probability_kaggle.py", "scripts/shape_probability_loss.py", "scripts/shape_probability_validation.py", "scripts/shape_robust_data.py", "scripts/shape_constrained_data.py")
DATA_SHA256 = "ab2f4d1112c52bcd1b675e0f213ed8ee387fbc2601a3596f8f2b5e933ea7d13e"


def validate_contract(config, manifest, world_size):
    settings = config["training"]
    if config.get("training_platform") != "Kaggle":
        raise ValueError("An explicit Kaggle training contract is required")
    if world_size != settings["world_size"] or settings["batch_size"] % world_size:
        raise ValueError("Allocation must match frozen world size and global batch")
    if (manifest.get("holdout_packaged") is not False
            or set(row["split"] for row in manifest["rows"]) != {"train", "validation"}):
        raise ValueError("Only physically isolated training and validation data are allowed")


def verify_sam_inputs(args, manifest, masks, validation, config):
    if (masks.get("complete") is not True or masks.get("split") != "train"
            or masks["contract"]["data_manifest_sha256"] != DATA_SHA256
            or masks["contract"]["holdout_access"] is not False
            or masks["contract"]["benchmark_access"] is not False
            or sha256(args.sam_validation/"result.json") != config["validation_diagnostic_result_sha256"]
            or validation.get("complete") is not True):
        raise ValueError("SAM data contract changed")
    for split, records in (("train", masks["rows"]), ("validation", validation["rows"])):
        expected = {row["id"]:row for row in manifest["rows"] if row["split"] == split}
        if len(records) != len(expected) or {row["id"] for row in records} != set(expected):
            raise ValueError("SAM object inventory differs")
        for row in records:
            if any(row[key] != expected[row["id"]][key] for key in ("image_id", "image", "mask")):
                raise ValueError("SAM input identity changed")
            base = args.sam_training if split == "train" else args.sam_validation
            name = row["sam_mask"] if split == "train" else "objects/"+row["id"]+"/sam-visible.png"
            expected_hash = row["sam_sha256"] if split == "train" else row["artifacts"]["sam-visible.png"]
            path = (base/name).resolve()
            if not path.is_relative_to(base.resolve()) or sha256(path) != expected_hash:
                raise ValueError("SAM input hash changed")


def encoder_digest(model):
    digest = hashlib.sha256()
    for name,value in sorted(model.state_dict().items()):
        if name.startswith(('stem.','layers.')):
            digest.update(name.encode())
            digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / CONFIG)
    parser.add_argument("--sam-training", type=Path, required=True)
    parser.add_argument("--sam-validation", type=Path, required=True)
    parser.add_argument("--initial-checkpoint", type=Path, required=True)
    parser.add_argument("--expected-gpu", default="T4")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    if not Path("/kaggle/input").is_dir() or not os.environ.get("SHAPE_KAGGLE_RUN") or not torch.cuda.is_available():
        raise RuntimeError("A real Kaggle CUDA allocation is required")
    if not str(torch.__version__).startswith("1.13.1") or torch.version.cuda != "11.7" or np.__version__ != "1.23.5":
        raise RuntimeError("Pinned PyTorch1.13.1/CUDA11.7/NumPy1.23.5 required")
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
    if config.get('schema')!=7 or settings.get('probability_loss_version')!=1:
        raise ValueError('Separate proper probability-loss protocol required')
    if sha256(args.initial_checkpoint) != config["initial_checkpoint_sha256"]:
        raise ValueError("Warm start differs from the completed v2 checkpoint")
    inherited = training_receipt(args.initial_checkpoint)
    sam_training_path = args.sam_training/"manifest.json"
    sam_training = json.loads(sam_training_path.read_bytes())
    sam_validation = json.loads((args.sam_validation/"result.json").read_bytes())
    gpu = torch.cuda.get_device_name(local_rank)
    if args.expected_gpu.upper() not in gpu.upper():
        raise RuntimeError(f"Expected {args.expected_gpu}, received {gpu}")
    probe = torch.arange(4096, device=device, dtype=torch.float32).reshape(64, 64) / 4096
    if not torch.allclose(probe, torch.fft.irfft2(torch.fft.rfft2(probe), s=probe.shape), atol=1e-5, rtol=1e-5):
        raise RuntimeError("CUDA FFT round trip failed")
    dist.init_process_group("nccl")
    started = time.monotonic()
    contract = dict(config_sha256=sha256(args.config), data_sha256=sha256(manifest_path),
                    sources={name: sha256(ROOT / name) for name in SOURCES},
                    initial_checkpoint_sha256=sha256(args.initial_checkpoint),
                    inherited_training_contract=inherited["contract"],
                    sam_training_manifest_sha256=sha256(sam_training_path),
                    validation_sam_result_sha256=sha256(args.sam_validation/"result.json"))
    if rank == 0:
        verify_data(args.data, manifest)
        verify_sam_inputs(args, manifest, sam_training, sam_validation, config)
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
    model = make_network(pretrained=False).to(device)
    if not args.resume:
        initial = torch.load(args.initial_checkpoint, map_location="cpu", weights_only=False)
        if initial["contract"] != inherited["contract"]:
            raise ValueError("Warm-start checkpoint contract mismatch")
        model.load_state_dict(initial["model"], strict=True)
        del initial
    freeze_encoder(model)
    initial_encoder_sha256 = encoder_digest(model)
    model = DistributedDataParallel(model, device_ids=[local_rank])
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=settings["learning_rate"], weight_decay=settings["weight_decay"])
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
        schema=7, method="shape-probability-kaggle", architecture="resnet34_unet_5channel", platform="Kaggle", contract=contract,
        seed=seed, selection=settings["selection"], holdout_used=False, benchmark_used=False,
        empty_training_sam_masks=sam_training["empty_sam_masks"], encoder_sha256=initial_encoder_sha256,
        geometry_policy=config["inference_geometry"], runs=[], complete=False)
    devices = [None] * world
    dist.all_gather_object(devices, dict(rank=rank, gpu=gpu, cuda_fft_round_trip=True))
    receipt["runs"].append(dict(started=datetime.now(timezone.utc).isoformat(), gpu=gpu,
        expected_gpu=args.expected_gpu, torch=str(torch.__version__), cuda=torch.version.cuda,
        job_id=os.environ["SHAPE_KAGGLE_RUN"], devices=devices, world_size=world,
        cuda_fft_round_trip=True, resumed_epoch=start, precision="float16 with GradScaler"))
    if rank == 0:
        receipt["encoder_weights"] = {p.name: sha256(p) for p in (Path(torch.hub.get_dir()) / "checkpoints").glob("resnet34-*.pth")}
        write_json(receipt_path, receipt)
    train = RobustData(args.data, [r for r in manifest["rows"] if r["split"] == "train"],
        args.sam_training, sam_training["rows"], seed, settings["size"])
    validation = [r for r in manifest["rows"] if r["split"] == "validation"]
    validation_images = sorted({row["image_id"] for row in validation})
    rank_images = set(validation_images[rank::world])
    rank_validation = [row for row in validation if row["image_id"] in rank_images]
    batch_size = settings["batch_size"] // world
    sampler = DistributedSampler(train, num_replicas=world, rank=rank, shuffle=True, seed=seed)
    if args.resume:
        baseline = receipt["initial_validation"]
        initial_encoder_sha256 = receipt['encoder_sha256']
        if encoder_digest(model.module)!=initial_encoder_sha256:
            raise ValueError('Resume encoder differs from frozen initialization')
    else:
        initial_rows = evaluate_native(model.module, args.data, rank_validation, args.sam_validation,
            device, 3 if args.preflight else None)
        gathered_initial = [None] * world
        dist.all_gather_object(gathered_initial, initial_rows)
        initial_rows = sorted([row for group in gathered_initial for row in group], key=lambda row: row["id"])
        baseline = aggregate(initial_rows)
        expected_rows = {row["id"]:row for row in sam_validation["rows"]}
        for row in initial_rows:
            expected = expected_rows[row["id"]]["sam_input"]
            for metric in ("full_iou", "hidden_iou"):
                if row[metric] != expected[metric]:
                    raise ValueError("Initial checkpoint does not replay frozen SAM geometry")
        if not args.preflight and (len(initial_rows) != 1264 or len({r["id"] for r in initial_rows}) != 1264):
            raise ValueError("Incomplete initial native evaluation")
        best = baseline["hidden_iou"], baseline["full_iou"]
        receipt["initial_validation"] = baseline
        if rank == 0:
            atomic_save(dict(model=model.module.state_dict(), contract=contract, epoch=0, validation=baseline), args.output / "best.pt")
            write_json(args.output/"best-validation-native.json", dict(split="validation", epoch=0,
                contract=contract, aggregate=baseline, rows=initial_rows, benchmark_success=False))
            write_json(receipt_path, receipt)
        dist.barrier()
    for epoch in range(start, settings["epochs"]):
        epoch_start = time.monotonic()
        train.epoch = epoch
        sampler.set_epoch(epoch)
        loader = DataLoader(train, batch_size=batch_size, sampler=sampler,
                          num_workers=settings["workers_per_rank"], pin_memory=True)
        model.train()
        freeze_encoder(model.module)
        total = torch.zeros(3, device=device, dtype=torch.float64)
        for step, (x, y, annotated_visible) in enumerate(loader):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            annotated_visible = annotated_visible.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(x)
                loss = probability_loss(logits.float(), y, x[:,3:4])
            finite = torch.tensor(int(torch.isfinite(loss)), device=device)
            dist.all_reduce(finite, op=dist.ReduceOp.MIN)
            if not finite.item():
                raise RuntimeError("Nonfinite training loss on a worker")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            previous_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            total[2] += int(scaler.get_scale()<previous_scale)
            total[0] += loss.detach().double() * len(x)
            total[1] += len(x)
            if rank == 0 and step % 25 == 0:
                print(f"epoch {epoch+1} step {step}/{len(loader)} loss {loss.item():.5f} seconds {time.monotonic()-epoch_start:.1f}", flush=True)
            if args.preflight and step == 1:
                break
        dist.all_reduce(total)
        dist.barrier()
        # Native validation on every rank must use the exact rank-zero buffers
        # that will be saved, not independently updated per-rank BatchNorm state.
        for buffer in model.module.buffers():
            dist.broadcast(buffer, src=0)
        if encoder_digest(model.module)!=initial_encoder_sha256:
            raise ValueError('Frozen encoder parameters or buffers changed')
        native_rows = evaluate_native(model.module, args.data, rank_validation, args.sam_validation,
            device, 3 if args.preflight else None)
        gathered = [None] * world
        dist.all_gather_object(gathered, native_rows)
        native_rows = sorted([row for rows in gathered for row in rows], key=lambda row:row["id"])
        if len({row["id"] for row in native_rows}) != len(native_rows):
            raise ValueError("Repeated native validation object")
        if not args.preflight and len(native_rows) != 1264:
            raise ValueError("Incomplete native validation")
        result = aggregate(native_rows)
        if not all(np.isfinite(result[key]) for key in ("full_iou", "hidden_iou")):
            raise ValueError("Nonfinite native validation")
        rank_score = result["hidden_iou"], result["full_iou"]
        history.append(dict(epoch=epoch+1, loss=(total[0]/total[1]).item(), validation=result,
                            seconds=time.monotonic()-epoch_start, sampled_training_objects=int(total[1].item()),
                            skipped_optimizer_steps=int(total[2].item()), encoder_sha256=initial_encoder_sha256))
        improved = eligible_improvement(result, baseline, best)
        if improved:
            best = rank_score
        scheduler.step()
        rng = [None] * world
        dist.all_gather_object(rng, dict(torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state(local_rank)))
        if rank == 0:
            if args.preflight and (not optimizer.state or total[2].item()!=0):
                raise RuntimeError("Preflight did not perform an optimizer update")
            saved = dict(model=model.module.state_dict(), optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                         scaler=scaler.state_dict(), epoch=epoch+1, best=best, history=history, contract=contract, rng=rng)
            atomic_save(saved, last)
            if improved:
                atomic_save(dict(model=model.module.state_dict(), contract=contract, epoch=epoch+1, validation=result), args.output / "best.pt")
                write_json(args.output/"best-validation-native.json", dict(split="validation", epoch=epoch+1,
                    contract=contract, aggregate=result, rows=native_rows, benchmark_success=False))
            receipt.update(epochs_completed=epoch+1, history=history, best_sha256=sha256(args.output / "best.pt"),
                last_sha256=sha256(last), complete=epoch+1 == settings["epochs"] and not args.preflight,
                preflight=args.preflight)
            receipt["best_validation_sha256"] = sha256(args.output/"best-validation-native.json")
            receipt["selected_epoch"] = json.loads((args.output/"best-validation-native.json").read_bytes())["epoch"]
            receipt["new_checkpoint_eligible"] = receipt["selected_epoch"] > 0
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

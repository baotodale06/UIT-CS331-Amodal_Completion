"""Shape-only v2 checkpoint validation and progressive candidate selection.

This module deliberately does not import the legacy Shape/Surface runtime.
The original visible observation defines the prior; generated RGB is only an
output to select, never a new observation or a corrected output.
"""
from functools import wraps
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .dataset import sha256, write_json
from .shape_v2 import make_network, predict_probability, score_masks, visible_box

ROOT = Path(__file__).resolve().parents[1]
CONFIG = "configs/research/shape-v2-20261123.json"
TRAINING_SOURCES = ("scripts/train_shape_v2.py", "amodal_benchmark/shape_v2.py")
SLURM_CONFIG = "configs/research/shape-v2-slurm-20261123.json"
SLURM_TRAINING_SOURCES = ("scripts/train_shape_v2_slurm.py", *TRAINING_SOURCES)
# Exact manifest inside the verified Colab training ZIP, excluding holdout.
TRAINING_DATA_SHA256 = "ab2f4d1112c52bcd1b675e0f213ed8ee387fbc2601a3596f8f2b5e933ea7d13e"
SEED_SLOTS = (11, 22, 33, 44, 55, 66, 77, 88)


def training_receipt(checkpoint):
    """Verify completed authorized training without torch or opening labels."""
    checkpoint = Path(checkpoint)
    receipt = json.loads(checkpoint.with_name("training.json").read_bytes())
    platform = receipt.get("platform")
    config_name = SLURM_CONFIG if platform == "Slurm" else CONFIG
    source_names = SLURM_TRAINING_SOURCES if platform == "Slurm" else TRAINING_SOURCES
    config = json.loads((ROOT / config_name).read_bytes())
    if (receipt.get("schema") != 2 or receipt.get("architecture") != "resnet34_unet_5channel"
            or platform not in ("Google Colab", "Slurm") or receipt.get("holdout_used") is not False
            or receipt.get("complete") is not True or receipt.get("preflight", False)):
        raise ValueError("Shape-v2 requires completed, holdout-free authorized training")
    if sha256(checkpoint) != receipt.get("best_sha256"):
        raise ValueError("Shape-v2 checkpoint differs from the selected best checkpoint")
    expected = dict(config_sha256=sha256(ROOT / config_name), data_sha256=TRAINING_DATA_SHA256,
                    sources={name: sha256(ROOT / name) for name in source_names})
    if receipt.get("contract") != expected:
        raise ValueError("Shape-v2 training data, config, or implementation differs")
    epochs = config["training"]["epochs"]
    history = receipt.get("history", [])
    if (receipt.get("epochs_completed") != epochs or len(history) != epochs
            or [row.get("epoch") for row in history] != list(range(1, epochs + 1))
            or receipt.get("seed") != config["seed"]
            or receipt.get("selection") != config["training"]["selection"]):
        raise ValueError("Shape-v2 training history is incomplete or has a different selection rule")
    for row in history:
        for name in ("hidden_iou", "full_iou"):
            value = row.get("validation", {}).get(name)
            if not isinstance(value, (int, float)) or not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("Shape-v2 validation geometry must be finite and bounded")
    runs = receipt.get("runs", [])
    if not runs or any(not run.get("expected_gpu") or not run.get("gpu")
                       or run["expected_gpu"].upper() not in run["gpu"].upper()
                       or run.get("cuda_fft_round_trip") is not True for run in runs):
        raise ValueError("Shape-v2 GPU evidence is missing or inconsistent")
    if platform == "Slurm":
        world = config["training"]["world_size"]
        for run in runs:
            devices = run.get("devices", [])
            if (not run.get("job_id") or run.get("world_size") != world or len(devices) != world
                    or {device.get("rank") for device in devices} != set(range(world))
                    or any(device.get("cuda_fft_round_trip") is not True
                           or run["expected_gpu"].upper() not in device.get("gpu", "").upper() for device in devices)):
                raise ValueError("Shape-v2 Slurm allocation evidence is missing or inconsistent")
    return receipt


def best_epoch(receipt):
    # max preserves the first epoch on an exact tie, just like the trainer's >.
    return max(receipt["history"], key=lambda row: (row["validation"]["hidden_iou"],
                                                   row["validation"]["full_iou"]))


def shape_v2_contract(checkpoint):
    checkpoint = Path(checkpoint)
    receipt = training_receipt(checkpoint)
    files = (*receipt["contract"]["sources"], "amodal_benchmark/shape_v2_runtime.py", "evaluate_open_world.py")
    return dict(schema=2, method="shape-v2", checkpoint_sha256=receipt["best_sha256"],
                training_sha256=sha256(checkpoint.with_name("training.json")),
                training_contract=receipt["contract"], training_receipt=receipt,
                selected_epoch=best_epoch(receipt)["epoch"],
                implementations={name: sha256(ROOT / name) for name in files},
                seeds=list(SEED_SLOTS), seed_rule="sha256(surface-progressive-v1:target:iteration:slot); "
                "first 8 little-endian bytes modulo 2**63-1 (same seeds as legacy shape)",
                observation="original visible RGB and mask only; neutralize all other RGB before resizing",
                crop="square visible bounding box, padding 2.0; zero probability outside this box",
                unknown="all non-visible pixels, matching training; no occluder constraint",
                projection="native probability into current PMC crop, bilinear resize; visible pixels set to 1",
                selection="maximum expected mask IoU; exclude empty masks; first seed wins ties; slot 0 fallback",
                candidate_matching="highest IoU with original visible support; require positive overlap",
                surface=None, native_visible_restoration=False, rgb_correction=False)


def load_prior(checkpoint):
    import torch
    receipt = training_receipt(checkpoint)
    # PyTorch 1.13's experimental weights-only unpickler rejects BINFLOAT in
    # validation metadata. Only load the exact locally trained artifact after
    # verifying its digest, completed receipt, source and data contract above.
    weights_only = not str(torch.__version__).startswith("1.13.")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=weights_only)
    chosen = best_epoch(receipt)
    if (saved.get("contract") != receipt["contract"] or saved.get("epoch") != chosen["epoch"]
            or saved.get("validation") != chosen["validation"]):
        raise ValueError("Shape-v2 checkpoint payload differs from the validation-selected training receipt")
    if not torch.cuda.is_available():
        raise RuntimeError("Shape-v2 PMC inference requires a validated CUDA environment")
    model = make_network(pretrained=False)
    model.load_state_dict(saved["model"], strict=True)
    return model.cuda().eval()


def progressive_seed(target_seed, iteration, slot):
    payload = f"surface-progressive-v1:{target_seed}:{iteration}:{slot}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**63 - 1)


def choose(scores):
    eligible = [(index, value) for index, value in enumerate(scores) if value is not None]
    if not scores:
        raise ValueError("Shape-v2 requires a nonempty candidate pool")
    if any(not np.isfinite(value) or not 0 <= value <= 1 for _, value in eligible):
        raise ValueError("Shape-v2 candidate scores must be finite and bounded")
    return (max(eligible, key=lambda item: item[1])[0], False) if eligible else (0, True)


def original_prior(model, rgb, visible):
    """Use the same visible-based input crop as validation; return canvas geometry."""
    rgb, visible = np.asarray(rgb, np.uint8), np.asarray(visible, bool)
    if rgb.shape != (*visible.shape, 3):
        raise ValueError("Original shape observation and mask frames differ")
    box = visible_box(visible, padding=2.0)
    observed = np.array(Image.fromarray(rgb).crop(box))
    support = np.array(Image.fromarray(visible).crop(box))
    observed[~support] = 127
    probability = predict_probability(model, observed, support)
    if (probability.shape != support.shape or not np.isfinite(probability).all()
            or np.any((probability < 0) | (probability > 1))):
        raise ValueError("Invalid shape-v2 prior probability")
    x0, y0, x1, y1 = box
    height, width = visible.shape
    left, top, right, bottom = max(0, x0), max(0, y0), min(width, x1), min(height, y1)
    canvas = np.zeros(visible.shape, np.float32)
    canvas[top:bottom, left:right] = probability[top-y0:bottom-y0, left-x0:right-x0]
    canvas[visible] = 1
    return canvas, box, observed, support


def project_prior(probability, visible, crop_rc, size):
    r0, r1, c0, c1 = crop_rc
    height, width = visible.shape
    if probability.shape != visible.shape or not (0 <= r0 < r1 <= height and 0 <= c0 < c1 <= width):
        raise ValueError("Original prior and progressive crop frames differ")
    region = np.s_[r0:r1, c0:c1]
    prior = np.array(Image.fromarray(probability[region]).resize(size, Image.Resampling.BILINEAR))
    support = np.array(Image.fromarray(visible[region]).resize(size, Image.Resampling.NEAREST))
    prior = np.clip(prior, 0, 1)
    prior[support] = 1
    return prior, support


def match_candidate(instances, visible):
    masks = [] if instances is None else [np.asarray(mask, bool) for mask in instances]
    if any(mask.shape != visible.shape for mask in masks):
        raise ValueError("Candidate segmentation and original visible frames differ")
    overlap = [float((mask & visible).sum() / max(1, (mask | visible).sum())) for mask in masks]
    if not overlap or max(overlap) <= 0:
        return np.zeros_like(visible)
    return masks[int(np.argmax(overlap))]


def install_shape_v2(upstream, models, sam_path, checkpoint):
    """Replace only the diffusion-pool choice; leave all released gates intact."""
    import torch
    if getattr(upstream, "_shape_v2_installed", False) or getattr(upstream, "_surface_installed", False):
        raise ValueError("Shape-v2 requires an unmodified shape-only selector")
    model = load_prior(checkpoint)
    original_iteration, original_crop = upstream.run_iteration, upstream.crop
    original_mixed, candidate_segment = upstream.run_mixed_context_diffusion, upstream.segment
    mixed_signature = inspect.signature(original_mixed)
    active = None

    @wraps(original_iteration)
    def iteration(query_obj, *args, **kwargs):
        nonlocal active
        if active is not None:
            raise ValueError("Nested shape-v2 target execution")
        active = dict(query=query_obj, crop_rc=None)
        try:
            return original_iteration(query_obj, *args, **kwargs)
        finally:
            active = None

    @wraps(original_crop)
    def crop(*args, **kwargs):
        result = original_crop(*args, **kwargs)
        if active is not None:
            active["crop_rc"] = tuple(int(value) for value in result[1:])
        return result

    @wraps(original_mixed)
    def mixed(*args, **kwargs):
        bound = mixed_signature.bind(*args, **kwargs)
        bound.apply_defaults()
        query = bound.arguments["query_obj"]
        if active is None or active["query"] is not query or active["crop_rc"] is None:
            raise ValueError("Shape-v2 sampler has no matching progressive crop")
        crop_rc = active["crop_rc"]
        r0, r1, c0, c1 = crop_rc
        if (r1-r0, c1-c0) != bound.arguments["sd_occ_mask"].shape:
            raise ValueError("Shape-v2 sampler and progressive crop frames differ")
        visible = np.asarray(query.init_query_mask_canvas, bool)
        canvas, box, observed, input_support = original_prior(model, query.init_img_canvas, visible)
        target_seed = torch.initial_seed()
        seeds = [progressive_seed(target_seed, query.iter_id, slot) for slot in SEED_SLOTS]
        pipeline, label = bound.arguments["sd_inpaint_model"], bound.arguments["sd_prompt"]

        @torch.inference_mode()
        def sample_pool(**arguments):
            images, masks = [], []
            for index, seed in enumerate(seeds):
                result = pipeline(**dict(arguments, generator=torch.Generator(device="cuda").manual_seed(seed)))
                if len(result.images) != 1:
                    raise ValueError("Shape-v2 expects one generated image per seed")
                image = result.images[0].convert("RGB")
                if index == 0:
                    first, size = result, image.size
                    prior, support = project_prior(canvas, visible, crop_rc, size)
                elif image.size != size:
                    raise ValueError("Shape-v2 candidates have inconsistent frames")
                _, instances, _, _ = candidate_segment(models[0], upstream.run_sam, str(sam_path), image,
                                                        upstream.transform_image(image), [label])
                images.append(np.array(image))
                masks.append(match_candidate(instances, support))
            scores = score_masks(prior, masks)
            index, fallback = choose(scores)
            folder = Path(query.output_img_dir) / "shape-v2" / f"iteration-{query.iter_id:02d}"
            folder.mkdir(parents=True, exist_ok=True)
            Image.fromarray(observed).save(folder / "observed-input.png")
            Image.fromarray(input_support.astype(np.uint8)*255).save(folder / "input-visible.png")
            np.save(folder / "prior.npy", prior, allow_pickle=False)
            Image.fromarray(np.rint(prior*255).astype(np.uint8)).save(folder / "prior.png")
            Image.fromarray(support.astype(np.uint8)*255).save(folder / "visible.png")
            for slot, (image, mask) in enumerate(zip(images, masks)):
                Image.fromarray(image).save(folder / f"candidate-{slot}.png")
                Image.fromarray(mask.astype(np.uint8)*255).save(folder / f"mask-{slot}.png")
            first.images[0] = Image.fromarray(images[index])
            first.images[0].save(folder / "selected.png")
            write_json(folder / "decision.json", dict(schema=2, method="shape-v2", iteration=int(query.iter_id),
                crop_rc=crop_rc, prior_box_xyxy=box, target_seed=target_seed, seed_slots=list(SEED_SLOTS),
                seeds=seeds, scores=scores, index=index, fallback=fallback,
                inference_truth_access=False, rgb_correction=False,
                artifacts={p.name: sha256(p) for p in sorted(folder.iterdir()) if p.suffix in (".png", ".npy")}))
            query.benchmark_diagnostics.setdefault("shape_v2_iterations", []).append(dict(
                index=index, fallback=fallback, decision_sha256=sha256(folder / "decision.json")))
            return first

        bound.arguments["sd_inpaint_model"] = sample_pool
        return original_mixed(*bound.args, **bound.kwargs)

    upstream.run_iteration, upstream.crop = iteration, crop
    upstream.run_mixed_context_diffusion = mixed
    upstream._shape_v2_installed = True

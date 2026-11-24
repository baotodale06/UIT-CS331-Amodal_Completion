"""Isolated shape-only SAM adaptation; require verified native admission first."""
from contextlib import contextmanager
from functools import wraps
import inspect
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .dataset import sha256, write_json
from . import shape_v2_runtime as v2
from .shape_v2 import make_network, score_masks
from .shape_v2_runtime import SEED_SLOTS, choose, match_candidate, progressive_seed, project_prior

ROOT = Path(__file__).resolve().parents[1]
METHOD = "shape-sam-constrained"
CONFIG = "configs/research/shape-sam-constrained-20261123.json"
TRAINING_SOURCES = ("scripts/train_shape_constrained_slurm.py", "scripts/shape_robust_data.py",
                    "scripts/shape_constrained_data.py")
NATIVE_SOURCE = "scripts/evaluate_shape_constrained_native.py"
REPLAY_SOURCE = "scripts/collect_shape_constrained.py"
EVIDENCE_FILES = ("training.json", "best-validation-native.json", "native-contract.json",
                  "native-result.json", "native-replay.json")
SAM_TRAINING_SHA256 = '77d75e18fd160c3113ff5813c35b83eb65c9181d8070713908183a64e5c134a7'
ARITHMETIC = dict(cudnn_deterministic=False, cudnn_benchmark=False,
                  cudnn_allow_tf32=True, matmul_allow_tf32=False, autocast=False,
                  precision="float32 inputs and weights; PyTorch 1.13 native validation defaults")


def selected_epoch(receipt):
    """Replay the stable constrained selection, including the initial checkpoint."""
    initial = dict(epoch=0, validation=receipt["initial_validation"])
    rows = [initial, *receipt["history"]]
    for row in rows:
        for name in ("full_iou", "hidden_iou"):
            value = row["validation"].get(name)
            if type(value) not in (float, int) or not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("Constrained native geometry must be finite and bounded")
    eligible = [row for row in rows
                if row["validation"]["full_iou"] >= initial["validation"]["full_iou"]]
    return max(eligible, key=lambda row: (row["validation"]["hidden_iou"], row["validation"]["full_iou"]))


def expected_training_contract(config):
    return dict(config_sha256=sha256(ROOT / CONFIG), data_sha256=v2.TRAINING_DATA_SHA256,
        sources={name: sha256(ROOT / name) for name in TRAINING_SOURCES},
        initial_checkpoint_sha256=config["initial_checkpoint_sha256"],
        inherited_training_contract=dict(config_sha256=sha256(ROOT / v2.SLURM_CONFIG),
            data_sha256=v2.TRAINING_DATA_SHA256,
            sources={name: sha256(ROOT / name) for name in v2.SLURM_TRAINING_SOURCES}),
        sam_training_manifest_sha256=SAM_TRAINING_SHA256,
        validation_sam_result_sha256=config["validation_diagnostic_result_sha256"])


def admission_sources():
    names = (*TRAINING_SOURCES, *v2.SLURM_TRAINING_SOURCES, CONFIG, v2.SLURM_CONFIG,
             NATIVE_SOURCE, REPLAY_SOURCE, "amodal_benchmark/shape_constrained_runtime.py",
             "amodal_benchmark/shape_v2_runtime.py", "scripts/package_shape_constrained.py")
    return {name: sha256(ROOT / name) for name in names}


def verify_evidence(checkpoint):
    """Read metadata only; no validation images, labels or model loading."""
    checkpoint = Path(checkpoint)
    folder = checkpoint.parent
    data = {name: json.loads((folder / name).read_bytes()) for name in EVIDENCE_FILES}
    receipt, best, native, result, replay = (data[name] for name in EVIDENCE_FILES)
    config = json.loads((ROOT / CONFIG).read_bytes())
    epochs = config["training"]["epochs"]
    if (receipt.get("schema") != 4 or receipt.get("method") != METHOD
            or receipt.get("architecture") != "resnet34_unet_5channel"
            or receipt.get("platform") != "Slurm" or receipt.get("complete") is not True
            or receipt.get("preflight") is not False or receipt.get("holdout_used") is not False
            or receipt.get("benchmark_used") is not False
            or receipt.get("epochs_completed") != epochs or receipt.get("seed") != config["seed"]
            or receipt.get("selection") != config["training"]["selection"]
            or receipt.get("geometry_policy") != config["inference_geometry"]
            or [row.get("epoch") for row in receipt.get("history", [])] != list(range(1, epochs+1))):
        raise ValueError("Completed holdout-free constrained training is required")
    if SAM_TRAINING_SHA256 is None or receipt.get("contract") != expected_training_contract(config):
        raise ValueError("Constrained training sources, data or configuration changed")
    if sha256(checkpoint) != receipt.get("best_sha256"):
        raise ValueError("Constrained checkpoint bytes changed")
    runs = receipt.get("runs", [])
    world = config["training"]["world_size"]
    if not runs:
        raise ValueError("Constrained allocation evidence is missing")
    for run in runs:
        devices = run.get("devices", [])
        if (not run.get("job_id") or run.get("world_size") != world
                or run.get("expected_gpu") != "Quadro RTX 6000"
                or "QUADRO RTX 6000" not in run.get("gpu", "").upper()
                or run.get("cuda_fft_round_trip") is not True or len(devices) != world
                or {device.get("rank") for device in devices} != set(range(world))
                or any(device.get("cuda_fft_round_trip") is not True
                       or "QUADRO RTX 6000" not in device.get("gpu", "").upper() for device in devices)):
            raise ValueError("Constrained four-Quadro allocation evidence changed")
    chosen = selected_epoch(receipt)
    if (chosen["epoch"] == 0 or receipt.get("new_checkpoint_eligible") is not True
            or receipt.get("selected_epoch") != chosen["epoch"]):
        raise ValueError("Epoch zero or an ineligible constrained checkpoint cannot be promoted")
    if (sha256(folder / "best-validation-native.json") != receipt.get("best_validation_sha256")
            or best.get("epoch") != chosen["epoch"] or best.get("aggregate") != chosen["validation"]
            or best.get("contract") != receipt["contract"] or best.get("split") != "validation"
            or best.get("benchmark_success") is not False):
        raise ValueError("Selected constrained validation evidence changed")
    native_required = dict(schema=1, split="validation", objects=1264, images=217,
        training_receipt_sha256=sha256(folder / "training.json"), checkpoint_sha256=receipt["best_sha256"],
        selected_epoch=chosen["epoch"], data_manifest_sha256=v2.TRAINING_DATA_SHA256,
        baseline_result_sha256=config["validation_diagnostic_result_sha256"],
        source_sha256=sha256(ROOT / NATIVE_SOURCE), geometry_policy=config["inference_geometry"],
        cuda_fft_round_trip=True, holdout_access=False, benchmark_access=False)
    if (any(native.get(key) != value for key, value in native_required.items())
            or not native.get("job_id") or "QUADRO RTX 6000" not in native.get("gpu", "").upper()
            or not native.get("torch", "").startswith("1.13.1") or native.get("cuda") != "11.7"):
        raise ValueError("Constrained native GPU replay contract changed")
    result_required = dict(complete=True, objects=1264, images=217, selected_epoch=chosen["epoch"],
        contract_sha256=sha256(folder / "native-contract.json"),
        max_selected_validation_difference=0., benchmark_success=False)
    if any(result.get(key) != value for key, value in result_required.items()):
        raise ValueError("Exact complete constrained native GPU replay required")
    rows, best_rows = result.get("rows", []), best.get("rows", [])
    if (len(rows) != 1264 or len(best_rows) != 1264
            or len({row["id"] for row in rows}) != 1264
            or len({row["image_id"] for row in rows}) != 217
            or {row["id"] for row in rows} != {row["id"] for row in best_rows}):
        raise ValueError("Constrained validation inventory changed")
    replay_required = dict(verified=True, objects=1264, images=217, selected_epoch=chosen["epoch"],
        new_checkpoint_eligible=True, training_sha256=sha256(folder / "training.json"),
        checkpoint_sha256=receipt["best_sha256"], result_sha256=sha256(folder / "native-result.json"),
        replay_source_sha256=sha256(ROOT / REPLAY_SOURCE), geometry=result["geometry"], benchmark_success=False)
    if any(replay.get(key) != value for key, value in replay_required.items()):
        raise ValueError("Independent constrained native replay is missing or changed")
    return receipt


def training_receipt(checkpoint):
    checkpoint = Path(checkpoint)
    receipt = verify_evidence(checkpoint)
    admission = json.loads(checkpoint.with_name("admission.json").read_bytes())
    if (admission.get("schema") != 1 or admission.get("method") != METHOD
            or admission.get("files") != {name: sha256(checkpoint.parent / name) for name in EVIDENCE_FILES}
            or admission.get("checkpoint_sha256") != receipt["best_sha256"]
            or admission.get("selected_epoch") != receipt["selected_epoch"]
            or admission.get("sources") != admission_sources()
            or admission.get("benchmark_success") is not False):
        raise ValueError("Constrained admission package changed; build a new isolated package")
    return receipt


def shape_constrained_contract(checkpoint):
    checkpoint = Path(checkpoint)
    receipt = training_receipt(checkpoint)
    return dict(schema=1, method=METHOD, checkpoint_sha256=receipt["best_sha256"],
        training_sha256=sha256(checkpoint.with_name("training.json")),
        admission_sha256=sha256(checkpoint.with_name("admission.json")),
        training_contract=receipt["contract"], training_receipt=receipt,
        selected_epoch=receipt["selected_epoch"], implementations={**admission_sources(),
            "evaluate_open_world.py": sha256(ROOT / "evaluate_open_world.py")},
        seeds=list(SEED_SLOTS), seed_rule="unchanged shape-v2 progressive_seed",
        observation="original visible RGB and SAM support only; other RGB neutralized before resizing",
        crop="square visible bounding box, padding 2.0; zero probability outside crop",
        projection="native probability into current PMC crop, bilinear resize; visible pixels set to 1",
        selection="maximum expected mask IoU; exclude empty masks; first seed wins ties; slot 0 fallback",
        candidate_matching="highest IoU with original visible support; require positive overlap",
        arithmetic=ARITHMETIC, surface=None, native_visible_restoration=False, rgb_correction=False,
        evidence_files=json.loads(checkpoint.with_name("admission.json").read_bytes())["files"],
        interpretation="Iterative shape-only research; validation admission does not establish benchmark improvement")


def load_prior(checkpoint):
    import torch
    receipt = training_receipt(checkpoint)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=not str(torch.__version__).startswith("1.13."))
    chosen = selected_epoch(receipt)
    if (saved.get("contract") != receipt["contract"] or saved.get("epoch") != chosen["epoch"]
            or saved.get("validation") != chosen["validation"]):
        raise ValueError("Constrained checkpoint payload differs from selected native validation")
    if (not torch.cuda.is_available() or not str(torch.__version__).startswith("1.13.1")
            or torch.version.cuda != "11.7" or "QUADRO RTX 6000" not in torch.cuda.get_device_name().upper()):
        raise RuntimeError("Constrained inference requires the validated Quadro/PyTorch 1.13.1/CUDA 11.7 environment")
    model = make_network(pretrained=False)
    model.load_state_dict(saved["model"], strict=True)
    return model.float().cuda().eval()


@contextmanager
def native_arithmetic():
    """Use native validation arithmetic for the prior, then restore diffusion."""
    import torch
    settings = ((torch.backends.cudnn, "deterministic", False),
                (torch.backends.cudnn, "benchmark", False),
                (torch.backends.cudnn, "allow_tf32", True),
                (torch.backends.cuda.matmul, "allow_tf32", False))
    previous = [(owner, key, getattr(owner, key)) for owner, key, _ in settings]
    try:
        for owner, key, value in settings:
            setattr(owner, key, value)
        with torch.autocast("cuda", enabled=False):
            yield
    finally:
        for owner, key, value in previous:
            setattr(owner, key, value)


def original_prior(model, rgb, visible):
    with native_arithmetic():
        return v2.original_prior(model, rgb, visible)


def install_shape_constrained(upstream, models, sam_path, checkpoint):
    """Replace only the diffusion-pool choice; leave all released gates intact."""
    import torch
    if any(getattr(upstream, name, False) for name in ("_shape_constrained_installed", "_shape_v2_installed", "_surface_installed", "_shape_installed")):
        raise ValueError("Constrained shape requires an unmodified shape-only selector")
    model = load_prior(checkpoint)
    original_iteration, original_crop = upstream.run_iteration, upstream.crop
    original_mixed, candidate_segment = upstream.run_mixed_context_diffusion, upstream.segment
    mixed_signature = inspect.signature(original_mixed)
    active = None

    @wraps(original_iteration)
    def iteration(query_obj, *args, **kwargs):
        nonlocal active
        if active is not None:
            raise ValueError("Nested shape-sam-constrained target execution")
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
            raise ValueError("Constrained shape sampler has no matching progressive crop")
        crop_rc = active["crop_rc"]
        r0, r1, c0, c1 = crop_rc
        if (r1-r0, c1-c0) != bound.arguments["sd_occ_mask"].shape:
            raise ValueError("Constrained shape sampler and progressive crop frames differ")
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
                    raise ValueError("Constrained shape expects one generated image per seed")
                image = result.images[0].convert("RGB")
                if index == 0:
                    first, size = result, image.size
                    prior, support = project_prior(canvas, visible, crop_rc, size)
                elif image.size != size:
                    raise ValueError("Constrained shape candidates have inconsistent frames")
                _, instances, _, _ = candidate_segment(models[0], upstream.run_sam, str(sam_path), image,
                                                        upstream.transform_image(image), [label])
                images.append(np.array(image))
                masks.append(match_candidate(instances, support))
            scores = score_masks(prior, masks)
            index, fallback = choose(scores)
            folder = Path(query.output_img_dir) / "shape-sam-constrained" / f"iteration-{query.iter_id:02d}"
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
            write_json(folder / "decision.json", dict(schema=2, method="shape-sam-constrained", iteration=int(query.iter_id),
                crop_rc=crop_rc, prior_box_xyxy=box, target_seed=target_seed, seed_slots=list(SEED_SLOTS),
                seeds=seeds, scores=scores, index=index, fallback=fallback,
                inference_truth_access=False, rgb_correction=False,
                artifacts={p.name: sha256(p) for p in sorted(folder.iterdir()) if p.suffix in (".png", ".npy")}))
            query.benchmark_diagnostics.setdefault("shape_constrained_iterations", []).append(dict(
                index=index, fallback=fallback, decision_sha256=sha256(folder / "decision.json")))
            return first

        bound.arguments["sd_inpaint_model"] = sample_pool
        return original_mixed(*bound.args, **bound.kwargs)

    upstream.run_iteration, upstream.crop = iteration, crop
    upstream.run_mixed_context_diffusion = mixed
    upstream._shape_constrained_installed = True

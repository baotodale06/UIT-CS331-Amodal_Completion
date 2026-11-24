"""Validation bridge for frozen Kaggle/local shape priors; no benchmark admission."""
from contextlib import contextmanager
import json
from pathlib import Path

import numpy as np

from .dataset import sha256
from .shape_v2 import make_network
from .shape_v2_runtime import original_prior as v2_prior
from .shape_v2_runtime import SLURM_CONFIG, SLURM_TRAINING_SOURCES

ROOT = Path(__file__).resolve().parents[1]
PROFILES = {
    5: ("shape-sam-constrained-kaggle", "Kaggle", "shape-sam-kaggle-20261123.json"),
    6: ("shape-local-constrained", "LocalWSL", "shape-local-constrained-20261123.json"),
    7: ("shape-probability-kaggle", "Kaggle", "shape-proper-loss-kaggle-20261123.json"),
}
DATA_SHA256 = "ab2f4d1112c52bcd1b675e0f213ed8ee387fbc2601a3596f8f2b5e933ea7d13e"
INITIAL_SHA256 = "f3caa75ac40f901a7595fffa2c199221b83e7fe5a57192f34e2b2317ffe937b7"
SAM_SHA256 = "77d75e18fd160c3113ff5813c35b83eb65c9181d8070713908183a64e5c134a7"
PROFILE_SOURCES = {
    5: ("scripts/train_shape_kaggle.py", "scripts/shape_robust_data.py", "scripts/shape_constrained_data.py"),
    6: ("scripts/train_shape_local.py", "scripts/shape_local_data.py", "scripts/shape_robust_data.py",
        "scripts/shape_constrained_data.py", "scripts/train_shape_kaggle.py", "scripts/train_shape_v2.py",
        "scripts/train_shape_v2_slurm.py", "amodal_benchmark/__init__.py", "amodal_benchmark/dataset.py",
        "amodal_benchmark/shape_v2.py", "amodal_benchmark/shape_v2_runtime.py",
        "configs/research/shape-v2-20261123.json", "configs/research/shape-v2-slurm-20261123.json"),
    7: ("scripts/train_shape_probability_kaggle.py", "scripts/shape_probability_loss.py",
        "scripts/shape_probability_validation.py", "scripts/shape_robust_data.py", "scripts/shape_constrained_data.py"),
}
REPLAY_SOURCES = {5: "scripts/collect_shape_kaggle.py", 6: "scripts/verify_shape_local.py",
                  7: "scripts/collect_shape_probability_training.py"}


def read(path):
    return json.loads(Path(path).read_bytes())


def selected_epoch(receipt):
    """Replay each platform's frozen selection; stable ties retain the first."""
    schema = receipt.get("schema")
    if schema not in PROFILES:
        raise ValueError("Unsupported external shape training schema")
    baseline = receipt["initial_validation"]
    floor = {key: baseline[key] for key in ("full_iou", "hidden_iou")}
    rows = [dict(epoch=0, validation=baseline), *receipt["history"]]
    if schema == 6:
        original = receipt["original_quadro_validation"]
        floor = {key: max(floor[key], original[key]) for key in floor}
        if receipt["selection_floor"] != floor:
            raise ValueError("Local selection floors changed")
        rows_to_check = [*rows, dict(validation=original)]
    else:
        rows_to_check = rows
    for row in rows_to_check:
        for key in floor:
            value = row["validation"][key]
            if type(value) not in (int, float) or not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("Finite bounded native geometry is required")
    chosen, best = rows[0], (floor["hidden_iou"], floor["full_iou"])
    for row in rows[1:]:
        values = row["validation"]
        score = values["hidden_iou"], values["full_iou"]
        if (values["full_iou"] >= floor["full_iou"]
                and (schema != 6 or values["hidden_iou"] >= floor["hidden_iou"])
                and score > best):
            chosen, best = row, score
    return chosen


def training_evidence(training, replay_path):
    """Require completed learned weights and their independently recovered evidence."""
    training = Path(training)
    receipt = read(training / "training.json")
    schema = receipt.get("schema")
    if schema not in PROFILES:
        raise ValueError("Unsupported external shape training schema")
    method, platform, config_name = PROFILES[schema]
    config_path = ROOT / "configs/research" / config_name
    config = read(config_path)
    if (receipt.get("method") != method or receipt.get("platform") != platform
            or receipt.get("complete") is not True or receipt.get("holdout_used") is not False
            or receipt.get("benchmark_used") is not False or receipt.get("epochs_completed") != 20
            or [row.get("epoch") for row in receipt.get("history", [])] != list(range(1, 21))):
        raise ValueError("Completed holdout-free external shape training required")
    contract = receipt["contract"]
    if (contract.get("config_sha256") != sha256(config_path)
            or contract.get("data_sha256") != DATA_SHA256
            or contract.get("initial_checkpoint_sha256") != INITIAL_SHA256
            or contract.get("sam_training_manifest_sha256") != SAM_SHA256
            or contract.get("validation_sam_result_sha256") != config["validation_diagnostic_result_sha256"]):
        raise ValueError("Frozen training inputs or configuration changed")
    expected_parent = dict(config_sha256=sha256(ROOT / SLURM_CONFIG), data_sha256=DATA_SHA256,
        sources={name: sha256(ROOT / name) for name in SLURM_TRAINING_SOURCES})
    if (set(contract.get("sources", {})) != set(PROFILE_SOURCES[schema])
            or contract.get("inherited_training_contract") != expected_parent):
        raise ValueError("Training source inventory or initialization contract changed")
    for name, digest in contract["sources"].items():
        path = (ROOT / name).resolve()
        if not path.is_relative_to(ROOT.resolve()) or sha256(path) != digest:
            raise ValueError("Frozen training source changed: " + name)
    if schema != 6:
        if (receipt.get("preflight") is not False or receipt.get("seed") != config["seed"]
                or receipt.get("geometry_policy") != config["inference_geometry"]
                or receipt.get("selection") != config["training"]["selection"]):
            raise ValueError("Kaggle training policy changed")
        runs = receipt.get("runs", [])
        if not runs:
            raise ValueError("Kaggle GPU execution evidence missing")
        for run in runs:
            devices = run.get("devices", [])
            if (run.get("world_size") != 2 or run.get("cuda_fft_round_trip") is not True
                    or not run.get("job_id") or len(devices) != 2
                    or {device.get("rank") for device in devices} != {0, 1}
                    or any("T4" not in device.get("gpu", "").upper()
                           or device.get("cuda_fft_round_trip") is not True for device in devices)):
                raise ValueError("Two-T4 execution evidence changed")
        if schema == 7 and (not receipt.get("encoder_sha256") or
                {row.get("encoder_sha256") for row in receipt["history"]} != {receipt["encoder_sha256"]}):
            raise ValueError("Frozen encoder evidence changed")
    else:
        runtime = receipt.get("runtime", {})
        if (runtime.get("torch") != "1.13.1" or runtime.get("cuda") != "11.7"
                or runtime.get("cudnn") != 8500 or "RTX 3060" not in runtime.get("gpu", "")
                or runtime.get("cuda_fft_round_trip") is not True):
            raise ValueError("Local GPU execution evidence changed")
    chosen = selected_epoch(receipt)
    if (chosen["epoch"] == 0 or receipt.get("new_checkpoint_eligible") is not True
            or receipt.get("selected_epoch") != chosen["epoch"]):
        raise ValueError("Epoch zero or an ineligible checkpoint cannot be promoted")
    if (sha256(training / "best.pt") != receipt.get("best_sha256")
            or sha256(training / "best-validation-native.json") != receipt.get("best_validation_sha256")):
        raise ValueError("Selected training artifacts changed")
    best = read(training / "best-validation-native.json")
    if (best.get("epoch") != chosen["epoch"] or best.get("aggregate") != chosen["validation"]
            or best.get("contract") != contract):
        raise ValueError("Selected validation evidence changed")
    replay = read(replay_path)
    expected = dict(verified=True, objects=1264, images=217, selected_epoch=chosen["epoch"],
                    new_checkpoint_eligible=True, training_sha256=sha256(training / "training.json"),
                    checkpoint_sha256=receipt["best_sha256"], benchmark_success=False,
                    replay_source_sha256=sha256(ROOT / REPLAY_SOURCES[schema]))
    if any(replay.get(key) != value for key, value in expected.items()):
        raise ValueError("Independent native replay changed or is missing")
    if schema != 6 and (replay.get("disposition") != "verified" or replay.get("state") != "COMPLETE"):
        raise ValueError("Completed versioned Kaggle recovery required")
    if schema == 6 and (replay.get("original_quadro_different_mask_objects") != 0
                        or replay.get("original_quadro_different_pixels") != 0):
        raise ValueError("Original cross-GPU pixel replay must be exact")
    return receipt, best, replay


@contextmanager
def native_arithmetic(schema):
    """Match native training arithmetic and restore the caller's diffusion flags."""
    import torch
    if schema not in PROFILES:
        raise ValueError("Unsupported external shape training schema")
    settings = ((torch.backends.cudnn, "deterministic", False),
                (torch.backends.cudnn, "benchmark", False),
                (torch.backends.cudnn, "allow_tf32", schema != 6),
                (torch.backends.cuda.matmul, "allow_tf32", False))
    previous = [(owner, name, getattr(owner, name)) for owner, name, _ in settings]
    try:
        for owner, name, value in settings:
            setattr(owner, name, value)
        with torch.inference_mode(), torch.autocast("cuda", enabled=False):
            yield
    finally:
        for owner, name, value in previous:
            setattr(owner, name, value)


def original_prior(model, rgb, support, schema):
    support = np.asarray(support, bool)
    if np.asarray(rgb).shape != (*support.shape, 3):
        raise ValueError("Original observation and support frames differ")
    if not support.any():
        return np.zeros(support.shape, np.float32), None, None, None
    with native_arithmetic(schema):
        return v2_prior(model, rgb, support)


def load_for_validation(training, receipt):
    """GPU validator only; benchmark integration requires a separate admission."""
    import torch
    if (not torch.cuda.is_available() or torch.cuda.device_count() != 1
            or "QUADRO RTX 6000" not in torch.cuda.get_device_name().upper()
            or not str(torch.__version__).startswith("1.13.1") or torch.version.cuda != "11.7"):
        raise RuntimeError("Use one allocated Quadro with PyTorch 1.13.1/CUDA 11.7")
    checkpoint = Path(training) / "best.pt"
    if sha256(checkpoint) != receipt["best_sha256"]:
        raise ValueError("Selected checkpoint changed before model loading")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    chosen = selected_epoch(receipt)
    if (saved.get("contract") != receipt["contract"] or saved.get("epoch") != chosen["epoch"]
            or saved.get("validation") != chosen["validation"]):
        raise ValueError("Checkpoint payload differs from selected validation")
    model = make_network(pretrained=False)
    model.load_state_dict(saved["model"], strict=True)
    return model.float().cuda().eval()

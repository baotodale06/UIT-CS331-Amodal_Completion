"""Replay an eligible external shape prior on one Quadro before benchmark admission."""
import argparse
from datetime import datetime, timezone
import hashlib
import io
from pathlib import Path
import sys
import zipfile

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from amodal_benchmark.dataset import sha256, write_json
from amodal_benchmark.shape_external_prior import (
    read, training_evidence, load_for_validation, original_prior)


def safe_asset(root, name):
    path = (Path(root) / name).resolve()
    if not path.is_relative_to(Path(root).resolve()):
        raise ValueError("Validation path escaped its root")
    return path


def binary_mask(payload, shape):
    with Image.open(io.BytesIO(payload)) as image:
        raw = np.array(image)
    if raw.shape != tuple(shape) or not np.isin(raw, [0, 255]).all():
        raise ValueError("Native prediction must be binary in the original frame")
    return raw > 0


def pixel_hash(prediction):
    return hashlib.sha256(np.asarray(prediction, np.uint8).tobytes()).hexdigest()


def index_rows(rows, selected):
    if len(rows) != len(selected) or {row["id"] for row in rows} != set(selected):
        raise ValueError("Complete unique native validation inventory required")
    mapped = {row["id"]: row for row in rows}
    if any(mapped[key]["image_id"] != source["image_id"] for key, source in selected.items()):
        raise ValueError("Native validation image identity changed")
    return mapped


def validate(args):
    receipt, best, replay = training_evidence(args.training, args.replay)
    schema, contract = receipt["schema"], receipt["contract"]
    manifest_path, baseline_path = args.data / "manifest.json", args.sam_validation / "result.json"
    if (sha256(manifest_path) != contract["data_sha256"]
            or sha256(baseline_path) != contract["validation_sam_result_sha256"]):
        raise ValueError("Frozen validation inputs changed")
    manifest = read(manifest_path)
    selected = {row["id"]: row for row in manifest["rows"] if row["split"] == "validation"}
    if len(selected) != 1264 or len({row["image_id"] for row in selected.values()}) != 217:
        raise ValueError("Expected 1264 validation objects / 217 images")
    baseline = index_rows(read(baseline_path)["rows"], selected)
    best_rows = index_rows(best["rows"], selected)
    result_path = args.native / "result.json"
    native = read(result_path)
    replay_key = "native_sha256" if schema == 6 else "result_sha256"
    if sha256(result_path) != replay[replay_key] or native.get("complete") is not True:
        raise ValueError("Independently replayed native evidence changed")
    native_rows = index_rows(native["rows"], selected)
    if native.get("selected_epoch") != receipt["selected_epoch"] or native.get("benchmark_success") is not False:
        raise ValueError("Native selected epoch or scope changed")
    archive = None
    if schema == 6:
        if (native.get("contract") != contract or native.get("runtime") != receipt["runtime"]
                or native.get("checkpoint_sha256") != receipt["best_sha256"]
                or native.get("training_sha256") != sha256(args.training / "training.json")
                or native.get("exact_pixel_replay") is not True or native["rows"] != best["rows"]
                or sha256(args.native / "selected-masks.zip") != native["selected_masks_sha256"]):
            raise ValueError("Local native checkpoint binding changed")
        archive = zipfile.ZipFile(args.native / "selected-masks.zip")
        if sorted(archive.namelist()) != sorted(key + ".png" for key in selected):
            archive.close()
            raise ValueError("Local native mask archive inventory changed")
    else:
        native_contract = read(args.native / "contract.json")
        expected = dict(training_receipt_sha256=sha256(args.training / "training.json"),
            checkpoint_sha256=receipt["best_sha256"], selected_epoch=receipt["selected_epoch"],
            data_manifest_sha256=contract["data_sha256"], split="validation", objects=1264, images=217,
            baseline_result_sha256=sha256(baseline_path), geometry_policy=receipt["geometry_policy"],
            holdout_access=False, benchmark_access=False, cuda_fft_round_trip=True)
        if (any(native_contract.get(key) != value for key, value in expected.items())
                or native["contract_sha256"] != sha256(args.native / "contract.json")
                or native.get("max_selected_validation_difference") != 0.):
            raise ValueError("Kaggle native checkpoint binding changed")
    if args.output.exists():
        if archive:
            archive.close()
        raise FileExistsError("Preserve previous GPU validation evidence")
    args.output.mkdir(parents=True)
    sources = {name: sha256(ROOT / name) for name in (
        "amodal_benchmark/shape_external_prior.py", "amodal_benchmark/shape_v2.py",
        "amodal_benchmark/shape_v2_runtime.py", "scripts/validate_shape_external_adapter.py")}
    execution = dict(schema=1, at=datetime.now(timezone.utc).isoformat(), training_schema=schema,
        training_sha256=sha256(args.training / "training.json"), checkpoint_sha256=receipt["best_sha256"],
        native_result_sha256=sha256(result_path), independent_replay_sha256=sha256(args.replay),
        selected_epoch=receipt["selected_epoch"], sources=sources, benchmark_success=False,
        scope="Inference-GPU equivalence only; separate benchmark admission and full evaluation required")
    write_json(args.output / "contract.json", execution)
    records = []
    try:
        import torch
        model = load_for_validation(args.training, receipt)
        signal = torch.arange(64, dtype=torch.float32, device="cuda").reshape(8, 8)
        if not torch.allclose(torch.fft.ifft2(torch.fft.fft2(signal)).real, signal, atol=1e-4):
            raise RuntimeError("CUDA FFT round trip failed")
        image_id, rgb = None, None
        for key, source in sorted(selected.items(), key=lambda item: (item[1]["image_id"], item[0])):
            if source["image_id"] != image_id:
                image_id = source["image_id"]
                image_path = safe_asset(args.data, source["image"])
                if sha256(image_path) != manifest["artifacts"][source["image"]]:
                    raise ValueError("Validation RGB changed")
                with Image.open(image_path) as image:
                    rgb = np.array(image.convert("RGB"))
            support_path = safe_asset(args.sam_validation, "objects/" + key + "/sam-visible.png")
            if sha256(support_path) != baseline[key]["artifacts"]["sam-visible.png"]:
                raise ValueError("Original SAM support changed")
            support = binary_mask(support_path.read_bytes(), rgb.shape[:2])
            row = native_rows[key]
            if schema == 6:
                expected_mask = binary_mask(archive.read(key + ".png"), support.shape)
                expected_hash = best_rows[key]["prediction_sha256"]
            else:
                path = safe_asset(args.native, "objects/" + key + "/prediction.png")
                if sha256(path) != row["prediction_sha256"] or sha256(support_path) != row["sam_sha256"]:
                    raise ValueError("Selected native artifact changed")
                expected_mask = binary_mask(path.read_bytes(), support.shape)
                expected_hash = (best_rows[key]["prediction_pixels_sha256"] if schema == 7
                                 else pixel_hash(expected_mask))
            if (pixel_hash(expected_mask) != expected_hash or np.any(support & ~expected_mask)
                    or (not support.any() and expected_mask.any())):
                raise ValueError("Selected native pixels or support policy changed")
            probability, box, _, _ = original_prior(model, rgb, support, schema)
            prediction = probability >= .5
            different = int(np.count_nonzero(prediction != expected_mask))
            record = dict(id=key, image_id=image_id, different_pixels=different,
                          expected_pixel_sha256=expected_hash, actual_pixel_sha256=pixel_hash(prediction))
            records.append(record)
            if different or (list(box) if box is not None else None) != row["box"]:
                write_json(args.output / "mismatch.json", record)
                raise ValueError(f"Inference GPU changed {key}: {different} pixels")
            if len(records) % 25 == 0:
                write_json(args.output / "progress.json", dict(objects=len(records), complete=False))
        if any(sha256(ROOT / name) != digest for name, digest in sources.items()):
            raise ValueError("Adapter sources changed during GPU validation")
        result = dict(verified=True, complete=True, objects=1264, images=217, different_pixels=0,
            contract_sha256=sha256(args.output / "contract.json"), rows=records,
            gpu=torch.cuda.get_device_name(), torch=str(torch.__version__), cuda=torch.version.cuda,
            cuda_fft_round_trip=True, training_pixel_hashes_available=schema in (6, 7),
            benchmark_success=False, benchmark_admitted=False)
        write_json(args.output / "result.json", result)
        return result
    finally:
        if archive:
            archive.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("training", "native", "replay", "data", "sam-validation", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    result = validate(parser.parse_args())
    print("Exact inference-GPU replay:", result["objects"], "objects")


if __name__ == "__main__":
    main()

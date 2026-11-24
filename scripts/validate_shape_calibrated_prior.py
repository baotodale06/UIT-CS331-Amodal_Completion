"""Check the frozen corrected prior against diagnostic pixels on local or Quadro GPUs."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from amodal_benchmark.dataset import sha256, write_json
from amodal_benchmark.shape_calibrated_prior import candidate_policy, original_prior
from amodal_benchmark.shape_v2 import make_network
from scripts.validate_shape_external_adapter import binary_mask, index_rows, pixel_hash, safe_asset


def validate(args):
    import torch
    import PIL
    read = lambda path: json.loads(path.read_bytes())
    policy = candidate_policy(args.candidate)
    if (str(torch.__version__) != "1.13.1" or torch.version.cuda != "11.7"
            or np.__version__ != "1.23.5" or PIL.__version__ != "9.5.0"
            or not torch.cuda.is_available() or torch.cuda.device_count() != 1):
        raise RuntimeError("One GPU and the pinned numerical environment are required")
    gpu = torch.cuda.get_device_name()
    expected_gpu = "RTX 3060" if args.device_family == "local" else "QUADRO RTX 6000"
    if expected_gpu not in gpu.upper():
        raise RuntimeError("Unexpected GPU family")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    if args.device_family == "local":
        torch.cuda.set_per_process_memory_fraction(.2)
    signal = torch.arange(64, dtype=torch.float32, device="cuda").reshape(8, 8)
    if not torch.allclose(torch.fft.ifft2(torch.fft.fft2(signal)).real, signal, atol=1e-4):
        raise RuntimeError("Real FFT round trip failed")
    if (sha256(args.data / "manifest.json") != policy["data_sha256"]
            or sha256(args.sam_validation / "result.json") != policy["baseline_result_sha256"]):
        raise ValueError("Frozen validation inputs changed")
    manifest = read(args.data / "manifest.json")
    selected = {row["id"]: row for row in manifest["rows"] if row["split"] == "validation"}
    if len(selected) != 1264 or len({row["image_id"] for row in selected.values()}) != 217:
        raise ValueError("Complete validation inventory required")
    baseline = index_rows(read(args.sam_validation / "result.json")["rows"], selected)
    native = index_rows(read(args.candidate / "selected-native.json")["rows"], selected)
    training = read(args.candidate / "training.json")
    saved = torch.load(args.candidate / "best.pt", map_location="cpu", weights_only=False)
    epoch = policy["checkpoint_epoch"]
    expected_validation = next(row["validation"] for row in training["history"] if row["epoch"] == epoch)
    if (saved.get("epoch") != epoch or saved.get("contract") != training["contract"]
            or saved.get("validation") != expected_validation):
        raise ValueError("Corrected checkpoint payload changed")
    model = make_network(pretrained=False).cuda().eval()
    model.load_state_dict(saved["model"], strict=True)
    if args.output.exists():
        raise FileExistsError("Preserve prior runtime validation evidence")
    args.output.mkdir(parents=True)
    sources = {name: sha256(ROOT / name) for name in (
        "scripts/validate_shape_calibrated_prior.py", "scripts/validate_shape_external_adapter.py",
        "amodal_benchmark/shape_calibrated_prior.py", "amodal_benchmark/shape_v2.py")}
    contract = dict(at=datetime.now(timezone.utc).isoformat(), gpu=gpu,
        device_family=args.device_family, torch=str(torch.__version__), cuda=torch.version.cuda,
        cuda_fft_round_trip=True, candidate_sha256=sha256(args.candidate / "candidate.json"),
        sources=sources, preflight=bool(args.preflight_objects), holdout_access=False,
        benchmark_access=False, benchmark_admitted=False, benchmark_success=False)
    write_json(args.output / "contract.json", contract)
    records = []
    with zipfile.ZipFile(args.candidate / "selected-masks.zip") as masks:
        if sorted(masks.namelist()) != sorted(key + ".png" for key in selected):
            raise ValueError("Selected diagnostic mask inventory changed")
        for key, source in sorted(selected.items(), key=lambda item: (item[1]["image_id"], item[0])):
            image_path = safe_asset(args.data, source["image"])
            support_path = safe_asset(args.sam_validation, "objects/" + key + "/sam-visible.png")
            if (sha256(image_path) != manifest["artifacts"][source["image"]]
                    or sha256(support_path) != baseline[key]["artifacts"]["sam-visible.png"]):
                raise ValueError("Validation RGB or SAM support changed")
            with Image.open(image_path) as image:
                rgb = np.array(image.convert("RGB"))
            support = binary_mask(support_path.read_bytes(), rgb.shape[:2])
            expected = binary_mask(masks.read(key + ".png"), support.shape)
            if pixel_hash(expected) != native[key]["prediction_sha256"]:
                raise ValueError("Selected diagnostic pixels changed")
            probability, box, _, _ = original_prior(model, rgb, support, policy["logit_bias"])
            actual = probability >= .5
            difference = int(np.count_nonzero(actual != expected))
            record = dict(id=key, image_id=source["image_id"], different_pixels=difference,
                expected_pixel_sha256=pixel_hash(expected), actual_pixel_sha256=pixel_hash(actual),
                box=list(box) if box is not None else None)
            records.append(record)
            if difference:
                write_json(args.output / "mismatch.json", record)
                raise ValueError(f"Corrected prior differs for {key}: {difference} pixels")
            if len(records) % 25 == 0:
                print(json.dumps(dict(objects=len(records), total=1264)), flush=True)
            if args.preflight_objects and len(records) >= args.preflight_objects:
                break
    candidate_policy(args.candidate)
    if any(sha256(ROOT / name) != digest for name, digest in sources.items()):
        raise ValueError("Runtime source changed during validation")
    result = dict(verified=True, complete=not args.preflight_objects, objects=len(records),
        images=len({row["image_id"] for row in records}), different_pixels=0, rows=records,
        contract_sha256=sha256(args.output / "contract.json"),
        inference_gpu_validated=args.device_family == "quadro" and not args.preflight_objects,
        benchmark_admitted=False, benchmark_success=False)
    write_json(args.output / "result.json", result)
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("candidate", "data", "sam-validation", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--device-family", choices=("local", "quadro"), required=True)
    parser.add_argument("--preflight-objects", type=int, default=0)
    args = parser.parse_args()
    if not 0 <= args.preflight_objects <= 1264:
        parser.error("preflight object count must be 0..1264")
    validate(args)


if __name__ == "__main__":
    main()

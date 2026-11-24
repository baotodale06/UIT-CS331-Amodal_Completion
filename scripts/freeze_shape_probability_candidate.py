"""Freeze one validation-selected correction for testing, without benchmark admission."""
import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory
import zipfile

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from amodal_benchmark.dataset import sha256, write_json
from scripts.diagnose_shape_probability import summarize


def select_candidate(summaries, biases):
    baseline = summaries["initial"]
    keys = ["adapted_bias_" + format(bias, "g") for bias in biases]
    if set(summaries) != {"initial", *keys}:
        raise ValueError("Frozen diagnostic variant inventory changed")
    for values in summaries.values():
        for metric in ("full_iou", "hidden_iou"):
            value = values[metric]
            if type(value) not in (int, float) or not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("Finite bounded geometry required")
    eligible = [(key, bias) for key, bias in zip(keys, biases)
                if all(summaries[key][metric] >= baseline[metric] for metric in ("full_iou", "hidden_iou"))]
    if not eligible:
        raise ValueError("No preset correction retained both native geometry floors")
    return max(eligible, key=lambda pair: (summaries[pair[0]]["hidden_iou"], summaries[pair[0]]["full_iou"]))


def freeze(diagnostic, output):
    read = lambda path: json.loads(path.read_bytes())
    if output.exists():
        raise FileExistsError("Preserve existing candidate packages")
    result_path, contract_path = diagnostic / "gpu-full/result.json", diagnostic / "gpu-full/contract.json"
    result, contract, replay = read(result_path), read(contract_path), read(diagnostic / "replay.json")
    expected = dict(verified=True, objects=1264, images=217, masks=7584,
        result_sha256=sha256(result_path), source_sha256=sha256(ROOT / "scripts/verify_shape_probability.py"),
        benchmark_success=False, promoted=False)
    if (any(replay.get(key) != value for key, value in expected.items())
            or result.get("complete") is not True or result.get("preflight") is not False
            or result.get("promoted") is not False or result.get("benchmark_success") is not False
            or result["contract_sha256"] != sha256(contract_path)
            or sha256(diagnostic / "gpu-full/masks.zip") != result["mask_archive_sha256"]):
        raise ValueError("Completed independently verified fixed-offset diagnostic required")
    for name, digest in contract["sources"].items():
        path = (ROOT / name).resolve()
        if not path.is_relative_to(ROOT.resolve()) or sha256(path) != digest:
            raise ValueError("Frozen diagnostic source changed")
    key, bias = select_candidate(result["summaries"], contract["config"]["logit_biases"])
    robust = ROOT / "output/shape-sam-robustness-20261123"
    training_path = robust / "recovered/training/training.json"
    training = read(training_path)
    checkpoint = robust / "training/best.pt"
    if (training.get("complete") is not True or training.get("schema") != 3
            or training.get("holdout_used") is not False or training.get("benchmark_used") is not False
            or training["best_sha256"] != contract["config"]["adapted_checkpoint_sha256"]
            or sha256(checkpoint) != training["best_sha256"]):
        raise ValueError("Completed fixed robust checkpoint required")
    rows = [dict(id=row["id"], image_id=row["image_id"], **row["variants"][key]) for row in result["rows"]]
    if len(rows) != 1264 or len({row["id"] for row in rows}) != 1264 or summarize(rows) != result["summaries"][key]:
        raise ValueError("Selected native validation evidence changed")
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as temporary:
        staged = Path(temporary) / "candidate"
        staged.mkdir()
        with zipfile.ZipFile(diagnostic / "gpu-full/masks.zip") as source, zipfile.ZipFile(
                staged / "selected-masks.zip", "w", zipfile.ZIP_DEFLATED) as destination:
            for row in rows:
                payload = source.read(key + "/" + row["id"] + ".png")
                with Image.open(io.BytesIO(payload)) as image:
                    raw = np.array(image)
                if (raw.ndim != 2 or not np.isin(raw, [0, 255]).all()
                        or hashlib.sha256((raw > 0).astype(np.uint8).tobytes()).hexdigest() != row["prediction_sha256"]):
                    raise ValueError("Selected native mask bytes changed")
                destination.writestr(row["id"] + ".png", payload)
        shutil.copyfile(checkpoint, staged / "best.pt")
        shutil.copyfile(training_path, staged / "training.json")
        shutil.copyfile(diagnostic / "replay.json", staged / "diagnostic-replay.json")
        write_json(staged / "selected-native.json", dict(rows=rows, summary=result["summaries"][key]))
        policy = dict(schema=1, method="shape-calibrated-probability", at=datetime.now(timezone.utc).isoformat(),
            scope="Candidate for runtime validation only; no benchmark admission or benchmark success",
            selection="After observing all five fixed diagnostic offsets, retain full/hidden floors; maximize hidden IoU, full IoU tie-break, stable first",
            validation_reused=True, selected_variant=key, logit_bias=bias, checkpoint_epoch=7,
            checkpoint_sha256=training["best_sha256"], data_sha256=contract["config"]["data_manifest_sha256"],
            baseline_result_sha256=contract["baseline_sha256"], native_baseline=result["summaries"]["initial"],
            native_selected=result["summaries"][key], diagnostic_result_sha256=sha256(result_path),
            diagnostic_contract_sha256=sha256(contract_path), diagnostic_sources=contract["sources"],
            diagnostic_reference_differences=result["native_reference_differences"],
            arithmetic=dict(cudnn_deterministic=True, cudnn_benchmark=False, cudnn_allow_tf32=False,
                            matmul_allow_tf32=False, autocast=False),
            geometry="Original visible RGB/SAM support only; square box padding2; resize logits before adding bias and sigmoid; zero outside crop; SAM support=1",
            files={path.name: sha256(path) for path in staged.iterdir()},
            sources={name: sha256(ROOT / name) for name in ("scripts/freeze_shape_probability_candidate.py",
                "amodal_benchmark/shape_calibrated_prior.py", "amodal_benchmark/shape_v2.py")},
            holdout_access=False, benchmark_access=False, benchmark_admitted=False, benchmark_success=False)
        write_json(staged / "candidate.json", policy)
        staged.rename(output)
    return policy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostic", type=Path, default=ROOT / "output/shape-probability-20261123")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(freeze(args.diagnostic, args.output), indent=2))


if __name__ == "__main__":
    main()

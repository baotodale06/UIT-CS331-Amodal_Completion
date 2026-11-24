"""Package an eligible constrained prior only after independent native mask replay."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from amodal_benchmark.dataset import sha256, write_json
from amodal_benchmark.shape_constrained_runtime import (
    METHOD, EVIDENCE_FILES, admission_sources, verify_evidence, training_receipt)


def package(study, output):
    from scripts import collect_shape_constrained as collector
    study, output = Path(study).resolve(), Path(output).resolve()
    if study != collector.STUDY.resolve():
        raise ValueError("This admission rule belongs to the isolated constrained-v2 study")
    if output.exists():
        raise FileExistsError("Preserve existing admission packages; choose a new output directory")
    status = json.loads((study / "collection-status.json").read_bytes())
    if status.get("disposition") != "ready" or not status.get("jobs") or any(
            value != ["COMPLETED", "0:0"] for value in status["jobs"].values()):
        raise ValueError("Training, native GPU replay and completed recovery are required")
    # Recompute all native masks, input identities, selection and bootstrap
    # statistics instead of trusting a stale or manually created replay receipt.
    replay = collector.replay(study / "recovered")
    recorded = json.loads((study / "native-replay.json").read_bytes())
    if replay != recorded or replay.get("new_checkpoint_eligible") is not True:
        raise ValueError("Independent replay changed or epoch zero is not eligible for promotion")
    sources = {
        "best.pt": study / "training/best.pt",
        "training.json": study / "recovered/training/training.json",
        "best-validation-native.json": study / "recovered/training/best-validation-native.json",
        "native-contract.json": study / "recovered/native-validation/contract.json",
        "native-result.json": study / "recovered/native-validation/result.json",
        "native-replay.json": study / "native-replay.json",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as temporary:
        staged = Path(temporary) / "admitted"
        staged.mkdir()
        for name, source in sources.items():
            shutil.copyfile(source, staged / name)
        receipt = verify_evidence(staged / "best.pt")
        admission = dict(schema=1, method=METHOD, at=datetime.now(timezone.utc).isoformat(),
            checkpoint_sha256=receipt["best_sha256"], selected_epoch=receipt["selected_epoch"],
            files={name: sha256(staged / name) for name in EVIDENCE_FILES},
            sources=admission_sources(), benchmark_success=False,
            scope="Validation admission only; full benchmark and native benchmark audit remain required")
        write_json(staged / "admission.json", admission)
        training_receipt(staged / "best.pt")
        staged.rename(output)
    return admission


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=ROOT / "output/shape-sam-constrained-v2-20261123")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(package(args.study, args.output), indent=2))


if __name__ == "__main__":
    main()

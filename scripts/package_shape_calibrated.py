"""Package the fixed correction only after complete exact local and Quadro replay."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from amodal_benchmark.dataset import write_json
from amodal_benchmark.shape_calibrated_runtime import (
    METHOD, admission_sources, evidence_files, verify_evidence, admitted_policy)
from scripts.verify_shape_calibrated_admission import replay


def package(study, output, data=None, sam=None):
    study, output = Path(study).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError("Preserve existing admission packages")
    # Fail before copying the checkpoint unless both complete native results exist.
    for name in ("local-native", "quadro-native"):
        result = json.loads((study / name / "result.json").read_bytes())
        if result.get("complete") is not True or result.get("verified") is not True:
            raise ValueError("Both complete native GPU replays are required")
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as temporary:
        staged = Path(temporary) / "admitted"
        staged.mkdir()
        shutil.copytree(study / "candidate", staged / "candidate")
        for family in ("local", "quadro"):
            for name in ("contract", "result"):
                shutil.copyfile(study / (family + "-native") / (name + ".json"),
                                staged / (family + "-" + name + ".json"))
        write_json(staged / "native-replay.json", replay(staged, data=data, sam=sam))
        policy = verify_evidence(staged)
        admission = dict(schema=1, method=METHOD, at=datetime.now(timezone.utc).isoformat(),
            checkpoint_sha256=policy["checkpoint_sha256"], files=evidence_files(staged),
            sources=admission_sources(), benchmark_success=False,
            scope="Native replay admission only; full diffusion smoke and benchmark evaluation remain required")
        write_json(staged / "admission.json", admission)
        admitted_policy(staged)
        staged.rename(output)
    return admission


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=ROOT / "output/shape-calibrated-20261123")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--sam-validation", type=Path)
    args = parser.parse_args()
    print(json.dumps(package(args.study, args.output, args.data, args.sam_validation), indent=2))


if __name__ == "__main__":
    main()

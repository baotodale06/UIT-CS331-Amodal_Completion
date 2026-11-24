"""Independently verify the Colab handoff and write its local validation receipt."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from amodal_benchmark.dataset import sha256, write_json


def main():
    import nbformat
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT/"output/shape-v2-20261123")
    args = parser.parse_args()
    package = args.root/"colab"
    receipt = json.loads((package/"package.json").read_bytes())
    if sha256(ROOT/"scripts/package_shape_v2_colab.py") != receipt["builder_sha256"]:
        raise ValueError("Package builder changed after archive creation")
    for name, record in receipt["files"].items():
        if sha256(package/name) != record["sha256"]:
            raise ValueError(f"Package hash mismatch: {name}")
    with zipfile.ZipFile(package/"shape-v2-code.zip") as archive:
        if set(archive.namelist()) != set(receipt["source_hashes"]):
            raise ValueError("Unexpected code archive members")
        for name, expected in receipt["source_hashes"].items():
            if hashlib.sha256(archive.read(name)).hexdigest() != expected or sha256(ROOT/name) != expected:
                raise ValueError("Packaged source differs from the current tested source")
    with zipfile.ZipFile(package/"shape-v2-training-data.zip") as archive:
        manifest = json.loads(archive.read("data/manifest.json"))
        if manifest["holdout_packaged"] or any(r["split"] == "holdout" for r in manifest["rows"]):
            raise ValueError("Holdout included in training bundle")
        expected = {"data/manifest.json", *("data/"+name for name in manifest["artifacts"])}
        if len(archive.namelist()) != len(expected) or set(archive.namelist()) != expected:
            raise ValueError("Unexpected or duplicate training archive members")
        for name, digest in manifest["artifacts"].items():
            if hashlib.sha256(archive.read("data/"+name)).hexdigest() != digest:
                raise ValueError("Corrupt training archive member")
    notebook = nbformat.read(package/"shape-v2-colab.ipynb", as_version=4)
    nbformat.validate(notebook)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            if cell.outputs or cell.execution_count is not None:
                raise ValueError("Unrun notebook contains claimed execution output")
            compile(cell.source, cell.id, "exec")
    text = (args.root/"check-linux.log").read_text()
    if "OK (skipped=1)" not in text or "CPU smoke passed" not in text or "FAILED (" in text:
        raise ValueError("Repository check evidence does not pass")
    focused = (args.root/"focused-tests.log").read_text()
    if "Ran 9 tests" not in focused or "\nOK\n" not in focused or "skipped" in focused:
        raise ValueError("Focused CPU model checks do not pass")
    result = dict(validated_at=datetime.now(timezone.utc).isoformat(), notebook_schema=True,
                  notebook_cells_compiled=True, notebook_executed_on_colab=False,
                  archived_training_objects=len(manifest["rows"]), archived_data_files=len(manifest["artifacts"]),
                  member_hashes_verified=True, holdout_excluded=True, current_sources_match=True,
                  tests=dict(repository_python=333, repository_optional_skips=1, focused_cpu=9,
                             bash_regressions=True, cpu_smoke=True, gpu_training=False, gpu_pipeline=False),
                  training_started=False, target_achieved=False,
                  input_sha256={p.name:sha256(p) for p in (package/"package.json", args.root/"check-linux.log", args.root/"focused-tests.log")})
    write_json(args.root/"validation.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

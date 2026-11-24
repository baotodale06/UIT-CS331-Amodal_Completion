"""Build verified Colab archives; reserve holdout pixels outside the training ZIP."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from amodal_benchmark.dataset import sha256, write_json

SOURCES = ("amodal_benchmark/__init__.py", "amodal_benchmark/dataset.py", "amodal_benchmark/shape_v2.py",
           "scripts/train_shape_v2.py", "configs/research/shape-v2-20261123.json")


def cell(kind, source):
    row = dict(cell_type=kind, metadata={}, source=source.splitlines(keepends=True))
    if kind == "code":
        row.update(execution_count=None, outputs=[])
    return row


def build_notebook(code_hash, data_hash):
    cells = [cell("markdown", """# Shape-only amodal training · 23 November 2026

Training has not been run and the 3% target has not been achieved by this package.
Use **Runtime → Change runtime type → H100** (the highest documented option).
If your account exposes a stronger accelerator, record it and explicitly update EXPECTED_GPU.
Do not accept a silently downgraded runtime. This notebook contains no Surface model.

Upload `shape-v2-code.zip` and `shape-v2-training-data.zip` when prompted.
Training uses true COCOA amodal outlines and fixed image-level validation;
holdout images and labels are excluded from the training archive.
Checkpoints persist in your Google Drive. Reconnect, run setup again, and set RESUME=True after interruption.
"""), cell("code", """import os, sys, subprocess
from pathlib import Path
from google.colab import files, drive
import torch
EXPECTED_GPU = 'H100'
assert torch.cuda.is_available(), 'Select the highest GPU runtime before continuing.'
GPU_NAME = torch.cuda.get_device_name(0)
assert EXPECTED_GPU.upper() in GPU_NAME.upper(), f'Expected {EXPECTED_GPU}; got {GPU_NAME}'
assert os.environ.get('COLAB_RELEASE_TAG') or Path('/var/colab').exists(), 'Colab runtime marker absent'
print(subprocess.check_output(['nvidia-smi'], text=True))
"""), cell("code", f"""import hashlib, zipfile
EXPECTED = {{'shape-v2-code.zip': '{code_hash}', 'shape-v2-training-data.zip': '{data_hash}'}}
missing = [name for name in EXPECTED if not Path('/content', name).exists()]
if missing:
    print('Upload both supplied ZIP files:', missing)
    uploaded = files.upload()
    del uploaded
WORK = Path('/content/shape-v2-20261123')
WORK.mkdir(exist_ok=True)
for name, expected in EXPECTED.items():
    archive = Path('/content', name)
    with archive.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    assert digest == expected, f'Archive changed: {{name}}'
    with zipfile.ZipFile(archive) as source:
        for member in source.infolist():
            assert (WORK/member.filename).resolve().is_relative_to(WORK.resolve()), 'Unsafe archive path'
            assert (member.external_attr >> 16) & 0o170000 != 0o120000, 'Symlink member rejected'
        source.extractall(WORK)
print('Verified source and training data archives.')
"""), cell("code", """# Keep the pinned training environment separate from Colab's installed packages.
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'uv==0.9.18'])
PYTHON = Path('/content/shape-v2-env/bin/python')
if not PYTHON.exists():
    subprocess.check_call(['uv', 'venv', '--python', '3.11', '/content/shape-v2-env'])
subprocess.check_call(['uv', 'pip', 'install', '--python', str(PYTHON),
    'torch==2.5.1', 'torchvision==0.20.1', '--index-url', 'https://download.pytorch.org/whl/cu121'])
subprocess.check_call(['uv', 'pip', 'install', '--python', str(PYTHON), 'numpy==1.26.4', 'Pillow==10.4.0'])
drive.mount('/content/drive')
OUTPUT = Path('/content/drive/MyDrive/shape-v2-20261123/training')
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
"""), cell("code", """# Only resume an existing run with unchanged source, configuration and data.
RESUME = False
command = [str(PYTHON), str(WORK/'scripts/train_shape_v2.py'),
           '--data', str(WORK/'data'), '--output', str(OUTPUT), '--expected-gpu', EXPECTED_GPU]
if RESUME:
    command.append('--resume')
subprocess.check_call(command, cwd=WORK)
"""), cell("code", """import json, shutil
receipt = json.loads((OUTPUT/'training.json').read_text())
assert receipt['complete'] is True, 'Training incomplete; resume from last.pt.'
assert receipt['holdout_used'] is False
with (OUTPUT/'best.pt').open('rb') as stream:
    assert hashlib.file_digest(stream, 'sha256').hexdigest() == receipt['best_sha256']
print('Finished epochs:', receipt['epochs_completed'])
print('Validation-only checkpoint:', receipt['best_sha256'])
print('Benchmark results: not evaluated by this notebook.')
archive = shutil.make_archive('/content/shape-v2-trained', 'zip', OUTPUT)
files.download(archive)
"""), cell("markdown", """## Required next step

Recover `best.pt`, `last.pt`, and `training.json`. Independently verify their hashes.
Freeze the shape-only selector using development data; then evaluate the reserved natural-amodal holdout once.
Then integrate the new checkpoint architecture into an isolated PMC runtime, verify GPU imports/FFT,
and evaluate all 2,565 targets with the original frozen scoring references and failure penalties.
The old `--shape-checkpoint` loader expects the v1 architecture and cannot load this checkpoint unchanged.
Run `scripts/assess_shape_v2.py` on verified score tables. All four metrics must improve at least 3%;
report bootstrap intervals, coverage, amodal IoU and the historical unseen subset as well.
Do not tune on the benchmark, call a validation gain a full-benchmark gain, or combine with Surface.
""")]
    return dict(nbformat=4, nbformat_minor=5, cells=cells,
                metadata=dict(kernelspec=dict(display_name="Python 3", language="python", name="python3"),
                              accelerator="GPU", colab=dict(name="shape-v2-colab.ipynb", gpuType="H100")))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "output/shape-v2-20261123/data")
    parser.add_argument("--output", type=Path, default=ROOT / "output/shape-v2-20261123/colab")
    args = parser.parse_args()
    manifest_path = args.data / "manifest.json"
    original = json.loads(manifest_path.read_bytes())
    if original["status"] != "complete":
        raise ValueError("Complete duplicate audit required before packaging")
    args.output.mkdir(parents=True, exist_ok=True)
    rows = [r for r in original["rows"] if r["split"] in ("train", "validation")]
    names = {r[key] for r in rows for key in ("image", "mask")}
    holdout_names = {r[key] for r in original["rows"] if r["split"] == "holdout" for key in ("image", "mask")}
    if names & holdout_names:
        raise ValueError("Holdout files overlap the training archive")
    packed = dict(original, rows=rows, objects=len(rows), counts=dict(Counter(r["split"] for r in rows)),
                  split_images={s:len({r["image_id"] for r in rows if r["split"] == s}) for s in ("train", "validation")},
                  split_images_before_content_screen=original["split_images"],
                  artifacts={name:original["artifacts"][name] for name in sorted(names)},
                  source_manifest_sha256=sha256(manifest_path), holdout_packaged=False,
                  held_out_objects=sum(r["split"] == "holdout" for r in original["rows"]))
    code_zip, data_zip = args.output / "shape-v2-code.zip", args.output / "shape-v2-training-data.zip"
    with zipfile.ZipFile(code_zip, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in SOURCES:
            archive.write(ROOT / name, name)
    with zipfile.ZipFile(data_zip, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("data/manifest.json", json.dumps(packed, indent=2))
        for name, expected in packed["artifacts"].items():
            path = args.data / name
            if sha256(path) != expected:
                raise ValueError(f"Data changed before packaging: {name}")
            archive.write(path, "data/" + name)
    notebook = build_notebook(sha256(code_zip), sha256(data_zip))
    for index, row in enumerate(notebook["cells"]):
        row["id"] = f"shape-v2-{index}"
        if row["cell_type"] == "code":
            compile("".join(row["source"]), f"notebook-cell-{index}", "exec")
    notebook_path = args.output / "shape-v2-colab.ipynb"
    write_json(notebook_path, notebook)
    receipt = dict(ready_for_training=True, training_started=False, benchmark_success=False,
                   manifest_sha256=sha256(manifest_path), counts=packed["counts"], holdout_objects=packed["held_out_objects"],
                   split_images=packed["split_images"], builder_sha256=sha256(Path(__file__)),
                   holdout_packaged=False, source_hashes={name:sha256(ROOT / name) for name in SOURCES},
                   files={p.name:dict(sha256=sha256(p), bytes=p.stat().st_size) for p in (code_zip, data_zip, notebook_path)})
    write_json(args.output / "package.json", receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()

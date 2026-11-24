"""Recover native SAM validation masks and independently replay their geometry."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile
from tempfile import TemporaryDirectory
import zipfile

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from amodal_benchmark.dataset import sha256, write_json
from scripts.collect_surface_job_evidence import recover_archive
from scripts.verify_shape_v2_metadata import equal

STUDY = ROOT / "output/shape-validation-sam-20261123"
HOST = "daoqm@slurm.csse.rose-hulman.edu"
REMOTE = "/home/daoqm/Vu_Hung/cv_nangcao/output/shape-validation-sam-20261123"
PROGRAM = r'''
import hashlib, io, json, os, subprocess, tarfile
from pathlib import Path
from tempfile import TemporaryDirectory
study = Path('/home/daoqm/Vu_Hung/cv_nangcao/output/shape-validation-sam-20261123')
job = json.loads((study/'job.json').read_bytes())['job_id']
os.environ['TZ'] = 'UTC'
raw = subprocess.check_output(['sacct','-X','-n','-P','-j',job,
    '--format=JobIDRaw,State%32,ExitCode,ElapsedRaw,AllocTRES,Start,End'], text=True)
rows = [line.split('|') for line in raw.splitlines() if line.strip()]
assert len(rows) == 1 and rows[0][:3] == [job,'COMPLETED','0:0'], raw
result = json.loads((study/'result/result.json').read_bytes())
assert result['complete'] and result['objects'] == 1264 and result['images'] == 217
archive, receipt = study/'native.tar.gz', study/'native-export.json'
if receipt.exists():
    data = json.loads(receipt.read_bytes())
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == data['sha256']
    assert data['accounting'] == rows[0]
else:
    assert not archive.exists(), 'Preserve archive without receipt for review'
    selected = [study/name for name in ('job.json','source-manifest.json','tests.log',f'validation-{job}.log')]
    selected += sorted(path for path in (study/'result').rglob('*') if path.is_file())
    assert all(not path.is_symlink() for path in selected)
    files = {p.relative_to(study).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in selected}
    with TemporaryDirectory(dir=study) as work:
        staged = Path(work)/'native.tar.gz'
        with tarfile.open(staged,'w:gz',compresslevel=1) as bundle:
            for name in files:
                bundle.add(study/name,arcname=name,recursive=False)
            manifest = json.dumps(files,sort_keys=True).encode()
            info = tarfile.TarInfo('member-sha256.json')
            info.size = len(manifest)
            bundle.addfile(info,io.BytesIO(manifest))
        data = dict(sha256=hashlib.sha256(staged.read_bytes()).hexdigest(),bytes=staged.stat().st_size,
            files=len(files),accounting=rows[0],result_sha256=files['result/result.json'])
        os.link(staged,archive)
    receipt.write_text(json.dumps(data,indent=2)+'\n')
print(json.dumps(data))
'''


def mask(path, shape):
    with Image.open(path) as source:
        values = np.array(source)
    if values.shape != shape or not np.isin(values, [0, 255]).all():
        raise ValueError("Invalid binary mask or native frame: " + str(path))
    return values > 0


def geometry(prediction, truth, visible):
    hidden_truth, hidden_prediction = truth & ~visible, prediction & ~visible
    scores = {}
    for name, first, second in (("full_iou", prediction, truth),
                                ("hidden_iou", hidden_prediction, hidden_truth)):
        scores[name] = (float(np.count_nonzero(first & second) / max(1, np.count_nonzero(first | second)))
                        if name == "full_iou" or hidden_truth.any() else None)
    return scores


def paired_summary(rows):
    summaries = {}
    for metric in ("full_iou", "hidden_iou"):
        groups = defaultdict(list)
        for row in rows:
            a, b = row["annotated_input"][metric], row["sam_input"][metric]
            if (a is None) != (b is None):
                raise ValueError("Mismatched geometry eligibility")
            if a is not None:
                groups[row["image_id"]].append([a, b])
        means = np.asarray([np.mean(groups[key], axis=0) for key in sorted(groups)])
        delta = means[:, 1] - means[:, 0]
        rng = np.random.default_rng(20261123)
        draws = np.asarray([delta[rng.integers(len(delta), size=len(delta))].mean() for _ in range(5000)])
        summaries[metric] = dict(images=len(means), objects=sum(len(g) for g in groups.values()),
            annotated_input=float(means[:, 0].mean()), sam_input=float(means[:, 1].mean()),
            sam_minus_annotated=float(delta.mean()), absolute_ci=np.quantile(draws, [.025, .975]).tolist())
    return summaries


def check_ids(rows, expected):
    if len(rows) != len(expected) or {row["id"] for row in rows} != set(expected):
        raise ValueError("Validation object inventory changed or duplicated")


def replay(native, data, packed_manifest, original_path, freeze_path, source_manifest):
    contract_path, result_path = native/"result/contract.json", native/"result/result.json"
    contract, result = (json.loads(p.read_bytes()) for p in (contract_path, result_path))
    original, freeze = (json.loads(p.read_bytes()) for p in (original_path, freeze_path))
    manifest = json.loads(packed_manifest)
    if (hashlib.sha256(packed_manifest).hexdigest() != contract["data_manifest_sha256"]
            or sha256(original_path) != contract["original_validation_sha256"]
            or sha256(original_path) != freeze["validation_sha256"]
            or contract["shape_policy"] != freeze["shape_policy"]
            or contract["shape_policy"] != original["shape_policy"]
            or result["contract_sha256"] != sha256(contract_path)
            or contract["source_sha256"] != source_manifest["files"]["scripts/audit_shape_sam_validation.py"]):
        raise ValueError("Diagnostic is not bound to the frozen validation source/data/model")
    if (contract["selected_split"] != "validation" or original["split"] != "validation"
            or any(contract[key] is not False for key in ("holdout_access", "benchmark_access", "checkpoint_modified"))
            or result["complete"] is not True or result["benchmark_success"] is not False
            or {row["split"] for row in manifest["rows"]} != {"train", "validation"}):
        raise ValueError("Diagnostic scope changed")
    selected = {row["id"]:row for row in manifest["rows"] if row["split"] == "validation"}
    if len(selected) != 1264 or len({row["image_id"] for row in selected.values()}) != 217:
        raise ValueError("Full validation split required")
    check_ids(result["rows"], selected)
    check_ids(original["rows"], selected)
    original_rows = {row["id"]:row for row in original["rows"]}
    recovered_rows, verified_assets, visible_scores = [], set(), defaultdict(list)
    for row in result["rows"]:
        source = selected[row["id"]]
        for field in ("image_id", "image", "mask"):
            equal(row[field], source[field], "validation identity/"+field)
        for name in (source["mask"], source["image"]):
            path = (data/name).resolve()
            if not path.is_relative_to(data.resolve()):
                raise ValueError("Validation asset escapes data root")
            if name not in verified_assets:
                if sha256(path) != manifest["artifacts"][name]:
                    raise ValueError("Validation source asset hash differs")
                verified_assets.add(name)
        directory = native/"result/objects"/row["id"]
        equal(json.loads((directory/"result.json").read_bytes()), row, "per-object record")
        if set(row["artifacts"]) != {"sam-visible.png", "annotated-prediction.png", "sam-prediction.png"}:
            raise ValueError("Unexpected native mask inventory")
        for name, expected in row["artifacts"].items():
            if sha256(directory/name) != expected:
                raise ValueError("Native mask hash differs")
        with Image.open(data/source["mask"]) as image:
            encoded = np.array(image)
        truth, visible = (encoded & 1) > 0, (encoded & 2) > 0
        sam = mask(directory/"sam-visible.png", truth.shape)
        annotated_prediction = mask(directory/"annotated-prediction.png", truth.shape)
        sam_prediction = mask(directory/"sam-prediction.png", truth.shape)
        yy, xx = np.where(visible)
        equal(row["sam_box_prompt"], [int(xx.min()), int(yy.min()), int(xx.max()+1), int(yy.max()+1)], "visible-only SAM prompt")
        if (visible & ~annotated_prediction).any() or (sam & ~sam_prediction).any():
            raise ValueError("Runtime prediction did not retain its own input support")
        if row["empty_sam_mask"] != (not bool(sam.any())) or (not sam.any() and sam_prediction.any()):
            raise ValueError("Empty SAM policy differs")
        score = float((sam & visible).sum()/max(1, (sam | visible).sum()))
        equal(score, row["visible_mask_iou"], "SAM/annotation agreement")
        visible_scores[row["image_id"]].append(score)
        recovered = dict(id=row["id"], image_id=row["image_id"],
            annotated_input=geometry(annotated_prediction, truth, visible),
            sam_input=geometry(sam_prediction, truth, visible))
        for condition in ("annotated_input", "sam_input"):
            equal(recovered[condition], row[condition], "native geometry/"+condition)
        for metric, value in recovered["annotated_input"].items():
            equal(value, original_rows[row["id"]][metric], "original validation geometry")
        recovered_rows.append(recovered)
    aggregate = paired_summary(recovered_rows)
    equal(aggregate, result["geometry"], "paired image geometry/bootstrap")
    equal(result["max_original_validation_difference"], 0., "original validation agreement")
    equal(result["empty_sam_masks"], sum(row["empty_sam_mask"] for row in result["rows"]), "empty masks")
    equal([result["objects"], result["images"]], [1264, 217], "result dimensions")
    return dict(verified=True, objects=1264, images=217, geometry=aggregate,
        input_mask_iou_image_mean=float(np.mean([np.mean(values) for values in visible_scores.values()])),
        empty_sam_masks=result["empty_sam_masks"], original_validation_max_difference=0.,
        result_sha256=sha256(result_path), freeze_sha256=sha256(freeze_path),
        verification_source_sha256=sha256(Path(__file__)), benchmark_success=False,
        limitation="Validation box prompts use annotated visible support; this is not GroundingDINO matching or a benchmark gain. Model predictions are replayed, not rerun locally.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-only", action="store_true")
    args = parser.parse_args()
    receipt_path, archive = STUDY/"native-export.json", STUDY/"native.tar.gz"
    if not args.local_only:
        response = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", HOST, "python3 -"],
            input=PROGRAM, text=True, capture_output=True, check=True, timeout=120)
        receipt = json.loads(response.stdout)
        if receipt_path.exists():
            equal(json.loads(receipt_path.read_bytes()), receipt, "existing export receipt")
        write_json(receipt_path, receipt)
        if not archive.exists():
            with TemporaryDirectory(dir=STUDY) as work:
                staged = Path(work)/archive.name
                subprocess.run(["scp", "-q", "-o", "BatchMode=yes", HOST+":"+REMOTE+"/"+archive.name, str(staged)],
                    check=True, timeout=300)
                if sha256(staged) != receipt["sha256"]:
                    raise ValueError("Downloaded archive checksum differs")
                staged.rename(archive)
    receipt = json.loads(receipt_path.read_bytes())
    if sha256(archive) != receipt["sha256"] or archive.stat().st_size != receipt["bytes"]:
        raise ValueError("Native archive differs from export receipt")
    with tarfile.open(archive, "r:gz") as bundle:
        payload = bundle.extractfile("member-sha256.json").read()
    members = json.loads(payload)
    if len(members) != receipt["files"] or members["result/result.json"] != receipt["result_sha256"]:
        raise ValueError("Native archive membership differs from receipt")
    members["member-sha256.json"] = hashlib.sha256(payload).hexdigest()
    job = json.loads((STUDY/"job.json").read_bytes())
    native = STUDY/"native"
    recover_archive(archive, native, ["result", "job.json", "source-manifest.json", "tests.log",
        "validation-"+job["job_id"]+".log", "member-sha256.json"], members)
    equal(json.loads((native/"job.json").read_bytes()), job, "submitted job")
    source = json.loads((STUDY/"source-manifest.json").read_bytes())
    equal(source, json.loads((native/"source-manifest.json").read_bytes()), "submitted source")
    if job["source_manifest_sha256"] != sha256(STUDY/"source-manifest.json"):
        raise ValueError("Submission source manifest differs")
    v2 = ROOT/"output/shape-v2-slurm-20261123"
    snapshot = v2/json.loads((v2/"latest-recovery.json").read_bytes())["snapshot"]
    with zipfile.ZipFile(ROOT/"output/shape-v2-20261123/colab/shape-v2-training-data.zip") as bundle:
        packed = bundle.read("data/manifest.json")
    result = replay(native, ROOT/"output/shape-v2-20261123/data", packed,
        snapshot/"validation-native.json", snapshot/"policy-freeze.json", source)
    result.update(export=receipt, allocated_gpu_hours=int(receipt["accounting"][3])/3600)
    write_json(STUDY/"native-replay.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

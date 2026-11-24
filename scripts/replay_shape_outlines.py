"""Check natural COCOA scores locally using the saved completion alpha in original coordinates."""
import argparse
import json
from pathlib import Path
from statistics import fmean
import subprocess
import sys
from tempfile import TemporaryDirectory

from PIL import Image, ImageChops

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from amodal_benchmark.cohort import select_manifest_rows
from amodal_benchmark.dataset import read_official, sha256, write_json
from scripts.collect_surface_job_evidence import HOST, PROJECT, recover_archive, ssh

STUDY = ROOT / "output/shape-20261121"


def ensure_files(run, expected, fetch):
    missing = {}
    for name, digest in expected.items():
        path = run / name
        if not path.is_file():
            missing[name] = digest
        elif sha256(path) != digest:
            raise ValueError(f"Local file differs from the closed execution: {path}")
    if not missing:
        return
    if not fetch:
        raise FileNotFoundError(f"{len(missing)} native files are missing under {run}")
    directory = run.relative_to(ROOT).as_posix()
    names = b"".join(name.encode() + b"\0" for name in sorted(missing))
    with TemporaryDirectory(dir=run) as work:
        archive = Path(work) / "native.tar.gz"
        with archive.open("wb") as output:
            ssh(HOST, f"cd ~/{PROJECT}/{directory} && tar --null --verbatim-files-from "
                "--no-recursion --hard-dereference -czf - -T -",
                input=names, stdout=output, stderr=subprocess.PIPE, timeout=1800)
        recover_archive(archive, run, ["targets/COCOA"], missing)


def original_frame_overlap(completion, encoded, geometry):
    if encoded.mode != "L" or list(encoded.size) != geometry["original_size"]:
        raise ValueError("Annotation differs from the original image frame")
    if completion.mode != "RGBA":
        raise ValueError("Native completion has no RGBA silhouette")
    outline = encoded.point(lambda value: 255 * (value & 1))
    prediction = Image.new("L", outline.size)
    row, column = geometry["offset_rc"]
    prediction.paste(completion.getchannel("A").point(lambda value: 255 if value else 0), (column, row))
    intersection = ImageChops.darker(outline, prediction).histogram()[255]
    union = ImageChops.lighter(outline, prediction).histogram()[255]
    if not outline.getbbox():
        raise ValueError("Empty annotated amodal outline")
    return intersection, union


def replay(arm, cohort, fetch):
    stage = "cohort-vs-pmc" if cohort else "full"
    run = STUDY / "progressive-l40s" / (f"{arm}-cohort" if cohort else arm)
    execution_path = run / "execution.json"
    execution = json.loads(execution_path.read_bytes())
    pairs_path = STUDY / f"{arm}-{stage}-pairs.json"
    summary = json.loads((STUDY / f"{arm}-{stage}-summary.json").read_bytes())
    if execution["status"] != "complete" or summary["pairs_sha256"] != sha256(pairs_path):
        raise ValueError("Replay requires a closed execution and its audited paired table")
    if not cohort and sha256(execution_path) != summary["cohort_reuse"]["full_execution_sha256"]:
        raise ValueError("The paired audit refers to a different full execution")
    official, _ = read_official(ROOT / "datasets/open_world_amodal")
    if cohort:
        official = select_manifest_rows(official, ROOT / "output/surface-annotated-20261112-design/cohort.json")
    ids = [row["id"] for row in official]
    pairs = json.loads(pairs_path.read_bytes())
    if (ids != execution["target_ids"] or set(ids) != execution["result_artifacts"].keys()
            or ids != [row["id"] for row in pairs]):
        raise ValueError("Paired table and native execution must cover the official ordered targets")
    geometry = STUDY / "evaluation_only/cocoa"
    manifest = geometry / "manifest.json"
    control = json.loads((STUDY / "cocoa-historical-controls.json").read_bytes())
    if sha256(manifest) != control["manifest_sha256"]:
        raise ValueError("Different COCOA annotations used for the historical and L40S runs")
    annotations = json.loads(manifest.read_bytes())["rows"]
    outlines = {row["id"]: row for row in annotations}
    if len(annotations) != 321 or len(outlines) != 321:
        raise ValueError("Expected 321 unique COCOA annotations")
    joined = {row["id"]: row for row in pairs if "amodal_iou" in row[arm]}
    if joined.keys() != outlines.keys() & set(ids) or len(joined) != (22 if cohort else 321):
        raise ValueError("The audited polygon subset differs from the annotated targets")
    ensure_files(run, {f"targets/{tid}/result.json": execution["result_artifacts"][tid] for tid in joined}, fetch)
    records = {}
    completions = {}
    for tid, pair in joined.items():
        record = json.loads((run / "targets" / tid / "result.json").read_bytes())
        if (record["id"], record["status"], record["input_sha256"], record["seed"]) != (
            tid, pair["status"][arm], pair["input_sha256"], pair["seed"]
        ):
            raise ValueError(f"Audited row and native result disagree for {tid}")
        records[tid] = record
        if record["status"] == "success":
            completions[f"targets/{tid}/completion.png"] = record["completion_sha256"]
    ensure_files(run, completions, fetch)
    rows = []
    for tid, record in records.items():
        outline = outlines[tid]
        path = geometry / outline["case"]
        if sha256(path) != outline["sha256"]:
            raise ValueError(f"Annotation changed for {tid}")
        intersection = union = None
        iou = 0.0
        if record["status"] == "success":
            with Image.open(run / "targets" / tid / "completion.png") as native, Image.open(path) as encoded:
                intersection, union = original_frame_overlap(native, encoded, record["geometry"])
            iou = intersection / union
        if iou != joined[tid][arm]["amodal_iou"]:
            raise ValueError(f"Native completion alpha and audited IoU disagree for {tid}")
        rows.append(dict(id=tid, status=record["status"], intersection=intersection, union=union, amodal_iou=iou))
    return dict(method=arm, targets=len(rows), saved=len(completions), mean_amodal_iou=fmean(r["amodal_iou"] for r in rows),
                execution_sha256=sha256(execution_path), pairs_sha256=sha256(pairs_path),
                manifest_sha256=sha256(manifest), rows=rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arm", choices=("shape", "combined"))
    parser.add_argument("--cohort", action="store_true", help="Replay the closed 204-target snapshot")
    parser.add_argument("--fetch", action="store_true", help="Recover missing records and completions from the cluster")
    parser.add_argument("--receipt", type=Path, help="Save per-target overlap counts and input digests")
    args = parser.parse_args()
    result = replay(args.arm, args.cohort, args.fetch)
    if args.receipt:
        write_json(args.receipt, result)
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}))

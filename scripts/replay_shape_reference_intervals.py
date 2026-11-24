"""Replay full LPIPS comparisons using reference hashes and native score CSVs."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from amodal_benchmark.dataset import read_official, sha256, write_json
from scripts.audit_surface_lastpass_result import paired_estimate
from scripts.compare_shape_progressive import execution_scores

STUDY = ROOT / "output/shape-20261121"
HISTORICAL = ROOT / "output/surface-annotated-20261112-freshrefs"
CONTROLS = ("baseline", "surface")
METHODS = ("shape", "combined")


def read(path):
    return json.loads(path.read_bytes())


def checked_references(run, execution, tid):
    raw = (run / "targets" / tid / "result.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != execution["result_artifacts"][tid]:
        raise ValueError(f"Native record differs from {run.name} execution for {tid}")
    result = json.loads(raw)
    refs = result["evaluation_reference_artifacts"]
    if result["id"] != tid or set(refs) != {"crop", "mask", "full"}:
        raise ValueError(f"Missing reference hashes for {run.name} / {tid}")
    return refs


def reference_groups(official, runs, early):
    groups = {"same-reference": [], "same-scoring-crop": []}
    checked = set()
    for target in official:
        tid = target["id"]
        refs = {arm: checked_references(run, execution, tid)
                for arm, (run, execution, _) in runs.items()}
        if refs["baseline"] != refs["surface"] or refs["shape"] != refs["combined"]:
            raise ValueError(f"Within-device references differ for {tid}")
        same = refs["baseline"] == refs["shape"]
        crop = refs["baseline"]["crop"] == refs["shape"]["crop"]
        if tid in early:
            if (same, crop) != (early[tid]["all_files"], early[tid]["scoring_crop"]):
                raise ValueError(f"Reference membership differs from native PNG audit for {tid}")
            checked.add(tid)
        row = dict(id=tid, source=target["source"],
                   image_identity=target["source"] + "/" + target["original_id"],
                   scores={arm: scores[tid] for arm, (_, _, scores) in runs.items()})
        if same:
            groups["same-reference"].append(row)
        if crop:
            groups["same-scoring-crop"].append(row)
    if checked != early.keys():
        raise ValueError("The paired run is missing targets from the native PNG audit")
    return groups


def check_estimate(actual, expected):
    if actual.keys() != expected.keys():
        raise ValueError("The bootstrap estimates have different fields")
    for key, value in actual.items():
        np.testing.assert_allclose(value, expected[key], rtol=0., atol=1e-10, err_msg=key)


def main():
    directory = STUDY / "progressive-l40s"
    official, _ = read_official(ROOT / "datasets/open_world_amodal")
    ids = [row["id"] for row in official]
    early_path = directory / "shape-reference-overlap-1136.json"
    if sha256(early_path) != "404e693ff3d421fd44693e6cd0a0a8541aff589ef56b2d26f5e870ea64271c0b":
        raise ValueError("The native PNG audit differs from job 45311")
    early = {r["id"]: r for r in read(early_path)["records"]}
    runs = {}
    for arm in (*CONTROLS, *METHODS):
        run = (HISTORICAL if arm in CONTROLS else directory) / arm
        execution = read(run / "execution.json")
        scores = execution_scores(run, execution)
        if (execution["status"] != "complete" or len(ids) != 2565
                or execution["target_ids"] != ids or list(scores) != ids):
            raise ValueError(f"Expected all scored targets in official order for {arm}")
        runs[arm] = run, execution, scores
    groups = reference_groups(official, runs, early)
    estimates = {}
    for arm in METHODS:
        pairs_path = STUDY / f"{arm}-full-pairs.json"
        summary = read(STUDY / f"{arm}-full-summary.json")
        if sha256(pairs_path) != summary["pairs_sha256"]:
            raise ValueError(f"The comparison summary uses different target records for {arm}")
        pairs = read(pairs_path)
        if [p["id"] for p in pairs] != ids:
            raise ValueError(f"Full comparison target order differs for {arm}")
        memberships = {
            "same-reference": [p["id"] for p in pairs if p["reference_equal"]],
            "same-scoring-crop": [p["id"] for p in pairs
                                  if p["historical_scoring_crop_sha256"] == p["method_reference"]["crop"]],
        }
        for group, rows in groups.items():
            if memberships[group] != [row["id"] for row in rows]:
                raise ValueError(f"Full {group} membership differs for {arm}")
            for control in CONTROLS:
                actual = paired_estimate(rows, control, arm, "LPIPS")
                check_estimate(actual, summary["metrics"][f"{control}-{group}"]["LPIPS"])
                estimates[f"{arm}-vs-{control}-{group}"] = actual
    receipt = dict(early_native_targets=len(early), groups={key: len(rows) for key, rows in groups.items()},
                   estimates=estimates, execution_sha256={arm: sha256(run / "execution.json")
                                                         for arm, (run, _, _) in runs.items()},
                   pairs_sha256={arm: sha256(STUDY / f"{arm}-full-pairs.json") for arm in METHODS})
    write_json(STUDY / "full-reference-interval-replay.json", receipt)
    print("Verified both reference subsets and all eight paired LPIPS comparisons.")


if __name__ == "__main__":
    main()

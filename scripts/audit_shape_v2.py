"""Audit full shape-only outputs, frozen references, selection and metric gains."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from amodal_benchmark.dataset import read_official, sha256, write_json
from amodal_benchmark.shape_v2 import score_masks
from amodal_benchmark.shape_v2_runtime import choose, progressive_seed, SEED_SLOTS
from amodal_benchmark.verify import verify_run
from scripts.assess_shape_v2 import assess, table, METRICS

UNSEEN_SHA256 = "393dbc2d9ab38f568e93a2ae773315d0aec09327e27cf4cff8b96cad862eb437"
GEOMETRY_SHA256 = "f0e0bc47f7f4fdf9fb0cb8f6388826e16c06b6a670fd40a95402e6d091e57a9c"


def paired_estimate(rows, metric, draws=5000):
    """Pair image resampling while keeping each arm's successful CLIP denominator."""
    if not rows:
        raise ValueError("No paired records")
    totals, sampled = np.zeros(4), np.zeros((draws, 4))
    direction = -1 if metric == "LPIPS" else 1
    rng = np.random.default_rng(20261110)
    for source in sorted({row["source"] for row in rows}):
        groups = defaultdict(lambda: np.zeros(4))
        for row in rows:
            if row["source"] != source:
                continue
            for side, name in enumerate(("control", "method")):
                value = row[name][metric]
                if value not in (None, ""):
                    groups[row["image_identity"]][side*2] += float(value)
                    groups[row["image_identity"]][side*2+1] += 1
                else:
                    groups[row["image_identity"]]  # Include all-failure images in resampling.
        values = np.array([groups[key] for key in sorted(groups)])
        totals += values.sum(axis=0)
        sampled += values[rng.integers(len(values), size=(draws, len(values)))].sum(axis=1)
    if not totals[1] or not totals[3]:
        raise ValueError("Metric has an empty denominator")
    first, second = totals[0]/totals[1], totals[2]/totals[3]
    valid = (sampled[:, 1] > 0) & (sampled[:, 3] > 0)
    a, b = sampled[valid, 0]/sampled[valid, 1], sampled[valid, 2]/sampled[valid, 3]
    delta = direction * (b-a)
    relative_valid = np.abs(a) > 0
    return dict(control=float(first), method=float(second), control_n=int(totals[1]), method_n=int(totals[3]),
        relative_gain_percent=float(100*direction*(second-first)/abs(first)) if first else None,
        absolute_ci=np.percentile(delta, [2.5, 97.5]).tolist() if valid.all() else [None, None],
        relative_ci_percent=np.percentile(100*delta/np.abs(a), [2.5, 97.5]).tolist()
            if valid.all() and relative_valid.all() else [None, None],
        undefined_bootstrap_samples=int(draws-valid.sum()+(~relative_valid).sum()),
        image_clusters=len({row["image_identity"] for row in rows}))


def audit_decisions(folder, method_name="shape-v2"):
    if method_name not in ("shape-v2", "shape-calibrated-probability"):
        raise ValueError("Unknown shape selector")
    count, fallbacks = 0, 0
    for path in sorted(folder.glob(f"**/{method_name}/iteration-*/decision.json")):
        record = json.loads(path.read_bytes())
        if method_name == "shape-calibrated-probability" and (
                record.get("method") != method_name or record.get("logit_bias") != -2.):
            raise ValueError("Calibrated decision policy changed")
        if record.get("rgb_correction") is not False or record.get("inference_truth_access") is not False:
            raise ValueError("Decision is not the frozen shape-only policy")
        if record["seed_slots"] != list(SEED_SLOTS) or record["seeds"] != [
                progressive_seed(record["target_seed"], record["iteration"], slot) for slot in SEED_SLOTS]:
            raise ValueError("Candidate seed inventory changed")
        for name, expected in record["artifacts"].items():
            if Path(name).name != name or sha256(path.parent/name) != expected:
                raise ValueError("Candidate artifact changed: " + str(path.parent/name))
        prior = np.load(path.parent/"prior.npy", allow_pickle=False)
        masks = []
        for slot in range(8):
            with Image.open(path.parent/f"mask-{slot}.png") as image:
                masks.append(np.array(image) > 0)
        scores = score_masks(prior, masks)
        if any((a is None) != (b is None) or (a is not None and abs(a-b) > 1e-12)
               for a, b in zip(scores, record["scores"])) or len(record["scores"]) != 8:
            raise ValueError("Stored shape scores do not replay")
        index, fallback = choose(scores)
        if index != record["index"] or fallback != record["fallback"]:
            raise ValueError("Shape choice does not replay")
        with Image.open(path.parent/"selected.png") as selected, Image.open(path.parent/f"candidate-{index}.png") as candidate:
            if not np.array_equal(selected, candidate):
                raise ValueError("Selected RGB differs from the uncorrected candidate")
        count += 1
        fallbacks += int(fallback)
    return dict(pools=count, candidates=count*8, fallbacks=fallbacks)


def audit_geometry(control, method, folder, official, unseen_ids):
    from scripts.compare_shape_progressive import natural_geometry
    path = folder/"manifest.json"
    if sha256(path) != GEOMETRY_SHA256:
        raise ValueError("Frozen 321-object natural geometry inventory changed")
    inventory = json.loads(path.read_bytes())
    objects = []
    by_id = {r["id"]:r for r in official}
    for row in inventory["rows"]:
        target = by_id[row["id"]]
        mask = folder/row["case"]
        if sha256(mask) != row["sha256"]:
            raise ValueError("Natural geometry annotation changed")
        with Image.open(mask) as image:
            encoded = np.array(image)
        truth, visible = (encoded & 1) > 0, (encoded & 2) > 0
        current = dict(id=row["id"], source=target["source"],
            image_identity=target["source"]+"/"+target["original_id"], occlusion_bin=row["bin"])
        for label, run in (("control", control), ("method", method)):
            location = run/"targets"/row["id"]
            record = json.loads((location/"result.json").read_bytes())
            current[label] = dict(full_iou=float(natural_geometry(location, record, truth)),
                hidden_iou=natural_geometry(location, record, truth, exclude=visible))
        objects.append(current)
    result = dict(objects=len(objects), annotation_manifest_sha256=GEOMETRY_SHA256, rows=objects)
    for name, rows in (("full", objects), ("historical_unseen", [r for r in objects if r["id"] in unseen_ids])):
        result[name] = {metric:paired_estimate(rows, metric) for metric in ("full_iou", "hidden_iou")}
    result["occlusion_strata"] = {
        str(bin_name): dict(objects=len(rows), **{metric:paired_estimate(rows, metric)
            for metric in ("full_iou", "hidden_iou")
            if any(r["control"][metric] is not None for r in rows)})
        for bin_name in sorted({r["occlusion_bin"] for r in objects})
        for rows in [[r for r in objects if r["occlusion_bin"] == bin_name]]}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--method", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--unseen-inventory", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--geometry-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selector-method", choices=("shape-v2", "shape-calibrated-probability"), default="shape-v2")
    args = parser.parse_args()
    freeze = json.loads(args.freeze.read_bytes())
    if sha256(args.control/"per_target.csv") != freeze["control_scores_sha256"]:
        raise ValueError("Original control table changed")
    verifications = [verify_run(run, args.dataset_root, require_full=True, require_provenance=True)
                     for run in (args.control, args.method)]
    for key in ("targets_sha256", "inputs_sha256", "references_sha256"):
        if verifications[0]["data_identity"][key] != verifications[1]["data_identity"][key]:
            raise ValueError("Matched benchmark identity differs: " + key)
    configs = [json.loads((run/"run_config.json").read_bytes()) for run in (args.control, args.method)]
    policy = configs[1].pop("shape_policy")
    if policy.get("method") != args.selector_method:
        raise ValueError("Audit selector differs from the saved method")
    if policy != freeze["shape_policy"] or configs[0] != configs[1]:
        raise ValueError("Inference is not the frozen shape-only change to original PMC")
    if (args.control/"metric_config.json").read_bytes() != (args.method/"metric_config.json").read_bytes():
        raise ValueError("Metric settings differ")
    official, _ = read_official(args.dataset_root)
    if len(official) != 2565:
        raise ValueError("Full audit requires all 2565 targets")
    first, second = table(args.control/"per_target.csv"), table(args.method/"per_target.csv")
    gate = assess(first, second, [r["id"] for r in official])
    pairs, transitions = [], Counter()
    for row in official:
        a, b = first[row["id"]], second[row["id"]]
        transitions[(a["status"], b["status"])] += 1
        pairs.append(dict(id=row["id"], source=row["source"], image_identity=row["source"]+"/"+row["original_id"],
                          control=a, method=b))
    if sha256(args.unseen_inventory) != UNSEEN_SHA256:
        raise ValueError("Historical unseen inventory changed")
    unseen_ids = set(json.loads(args.unseen_inventory.read_bytes())["target_ids"])
    unseen = [r for r in pairs if r["id"] in unseen_ids]
    if len(unseen) != 2017:
        raise ValueError("Historical unseen inventory is incomplete")
    report = dict(verified=True, freeze_sha256=sha256(args.freeze), gate=gate,
        targets=len(pairs), unseen_targets=len(unseen),
        full={metric:paired_estimate(pairs, metric) for metric in METRICS},
        unseen={metric:paired_estimate(unseen, metric) for metric in METRICS},
        transitions=[dict(control=a, method=b, targets=n) for (a,b),n in sorted(transitions.items())],
        selector=audit_decisions(args.method/"targets", args.selector_method),
        geometry=audit_geometry(args.control, args.method, args.geometry_dir, official, unseen_ids),
        verification=verifications,
        interpretation="Iterated benchmark; nominal intervals are separate from the >=3% point threshold")
    if args.selector_method == "shape-calibrated-probability":
        from scripts.prepare_shape_calibrated_evaluation import audit_target_links
        report["selector_links"] = audit_target_links(args.method, policy)
    write_json(args.output, report)
    print(json.dumps(dict(targets=len(pairs), gate=gate, selector=report["selector"]), indent=2))


if __name__ == "__main__":
    main()

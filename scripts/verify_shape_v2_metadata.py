"""Replay a recovered full shape-v2 archive and bind it to the native audit."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from amodal_benchmark.dataset import read_official, sha256, write_json
from amodal_benchmark.protocol import failure_metrics
from amodal_benchmark.provenance import verify_provenance
from scripts.assess_shape_v2 import assess, table, METRICS
from scripts.audit_shape_v2 import paired_estimate, UNSEEN_SHA256, GEOMETRY_SHA256
from scripts.compare_shape_progressive import verify_full_registration
from scripts.export_shape_metadata import inventory


def equal(actual, expected, context):
    if isinstance(actual, dict) and isinstance(expected, dict):
        if actual.keys() != expected.keys():
            raise ValueError("Different fields: " + context)
        for key in actual:
            equal(actual[key], expected[key], context+"/"+key)
    elif isinstance(actual, list) and isinstance(expected, list):
        if len(actual) != len(expected):
            raise ValueError("Different lengths: " + context)
        for index, (a, b) in enumerate(zip(actual, expected)):
            equal(a, b, context+"/"+str(index))
    elif type(actual) in (int, float) and type(expected) in (int, float):
        if not math.isfinite(actual) or not math.isfinite(expected) or not math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-10):
            raise ValueError("Different numeric evidence: " + context)
    elif type(actual) is not type(expected) or actual != expected:
        raise ValueError("Different evidence: " + context)


def check_target(directory, row, control_record):
    record = json.loads((directory/"result.json").read_bytes())
    for name in ("id", "seed", "input_sha256", "evaluation_reference_artifacts"):
        equal(record[name], control_record[name], row["id"]+"/matched "+name)
    if record["status"] != row["status"] or record["seed"] != int(row["seed"]):
        raise ValueError("Target status or seed differs from score table")
    if record["status"] == "success":
        cache = json.loads((directory/"metrics.json").read_bytes())
        clip_only = row["reference_status"] != "ready"
        expected = dict(completion_sha256=record["completion_sha256"],
                        reference_sha256=record["evaluation_reference_artifacts"]["full" if clip_only else "crop"])
        if clip_only:
            expected["clip_only"] = True
        equal(cache["inputs"], expected, row["id"]+"/score inputs")
        values = cache["metrics"]
        coverage = {key for key, value in values.items() if value is not None}
        if coverage != ({"CLIP"} if clip_only else set(METRICS)):
            raise ValueError("Score cache metric coverage changed")
    elif record["status"] in ("missing_image", "reference_failed"):
        values = dict.fromkeys(METRICS)
    else:
        values = failure_metrics()
    numbers = {metric:float(row[metric]) if row[metric] else None for metric in METRICS}
    equal(numbers, values, row["id"]+"/failure-inclusive scores")
    return record


def verify_prior_evidence(snapshot, freeze, calibrated_data=None, sam_validation=None):
    read = lambda path: json.loads(path.read_bytes())
    calibrated = freeze["shape_policy"].get("method") == "shape-calibrated-probability"
    if calibrated:
        from amodal_benchmark.shape_calibrated_runtime import shape_calibrated_contract
        from scripts.verify_shape_calibrated_admission import replay
        if calibrated_data is None or sam_validation is None:
            raise ValueError("Independent calibrated validation labels and SAM evidence are required")
        package = snapshot / "admitted"
        equal(shape_calibrated_contract(package), freeze["shape_policy"], "admitted calibrated policy")
        equal(replay(package, calibrated_data, sam_validation), read(package / "native-replay.json"),
              "independent calibrated native replay")
        names = (("holdout-native.json", "holdout"),)
    else:
        names = (("validation-native.json", "validation"), ("holdout-native.json", "holdout"))
    prior = {}
    for name, split in names:
        geometry = read(snapshot/name)
        equal(geometry["shape_policy"], freeze["shape_policy"], name+"/policy")
        if geometry["split"] != split or geometry["geometry_source_sha256"] != freeze["geometry_source_sha256"]:
            raise ValueError("Native prior geometry uses a different split or evaluator")
        if split == "validation" and sha256(snapshot/name) != freeze["validation_sha256"]:
            raise ValueError("Development geometry changed after freezing")
        if split == "holdout" and (geometry["freeze_sha256"] != sha256(snapshot/"policy-freeze.json")
                or geometry["manifest_sha256"] != freeze["holdout_manifest_sha256"]):
            raise ValueError("Holdout evidence differs from the frozen model or data")
        from scripts.evaluate_shape_v2_geometry import grouped_means
        equal(grouped_means(geometry["rows"]), geometry["aggregate"], name+"/image-mean aggregation")
        prior[split] = geometry["aggregate"]
    if calibrated:
        prior["validation"] = read(snapshot / "admitted/candidate/candidate.json")["native_selected"]
        if prior["holdout"]["objects"] != 1183 or prior["holdout"]["images"] != 195:
            raise ValueError("Reserved native geometry inventory is incomplete")
    return prior


def verify_benchmark_geometry(geometry, official, unseen_ids, folder):
    if folder is None or sha256(folder / "manifest.json") != GEOMETRY_SHA256:
        raise ValueError("Frozen 321-target geometry annotations required")
    manifest = json.loads((folder / "manifest.json").read_bytes())
    expected = {row["id"]: row for row in manifest["rows"]}
    rows = geometry["rows"]
    if (len(expected) != 321 or len(rows) != 321 or geometry.get("objects") != 321
            or len({row["id"] for row in rows}) != 321 or {row["id"] for row in rows} != set(expected)
            or geometry.get("annotation_manifest_sha256") != GEOMETRY_SHA256):
        raise ValueError("Native benchmark geometry coverage changed")
    indexed = {row["id"]: row for row in official}
    for row in rows:
        target = indexed[row["id"]]
        if (row["source"] != target["source"] or row["image_identity"] != target["source"]+"/"+target["original_id"]
                or row["occlusion_bin"] != expected[row["id"]]["bin"]):
            raise ValueError("Native geometry identity or occlusion stratum changed")
        for arm in ("control", "method"):
            for metric in ("full_iou", "hidden_iou"):
                value = row[arm][metric]
                if value is None and metric == "hidden_iou":
                    continue
                if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
                    raise ValueError("Invalid native geometry value")
        if (row["control"]["hidden_iou"] is None) != (row["method"]["hidden_iou"] is None):
            raise ValueError("Hidden geometry eligibility differs across arms")
    for name, selected in (("full", rows), ("historical_unseen", [row for row in rows if row["id"] in unseen_ids])):
        equal({metric: paired_estimate(selected, metric) for metric in ("full_iou", "hidden_iou")},
              geometry[name], "native geometry " + name)
    strata = {str(bin_name): dict(objects=len(selected), **{metric: paired_estimate(selected, metric)
              for metric in ("full_iou", "hidden_iou")
              if any(row["control"][metric] is not None for row in selected)})
              for bin_name in sorted({row["occlusion_bin"] for row in rows})
              for selected in [[row for row in rows if row["occlusion_bin"] == bin_name]]}
    equal(strata, geometry["occlusion_strata"], "native geometry strata")
    return dict(verified=True, objects=321, native_pixel_metrics_recomputed=False,
                scope="Inventory, eligibility and image-cluster aggregation replay; pixel scores remain the cluster audit")


def reconcile(method, control, snapshot, official, unseen_path, calibrated_data=None, sam_validation=None, geometry_dir=None):
    read = lambda path: json.loads(path.read_bytes())
    freeze, native = read(snapshot/"policy-freeze.json"), read(snapshot/"full-audit.json")
    if (sha256(control/"per_target.csv") != freeze["control_scores_sha256"]
            or native["freeze_sha256"] != sha256(snapshot/"policy-freeze.json")
            or native.get("verified") is not True or native.get("targets") != 2565):
        raise ValueError("Original control, native audit or freeze differs")
    ids = [row["id"] for row in official]
    if len(ids) != 2565 or len(set(ids)) != 2565:
        raise ValueError("Full official 2,565-target contract required")
    executions = [verify_provenance(run) for run in (control, method)]
    if any(execution["target_ids"] != ids for execution in executions):
        raise ValueError("Closed executions differ from the ordered full benchmark")
    files = inventory(method, executions[1])
    declared = read(method/"member-sha256.json")
    if files != declared or any(sha256(method/name) != expected for name, expected in files.items()):
        raise ValueError("Recovered archive membership or contents differ from the closed run")
    configs = [read(run/"run_config.json") for run in (control, method)]
    equal(configs[1].pop("shape_policy"), freeze["shape_policy"], "frozen shape-only policy")
    equal(configs[1], configs[0], "original PMC settings")
    if (control/"metric_config.json").read_bytes() != (method/"metric_config.json").read_bytes():
        raise ValueError("Scoring settings changed")
    prior_geometry = verify_prior_evidence(snapshot, freeze, calibrated_data, sam_validation)
    selector_method = freeze["shape_policy"]["method"]
    trace_key = "shape_calibrated_iterations" if selector_method == "shape-calibrated-probability" else "shape_v2_iterations"
    first, second = table(control/"per_target.csv"), table(method/"per_target.csv")
    gate = assess(first, second, ids)
    equal(gate, native["gate"], "four-metric gate")
    pairs, saved, decisions, transitions = [], 0, 0, Counter()
    for row in official:
        tid = row["id"]
        original = read(control/"targets"/tid/"result.json")
        record = check_target(method/"targets"/tid, second[tid], original)
        transitions[(first[tid]["status"], second[tid]["status"])] += 1
        saved += record["status"] == "success"
        decisions += len(record.get("diagnostics", {}).get(trace_key, []))
        pairs.append(dict(id=tid, source=row["source"], image_identity=row["source"]+"/"+row["original_id"],
                          control=first[tid], method=second[tid]))
    if decisions != native["selector"]["pools"]:
        raise ValueError("Native selection audit is not bound to all recovered target traces")
    equal([dict(control=a, method=b, targets=n) for (a, b), n in sorted(transitions.items())],
          native["transitions"], "failure transitions")
    if sha256(unseen_path) != UNSEEN_SHA256:
        raise ValueError("Historical unseen inventory changed")
    unseen_ids = set(read(unseen_path)["target_ids"])
    unseen = [row for row in pairs if row["id"] in unseen_ids]
    if len(unseen) != 2017 or native["unseen_targets"] != 2017:
        raise ValueError("Historical unseen coverage changed")
    intervals = {name:{metric:paired_estimate(rows, metric) for metric in METRICS}
                 for name, rows in (("full", pairs), ("unseen", unseen))}
    for name in intervals:
        equal(intervals[name], native[name], name+" intervals")
    verification = read(snapshot/"full-verification.json")
    equal(verification, native["verification"][1], "native full verification")
    equal(read(method/"summary.json")["data_identity"], verification["data_identity"], "native verified identity")
    registration = verify_full_registration(method, snapshot/"registry.csv", verification)
    if selector_method == "shape-calibrated-probability":
        from scripts.prepare_shape_calibrated_evaluation import audit_target_links
        equal(audit_target_links(method, freeze["shape_policy"]), native["selector_links"], "calibrated target bindings")
        geometry_replay = verify_benchmark_geometry(native["geometry"], official, unseen_ids, geometry_dir)
    else:
        geometry_replay = None
    return dict(verified=True, targets=2565, saved=int(saved), failures=2565-int(saved),
        decisions=decisions, gate=gate, registration=registration, **intervals,
        method=selector_method, prior_geometry=prior_geometry, geometry_aggregation_replay=geometry_replay,
        method_execution=executions[1]["id"], method_scores_sha256=sha256(method/"per_target.csv"),
        freeze_sha256=sha256(snapshot/"policy-freeze.json"), native_audit_sha256=sha256(snapshot/"full-audit.json"),
        local_native_image_or_neural_metric_replay=False,
        scope="Local metadata, score-cache inputs, failure penalties, selector traces, registration and statistical replay. Native image/mask replay remains the separate cluster audit.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", type=Path, required=True, help="Safely extracted metadata archive")
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True, help="Dated recovered final evidence")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--unseen-inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    official, _ = read_official(args.dataset_root)
    report = reconcile(args.method, args.control, args.snapshot, official, args.unseen_inventory)
    write_json(args.output, report)
    print(json.dumps(dict(verified=True, targets=report["targets"], saved=report["saved"], gate=report["gate"]), indent=2))


if __name__ == "__main__":
    main()

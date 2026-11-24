"""Reconcile a recovered Shape archive with its closed execution and reported scores."""
import csv
import hashlib
import json
from pathlib import Path
import sys
import tarfile
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from amodal_benchmark.dataset import read_official, sha256
from amodal_benchmark.protocol import failure_metrics, METRICS
from amodal_benchmark.provenance import verify_provenance
from amodal_benchmark.shape import choose
from amodal_benchmark.surface_runtime import progressive_seed
from scripts.collect_surface_job_evidence import recover_archive


def reconcile(run, arm, ids, pairs, members):
    execution = verify_provenance(run)
    with (run / "per_target.csv").open(encoding="utf8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if (ids != execution["target_ids"] or set(ids) != execution["result_artifacts"].keys()
            or ids != [r["id"] for r in rows] or ids != [r["id"] for r in pairs]):
        raise ValueError("The archive and paired report must cover the same ordered official targets")
    policy = json.loads((run / "run_config.json").read_bytes())["shape_policy"]
    seeds = list(range(11, 89, 11))
    if policy["method"] != arm or policy["seeds"] != seeds:
        raise ValueError("Archived selector configuration differs from the study")
    expected = {"execution.json", "member-sha256.json", *execution["artifacts"]}
    expected.update(name for name in members if name.startswith("provenance/"))
    decisions = fallbacks = saved = 0
    for row, pair in zip(rows, pairs):
        tid = row["id"]
        directory = f"targets/{tid}"
        path = f"{directory}/result.json"
        expected.add(path)
        record = json.loads((run / path).read_bytes())
        status = record["status"]
        if ((record["id"], status, record["input_sha256"], record["seed"], record["evaluation_reference_artifacts"])
                != (tid, pair["status"][arm], pair["input_sha256"], pair["seed"], pair["method_reference"])
                or status != row["status"] or int(row["seed"]) != record["seed"]):
            raise ValueError(f"Archived result differs from the scored target {tid}")
        traces = record.get("diagnostics", {}).get("shape_iterations", [])
        for iteration, trace in enumerate(traces):
            path = f"{directory}/shape/iteration-{iteration:02d}/decision.json"
            expected.add(path)
            if members.get(path) != trace["decision_sha256"]:
                raise ValueError(f"Selector decision hash differs for {tid} iteration {iteration}")
            decision = json.loads((run / path).read_bytes())
            losses = decision.get("surface_losses")
            if ((losses is None) != (arm == "shape") or len(decision["scores"]) != len(seeds)
                    or (losses is not None and len(losses) != len(seeds))
                    or decision["iteration"] != iteration or decision["target_seed"] != record["seed"]
                    or decision["seeds"] != [progressive_seed(record["seed"], iteration, slot) for slot in seeds]):
                raise ValueError(f"Different candidate pool for {tid} iteration {iteration}")
            choice = choose(decision["scores"], losses)
            if choice != (decision["index"], decision["fallback"]) or choice != (trace["index"], trace["fallback"]):
                raise ValueError(f"Archived selector choice differs for {tid} iteration {iteration}")
            decisions += 1
            fallbacks += int(choice[1])
        if status == "success":
            if not traces:
                raise ValueError(f"Missing selector history for {tid}")
            saved += 1
            path = f"{directory}/metrics.json"
            expected.add(path)
            cache = json.loads((run / path).read_bytes())
            clip_only = row["reference_status"] != "ready"
            reference = record["evaluation_reference_artifacts"]["full" if clip_only else "crop"]
            inputs = {"completion_sha256": record["completion_sha256"], "reference_sha256": reference}
            if clip_only:
                inputs["clip_only"] = True
            values = cache["metrics"]
            available = {key for key, value in values.items() if value is not None}
            if cache["inputs"] != inputs or available != ({"CLIP"} if clip_only else set(METRICS)):
                raise ValueError(f"Score cache inputs or metric coverage differ for {tid}")
        elif status in {"missing_image", "reference_failed"}:
            values = dict.fromkeys(METRICS)
        else:
            values = failure_metrics()
        csv_values = {key: float(row[key]) if row[key] else None for key in METRICS}
        if values != csv_values or values != {key: pair[arm][key] for key in METRICS}:
            raise ValueError(f"Archived metrics differ from the CSV or paired report for {tid}")
    if expected != members.keys():
        raise ValueError("Archive membership differs from the native results and scored execution")
    return dict(execution=execution["id"], targets=len(ids), saved=saved, decisions=decisions, fallbacks=fallbacks)


def verify_export(archive, receipt, execution_digest, arm, ids, pairs):
    if sha256(archive) != receipt["sha256"] or receipt["targets"] != len(ids):
        raise ValueError("Archive digest or target count differs from the export receipt")
    with tarfile.open(archive, "r:gz") as bundle:
        manifest = bundle.extractfile("member-sha256.json").read()
    declared = json.loads(manifest)
    if (declared.get("execution.json") != execution_digest or len(declared) != receipt["files"]
            or "member-sha256.json" in declared):
        raise ValueError("Archive manifest differs from the collected execution or receipt")
    declared["member-sha256.json"] = hashlib.sha256(manifest).hexdigest()
    prefixes = ["member-sha256.json", "execution.json", "per_target.csv", "summary.json", "metric_config.json",
                "run_config.json", "resolved_config.json", "provenance", "targets"]
    with TemporaryDirectory() as work:
        run = Path(work)
        members = recover_archive(archive, run, prefixes, declared)
        result = reconcile(run, arm, ids, pairs, members)
    if result["execution"] != receipt["execution"]:
        raise ValueError("Archive contains a different execution than the export receipt")
    return dict(result, archive_sha256=receipt["sha256"], execution_sha256=execution_digest,
                manifest_sha256=declared["member-sha256.json"], files=receipt["files"])


def verify_collected(study, arm, job):
    directory = study / "progressive-l40s"
    archive = directory / f"{arm}-full-metadata.tar.gz"
    receipt = json.loads((directory / f"{arm}-export-{job}.log").read_bytes())
    execution_digest = sha256(directory / arm / "execution.json")
    pairs = study / f"{arm}-full-pairs.json"
    summary = json.loads((study / f"{arm}-full-summary.json").read_bytes())
    if (receipt["archive"] != archive.relative_to(ROOT).as_posix()
            or summary["cohort_reuse"]["full_execution_sha256"] != execution_digest
            or summary["pairs_sha256"] != sha256(pairs)):
        raise ValueError(f"Collected archive and paired audit disagree for {arm}")
    rows, _ = read_official(ROOT / "datasets/open_world_amodal")
    result = verify_export(archive, receipt, execution_digest, arm, [r["id"] for r in rows],
                           json.loads(pairs.read_bytes()))
    return dict(result, method=arm, export_job=job, pairs_sha256=summary["pairs_sha256"])

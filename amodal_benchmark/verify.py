"""Independently audit saved artifacts on CPU; never rerun learned metric models."""
import csv
import json
import math
from pathlib import Path

from PIL import Image

from .dataset import SOURCES, read_official, sha256
from .protocol import METRICS, _optional_bool, failure_metrics, failure_stage, read_cached_result
from .references import cached_reference, reference_paths
from .reproduction import data_identity, load_run


STATUSES = {"success", "missing_image", "reference_failed", "not_detected", "ambiguous_target", "no_completion"}


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _number(value):
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        raise ValueError("Metric values cannot be booleans")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("Metric values must be finite")
    return number


def _equal(actual, expected, context):
    if actual == expected:
        return
    if (isinstance(actual, (int, float)) and not isinstance(actual, bool) and
            isinstance(expected, (int, float)) and not isinstance(expected, bool) and
            math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12)):
        return
    raise ValueError(f"Artifact mismatch ({context}): {actual!r} != {expected!r}")


def _validate_metrics(values, context):
    if set(values) != set(METRICS):
        raise ValueError(f"Metric keys differ for {context}")
    numbers = {key: _number(value) for key, value in values.items()}
    if numbers["LPIPS"] is not None and numbers["LPIPS"] < 0:
        raise ValueError(f"Negative LPIPS for {context}")
    for name in ("Feature Similarity", "SSIM"):
        if numbers[name] is not None and not -1.000001 <= numbers[name] <= 1.000001:
            raise ValueError(f"Out-of-range {name} for {context}")
    return numbers


def verify_artifacts(run_dir, dataset_root, official_rows, reference_dir=None, *,
                     require_full=False, require_provenance=False):
    """Audit a known target contract. Public real-data callers use verify_run.

    The synthetic smoke harness supplies its own two-row contract here, and
    cannot register it as an official full benchmark.
    """
    output, dataset_root = Path(run_dir).resolve(), Path(dataset_root).resolve()
    run = load_run(output)  # Independently recomputes aggregate metrics/counts.
    config, summary = run["run_config"], run["summary"]
    # load_run normalizes historical summaries. Audit their recorded optional
    # statistics too, so a stale failure rate or status histogram cannot vanish
    # during normalization and then appear to have passed verification.
    saved_summary = _read(output / "summary.json")
    for source, statistics in summary["statistics"].items():
        for key, expected in statistics.items():
            if key in saved_summary["statistics"][source]:
                _equal(saved_summary["statistics"][source][key], expected, f"summary {source}/{key}")
    with (output / "per_target.csv").open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        columns = reader.fieldnames or []
        if len(columns) != len(set(columns)) or not {"id", "source", "label", "status", *METRICS} <= set(columns):
            raise ValueError("per_target.csv has missing or duplicate columns")
        csv_rows = list(reader)
    if not csv_rows:
        raise ValueError("Cannot verify an empty run")
    by_id = {row["id"]: row for row in official_rows}
    identifiers = [row["id"] for row in csv_rows]
    if any(identifier not in by_id for identifier in identifiers):
        raise ValueError("Run contains target IDs outside the dataset contract")
    selected = set(identifiers)
    if identifiers != [row["id"] for row in official_rows if row["id"] in selected]:
        raise ValueError("Target order differs from the official annotations")
    if config.get("annotation_sha256") != {source: values[4] for source, values in SOURCES.items()}:
        raise ValueError("Run annotation checksums differ from the official benchmark")
    sam = config.get("pipeline_profile") == "released-sam"
    backend = "grounded-sam" if sam else "lisa"
    _equal(summary.get("reference_backend", "lisa"), backend, "reference backend")
    _equal(summary.get("target_selection"), config["target_selection"], "target selection")
    _equal(summary.get("boundary_check"), config.get("boundary_check", "released"), "boundary check")
    scored = summary.get("evaluation_performed")
    if type(scored) is not bool:
        raise ValueError("evaluation_performed must be a boolean")
    if scored:
        if not run["metric_config"]:
            raise ValueError("Scored run is missing metric_config.json")
        _equal(summary["metric_alignment"], run["metric_config"]["alignment"], "metric alignment")
    refs = Path(reference_dir).resolve() if reference_dir is not None else (
        output / "visible_references" if sam else dataset_root / "visible_references")
    needs_refs = sam or scored or config["target_selection"].startswith("reference")
    marker = refs / "reference_config.json"
    if not sam and needs_refs and marker.is_file() and _read(marker).get("backend") == "grounded-sam":
        raise ValueError("SAM references cannot be reported as the independent LISA protocol")
    if sam:
        from evaluate_open_world import sam_reference_config
        _equal(_read(refs / "reference_config.json"), sam_reference_config(config), "SAM reference configuration")
    records, reference_records, image_hashes = [], {}, {}
    clip_only = uncached_clip_only = 0
    for csv_row in csv_rows:
        row = by_id[csv_row["id"]]
        identifier = row["id"]
        image_path = dataset_root / row["image"]
        if row["image"] not in image_hashes:
            if image_path.is_file():
                with Image.open(image_path) as image:
                    image.load()
                image_hashes[row["image"]] = sha256(image_path)
            else:
                image_hashes[row["image"]] = None
        image_hash = image_hashes[row["image"]]
        reference = None
        paths = reference_paths(refs, row)
        if needs_refs and image_hash is not None:
            reference = cached_reference(refs, row, image_hash)
            if reference is not None:
                if not sam and reference.get("provenance", {}).get("backend") == "grounded-sam":
                    raise ValueError(f"SAM reference cannot be reported as LISA: {identifier}")
                if set(reference["artifacts"]) != {"full", "crop", "mask"}:
                    raise ValueError(f"Incomplete reference artifact inventory: {identifier}")
                if reference["status"] not in ("ready", "empty_mask", "eroded_empty"):
                    raise ValueError(f"Unknown reference status: {identifier}")
                reference_records[identifier] = reference
            elif sam or summary.get("metric_alignment") == "released":
                raise ValueError(f"Generated reference missing: {identifier}")
            with Image.open(paths["full"]) as image:
                image.load()
                if image.mode != "RGBA":
                    raise ValueError(f"Reference must be RGBA: {identifier}")
                with Image.open(image_path) as original:
                    _equal(image.size, original.size, f"reference dimensions {identifier}")
        selection_hash = None
        if image_hash and config["target_selection"].startswith("reference"):
            selection_hash = sha256(paths["mask"] if reference is not None else paths["full"])
        folder = output / "targets" / identifier
        record = read_cached_result(folder / "result.json", row, image_hash, selection_hash)
        if record is None:
            raise ValueError(f"Result missing or newly available input requires inference: {identifier}")
        for field, value in row.items():
            _equal(record.get(field), value, f"annotation {identifier}/{field}")
        if record["status"] not in STATUSES:
            raise ValueError(f"Unknown result status: {identifier}/{record['status']}")
        if (record["status"] == "missing_image") != (image_hash is None):
            raise ValueError(f"Input availability does not match saved status: {identifier}")
        if record["status"] == "reference_failed" and (
                reference is None or reference["status"] == "ready" or
                record.get("reference_status") != reference["status"] or
                not config["target_selection"].startswith("reference")):
            raise ValueError(f"Reference-failure status lacks a failed selection reference: {identifier}")
        for field in ("source", "label", "status", "failure_reason", "target_selection", "detected_class"):
            if field in csv_row:
                _equal(csv_row[field] or None, record.get(field) or None, f"CSV {identifier}/{field}")
        for field in ("seed", "iterations", "selected_iou", "detector_score"):
            if field in csv_row:
                _equal(_number(csv_row[field]), _number(record.get(field)), f"CSV {identifier}/{field}")
        if "iteration_limit_reached" in csv_row:
            _equal(_optional_bool(csv_row["iteration_limit_reached"]), record.get("iteration_limit_reached"),
                   f"CSV iteration flag {identifier}")
        if "failure_stage" in csv_row:
            _equal(csv_row["failure_stage"], failure_stage(record), f"failure stage {identifier}")
        if sam and image_hash:
            _equal(record.get("evaluation_reference_artifacts"), reference["artifacts"], f"SAM artifacts {identifier}")
        if record["status"] == "success":
            with Image.open(folder / "completion.png") as image:
                image.load()
                if image.mode != "RGBA" or not image.getchannel("A").getbbox():
                    raise ValueError(f"Completion must be a nonempty RGBA image: {identifier}")
            geometry = record.get("geometry", {})
            with Image.open(image_path) as original:
                _equal(geometry.get("original_size"), list(original.size), f"completion geometry {identifier}")
            offset = geometry.get("offset_rc")
            if not isinstance(offset, list) or len(offset) != 2 or any(type(n) is not int for n in offset):
                raise ValueError(f"Invalid completion offset: {identifier}")
        numbers = _validate_metrics({name: csv_row[name] for name in METRICS}, identifier)
        excluded = record["status"] in ("missing_image", "reference_failed")
        unavailable = reference is not None and reference["status"] != "ready"
        if not scored or excluded:
            expected = dict.fromkeys(METRICS)
        elif sam and record["status"] != "success":
            expected = failure_metrics()
        elif unavailable:
            expected = dict.fromkeys(METRICS)
            if record["status"] == "success":
                if numbers["CLIP"] is None:
                    raise ValueError(f"Successful completion has no CLIP score: {identifier}")
                expected["CLIP"] = numbers["CLIP"]
                clip_only += 1
                if (folder / "metrics.json").is_file():
                    cache = _read(folder / "metrics.json")
                    _equal(cache.get("inputs"), {"reference_sha256": sha256(paths["full"]),
                                                "completion_sha256": record["completion_sha256"], "clip_only": True},
                           f"CLIP-only input hashes {identifier}")
                    expected = _validate_metrics(cache["metrics"], identifier)
                    if expected["CLIP"] is None or any(expected[name] is not None for name in METRICS[1:]):
                        raise ValueError(f"Invalid CLIP-only metric cache: {identifier}")
                else:
                    uncached_clip_only += 1
        elif record["status"] != "success":
            expected = failure_metrics()
        else:
            cache = _read(folder / "metrics.json")
            ref_path = paths["crop"] if summary["metric_alignment"] == "released" else paths["full"]
            _equal(cache.get("inputs"), {"reference_sha256": sha256(ref_path),
                                        "completion_sha256": record["completion_sha256"]},
                   f"metric input hashes {identifier}")
            expected = _validate_metrics(cache["metrics"], identifier)
            if any(value is None for value in expected.values()):
                raise ValueError(f"Incomplete successful metric cache: {identifier}")
        for metric in METRICS:
            _equal(numbers[metric], expected[metric], f"metric {identifier}/{metric}")
        records.append(record)
    identity = data_identity(records, {"reference_records": reference_records})
    _equal(summary.get("data_identity"), identity, "dataset/reference identity")
    full = len(records) == 2565 and set(identifiers) == set(by_id) and all(image_hashes.values())
    _equal(summary.get("full_image_coverage"), full, "full image coverage")
    if require_full and (not full or not scored):
        raise ValueError("A full, scored 2,565-target evaluation is required")
    resolved_path = output / "resolved_config.json"
    if resolved_path.is_file():
        from .config import load_config
        resolved = load_config(resolved_path)
        if require_full and resolved["evaluation_scope"] != "full":
            raise ValueError("Smoke runs cannot be registered as full evaluations")
        expected_ids = [row["id"] for row in official_rows if row["source"] in resolved["dataset"]["sources"]]
        if resolved["dataset"]["limit"] is not None:
            expected_ids = expected_ids[:resolved["dataset"]["limit"]]
        _equal(identifiers, expected_ids, "resolved target selection")
        from .config import check_saved_inference
        check_saved_inference(resolved, config)
        _equal(summary["metric_alignment"], resolved["metrics"]["alignment"], "resolved metric alignment")
    provenance = None
    if (output / "execution.json").exists() or require_provenance:
        from .provenance import verify_provenance
        provenance = verify_provenance(output)
        _equal(provenance["target_ids"], identifiers, "execution target inventory")
    return {"schema": 1, "verified": True, "run_dir": str(output), "selected_targets": len(records),
            "full_evaluation": bool(full and scored), "evaluation_performed": scored,
            "reference_backend": backend, "data_identity": identity,
            "statistics": summary["statistics"], "provenance_verified": provenance is not None,
            "clip_only_targets": clip_only,
            "uncached_clip_only_targets": uncached_clip_only,
            "limitations": ["Learned metrics are checked against saved caches, not recomputed on CPU.",
                            "Historical uncached CLIP-only values can only be checked for finite values and aggregation.",
                            "Checksums establish consistency, not independent segmentation or hidden-pixel ground truth."]}


def verify_run(run_dir, dataset_root, reference_dir=None, *, require_full=False, require_provenance=False):
    rows, _ = read_official(dataset_root)
    return verify_artifacts(run_dir, dataset_root, rows, reference_dir,
                            require_full=require_full, require_provenance=require_provenance)

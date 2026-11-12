"""An explicit, locked research ledger; smoke runs never enter the full ledger."""
import csv
from contextlib import nullcontext
from datetime import datetime, timezone
import json
from pathlib import Path

from .config import load_config
from .dataset import read_official, sha256
from .parallel import output_lock
from .provenance import digest, verify_provenance
from .verify import verify_run


FIELDS = ("entry_id", "recorded_utc", "run_dir", "status", "hypothesis", "reason", "baseline_dir",
          "changed_factor", "comparison", "protocol_sha256", "inputs_sha256", "references_sha256",
          "summary_sha256", "targets", "successes", "failures", "failure_rate", "CLIP", "CLIP_n",
          "LPIPS", "LPIPS_n", "Feature Similarity", "Feature Similarity_n", "SSIM", "SSIM_n")


def compare_baseline(output, baseline, dataset_root, reference_dir=None, baseline_reference_dir=None):
    current = load_config(Path(output) / "resolved_config.json")
    previous = load_config(Path(baseline) / "resolved_config.json")
    changes = [f"{section}.{key}" for section in ("inference", "metrics")
               for key, value in current[section].items() if previous[section][key] != value]
    factor = current["experiment"]["changed_factor"]
    if len(changes) != 1 or changes[0] != factor:
        raise ValueError(f"Baseline comparison requires exactly the declared changed factor {factor!r}; got {changes}")
    first = verify_run(output, dataset_root, reference_dir, require_full=True, require_provenance=True)
    second = verify_run(baseline, dataset_root, baseline_reference_dir, require_full=True, require_provenance=True)
    for key in ("targets_sha256", "inputs_sha256"):
        if first["data_identity"][key] != second["data_identity"][key]:
            raise ValueError("Baseline comparison requires the same original targets and input bytes")
    reference_changed = first["data_identity"]["references_sha256"] != second["data_identity"]["references_sha256"]
    if reference_changed and current["inference"]["pipeline_profile"] == "benchmark":
        raise ValueError("LISA baseline comparisons require identical frozen references")
    # Declaring one factor in a JSON file cannot hide a changed checkpoint,
    # vocabulary file, model revision, or matcher implementation in the run.
    allowed = {
        "inference.seed": {"seed"}, "inference.mc_timestep": {"mc_timestep"},
        "inference.max_iter": {"max_iter_id"}, "inference.background_sizing": {"resize_background"},
        "inference.boundary_check": {"boundary_check", "adapter_version"},
        "inference.target_selection": {"target_selection", "label_matching", "adapter_version"},
        "inference.detector_vocabulary": {"detector_vocabulary", "adapter_version"},
        "inference.sd_model": {"sd_model", "sd_revision", "local_sd_files"},
        "metrics.alignment": set(),
    }.get(factor)
    if allowed is None:
        raise ValueError("A pipeline-profile switch defines a separate baseline, not a single-factor comparison")
    for name in ("run_config.json", "metric_config.json"):
        left = json.loads((Path(output) / name).read_text(encoding="utf-8"))
        right = json.loads((Path(baseline) / name).read_text(encoding="utf-8"))
        permitted = allowed if name == "run_config.json" else ({"alignment"} if factor == "metrics.alignment" else set())
        extra = {key for key in set(left) | set(right) if left.get(key) != right.get(key)} - permitted
        if extra:
            raise ValueError(f"Baseline artifacts differ beyond the declared factor: {name}/{sorted(extra)}")
    return "same_inputs_different_references" if reference_changed else "same_inputs_and_references"


def _read_ledger(path):
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != list(FIELDS):
            raise ValueError("Existing registry has a different schema; choose a separate registry")
        rows = list(reader)
    ids = [row["entry_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Existing registry contains duplicate entries")
    return rows


def register_run(run_dir, dataset_root, registry, *, reference_dir=None, hypothesis=None,
                 baseline_dir=None, baseline_reference_dir=None, reason=None):
    # Registration must not race a resumed coordinator while verifying files
    # and publishing their identities. This is the same nonblocking OS lock
    # used by evaluation, so a running experiment is reported immediately.
    with output_lock(Path(run_dir).resolve()):
        return _register_locked(run_dir, dataset_root, registry, reference_dir=reference_dir,
                                hypothesis=hypothesis, baseline_dir=baseline_dir,
                                baseline_reference_dir=baseline_reference_dir, reason=reason)


def _register_locked(run_dir, dataset_root, registry, *, reference_dir=None, hypothesis=None,
                     baseline_dir=None, baseline_reference_dir=None, reason=None):
    output, registry = Path(run_dir).resolve(), Path(registry).resolve()
    execution = json.loads((output / "execution.json").read_text(encoding="utf-8"))
    resolved_path = output / "resolved_config.json"
    resolved = load_config(resolved_path) if resolved_path.is_file() else None
    if resolved is not None and resolved["evaluation_scope"] != "full":
        raise ValueError("Smoke runs cannot enter the full experiment registry")
    if len(execution.get("target_ids", [])) != 2565:
        raise ValueError("Subset runs cannot enter the full experiment registry")
    hypothesis = hypothesis or (resolved["experiment"]["hypothesis"] if resolved else None)
    if not hypothesis or not hypothesis.strip():
        raise ValueError("Record an experiment hypothesis before registering a run")
    entry = dict.fromkeys(FIELDS, "")
    entry.update(entry_id=digest({"run": str(output), "execution": execution["id"]}),
                 recorded_utc=datetime.now(timezone.utc).isoformat(), run_dir=str(output),
                 hypothesis=hypothesis, status=execution["status"],
                 changed_factor=resolved["experiment"]["changed_factor"] or "" if resolved else "")
    if execution["status"] == "complete":
        report = verify_run(output, dataset_root, reference_dir, require_full=True, require_provenance=True)
        if not all(report["data_identity"].values()):
            raise ValueError("Full registry entries require recorded input and reference identities")
        if report["uncached_clip_only_targets"]:
            raise ValueError("Score the historical CLIP-only targets once to create verifiable metric caches")
        entry.update({key: report["statistics"]["Overall"][key] for key in FIELDS
                      if key in report["statistics"]["Overall"]})
        entry.update(inputs_sha256=report["data_identity"]["inputs_sha256"],
                     references_sha256=report["data_identity"]["references_sha256"],
                     summary_sha256=sha256(output / "summary.json"))
        config = json.loads((output / "run_config.json").read_text(encoding="utf-8"))
        metric = json.loads((output / "metric_config.json").read_text(encoding="utf-8"))
        entry["protocol_sha256"] = digest({"inference": config, "metrics": metric,
                                           "data": report["data_identity"]})
        if baseline_dir is not None:
            baseline = Path(baseline_dir).resolve()
            if baseline == output:
                raise ValueError("Baseline must be a different run")
            with output_lock(baseline):
                entry["comparison"] = compare_baseline(output, baseline, dataset_root, reference_dir,
                                                         baseline_reference_dir)
            entry["baseline_dir"] = str(baseline)
        elif entry["changed_factor"]:
            raise ValueError("An experiment with a changed_factor requires --baseline-dir")
    elif execution["status"] in ("failed", "interrupted"):
        verify_provenance(output, require_complete=False)
        official_rows, _ = read_official(dataset_root)
        if execution["target_ids"] != [row["id"] for row in official_rows]:
            raise ValueError("Failed full attempts require every official target in annotation order")
        entry["reason"] = reason or execution.get("error", "")
        if not entry["reason"]:
            raise ValueError("Failed runs need a recorded reason")
        entry["targets"] = len(execution["target_ids"])
    else:
        raise ValueError("Run is still running or its termination is unknown; do not register it as completed")
    entry = {key: "" if value is None else value for key, value in entry.items()}
    registry.parent.mkdir(parents=True, exist_ok=True)
    with nullcontext() if registry.parent == output else output_lock(registry.parent):
        rows = _read_ledger(registry)
        existing = next((row for row in rows if row["entry_id"] == entry["entry_id"]), None)
        if existing is not None:
            # Registration is idempotent, but never silently rewrites history.
            for key in FIELDS:
                if key != "recorded_utc" and existing[key] != str(entry[key]):
                    raise ValueError("This execution is already registered with different metadata")
        else:
            rows.append(entry)
            temporary = registry.with_suffix(registry.suffix + ".tmp")
            with temporary.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, FIELDS)
                writer.writeheader()
                writer.writerows(rows)
            temporary.replace(registry)
        render_ledger(rows, registry.with_name("experiment_report.md"))
    return entry


def render_ledger(rows, path):
    def cell(value):
        return str(value).replace("|", "\\|").replace("\r", " ").replace("\n", " ")
    lines = ["# Research experiment ledger", "",
             "Runs are listed in registration order. There is no automatic best-model ranking.",
             "Compare metrics only after inspecting protocol, reference identity and denominators.", "",
             "| Run | Status | Hypothesis | Saved / targets | Failure rate | CLIP (n) | LPIPS (n) | Comparison |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for row in rows:
        values = [row["run_dir"], row["status"], row["hypothesis"],
                  f"{row['successes']} / {row['targets']}", row["failure_rate"],
                  f"{row['CLIP']} ({row['CLIP_n']})", f"{row['LPIPS']} ({row['LPIPS_n']})", row["comparison"]]
        lines.append("| " + " | ".join(cell(value) for value in values) + " |")
    temporary = Path(path).with_suffix(".md.tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporary.replace(path)

"""Strict experiment contracts; JSON keeps CPU and pinned Python 3.10 setup small."""
import copy
import json
from pathlib import Path
import re

from .dataset import SOURCES


FIELDS = {
    "project": {"slug", "task"},
    "dataset": {"root", "sources", "limit", "allow_incomplete"},
    "inference": {"pipeline_profile", "target_selection", "boundary_check", "background_sizing",
                  "detector_vocabulary", "seed", "mc_timestep", "max_iter", "sd_model"},
    "metrics": {"alignment"},
    "execution": {"workers"},
    "experiment": {"hypothesis", "changed_factor"},
}


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate configuration key: {key}")
        result[key] = value
    return result


def _merge(base, child):
    result = copy.deepcopy(base)
    for key, value in child.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(path, _parents=()):
    """Resolve one-parent inheritance; only extends paths are config-relative.

    Dataset and runtime paths are relative to the caller's working directory,
    matching the existing command-line workflow on Windows and Slurm.
    """
    path = Path(path).resolve()
    if path in _parents or len(_parents) >= 16:
        raise ValueError(f"Configuration inheritance cycle or excessive depth: {path}")
    value = json.loads(path.read_text(encoding="utf-8-sig"), object_pairs_hook=_object)
    if not isinstance(value, dict):
        raise ValueError("Configuration must be a JSON object")
    parent = value.pop("extends", None)
    if parent is not None:
        if not isinstance(parent, str) or not parent.strip():
            raise ValueError("extends must be a nonempty configuration filename")
        value = _merge(load_config(path.parent / parent, (*_parents, path)), value)
    validate_config(value)
    return value


def validate_config(value):
    expected = {"schema", "evaluation_scope", *FIELDS}
    if set(value) != expected:
        raise ValueError(f"Configuration fields differ: missing={sorted(expected - set(value))}, "
                         f"unknown={sorted(set(value) - expected)}")
    if type(value["schema"]) is not int or value["schema"] != 1:
        raise ValueError("Unsupported configuration schema; expected 1")
    for section, keys in FIELDS.items():
        if not isinstance(value[section], dict) or set(value[section]) != keys:
            raise ValueError(f"{section} must contain exactly: {', '.join(sorted(keys))}")
    project, data, infer = value["project"], value["dataset"], value["inference"]
    if not isinstance(project["slug"], str) or not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", project["slug"]):
        raise ValueError("project.slug must be a lowercase, hyphen-separated identifier")
    if project["task"] != "amodal-appearance-completion":
        raise ValueError("Only amodal-appearance-completion is supported; extend the contract for another task")
    if value["evaluation_scope"] not in ("full", "smoke"):
        raise ValueError("evaluation_scope must be full or smoke")
    if not isinstance(data["root"], str) or not data["root"].strip():
        raise ValueError("dataset.root must be a nonempty path")
    sources = data["sources"]
    if (not isinstance(sources, list) or not sources or
            any(not isinstance(s, str) or s not in SOURCES for s in sources) or
            len(set(sources)) != len(sources)):
        raise ValueError("dataset.sources must list unique official source names")
    if data["limit"] is not None and (type(data["limit"]) is not int or data["limit"] < 1):
        raise ValueError("dataset.limit must be null or a positive integer")
    if type(data["allow_incomplete"]) is not bool:
        raise ValueError("dataset.allow_incomplete must be a boolean")
    if value["evaluation_scope"] == "full" and (
            set(sources) != set(SOURCES) or data["limit"] is not None or data["allow_incomplete"]):
        raise ValueError("Full evaluation requires every source, no limit, and complete inputs")
    choices = {
        "pipeline_profile": ("benchmark", "released-sam"),
        "target_selection": ("label", "label-semantic", "reference", "reference-best"),
        "boundary_check": ("paper", "released"),
        "background_sizing": ("released", "auto"),
        "detector_vocabulary": ("released", "annotated"),
    }
    for name, allowed in choices.items():
        if infer[name] not in allowed:
            raise ValueError(f"inference.{name} must be one of {allowed}")
    for name, minimum, maximum in (("seed", 0, None), ("mc_timestep", 0, 49), ("max_iter", 1, None)):
        number = infer[name]
        if type(number) is not int or number < minimum or maximum is not None and number > maximum:
            raise ValueError(f"Invalid inference.{name}: {number!r}")
    if not isinstance(infer["sd_model"], str) or not infer["sd_model"].strip():
        raise ValueError("inference.sd_model must be a model ID or directory")
    if infer["pipeline_profile"] == "released-sam":
        if infer["boundary_check"] != "released" or infer["target_selection"] not in ("label", "label-semantic"):
            raise ValueError("released-sam requires released boundaries and label or label-semantic selection")
    elif infer["background_sizing"] != "auto" or infer["target_selection"] == "label-semantic":
        raise ValueError("benchmark requires auto background sizing and does not support semantic label selection")
    if infer["detector_vocabulary"] == "annotated" and (
            infer["pipeline_profile"] != "released-sam" or infer["target_selection"] != "label-semantic"):
        raise ValueError("Annotated vocabulary requires released-sam and label-semantic selection")
    if value["metrics"]["alignment"] not in ("released", "original-frame"):
        raise ValueError("metrics.alignment must be released or original-frame")
    workers = value["execution"]["workers"]
    if workers != "auto" and (type(workers) is not int or workers < 1):
        raise ValueError("execution.workers must be auto or a positive integer")
    experiment = value["experiment"]
    if not isinstance(experiment["hypothesis"], str) or not experiment["hypothesis"].strip():
        raise ValueError("experiment.hypothesis must explain the run")
    factor = experiment["changed_factor"]
    factors = {f"{section}.{key}" for section in ("inference", "metrics") for key in FIELDS[section]}
    if factor is not None and (not isinstance(factor, str) or factor not in factors):
        raise ValueError("experiment.changed_factor must name an inference or metrics field, or be null")
    return value


def evaluator_arguments(config):
    validate_config(config)
    data = config["dataset"]
    args = ["--dataset-root", data["root"], "--sources", ",".join(data["sources"]),
            "--workers", str(config["execution"]["workers"]),
            "--metric-alignment", config["metrics"]["alignment"]]
    for key, value in config["inference"].items():
        args += ["--" + key.replace("_", "-"), str(value)]
    if data["limit"] is not None:
        args += ["--limit", str(data["limit"])]
    if data["allow_incomplete"]:
        args.append("--allow-incomplete")
    return args


def effective_config(config, args):
    """Record explicit CLI overrides as the actual, validated experiment."""
    value = copy.deepcopy(config)
    value["dataset"].update(root=str(args.dataset_root), sources=args.sources.split(","),
                            limit=args.limit, allow_incomplete=args.allow_incomplete)
    value["inference"].update({key: getattr(args, key) for key in FIELDS["inference"]})
    value["metrics"]["alignment"] = args.metric_alignment
    value["execution"]["workers"] = "auto" if args.workers == "auto" else int(args.workers)
    return validate_config(value)


def check_saved_inference(config, saved):
    """Reject incompatible config adoption before modifying a legacy output."""
    defaults = {"pipeline_profile": saved.get("pipeline_profile", "benchmark"),
                "background_sizing": "auto" if saved.get("resize_background", True) else "released",
                "max_iter": saved.get("max_iter_id"),
                "boundary_check": saved.get("boundary_check", "released"),
                "detector_vocabulary": saved.get("detector_vocabulary", {}).get("mode", "released")}
    for key, value in config["inference"].items():
        if defaults.get(key, saved.get(key)) != value:
            raise ValueError(f"Resolved inference.{key} differs from the saved run; use a new output directory")

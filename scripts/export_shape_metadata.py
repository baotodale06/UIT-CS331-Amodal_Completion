"""Archive the execution, native score caches and selector decisions for local review."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tarfile
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from amodal_benchmark.dataset import sha256
from amodal_benchmark.provenance import verify_provenance


def inventory(run, execution):
    targets = execution["target_ids"]
    if len(set(targets)) != len(targets) or set(targets) != execution["result_artifacts"].keys():
        raise ValueError("Incomplete target inventory")
    files = dict(execution["artifacts"])
    files["execution.json"] = sha256(run / "execution.json")
    config_path = run / "run_config.json"
    config = json.loads(config_path.read_bytes()) if config_path.exists() else {}
    method = config.get("shape_policy", {}).get("method")
    v2 = method in ("shape-v2", "shape-calibrated-probability")
    selector_folder = method if v2 else "shape"
    trace_key = ("shape_calibrated_iterations" if method == "shape-calibrated-probability"
                 else "shape_v2_iterations" if v2 else "shape_iterations")
    for path in sorted((run / "provenance").rglob("*")):
        if path.is_file():
            files[path.relative_to(run).as_posix()] = sha256(path)
    for tid in targets:
        directory = run / "targets" / tid
        record = json.loads((directory / "result.json").read_bytes())
        if record["id"] != tid:
            raise ValueError(f"Result has a different target ID for {tid}")
        files[f"targets/{tid}/result.json"] = execution["result_artifacts"][tid]
        decisions = sorted((directory / selector_folder).glob("iteration-*/decision.json"))
        traces = record.get("diagnostics", {}).get(trace_key, [])
        if len(decisions) != len(traces) or (record["status"] == "success" and not traces):
            raise ValueError(f"Selector decision inventory differs for {tid}")
        for path, trace in zip(decisions, traces):
            files[path.relative_to(run).as_posix()] = trace["decision_sha256"]
            if v2:
                verify_v2_decision(path, trace, record, method=method)
        if record["status"] == "success":
            path = directory / "metrics.json"
            data = path.read_bytes()
            cache = json.loads(data)
            inputs = cache["inputs"]
            clip_only = inputs.get("clip_only") is True
            reference = record["evaluation_reference_artifacts"]["full" if clip_only else "crop"]
            expected = {"reference_sha256": reference, "completion_sha256": record["completion_sha256"]}
            if clip_only:
                expected["clip_only"] = True
            if inputs != expected:
                raise ValueError(f"Score cache inputs differ for {tid}")
            files[path.relative_to(run).as_posix()] = hashlib.sha256(data).hexdigest()
    return files


def verify_v2_decision(path, trace, record, method="shape-v2"):
    """Bind v2 decisions to the closed target, rather than an unbound glob."""
    from amodal_benchmark.shape_v2_runtime import choose, progressive_seed, SEED_SLOTS
    if sha256(path) != trace["decision_sha256"]:
        raise ValueError("Shape-v2 decision differs from its target trace")
    decision = json.loads(path.read_bytes())
    iteration = int(path.parent.name.removeprefix("iteration-"))
    if (method not in ("shape-v2", "shape-calibrated-probability")
            or decision.get("method") != method or decision.get("schema") != 2
            or (method == "shape-calibrated-probability" and decision.get("logit_bias") != -2.)
            or decision["iteration"] != iteration or decision["target_seed"] != record["seed"]
            or decision["seed_slots"] != list(SEED_SLOTS)
            or decision["seeds"] != [progressive_seed(record["seed"], iteration, slot) for slot in SEED_SLOTS]
            or len(decision["scores"]) != len(SEED_SLOTS)
            or decision.get("rgb_correction") is not False or decision.get("inference_truth_access") is not False):
        raise ValueError("Shape-v2 decision differs from its target policy")
    selected = choose(decision["scores"])
    if selected != (decision["index"], decision["fallback"]) or selected != (trace["index"], trace["fallback"]):
        raise ValueError("Shape-v2 decision does not replay")


def add_bytes(bundle, name, data):
    info = tarfile.TarInfo(name)
    info.size = len(data)
    bundle.addfile(info, io.BytesIO(data))


def export(run, destination):
    if destination.exists():
        raise FileExistsError(destination)
    execution = verify_provenance(run)
    files = inventory(run, execution)
    with TemporaryDirectory(dir=destination.parent) as work:
        staged = Path(work) / "metadata.tar.gz"
        with tarfile.open(staged, "w:gz", compresslevel=1) as bundle:
            for name, expected in sorted(files.items()):
                data = (run / name).read_bytes()
                if hashlib.sha256(data).hexdigest() != expected:
                    raise ValueError(f"Native metadata differs from its inventory for {name}")
                add_bytes(bundle, name, data)
            add_bytes(bundle, "member-sha256.json", json.dumps(files, sort_keys=True).encode())
        archive_hash = sha256(staged)
        os.link(staged, destination)
    return dict(archive=str(destination), sha256=archive_hash, execution=execution["id"],
                targets=len(execution["target_ids"]), files=len(files))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    print(json.dumps(export(args.run, args.destination)))

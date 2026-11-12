"""Small, immutable execution snapshots without model downloads or secret capture."""
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import signal
import sys
import threading
import uuid

from .dataset import sha256, write_json


SOURCE_FILES = (
    "evaluate_open_world.py", "evaluate_pdmc_open_world.py", "prepare_open_world.py",
    "prepare_visible_references.py", "compare_open_world.py", "run_open_world_controls.py",
    "run_amodal_slurm.sh", "run_pdmc_open_world.sh", "pyproject.toml", "requirements.txt",
)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def environment_snapshot():
    # Distribution names/versions suffice for exact package inventory. Avoid
    # pip freeze's credential-bearing VCS URLs and arbitrary environment values.
    packages = sorted({(dist.metadata.get("Name", "unknown"), dist.version)
                       for dist in importlib.metadata.distributions()})
    allowed = ("SLURM_JOB_ID", "SLURM_JOB_GPUS", "SLURM_CPUS_PER_TASK", "CUDA_VISIBLE_DEVICES",
               "OMP_NUM_THREADS", "MKL_NUM_THREADS", "CUBLAS_WORKSPACE_CONFIG")
    return {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
            "packages": [{"name": name, "version": version} for name, version in packages],
            "execution_environment": {key: os.environ[key] for key in allowed if key in os.environ}}


def snapshot_source(destination):
    package = Path(__file__).resolve().parent
    root = package.parent
    paths = [*package.glob("*.py"), *(root / name for name in SOURCE_FILES),
             *(root / "configs").glob("*.json")]
    files = {}
    for path in sorted(paths):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        target = Path(destination) / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        files[relative] = sha256(target)
    return {"files": files, "sha256": digest(files)}


def _scientific_contract(config):
    value = {key: item for key, item in config.items() if key != "execution"}
    value["dataset"] = {key: item for key, item in config["dataset"].items() if key != "root"}
    return value


@contextmanager
def execution_snapshot(output, args, rows, resolved_config=None):
    """Call under the coordinator output lock; workers never publish snapshots."""
    output = Path(output)
    if resolved_config is not None:
        if (output / "run_config.json").exists():
            from .config import check_saved_inference
            check_saved_inference(resolved_config, json.loads((output / "run_config.json").read_text(encoding="utf-8")))
        path = output / "resolved_config.json"
        if path.exists():
            saved = json.loads(path.read_text(encoding="utf-8"))
            if _scientific_contract(saved) != _scientific_contract(resolved_config):
                raise ValueError("Resolved experiment configuration changed; use a new output directory")
        else:
            write_json(path, resolved_config)
    identifier = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") + "-" + uuid.uuid4().hex[:8]
    folder = output / "provenance" / identifier
    folder.mkdir(parents=True, exist_ok=False)
    source = snapshot_source(folder / "source")
    write_json(folder / "source.json", source)
    write_json(folder / "environment.json", environment_snapshot())
    if resolved_config is not None:
        write_json(folder / "resolved_config.json", resolved_config)
    arguments = {key: str(value) if isinstance(value, Path) else value
                 for key, value in vars(args).items()}
    def results():
        return {row["id"]: sha256(output / "targets" / row["id"] / "result.json") for row in rows
                if (output / "targets" / row["id"] / "result.json").is_file()}
    initial_results = results()
    record = {"schema": 1, "id": identifier, "status": "running",
              "started_utc": datetime.now(timezone.utc).isoformat(), "arguments": arguments,
              "target_ids": [row["id"] for row in rows], "source_sha256": source["sha256"],
              "environment_sha256": sha256(folder / "environment.json")}
    if resolved_config is not None:
        record["resolved_config_sha256"] = sha256(folder / "resolved_config.json")
    write_json(folder / "execution.json", record)
    write_json(output / "execution.json", record)
    original_cwd = Path.cwd()
    previous_handler = None
    if threading.current_thread() is threading.main_thread():
        previous_handler = signal.getsignal(signal.SIGTERM)
        def cancelled(signum, _frame):
            raise SystemExit(128 + signum)
        signal.signal(signal.SIGTERM, cancelled)
    try:
        yield record
    except BaseException as error:
        record.update(status="interrupted" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "failed",
                      error_type=type(error).__name__, error=str(error))
        raise
    else:
        record["status"] = "complete"
    finally:
        if previous_handler is not None:
            signal.signal(signal.SIGTERM, previous_handler)
        # Upstream imports require its working directory. Do not leak that
        # process-global change to callers of the installed Python entry point.
        os.chdir(original_cwd)
        record["finished_utc"] = datetime.now(timezone.utc).isoformat()
        record["result_artifacts"] = results()
        record["new_or_updated_target_ids"] = [identifier for identifier, value in record["result_artifacts"].items()
                                                if initial_results.get(identifier) != value]
        record["artifacts"] = {name: sha256(output / name) for name in
                               ("run_config.json", "metric_config.json", "summary.json", "per_target.csv", "resolved_config.json")
                               if (output / name).is_file()}
        write_json(folder / "execution.json", record)
        write_json(output / "execution.json", record)


def verify_provenance(output, *, require_complete=True):
    output = Path(output)
    record = json.loads((output / "execution.json").read_text(encoding="utf-8"))
    identifier = record.get("id", "")
    if not identifier or Path(identifier).name != identifier or "/" in identifier or "\\" in identifier:
        raise ValueError("Invalid execution snapshot ID")
    folder = output / "provenance" / identifier
    if record != json.loads((folder / "execution.json").read_text(encoding="utf-8")):
        raise ValueError("Execution snapshot differs from the latest execution record")
    if record["status"] not in ("complete", "failed", "interrupted"):
        raise ValueError("Execution has not reached a recorded terminal status")
    if require_complete and record["status"] != "complete":
        raise ValueError(f"Latest execution is {record['status']}; finish the run before verification")
    source = json.loads((folder / "source.json").read_text(encoding="utf-8"))
    if (not source.get("files") or digest(source["files"]) != record["source_sha256"] or
            source.get("sha256") != record["source_sha256"]):
        raise ValueError("Source snapshot inventory changed")
    for relative, expected in source["files"].items():
        path = (folder / "source" / relative).resolve()
        if not path.is_relative_to((folder / "source").resolve()) or sha256(path) != expected:
            raise ValueError(f"Source snapshot missing or changed: {relative}")
    if sha256(folder / "environment.json") != record["environment_sha256"]:
        raise ValueError("Environment snapshot changed")
    if "resolved_config_sha256" in record:
        if sha256(folder / "resolved_config.json") != record["resolved_config_sha256"]:
            raise ValueError("Resolved configuration snapshot changed")
    for relative, expected in record["artifacts"].items():
        if relative not in ("run_config.json", "metric_config.json", "summary.json", "per_target.csv", "resolved_config.json"):
            raise ValueError(f"Unexpected execution artifact: {relative}")
        if sha256(output / relative) != expected:
            raise ValueError(f"Execution artifact changed: {relative}")
    for identifier, expected in record.get("result_artifacts", {}).items():
        path = (output / "targets" / identifier / "result.json").resolve()
        if not path.is_relative_to((output / "targets").resolve()) or sha256(path) != expected:
            raise ValueError(f"Execution result changed: {identifier}")
    return record

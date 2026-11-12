"""Validate the allocated runtime, score a smoke run, then verify/register a baseline."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path, PurePosixPath
import shlex
import shutil
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from amodal_benchmark.config import load_config
from amodal_benchmark.dataset import sha256, write_json
from amodal_benchmark.planning import slurm_command


GPU_CHECK = """
import json, torch
assert torch.cuda.device_count() >= 4, 'The baseline requires four allocated CUDA-visible GPUs'
devices = []
for index in range(4):
    with torch.cuda.device(index):
        value = torch.ones((32, 32), device=f'cuda:{index}')
        assert torch.allclose(value @ value, torch.full_like(value, 32))
        signal = torch.arange(16 * 64 * 64, dtype=torch.float32, device=f'cuda:{index}').reshape(1, 16, 64, 64)
        signal = signal / signal.numel()
        spectrum = torch.fft.rfftn(signal, dim=(-2, -1))
        reconstructed = torch.fft.irfftn(spectrum, s=signal.shape[-2:], dim=(-2, -1))
        assert torch.allclose(signal, reconstructed, atol=1e-5, rtol=1e-5)
        torch.cuda.synchronize()
        props = torch.cuda.get_device_properties(index)
        devices.append(dict(index=index, name=props.name, total_memory=props.total_memory,
                            capability=list(torch.cuda.get_device_capability(index))))
print(json.dumps(dict(torch=torch.__version__, cuda=torch.version.cuda, devices=devices), indent=2))
"""


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--smoke-output-dir")
    parser.add_argument("--benchmark-workers", type=int, choices=[4], default=4)
    parser.add_argument("--cpus", type=int, default=16)
    parser.add_argument("--run-only", action="store_true", help="Require the existing prepared cache")
    parser.add_argument("--plan", action="store_true", help="Print every stage without running anything")
    args = parser.parse_args(argv)
    if args.cpus < 4:
        parser.error("Four workers require at least four CPUs")
    args.smoke_output_dir = args.smoke_output_dir or args.output_dir.rstrip("/") + "-smoke"
    paths = [Path(args.output_dir).resolve(), Path(args.smoke_output_dir).resolve(),
             Path(args.cache_dir).resolve(), (ROOT / "datasets/open_world_amodal").resolve()]
    for index, path in enumerate(paths[:2]):
        for other in paths[index + 1:]:
            if path == other or path in other.parents or other in path.parents:
                parser.error("Full output, smoke output, dataset and cache must be separate directories")
    return args


def runner(config, output, args, *, setup=False):
    command = slurm_command(load_config(ROOT / config), config, output, args.cache_dir,
                            cpus=args.cpus, run_only=not setup)
    command = ["bash", "run_amodal_slurm.sh", "--local",
               *command[command.index("run_amodal_slurm.sh") + 1:]]
    if setup:
        command.append("--setup-only")
    return command


def steps(args):
    # Planning also runs from Windows; every executable path is for Linux.
    prefix = PurePosixPath(args.cache_dir) / "conda-env"
    checkout = str(PurePosixPath(args.cache_dir) / "amodal")
    python = str(prefix / "bin/python")
    data = "datasets/open_world_amodal"
    verify = [python, "-m", "amodal_benchmark", "verify", "--dataset-root", data,
              "--require-provenance"]
    # A prepared-cache run skips provisioning. The shared runner deliberately
    # rejects --setup-only combined with --run-only; later smoke/full launches
    # perform its normal run-only cache, checkpoint and package preflight.
    preparation = [] if args.run_only else [
        ("setup", runner("configs/baseline.json", args.output_dir, args, setup=True))]
    return preparation + [
        ("gpu-inventory", [python, "-c", GPU_CHECK]),
        ("cuda-extension", ["bash", "tests/validate_cuda_extension.sh", str(prefix), checkout]),
        ("pipeline-imports", ["bash", "tests/validate_pipeline_imports.sh", str(prefix), checkout]),
        ("data-preflight", [python, "evaluate_open_world.py", "--config", "configs/baseline.json",
                            "--output-dir", args.output_dir, "--check-only"]),
        ("smoke", runner("configs/baseline-gpu-smoke.json", args.smoke_output_dir, args)),
        ("verify-smoke", [*verify, "--run-dir", args.smoke_output_dir]),
        ("full", runner("configs/baseline.json", args.output_dir, args)),
        ("verify-full", [*verify, "--run-dir", args.output_dir, "--require-full"]),
        ("register", [python, "-m", "amodal_benchmark", "register", "--run-dir", args.output_dir,
                      "--dataset-root", data]),
    ]


def require_scored_completion(smoke_dir):
    folder = Path(smoke_dir)
    summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    overall = summary["statistics"]["Overall"]
    if not summary["evaluation_performed"] or overall["successes"] < 1 or overall["CLIP_n"] < 1:
        raise ValueError("Smoke did not generate and score a completion; full baseline was not started")
    # Failure penalties can populate aggregate appearance denominators. Require
    # a real successful target's cache to exercise all four learned/image metrics.
    for path in (folder / "targets").glob("*/*/metrics.json"):
        metrics = json.loads(path.read_text(encoding="utf-8"))["metrics"]
        if all(metrics.get(name) is not None for name in ("CLIP", "LPIPS", "Feature Similarity", "SSIM")):
            return
    raise ValueError("Smoke has no completion scored by all four metrics; full baseline was not started")


def run_stage(name, command, log, environment):
    print(f"Starting {name}; log: {log}", flush=True)
    with log.open("x", encoding="utf-8") as stream:
        stream.write(shlex.join(command) + "\n")
        stream.flush()
        # No wall-clock timeout: a valid full run may continue beyond the work session.
        result = subprocess.run(command, cwd=ROOT, env=environment, stdin=subprocess.DEVNULL,
                                stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"Stage {name} exited {result.returncode}; inspect {log}")


def run_workflow(args):
    import fcntl
    output = Path(args.output_dir)
    smoke = Path(args.smoke_output_dir)
    for path in (output, smoke):
        if (path / "targets").is_dir() and not (path / "execution.json").is_file():
            raise ValueError(f"Refusing to adopt historical targets without execution provenance: {path}")
        path.mkdir(parents=True, exist_ok=True)
    # Lock both destinations for the whole workflow, including its smoke stage.
    locks = []
    try:
        for path in sorted((output.resolve(), smoke.resolve())):
            lock = (path / ".baseline-workflow.lock").open("a")
            locks.append(lock)
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        identifier = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
        evidence = output / "workflow" / identifier
        evidence.mkdir(parents=True)
        source = {}
        for relative in ("scripts/run_baseline_slurm.sh", "scripts/run_baseline_workflow.py",
                         "tests/validate_cuda_extension.sh", "tests/validate_pipeline_imports.sh",
                         "configs/baseline.json", "configs/baseline-gpu-smoke.json"):
            target = evidence / "source" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / relative, target)
            source[relative] = sha256(target)
        record = dict(id=identifier, status="running", started_utc=datetime.now(timezone.utc).isoformat(),
                      slurm_job_id=os.environ.get("SLURM_JOB_ID"), source_sha256=source, stages=[])
        environment = os.environ.copy()
        environment.pop("PYTHONPATH", None)
        environment.pop("PYTHONHOME", None)
        environment.pop("AMODAL_ENV_DIR", None)
        environment.pop("AMODAL_REPO_DIR", None)
        environment.update(PYTHONNOUSERSITE="1", PYTHONUNBUFFERED="1", AMODAL_SCRIPT_DIR=str(ROOT))
        lib = str(Path(args.cache_dir).resolve() / "conda-env/lib")
        environment["LD_LIBRARY_PATH"] = lib + (":" + environment["LD_LIBRARY_PATH"] if environment.get("LD_LIBRARY_PATH") else "")
        try:
            for name, command in steps(args):
                stage = dict(name=name, command=command, status="running",
                             started_utc=datetime.now(timezone.utc).isoformat())
                record["stages"].append(stage)
                write_json(evidence / "workflow.json", record)
                log = evidence / f"{name}.log"
                run_stage(name, command, log, environment)
                if name == "verify-smoke":
                    require_scored_completion(smoke)
                stage.update(status="complete", finished_utc=datetime.now(timezone.utc).isoformat(),
                             log_sha256=sha256(log))
                write_json(evidence / "workflow.json", record)
            record["status"] = "complete"
        except BaseException as error:
            record.update(status="failed", error_type=type(error).__name__, error=str(error))
            if record["stages"]:
                record["stages"][-1]["status"] = "failed"
            raise
        finally:
            record["finished_utc"] = datetime.now(timezone.utc).isoformat()
            write_json(evidence / "workflow.json", record)
        print(f"Full baseline verified and registered: {output}; workflow evidence: {evidence}", flush=True)
    finally:
        for lock in locks:
            lock.close()


def main(argv=None):
    args = arguments(argv)
    if args.plan:
        for name, command in steps(args):
            print(f"# {name}\n{shlex.join(command)}")
        return 0
    if not os.environ.get("SLURM_JOB_ID"):
        raise ValueError("Submit scripts/run_baseline_slurm.sh inside a four-GPU Slurm allocation; --plan is safe locally")
    os.chdir(ROOT)
    run_workflow(args)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)

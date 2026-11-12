"""Build reviewable cluster commands without contacting a remote host."""
from pathlib import Path, PureWindowsPath
import shlex

from .config import evaluator_arguments


def slurm_command(config, config_path, output_dir, cache_dir, *, partition="debug-gpu",
                  cpus=16, memory="64G", time_limit="24:00:00", run_only=False,
                  reference_dir=None):
    if cpus < 1:
        raise ValueError("CPU count must be positive")
    workers = config["execution"]["workers"]
    if workers == "auto":
        raise ValueError("A Slurm plan needs an explicit execution.workers GPU allocation")
    paths = [str(config_path), output_dir, cache_dir, config["dataset"]["root"]]
    if reference_dir:
        paths.append(reference_dir)
    if any(PureWindowsPath(path).drive or "\\" in path for path in paths):
        raise ValueError("Cluster plans require Linux paths; use a project-relative config path")
    command = ["sbatch", f"--partition={partition}", f"--gres=gpu:{workers}",
               f"--cpus-per-task={cpus}", f"--mem={memory}", f"--time={time_limit}",
               "--output=logs/%x-%j.out", "run_amodal_slurm.sh",
               "--benchmark-config", str(config_path), "--output-dir", output_dir,
               "--cache-dir", cache_dir, "--cpus", str(cpus)]
    if run_only:
        command.append("--run-only")
    if reference_dir:
        command += ["--reference-dir", reference_dir]
    arguments = evaluator_arguments(config)
    shared = {"--dataset-root": "--benchmark-root", "--mc-timestep": "--mc-timestep",
              "--max-iter": "--max-iter", "--sd-model": "--sd-model"}
    for value in arguments:
        command.append(shared.get(value, "--benchmark-" + value[2:]) if value.startswith("--") else value)
    return command


def render_plan(command):
    return ("#!/usr/bin/env bash\nset -Eeuo pipefail\n"
            "# Run from your remote project directory after reviewing resources.\n"
            "mkdir -p -- logs\n" + shlex.join(command) + "\n")

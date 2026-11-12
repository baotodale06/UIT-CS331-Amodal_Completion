#!/usr/bin/env python3
"""Plan or launch boundary/matching controls with fixed visible references.

The default only writes a plan and shell commands. --execute validates the
existing dataset/references, then invokes the shared Slurm runner. No reference
generation is requested, even when the inference seed changes.
"""
import argparse
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys

from amodal_benchmark.dataset import SOURCES, write_json

EXPERIMENTS = {
    "boundary-released": ("released", "reference-best"),
    "matching-strict": ("paper", "reference"),
    "corrected": ("paper", "reference-best"),
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True,
                        help="Existing generated references, reused unchanged for every run")
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--plan-dir", type=Path, help="Default: a sibling of OUTPUT_ROOT named OUTPUT_ROOT-plan")
    parser.add_argument("--experiments", nargs="+", choices=EXPERIMENTS,
                        default=["boundary-released", "matching-strict"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--sources", default=",".join(SOURCES))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", default="auto")
    parser.add_argument("--mc-timestep", type=int, default=35)
    parser.add_argument("--max-iter", type=int, default=5)
    parser.add_argument("--partition")
    parser.add_argument("--gres", default="gpu:4")
    parser.add_argument("--cpus", type=int, default=16)
    parser.add_argument("--mem", default="64G")
    parser.add_argument("--time", default="24:00:00")
    parser.add_argument("--local", action="store_true", help="Run sequentially inside an existing GPU allocation")
    parser.add_argument("--execute", action="store_true", help="Validate data and invoke Slurm; otherwise only write the plan")
    args = parser.parse_args(argv)
    if min(args.seeds) < 0 or len(set(args.seeds)) != len(args.seeds):
        parser.error("--seeds must contain unique non-negative integers")
    if len(set(args.experiments)) != len(args.experiments):
        parser.error("--experiments must contain unique names")
    if args.workers != "auto" and (not args.workers.isascii() or not args.workers.isdigit() or int(args.workers) < 1):
        parser.error("--workers must be auto or a positive integer")
    if args.cpus < 1 or args.max_iter < 1 or not 0 <= args.mc_timestep <= 49 or args.limit is not None and args.limit < 1:
        parser.error("Require positive CPUs/iterations/limit and 0 <= mc-timestep <= 49")
    sources = args.sources.split(",")
    if len(set(sources)) != len(sources) or not sources or set(sources) - set(SOURCES):
        parser.error("--sources must list unique VG,COCOA,free_images,laion")
    return args


def build_plan(args):
    project = Path(__file__).resolve().parent
    runs = []
    for experiment in args.experiments:
        boundary, selection = EXPERIMENTS[experiment]
        for seed in args.seeds:
            output = args.output_root.resolve() / experiment / f"seed-{seed}"
            command = ["bash", str(project / "run_amodal_slurm.sh"),
                       "--benchmark-root", str(args.dataset_root.resolve()),
                       "--reference-dir", str(args.reference_dir.resolve()),
                       "--cache-dir", str(args.cache_dir.resolve()), "--output-dir", str(output),
                       "--benchmark-boundary-check", boundary, "--benchmark-target-selection", selection,
                       "--benchmark-seed", str(seed), "--benchmark-workers", args.workers,
                       "--benchmark-sources", args.sources, "--benchmark-metric-alignment", "released",
                       "--mc-timestep", str(args.mc_timestep), "--max-iter", str(args.max_iter),
                       "--gres", args.gres, "--cpus", str(args.cpus), "--mem", args.mem,
                       "--time", args.time, "--job-name", f"pdmc-{experiment}-s{seed}"]
            if args.partition:
                command += ["--partition", args.partition]
            if args.local:
                command.append("--local")
            if args.limit is not None:
                command += ["--benchmark-limit", str(args.limit)]
            runs.append({"experiment": experiment, "seed": seed, "boundary_check": boundary,
                         "target_selection": selection, "output_dir": str(output), "command": command,
                         "status": "planned"})
    return {"schema": 1, "project_dir": str(project), "reference_dir": str(args.reference_dir.resolve()),
            "reference_generation": False, "runs": runs,
            "notes": ["Local adapter controls, not a claimed author-supplied baseline adapter.",
                      "Keep the reference directory fixed across seeds; compare verified data identities after completion.",
                      "The shared runner submits Slurm jobs on a login node and runs directly inside an allocation."]}


def main(argv=None):
    args = parse_args(argv)
    plan = build_plan(args)
    if args.execute:
        if not args.reference_dir.is_dir():
            raise ValueError(f"Existing reference directory is required: {args.reference_dir}")
        if os.name == "nt":
            raise ValueError("Execute controls from Linux/WSL on the Slurm host or allocated GPU node")
        # Reuse the real evaluator preflight, including annotation/artifact
        # checksums and full image decoding, before submitting any jobs.
        from evaluate_open_world import parse_args as evaluation_args, preflight
        from amodal_benchmark.reproduction import data_identity
        check = evaluation_args(["--dataset-root", str(args.dataset_root), "--reference-dir", str(args.reference_dir),
                                 "--output-dir", plan["runs"][0]["output_dir"], "--sources", args.sources,
                                 *(["--limit", str(args.limit)] if args.limit is not None else [])])
        rows, _, audit = preflight(check)
        identified = [{**r, "input_sha256": audit["reference_records"][r["id"]]["input_sha256"]} for r in rows]
        plan["data_identity"] = data_identity(identified, audit)
    plan_dir = (args.plan_dir or args.output_root.with_name(args.output_root.name + "-plan")).resolve()
    write_json(plan_dir / "plan.json", plan)
    commands = [shlex.join(run["command"]) for run in plan["runs"]]
    shell = "#!/usr/bin/env bash\nset -Eeuo pipefail\n"
    shell += "# Re-enter the launcher for preflight and serial Slurm dependencies.\n"
    shell += "cd -- " + shlex.quote(str(Path.cwd())) + "\n"
    launcher = ["python3", str(Path(__file__).resolve()), *(sys.argv[1:] if argv is None else argv), "--execute"]
    shell += shlex.join(launcher) + "\n"
    (plan_dir / "commands.sh").write_text(shell, encoding="utf-8", newline="\n")
    print(f"Plan: {plan_dir / 'plan.json'}", flush=True)
    in_allocation = args.local or bool(os.environ.get("SLURM_JOB_ID"))
    previous_job = None
    for run, command in zip(plan["runs"], commands):
        print(command, flush=True)
        if args.execute:
            launch = list(run["command"])
            if not in_allocation:
                launch.append("--parsable")
                if previous_job is not None:
                    launch += ["--dependency", f"afterok:{previous_job}"]
            run["submission_command"] = launch
            run["status"] = "launching"
            write_json(plan_dir / "plan.json", plan)
            result = subprocess.run(launch, cwd=plan["project_dir"], capture_output=not in_allocation, text=True)
            run["returncode"] = result.returncode
            run["status"] = "failed" if result.returncode else "completed" if in_allocation else "submitted"
            if not in_allocation:
                print(result.stdout, end="", flush=True)
                print(result.stderr, end="", file=sys.stderr, flush=True)
                if result.returncode == 0:
                    job_ids = re.findall(r"(?m)^([1-9][0-9]*)(?:;[^\s]+)?\s*$", result.stdout)
                    if not job_ids:
                        run["status"] = "submission_job_id_unknown"
                        write_json(plan_dir / "plan.json", plan)
                        raise ValueError("Slurm accepted a submission but returned no parsable job ID; "
                                         "stopping before submitting other jobs sharing the cache")
                    previous_job = run["job_id"] = job_ids[-1]
            write_json(plan_dir / "plan.json", plan)
            if result.returncode:
                return result.returncode
    if not args.execute:
        print("Plan only; no inference or Slurm submission was performed.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, KeyError, OSError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)

"""Project workflow, with GPU imports confined to explicit evaluation commands."""
import argparse
import json
from pathlib import Path
import sys


def _main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    # Forward full argument handling to the established entry points.
    forwarding = {"run": "evaluate_open_world", "run-sam": "evaluate_pdmc_open_world",
                  "prepare": "prepare_open_world", "references": "prepare_visible_references",
                  "compare": "compare_open_world"}
    if argv and argv[0] in forwarding:
        import importlib
        return importlib.import_module(forwarding[argv[0]]).main(argv[1:])
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in forwarding:
        commands.add_parser(name, help=f"Forward to {forwarding[name]}.py")
    config = commands.add_parser("config", help="Validate and print a fully resolved experiment")
    config.add_argument("path", type=Path)
    plan = commands.add_parser("plan", help="Print a Slurm script; never submit it")
    plan.add_argument("--config", type=Path, required=True)
    plan.add_argument("--output-dir", required=True)
    plan.add_argument("--cache-dir", required=True)
    plan.add_argument("--reference-dir")
    plan.add_argument("--partition", default="debug-gpu")
    plan.add_argument("--cpus", type=int, default=16)
    plan.add_argument("--mem", default="64G")
    plan.add_argument("--time", default="24:00:00")
    plan.add_argument("--run-only", action="store_true")
    plan.add_argument("--output-script", type=Path)
    verify = commands.add_parser("verify", help="Verify saved results against original inputs on CPU")
    verify.add_argument("--run-dir", type=Path, required=True)
    verify.add_argument("--dataset-root", type=Path, required=True)
    verify.add_argument("--reference-dir", type=Path)
    verify.add_argument("--require-full", action="store_true")
    verify.add_argument("--require-provenance", action="store_true")
    register = commands.add_parser("register", help="Register a verified full run or recorded failed attempt")
    register.add_argument("--run-dir", type=Path, required=True)
    register.add_argument("--dataset-root", type=Path, required=True)
    register.add_argument("--reference-dir", type=Path)
    register.add_argument("--registry", type=Path, default=Path("experiments.csv"))
    register.add_argument("--hypothesis")
    register.add_argument("--baseline-dir", type=Path)
    register.add_argument("--baseline-reference-dir", type=Path)
    register.add_argument("--reason")
    smoke = commands.add_parser("smoke", help="Run a synthetic CPU pipeline and artifact verification")
    smoke.add_argument("--output-dir", type=Path, help="Retain debugging artifacts in a new directory")
    notes = commands.add_parser("notes", help="Initialize research notes without overwriting files")
    notes.add_argument("--directory", type=Path, default=Path("."))
    notes.add_argument("--project", default="open-world-amodal")
    args = parser.parse_args(argv)
    if args.command == "config":
        from .config import load_config
        print(json.dumps(load_config(args.path), indent=2))
    elif args.command == "plan":
        from .config import load_config
        from .planning import render_plan, slurm_command
        command = slurm_command(load_config(args.config), args.config.as_posix(), args.output_dir,
                                args.cache_dir, partition=args.partition, cpus=args.cpus,
                                memory=args.mem, time_limit=args.time, run_only=args.run_only,
                                reference_dir=args.reference_dir)
        script = render_plan(command)
        if args.output_script:
            # Exclusive creation prevents clobbering an edited submission script.
            with args.output_script.open("x", encoding="utf-8", newline="\n") as stream:
                stream.write(script)
        print(script, end="")
    elif args.command == "verify":
        from .verify import verify_run
        report = verify_run(args.run_dir, args.dataset_root, args.reference_dir,
                            require_full=args.require_full, require_provenance=args.require_provenance)
        print(json.dumps(report, indent=2))
    elif args.command == "register":
        from .registry import register_run
        entry = register_run(args.run_dir, args.dataset_root, args.registry, reference_dir=args.reference_dir,
                             hypothesis=args.hypothesis, baseline_dir=args.baseline_dir,
                             baseline_reference_dir=args.baseline_reference_dir, reason=args.reason)
        print(f"Registered {entry['status']} execution in {args.registry}")
    elif args.command == "smoke":
        from .smoke import smoke_check
        smoke_check(args.output_dir)
    elif args.command == "notes":
        from .knowledge import init_notes
        print("Created: " + ", ".join(init_notes(args.directory, args.project)))
    return 0


def main(argv=None):
    try:
        return _main(argv)
    except (ValueError, OSError, KeyError, RuntimeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2

"""Resume a timed-out L40S arm and keep its existing Slurm dependants queued."""
import argparse
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from amodal_benchmark.parallel import output_lock
from scripts.snapshot_shape_timeout import read

STUDY = Path("output/shape-20261121/progressive-l40s")


def after_either(run, recovery):
    return f"afterok:{run}?afterok:{recovery}"


def recovery_command(arm, job, children, dependency):
    wrapped = [".amodal-cache/surface-metrics-env/bin/python", "scripts/resume_shape_timeout.py",
               arm, job, *children]
    return ["sbatch", "--parsable", "-p", "cpu", "-c", "1", "--mem=1G", "--time=00:25:00",
            "--dependency=" + dependency, "--kill-on-invalid-dep=yes", "-J", "resume-" + arm,
            "-o", str(STUDY / f"resume-{arm}-%j.log"), "--wrap", shlex.join(wrapped)]


def submit(command):
    job = subprocess.check_output(command, cwd=ROOT, text=True).strip().split(";")[0]
    if not job.isdigit():
        raise ValueError(f"Unexpected sbatch response {job!r}")
    print(job, shlex.join(command), flush=True)
    return job


def resume(arm, job, children, controller):
    raw = subprocess.check_output(["sacct", "-XnP", "-j", job, "--format=JobIDRaw,State%32"], text=True)
    state = dict(line.split("|") for line in raw.splitlines())[job].split()[0]
    if state == "COMPLETED":
        print(f"Job {job} completed. Its downstream jobs can use those results.")
        return
    if state != "TIMEOUT":
        raise ValueError(f"Job {job} is {state}. Only a confirmed timeout can be resumed here.")
    study = ROOT / STUDY
    ledger = study / "submissions.tsv"
    with output_lock(study):
        original = ledger.read_text(encoding="utf8")
        entries = [row.split("\t", 3) for row in original.splitlines()]
        current = {role: (jid, command) for jid, role, stage, command in entries if stage == "full"}
        job_roles = ["audit-shape", "combined"] if arm == "shape" else ["audit-combined"]
        if current[arm][0] != job or children != [current[role][0] for role in job_roles]:
            raise ValueError("Recovery arguments differ from the current recorded job IDs")
        for child in children:
            info = subprocess.check_output(["scontrol", "show", "job", "-o", child], text=True)
            fields = dict(word.split("=", 1) for word in info.split() if "=" in word)
            if (fields["JobState"] != "PENDING" or fields["WorkDir"] != str(ROOT)
                    or controller not in re.findall(r"afterok:([0-9]+)", fields["Dependency"])):
                raise ValueError(f"Job {child} is no longer waiting for this recovery")

        subprocess.run([sys.executable, "scripts/snapshot_shape_timeout.py", arm, job], cwd=ROOT, check=True)
        receipt = study / f"{arm}-timeout-{job}.json"
        progress = read(receipt)
        for older in study.glob(f"{arm}-timeout-*.json"):
            if older != receipt:
                previous = read(older)
                if (len(previous["completed"]) >= len(progress["completed"])
                        and previous.get("cached_scores", -1) >= progress.get("cached_scores", 0)):
                    raise ValueError("Consecutive timeouts added no results or scores. Inspect the worker and scoring logs.")

        command = shlex.split(current[arm][1])
        expected_tail = ["scripts/run_shape_progressive.sh", "--benchmark-workers", "4", arm, "full"]
        if command[-5:] != expected_tail or "--gres=gpu:nvidia_l40s:4" not in command:
            raise ValueError("The recorded submission differs from the four-worker L40S experiment")
        command = [arg for arg in command if not arg.startswith("--dependency=")]
        command.insert(1, "--dependency=afterok:" + controller)
        retry = submit(command)
        follow = recovery_command(arm, retry, children, "afterany:" + retry)
        next_controller = submit(follow)
        # The GPU allocation can only start after this controller exits successfully.
        # Preserve both job IDs and commands before releasing any downstream work.
        additional = [(retry, arm, command), (next_controller, "resume-" + arm, follow)]
        temp = ledger.with_suffix(".tsv.tmp")
        temp.write_text(original.rstrip("\n") + "\n" + "".join(
            f"{jid}\t{role}\tfull\t{shlex.join(cmd)}\n" for jid, role, cmd in additional), encoding="utf8")
        temp.replace(ledger)
        for child in children:
            subprocess.run(["scontrol", "update", "JobId=" + child,
                            "Dependency=" + after_either(retry, next_controller)], check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arm", choices=["shape", "combined"])
    parser.add_argument("job")
    parser.add_argument("children", nargs="+")
    args = parser.parse_args()
    resume(args.arm, args.job, args.children, os.environ["SLURM_JOB_ID"])

"""Resume only infrastructure-interrupted frozen shape-only benchmark jobs."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from amodal_benchmark.dataset import sha256, write_json
from amodal_benchmark.parallel import output_lock

RESUMABLE = {"TIMEOUT", "PREEMPTED", "NODE_FAIL"}


def accounting(job):
    result = subprocess.check_output(["sacct", "-j", str(job), "-n", "-P",
        "--format=JobIDRaw,State,ExitCode,Elapsed,AllocTRES"], text=True)
    for line in result.splitlines():
        fields = line.split("|")
        if fields[0] == str(job):
            return dict(job_id=str(job), state=fields[1].split()[0].rstrip("+"),
                        exit_code=fields[2], elapsed=fields[3], allocation=fields[4])
    raise ValueError("Terminal job accounting is not yet available")


def verify_frozen_source(study):
    freeze = json.loads((study/"policy-freeze.json").read_bytes())
    if sha256(study/"evaluation-source-manifest.json") != freeze["evaluation_source_manifest_sha256"]:
        raise ValueError("Evaluation source manifest changed after policy freezing")
    receipt = json.loads((study/"evaluation-source-manifest.json").read_bytes())
    for name, expected in receipt["files"].items():
        if sha256(study/"evaluation-source"/name) != expected:
            raise ValueError("Frozen evaluation source changed before resume: " + name)


def submit(command, cwd):
    job = subprocess.check_output(command, cwd=cwd, text=True).strip().split(";")[0]
    if not job.isdigit():
        raise ValueError("Unrecognized sbatch response")
    return job


def main(method="shape-v2"):
    if method not in ("shape-v2", "shape-calibrated"):
        raise ValueError("Unknown continuation method")
    runner = "scripts/run_" + method.replace("-", "_") + "_evaluation.sh"
    controller_runner = "scripts/run_" + method.replace("-", "_") + "_continuation.sh"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--max-resumes", type=int, default=4)
    args = parser.parse_args()
    study = args.study.resolve()
    (study/"continuation-control").mkdir(parents=True, exist_ok=True)
    with output_lock(study/"continuation-control"):
        path = study/"continuations.json"
        log = json.loads(path.read_bytes()) if path.exists() else dict(schema=1, events=[])
        for row in log["events"]:
            if row["previous"]["job_id"] == args.job_id:
                if row.get("action") == "resumed" and row.get("controller_command") and not row.get("controller_job_id"):
                    row["controller_job_id"] = submit(row["controller_command"], study/"evaluation-source")
                    write_json(path, log)
                print("This job already has a recorded continuation decision", flush=True)
                return
        previous = accounting(args.job_id)
        event = dict(at=datetime.now(timezone.utc).isoformat(), previous=previous)
        if previous["state"] == "COMPLETED":
            audit = json.loads((study/"full-audit.json").read_bytes())
            if audit.get("verified") is not True or audit.get("targets") != 2565:
                raise ValueError("Completed benchmark job has no full verified audit")
            event.update(action="finished", four_metric_gate=audit["gate"]["eligible_for_success_claim"])
        elif previous["state"] in RESUMABLE:
            if sum(row.get("action") == "resumed" for row in log["events"]) >= args.max_resumes:
                event.update(action="needs_review", reason="Configured continuation limit reached")
            else:
                verify_frozen_source(study)
                if method == "shape-calibrated":
                    retained = {p.relative_to(study/"benchmark").as_posix():sha256(p)
                        for record in (study/"benchmark/targets").glob("**/result.json")
                        for p in record.parent.rglob("*") if p.is_file()}
                else:
                    retained = {p.relative_to(study/"benchmark").as_posix():sha256(p)
                        for pattern in ("targets/**/result.json", "targets/**/metrics.json")
                        for p in (study/"benchmark").glob(pattern)}
                inventory = study/f"preserved-before-resume-{args.job_id}.json"
                write_json(inventory, dict(previous=previous, files=retained,
                    freeze_sha256=sha256(study/"policy-freeze.json"),
                    run_config_sha256=sha256(study/"benchmark/run_config.json")))
                command = ["sbatch", "--parsable", "--partition=debug-gpu", "--gres=gpu:quadro_rtx_6000:4",
                    *(["--nodelist=gebru"] if method == "shape-calibrated" else []),
                    "--cpus-per-task=16", "--mem=96G", "--time=24:00:00", "--job-name=" + method + "-full",
                    f"--output={study}/full-%j.log", runner, "--benchmark-workers", "4", "full"]
                next_job = submit(command, study/"evaluation-source")
                event.update(action="resumed", next_job_id=next_job, command=command,
                             preserved_inventory_sha256=sha256(inventory), retained_records=len(retained))
                # Persist the submitted GPU job before scheduling its controller.
                log["events"].append(event)
                write_json(path, log)
                controller = ["sbatch", "--parsable", "--partition=debug-cpu", "--cpus-per-task=2", "--mem=8G",
                    "--time=00:10:00", "--job-name=" + method + "-resume", f"--dependency=afterany:{next_job}",
                    f"--output={study}/continuation-%j.log", controller_runner,
                    "--study", str(study), "--job-id", next_job, "--max-resumes", str(args.max_resumes)]
                event["controller_command"] = controller
                write_json(path, log)
                event["controller_job_id"] = submit(controller, study/"evaluation-source")
                write_json(path, log)
                print(json.dumps(event), flush=True)
                return
        else:
            event.update(action="needs_review", reason="No automatic retry for application failures or cancellation")
        log["events"].append(event)
        write_json(path, log)
        print(json.dumps(event), flush=True)


if __name__ == "__main__":
    main()

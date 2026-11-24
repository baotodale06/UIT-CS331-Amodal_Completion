"""Recover versioned shape-v2 evidence without changing or resubmitting jobs."""
import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from amodal_benchmark.dataset import sha256, write_json

HOST = "daoqm@slurm.csse.rose-hulman.edu"
REMOTE_STUDY = "/home/daoqm/Vu_Hung/cv_nangcao/output/shape-v2-slurm-20261123"
REMOTE = r'''
import base64, hashlib, json, os, subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
study = Path("/home/daoqm/Vu_Hung/cv_nangcao/output/shape-v2-slurm-20261123")
os.environ["TZ"] = "UTC"
jobs = json.loads((study/"evaluation-jobs.json").read_bytes())
roles = {str(job): role for role, job in jobs.items()}
roles.update({"45631": "preflight", "45633": "runtime_check"})
continuations = study/"continuations.json"
if continuations.exists():
    for event in json.loads(continuations.read_bytes())["events"]:
        for key, role in (("next_job_id", "full"), ("controller_job_id", "controller")):
            if event.get(key):
                roles[str(event[key])] = role
finalization_jobs = study/"finalization-jobs.json"
if finalization_jobs.exists():
    roles[str(json.loads(finalization_jobs.read_bytes())["initial_job_id"])] = "finalization"
backfill_job = study/"backfill-job.json"
if backfill_job.exists():
    roles[str(json.loads(backfill_job.read_bytes())["job_id"])] = "backfill_amendment"
for path in (study/"finalization-control").glob("*.json"):
    event = json.loads(path.read_bytes())
    if event.get("follow_job_id"):
        roles[str(event["follow_job_id"])] = "finalization"
command = ["sacct", "-X", "-j", ",".join(sorted(roles)), "-P",
           "--format=JobIDRaw,State%32,ExitCode,ElapsedRaw,AllocTRES%120,NodeList,Start,End"]
accounting = subprocess.check_output(command, text=True)
queue_command = ["squeue", "-j", ",".join(sorted(roles)), "-h", "-o", "%i|%T|%M|%l|%R"]
queue = subprocess.check_output(queue_command, text=True)
files = {}
names = ["training/training.json", "validation-native.json", "holdout-native.json",
         "policy-freeze.json", "references.json", "smoke-manifest.json", "smoke-validation.json",
         "full-verification.json", "full-audit.json", "registry.csv", "continuations.json",
         "evaluation-jobs.json", "evaluation-source-manifest.json", "finalization-jobs.json",
         "metadata-export.json", "postprocessing-source-manifest.json", "operational-source-manifest.json"]
names += ["unseen-inventory.json", "backfill-job.json", "backfill-result.json", "smoke-native-export.json"]
for stage in ("smoke", "benchmark"):
    names += [stage+"/"+name for name in ("execution.json", "run_config.json", "metric_config.json")]
    execution = study/stage/"execution.json"
    if execution.exists() and json.loads(execution.read_bytes()).get("status") == "complete":
        names += [stage+"/summary.json", stage+"/per_target.csv"]
names += [p.name for p in study.glob("preserved-before-resume-*.json")]
names += [p.relative_to(study).as_posix() for p in (study/"finalization-control").glob("*.json")]
for name in names:
    path = study/name
    if path.is_file():
        data = path.read_bytes()
        files[name] = dict(sha256=hashlib.sha256(data).hexdigest(), bytes=len(data),
                           base64=base64.b64encode(data).decode())
logs = {}
for path in sorted(study.glob("*-*.log")):
    with path.open("rb") as stream:
        size = path.stat().st_size
        stream.seek(max(0, size-65536))
        data = stream.read(65536)
    logs[path.name] = dict(bytes_observed=size, tail=data.decode("utf8", errors="replace"))
progress = {}
for stage in ("smoke", "benchmark"):
    counts = Counter()
    for path in (study/stage/"targets").glob("**/result.json"):
        counts[json.loads(path.read_bytes())["status"]] += 1
    progress[stage] = dict(records=sum(counts.values()), statuses=dict(counts))
checkpoint_files = {}
training = json.loads((study/"training/training.json").read_bytes())
if training.get("complete") is True:
    for name, key in (("best.pt", "best_sha256"), ("last.pt", "last_sha256")):
        checkpoint_files[name] = dict(sha256=training[key], bytes=(study/"training"/name).stat().st_size)
print(json.dumps(dict(at=datetime.now(timezone.utc).isoformat(), files=files, logs=logs,
    accounting=accounting, accounting_command=command, queue=queue, queue_command=queue_command,
    roles=roles, progress=progress, checkpoint_files=checkpoint_files)))
'''


def allocation_cost(accounting, roles):
    """Count allocations once, including interrupted and failed work, not steps."""
    lines = [line.split("|") for line in accounting.splitlines() if line.strip()]
    header, records = lines[0], lines[1:]
    jobs, totals = {}, {}
    for fields in records:
        row = dict(zip(header, fields))
        job = row["JobIDRaw"]
        if job not in roles or job in jobs:
            raise ValueError("Unexpected or repeated allocation: " + job)
        tres = dict(item.split("=", 1) for item in row["AllocTRES"].split(",") if "=" in item)
        gpus = int(tres.get("gres/gpu", 0))
        if not gpus:
            gpus = sum(int(value) for key, value in tres.items() if key.startswith("gres/gpu:"))
        seconds = int(row["ElapsedRaw"])
        role = roles[job]
        gpu_hours = gpus*seconds/3600
        jobs[job] = dict(role=role, state=row["State"], exit_code=row["ExitCode"],
                         seconds=seconds, gpus=gpus, reserved_gpu_hours=gpu_hours,
                         nodes=row["NodeList"], start=row["Start"], end=row["End"])
        totals[role] = totals.get(role, 0.) + gpu_hours
    if jobs.keys() != roles.keys():
        raise ValueError("Accounting does not cover every tracked job")
    return dict(jobs=jobs, reserved_gpu_hours_by_role=totals,
                total_reserved_gpu_hours=sum(totals.values()),
                final=all(row["state"].split()[0].rstrip("+") in
                    {"COMPLETED", "FAILED", "TIMEOUT", "CANCELLED", "NODE_FAIL", "OUT_OF_MEMORY", "PREEMPTED"}
                    for row in jobs.values()),
                scope="Elapsed allocated GPU-hours, including failures and interruptions; not GPU utilization. Running totals are provisional.")


def save_member(destination, name, record):
    relative = PurePosixPath(name)
    if relative.is_absolute() or ".." in relative.parts or "\\" in name or ":" in name:
        raise ValueError("Unexpected recovery path")
    data = base64.b64decode(record["base64"], validate=True)
    if len(data) != record["bytes"] or hashlib.sha256(data).hexdigest() != record["sha256"]:
        raise ValueError("Recovery checksum differs: " + name)
    path = destination.joinpath(*relative.parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(data)
    return {key: record[key] for key in ("bytes", "sha256")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=ROOT/"output/shape-v2-slurm-20261123")
    parser.add_argument("--checkpoints", action="store_true")
    args = parser.parse_args()
    command = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", "-o", "ServerAliveInterval=30", HOST, "python3 -"]
    response = subprocess.run(command, input=REMOTE, text=True, capture_output=True, timeout=120, check=True)
    payload = json.loads(response.stdout)
    destination = args.study/"recovery"/datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    members = {name:save_member(destination, name, record) for name, record in payload.pop("files").items()}
    cost = allocation_cost(payload["accounting"], payload["roles"])
    write_json(destination/"allocation-cost.json", cost)
    checkpoints = {}
    if args.checkpoints:
        training_jobs = [row for row in cost["jobs"].values() if row["role"] == "training"]
        if not training_jobs or any(row["state"] != "COMPLETED" or row["exit_code"] != "0:0" for row in training_jobs):
            raise ValueError("Checkpoint recovery requires completed successful training")
        if set(payload["checkpoint_files"]) != {"best.pt", "last.pt"}:
            raise ValueError("Completed training checkpoints are unavailable")
        folder = args.study/"training"
        folder.mkdir(parents=True, exist_ok=True)
        for name, record in payload["checkpoint_files"].items():
            target = folder/name
            if not target.exists():
                with tempfile.TemporaryDirectory(dir=folder) as temporary:
                    staged = Path(temporary)/name
                    subprocess.run(["scp", "-q", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20",
                        HOST+":"+REMOTE_STUDY+"/training/"+name, str(staged)], check=True, timeout=1800)
                    if staged.stat().st_size != record["bytes"] or sha256(staged) != record["sha256"]:
                        raise ValueError("Checkpoint download checksum differs")
                    staged.rename(target)
            if target.stat().st_size != record["bytes"] or sha256(target) != record["sha256"]:
                raise ValueError("Preserve differing local checkpoint")
            checkpoints[name] = record
        data = (destination/"training/training.json").read_bytes()
        target = folder/"training.json"
        if target.exists() and target.read_bytes() != data:
            raise ValueError("Preserve differing local completed training receipt")
        if not target.exists():
            target.write_bytes(data)
    write_json(destination/"receipt.json", dict(**payload, members=members, checkpoints=checkpoints,
        source_sha256=sha256(Path(__file__)), ssh_command=command, remote_program=REMOTE))
    write_json(args.study/"latest-recovery.json", dict(snapshot=destination.relative_to(args.study).as_posix(),
        receipt_sha256=sha256(destination/"receipt.json")))
    print(json.dumps(dict(snapshot=str(destination), progress=payload["progress"], jobs=cost["jobs"],
                         recovered_checkpoints=list(checkpoints)), indent=2))


if __name__ == "__main__":
    main()

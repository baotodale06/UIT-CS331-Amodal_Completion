"""Recover and replay the frozen full shape-v2 export, optionally watching Slurm."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile
from tempfile import TemporaryDirectory
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from amodal_benchmark.dataset import read_official, sha256, write_json
from amodal_benchmark.parallel import output_lock
from scripts.collect_surface_job_evidence import recover_archive
from scripts.recover_shape_v2 import HOST, REMOTE_STUDY
from scripts.verify_shape_v2_metadata import reconcile

STUDY = ROOT / "output/shape-v2-slurm-20261123"
TERMINAL = {"COMPLETED", "FAILED", "TIMEOUT", "CANCELLED", "NODE_FAIL", "OUT_OF_MEMORY", "PREEMPTED"}
PROBE = r'''
import json, os, subprocess
from collections import Counter
from pathlib import Path
study = Path("/home/daoqm/Vu_Hung/cv_nangcao/output/shape-v2-slurm-20261123")
os.environ["TZ"] = "UTC"
ids = set(json.loads((study/"evaluation-jobs.json").read_bytes()).values())
finalization = json.loads((study/"finalization-jobs.json").read_bytes())
ids.add(finalization["initial_job_id"])
if (study/"backfill-job.json").exists():
    ids.add(json.loads((study/"backfill-job.json").read_bytes())["job_id"])
for name in ("continuations.json",):
    if (study/name).exists():
        for event in json.loads((study/name).read_bytes())["events"]:
            for key in ("next_job_id", "controller_job_id"):
                if event.get(key):
                    ids.add(event[key])
for path in (study/"finalization-control").glob("*.json"):
    event = json.loads(path.read_bytes())
    if event.get("follow_job_id"):
        ids.add(event["follow_job_id"])
raw = subprocess.check_output(["sacct", "-X", "-n", "-P", "-j", ",".join(sorted(ids)),
                              "--format=JobIDRaw,State%32,ExitCode"], text=True)
jobs = {line.split("|")[0]:line.split("|")[1:3] for line in raw.splitlines() if line.strip()}
assert jobs.keys() == ids, "Incomplete scheduler accounting; do not infer termination"
counts = Counter(json.loads(p.read_bytes())["status"] for p in (study/"benchmark/targets").glob("**/result.json"))
print(json.dumps(dict(jobs=jobs, counts=dict(counts), export_ready=(study/"metadata-export.json").is_file())))
'''


def disposition(status):
    states = [row[0].split()[0].rstrip("+") for row in status["jobs"].values()]
    if not states or any(state not in TERMINAL for state in states):
        return "waiting"
    return "ready" if status["export_ready"] else "needs_review"


def recover_full(snapshot):
    receipt = json.loads((snapshot/"metadata-export.json").read_bytes())
    freeze = snapshot/"policy-freeze.json"
    if (receipt["targets"] != 2565 or receipt["freeze_sha256"] != sha256(freeze)
            or receipt["audit_sha256"] != sha256(snapshot/"full-audit.json")
            or receipt["postprocessing_manifest_sha256"] != sha256(snapshot/"postprocessing-source-manifest.json")
            or receipt["archive"] != REMOTE_STUDY+"/benchmark-metadata.tar.gz"):
        raise ValueError("Final archive receipt differs from recovered frozen audit")
    archive = STUDY/"benchmark-metadata.tar.gz"
    if not archive.exists():
        with TemporaryDirectory(dir=STUDY) as temporary:
            staged = Path(temporary)/archive.name
            subprocess.run(["scp", "-q", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20",
                            HOST+":"+receipt["archive"], str(staged)], check=True, timeout=1800)
            if sha256(staged) != receipt["sha256"]:
                raise ValueError("Downloaded benchmark metadata changed")
            staged.rename(archive)
    if sha256(archive) != receipt["sha256"]:
        raise ValueError("Preserve differing local metadata archive")
    with tarfile.open(archive, "r:gz") as bundle:
        manifest = bundle.extractfile("member-sha256.json").read()
    members = json.loads(manifest)
    if (len(members) != receipt["files"] or "member-sha256.json" in members
            or members["execution.json"] != sha256(snapshot/"benchmark/execution.json")):
        raise ValueError("Full export inventory disagrees with recovered execution")
    members["member-sha256.json"] = hashlib.sha256(manifest).hexdigest()
    method = STUDY/"benchmark-metadata"
    recover_archive(archive, method, ["execution.json", "run_config.json", "metric_config.json",
        "resolved_config.json", "summary.json", "per_target.csv", "provenance", "targets", "member-sha256.json"], members)
    official, _ = read_official(ROOT/"datasets/open_world_amodal")
    result = reconcile(method, ROOT/"output/surface-annotated-20261112-freshrefs/baseline", snapshot,
                       official, snapshot/"unseen-inventory.json")
    if result["method_execution"] != receipt["execution"]:
        raise ValueError("Local replay execution differs from exported execution")
    destination = STUDY/"full-metadata-replay.json"
    if destination.exists() and json.loads(destination.read_bytes()) != result:
        raise ValueError("Preserve differing existing full replay")
    write_json(destination, result)
    write_json(STUDY/"final-recovery.json", dict(snapshot=snapshot.relative_to(STUDY).as_posix(),
        replay_sha256=sha256(destination), archive_sha256=receipt["sha256"],
        allocation_final=json.loads((snapshot/"allocation-cost.json").read_bytes())["final"]))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=int, default=60)
    args = parser.parse_args()
    if args.interval < 30:
        raise ValueError("Polling interval must be at least 30 seconds")
    lock = STUDY/"collection-control"
    lock.mkdir(parents=True, exist_ok=True)
    with output_lock(lock):
        while True:
            try:
                response = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", HOST, "python3 -"],
                    input=PROBE, text=True, capture_output=True, timeout=90, check=True)
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
                print(json.dumps(dict(at=datetime.now(timezone.utc).isoformat(), observation_error=str(error))), flush=True)
                if not args.watch:
                    raise
                time.sleep(args.interval)
                continue
            status = json.loads(response.stdout)
            status["at"] = datetime.now(timezone.utc).isoformat()
            status["disposition"] = disposition(status)
            write_json(STUDY/"collection-status.json", status)
            print(json.dumps(status), flush=True)
            if status["disposition"] != "waiting":
                subprocess.run([sys.executable, str(ROOT/"scripts/recover_shape_v2.py")], check=True, timeout=180)
                pointer = json.loads((STUDY/"latest-recovery.json").read_bytes())
                if status["disposition"] == "needs_review":
                    raise RuntimeError("The job chain ended without a completed export; inspect recovered logs")
                result = recover_full(STUDY/pointer["snapshot"])
                print(json.dumps(dict(verified=result["verified"], gate=result["gate"])), flush=True)
                return
            if not args.watch:
                return
            time.sleep(args.interval)


if __name__ == "__main__":
    main()

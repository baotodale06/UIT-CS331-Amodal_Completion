"""Recover a terminal calibrated export, independently replay it, then report it."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from amodal_benchmark.dataset import sha256, write_json
from scripts.collect_surface_job_evidence import recover_archive
from scripts.recover_shape_v2 import save_member, allocation_cost

HOST = "daoqm@slurm.csse.rose-hulman.edu"
REMOTE_STUDY = "/home/daoqm/Vu_Hung/cv_nangcao/output/shape-calibrated-benchmark-20261123"
TERMINAL = {"COMPLETED", "FAILED", "TIMEOUT", "CANCELLED", "NODE_FAIL", "PREEMPTED", "OUT_OF_MEMORY", "BOOT_FAIL", "DEADLINE"}


def disposition(report):
    if not report["jobs"] or set(report["jobs"].values()) != set(report["accounting"]):
        raise ValueError("Complete job accounting is required")
    states = {job: row[0].split()[0].rstrip("+") for job, row in report["accounting"].items()}
    if any(states[report["jobs"][role]] in TERMINAL - {"COMPLETED"}
           for role in ("native", "preflight", "smoke") if role in report["jobs"]):
        return "needs_review"
    if any(state not in TERMINAL for state in states.values()):
        return "waiting"
    return "export_ready" if report.get("export_ready") and report.get("audit_ready") else "needs_review"


PROBE = r'''
import base64, hashlib, json, os, subprocess
from pathlib import Path
study = Path('/home/daoqm/Vu_Hung/cv_nangcao/output/shape-calibrated-benchmark-20261123')
ledger = json.loads((study / 'evaluation-jobs.json').read_bytes())
roles = {job: role for role, job in ledger['jobs'].items()}
roles.update({'45680': 'native_validation', '45632': 'inherited_v2_training',
              '45662': 'inherited_robust_preflight', '45663': 'inherited_robust_training'})
finalizers = json.loads((study / 'finalization-jobs.json').read_bytes())
roles[finalizers['initial_job_id']] = 'finalization'
for event in json.loads((study / 'continuations.json').read_bytes())['events']:
    for key, role in (('next_job_id', 'full'), ('controller_job_id', 'controller')):
        if event.get(key):
            roles[event[key]] = role
for path in (study / 'finalization-control').glob('*.json'):
    event = json.loads(path.read_bytes())
    if event.get('follow_job_id'):
        roles[event['follow_job_id']] = 'finalization'
accounting = subprocess.check_output(['sacct', '-X', '-j', ','.join(sorted(roles)), '-P',
    '--format=JobIDRaw,State%32,ExitCode,ElapsedRaw,AllocTRES%120,NodeList,Start,End'],
    text=True, env=dict(os.environ, TZ='UTC'))
names = ['policy-freeze.json', 'full-audit.json', 'full-verification.json', 'registry.csv', 'holdout-native.json',
    'pipeline-validation.json', 'smoke-validation.json', 'unseen-inventory.json', 'metadata-export.json',
    'evaluation-source-manifest.json', 'postprocessing-source-manifest.json', 'operational-source-manifest.json',
    'evaluation-jobs.json', 'finalization-jobs.json', 'continuations.json', 'benchmark/execution.json',
    'admitted/admission.json', 'admitted/native-replay.json', 'admitted/local-contract.json',
    'admitted/local-result.json', 'admitted/quadro-contract.json', 'admitted/quadro-result.json',
    'admitted/candidate/candidate.json', 'cuda-validation.log', 'import-validation.log']
names += [p.relative_to(study).as_posix() for p in (study / 'finalization-control').glob('*.json')]
files = {}
for name in names:
    data = (study / name).read_bytes()
    files[name] = dict(sha256=hashlib.sha256(data).hexdigest(), bytes=len(data), base64=base64.b64encode(data).decode())
logs = {}
for path in study.glob('*-*.log'):
    with path.open('rb') as stream:
        stream.seek(max(0, path.stat().st_size - 65536))
        logs[path.name] = stream.read().decode('utf8', errors='replace')
print(json.dumps(dict(files=files, roles=roles, accounting=accounting, logs=logs)))
'''


def validate_export(receipt, snapshot):
    if (receipt.get("targets") != 2565 or receipt.get("archive") != REMOTE_STUDY + "/benchmark-metadata.tar.gz"
            or receipt.get("freeze_sha256") != sha256(snapshot / "policy-freeze.json")
            or receipt.get("audit_sha256") != sha256(snapshot / "full-audit.json")
            or receipt.get("postprocessing_manifest_sha256") != sha256(snapshot / "postprocessing-source-manifest.json")):
        raise ValueError("Export is not bound to the complete frozen calibrated audit")


def complete_candidate(snapshot, project):
    candidate = project / "output/shape-calibrated-20261123/candidate"
    target = snapshot / "admitted/candidate"
    if (candidate / "candidate.json").read_bytes() != (target / "candidate.json").read_bytes():
        raise ValueError("Recovered admitted candidate differs from the frozen local candidate")
    policy = json.loads((target / "candidate.json").read_bytes())
    expected = {"best.pt", "training.json", "diagnostic-replay.json", "selected-native.json", "selected-masks.zip"}
    if set(policy["files"]) != expected:
        raise ValueError("Candidate artifact inventory changed")
    for name, checksum in policy["files"].items():
        if sha256(candidate / name) != checksum:
            raise ValueError("Local frozen candidate changed")
        if (target / name).exists():
            if sha256(target / name) != checksum:
                raise ValueError("Existing recovered candidate differs")
        else:
            shutil.copyfile(candidate / name, target / name)


def recover(study, project):
    study, project = study.resolve(), project.resolve()
    completed = study / "final-recovery.json"
    if completed.exists():
        final = json.loads(completed.read_bytes())
        if sha256(study / "full-metadata-replay.json") != final["replay_sha256"]:
            raise ValueError("Existing independent replay changed")
        return final
    result = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", HOST, "python3 -"],
                            input=PROBE.encode(), capture_output=True, timeout=180, check=True)
    payload = json.loads(result.stdout)
    cost = allocation_cost(payload["accounting"], payload["roles"])
    if not cost["final"]:
        raise ValueError("Wait for terminal accounting of the entire calibrated export chain")
    cost["scope"] += (" Tracks calibrated native validation, pipeline checks, smoke, full benchmark, controllers and export, "
        "plus inherited v2 training and robust-adaptation preflight/training. Other exploratory Slurm work and local/Kaggle "
        "training are excluded; this is not a total of all research compute.")
    snapshot = study / "recovery" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    members = {name: save_member(snapshot, name, item) for name, item in payload.pop("files").items()}
    write_json(snapshot / "allocation-cost.json", cost)
    write_json(snapshot / "recovery.json", dict(**payload, files=members))
    receipt = json.loads((snapshot / "metadata-export.json").read_bytes())
    validate_export(receipt, snapshot)
    post_manifest = json.loads((snapshot / "postprocessing-source-manifest.json").read_bytes())
    if post_manifest["base_evaluation_manifest_sha256"] != sha256(snapshot / "evaluation-source-manifest.json"):
        raise ValueError("Postprocessing source does not match the inference snapshot")
    source = study / "postprocessing-source"
    for name, expected in post_manifest["files"].items():
        path = (source / name).resolve()
        if not path.is_relative_to(source.resolve()) or sha256(path) != expected:
            raise ValueError("Local postprocessing source differs: " + name)
    complete_candidate(snapshot, project)
    archive = study / "benchmark-metadata.tar.gz"
    if not archive.exists():
        with TemporaryDirectory(dir=study) as temporary:
            partial = Path(temporary) / archive.name
            subprocess.run(["scp", "-q", "-o", "BatchMode=yes", HOST + ":" + receipt["archive"], str(partial)],
                           check=True, timeout=1800)
            if sha256(partial) != receipt["sha256"]:
                raise ValueError("Downloaded metadata archive differs")
            partial.rename(archive)
    if sha256(archive) != receipt["sha256"]:
        raise ValueError("Preserve differing existing metadata archive")
    with tarfile.open(archive, "r:gz") as bundle:
        packed = bundle.extractfile("member-sha256.json").read()
    inventory = json.loads(packed)
    if (len(inventory) != receipt["files"] or "member-sha256.json" in inventory
            or inventory["execution.json"] != sha256(snapshot / "benchmark/execution.json")):
        raise ValueError("Archive membership differs from the closed benchmark")
    inventory["member-sha256.json"] = hashlib.sha256(packed).hexdigest()
    method = study / "benchmark-metadata"
    recover_archive(archive, method, ["execution.json", "run_config.json", "metric_config.json", "resolved_config.json",
        "summary.json", "per_target.csv", "provenance", "targets", "member-sha256.json"], inventory)
    replay_path = study / "full-metadata-replay.json"
    # Run the immutable verifier; future workspace edits cannot change this replay.
    if not replay_path.exists():
        subprocess.run([sys.executable, str(source / "scripts/verify_shape_calibrated_metadata.py"),
            "--method", str(method), "--control", str(project / "output/surface-annotated-20261112-freshrefs/baseline"),
            "--snapshot", str(snapshot), "--dataset-root", str(project / "datasets/open_world_amodal"),
            "--unseen-inventory", str(snapshot / "unseen-inventory.json"),
            "--native-data", str(project / "output/shape-v2-20261123/data"),
            "--sam-validation", str(project / "output/shape-validation-sam-20261123/native/result"),
            "--geometry-dir", str(project / "output/shape-20261121/evaluation_only/cocoa"),
            "--output", str(replay_path)], check=True, timeout=1800)
    replay = json.loads(replay_path.read_bytes())
    if (replay.get("verified") is not True or replay.get("targets") != 2565
            or replay["method_execution"] != receipt["execution"]
            or replay["freeze_sha256"] != sha256(snapshot / "policy-freeze.json")
            or replay["native_audit_sha256"] != sha256(snapshot / "full-audit.json")):
        raise ValueError("Replay does not match the completed export")
    final = dict(snapshot=snapshot.relative_to(study).as_posix(), replay_sha256=sha256(replay_path),
                 archive_sha256=receipt["sha256"], allocation_final=True)
    write_json(completed, final)
    return final


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()
    final = recover(args.study, args.project_root)
    source = args.study / "postprocessing-source"
    if not (args.study / "report").exists():
        subprocess.run([sys.executable, str(source / "scripts/build_shape_v2_report.py"), "--study", str(args.study)],
                       check=True, timeout=300)
    report = json.loads((args.study / "report/report-receipt.json").read_bytes())
    if report["replay_sha256"] != final["replay_sha256"]:
        raise ValueError("Final report is not bound to the recovered replay")
    for name, expected in report["files"].items():
        if Path(name).name != name or sha256(args.study / "report" / name) != expected:
            raise ValueError("Final report artifact changed")
    print(json.dumps(final), flush=True)


if __name__ == "__main__":
    main()

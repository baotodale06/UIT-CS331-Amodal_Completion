"""Recover audited Shape-PMC table updates while the L40S study runs."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from amodal_benchmark.dataset import write_json
from scripts.replay_shape_outlines import replay
from scripts.replay_shape_reference_intervals import check_estimate
from scripts.replay_shape_slide_tables import replay_full_slide_tables
from scripts.verify_shape_metadata import verify_collected

HOST="daoqm@slurm.csse.rose-hulman.edu"
REMOTE="Vu_Hung/cv_nangcao/output/shape-20261121/"
OUT=Path(__file__).resolve().parents[1]/"output/shape-20261121"
PROBE=r"""
import hashlib, json, subprocess
from pathlib import Path
root=Path.home()/"Vu_Hung/cv_nangcao/output/shape-20261121"
study=root/"progressive-l40s"
entries=[line.split()[:3] for line in (study/"submissions.tsv").read_text().splitlines()]
jobs=[job for job,role,stage in entries]
reports=dict(line.split() for line in (study/"report-jobs.txt").read_text().splitlines())
jobs+=list(reports.values())
latest={(role,stage):job for job,role,stage in entries}
latest.update({("report",stage):job for stage,job in reports.items()})
export_ledger = study/"metadata-export-jobs.tsv"
exports = {}
if export_ledger.exists():
    for line in export_ledger.read_text().splitlines():
        job, arm, audit, command = line.split("\t", 3)
        if arm not in {"shape", "combined"} or arm in exports or not job.isdecimal():
            raise ValueError("Unexpected metadata export job in ledger")
        exports[arm] = job
    jobs.extend(exports.values())
raw=subprocess.check_output(["sacct","-X","-nP","-j",",".join(jobs),"--format=JobIDRaw,State%32,Elapsed,ExitCode"],text=True)
account={r[0]:r[1:] for r in (line.split("|") for line in raw.splitlines())}
files=["progressive-l40s/submissions.tsv","progressive-l40s/report-jobs.txt","controlled-audit.json"]
for job, role, stage in entries:
    if role in {"shape", "combined"} and stage == "full":
        timeout = f"progressive-l40s/{role}-timeout-{job}.json"
        if (root/timeout).is_file():
            files.append(timeout)
    if role == "combined" and stage == "full":
        receipt = f"progressive-l40s/combined-refs-full-{job}.json"
        if (root/receipt).is_file() or account.get(job, [""])[0] == "COMPLETED":
            files.append(receipt)
for (role,stage),job in latest.items():
    if account.get(job,[""])[0]=="COMPLETED":
        if role in {"baseline","shape","combined"}:
            files.append(f"progressive-l40s/{role}-{stage}-verification.json")
            directory = f"progressive-l40s/{role}" + ("-cohort" if stage == "cohort" else "")
            files.extend(f"{directory}/{name}" for name in
                         ("execution.json", "per_target.csv", "run_config.json", "metric_config.json", "summary.json"))
            if stage=="full":
                files.append(f"progressive-l40s/{role}-registry.csv")
            if stage == "cohort" and role in {"shape", "combined"}:
                files.append(f"progressive-l40s/{role}-refs-cohort.json")
        if role=="audit-references":
            files+=["reference-drift-cohort.json"]
        if (role, stage) == ("audit-reference-intervals", "full"):
            files.extend(["full-reference-interval-replay.json", f"progressive-l40s/shape-reference-intervals-{job}.log"])
        if role=="audit-shape-vs-pmc":
            files+=["REPORT.md",f"shape-{stage}-vs-pmc-pairs.json",f"shape-{stage}-vs-pmc-summary.json",
                    f"progressive-l40s/baseline-{stage}-verification.json",f"progressive-l40s/shape-{stage}-verification.json"]
        if role in {"audit-shape", "audit-combined"} and stage == "full":
            arm = role.split("-", 1)[1]
            files += [f"{arm}-full-pairs.json", f"{arm}-full-summary.json"]
            if arm == "shape":
                files.append("REPORT.md")
for stage,job in reports.items():
    if account.get(job,[""])[0]=="COMPLETED":
        files+=["REPORT.md"]
        if stage=="full":
            files+=["slide-16.csv","slide-16.png"]
        for arm in ("shape","combined"):
            prefix=f"{arm}-{stage}" + ("-vs-pmc" if stage=="cohort" else "")
            files += [prefix+"-summary.json",prefix+"-pairs.json"]
inventory={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in set(files)}
if export_ledger.exists():
    inventory["progressive-l40s/metadata-export-jobs.tsv"] = hashlib.sha256(export_ledger.read_bytes()).hexdigest()
for arm, job in exports.items():
    if account.get(job, [""])[0] == "COMPLETED":
        log = f"progressive-l40s/{arm}-export-{job}.log"
        receipt = json.loads((root/log).read_bytes())
        archive = f"progressive-l40s/{arm}-full-metadata.tar.gz"
        execution = json.loads((study/arm/"execution.json").read_bytes())
        if (receipt["targets"] != 2565 or receipt["execution"] != execution["id"]
                or receipt["archive"] != f"output/shape-20261121/{archive}"):
            raise ValueError(f"Metadata export receipt disagrees with the {arm} execution")
        inventory[log] = hashlib.sha256((root/log).read_bytes()).hexdigest()
        inventory[archive] = receipt["sha256"]
recovering = []
for arm in ("shape", "combined"):
    job = latest.get((arm, "full"))
    controller = latest.get(("resume-" + arm, "full"))
    if account.get(job, [""])[0] == "TIMEOUT" and account.get(controller, [""])[0] in {"PENDING", "RUNNING"}:
        recovering.append(job)
print(json.dumps(dict(jobs=account,current_job_ids=sorted(set(latest.values()) | set(exports.values())),
 interval_job=latest.get(("audit-reference-intervals", "full")),
 export_jobs=list(exports.values()),export_jobs_by_arm=exports,recovering=recovering,hashes=inventory,full_report_job=reports["full"],counts={
 arm:sum(1 for _ in (study/arm/"targets").glob("*/*/result.json")) for arm in ("baseline","shape","surface","combined")})))
"""


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect():
    result=subprocess.run(["ssh","-o","BatchMode=yes","-o","ConnectTimeout=15",HOST,"python3 -"],
                          input=PROBE.encode(),capture_output=True,timeout=90,check=True)
    record=json.loads(result.stdout)
    record["observed_at"]=datetime.now(timezone.utc).isoformat()
    for name,digest in record["hashes"].items():
        path=OUT/name
        if path.is_file() and sha(path)==digest:
            continue
        path.parent.mkdir(parents=True,exist_ok=True)
        temp=path.with_suffix(path.suffix+".downloading")
        subprocess.run(["scp","-q","-o","BatchMode=yes","-o","ConnectTimeout=15",HOST+":"+REMOTE+name,str(temp)],
                       check=True,timeout=180)
        if sha(temp)!=digest:
            raise ValueError("File changed during retrieval: "+name)
        temp.replace(path)
    temp=OUT/"status.tmp"
    temp.write_text(json.dumps(record,indent=2)+"\n",encoding="utf8")
    temp.replace(OUT/"status.json")
    return record


def check_closed_outlines(hashes, checked):
    for arm in ("shape", "combined"):
        pair_digest = hashes.get(f"{arm}-full-pairs.json")
        if not pair_digest:
            continue
        inputs = (hashes[f"progressive-l40s/{arm}/execution.json"],
                  hashes[f"{arm}-full-summary.json"], pair_digest)
        if inputs == checked.get(arm):
            continue
        receipt = replay(arm, cohort=False, fetch=True)
        if (receipt["execution_sha256"], receipt["pairs_sha256"]) != (inputs[0], pair_digest):
            raise ValueError(f"Local {arm} outline inputs differ from the collected full execution")
        write_json(OUT / f"{arm}-full-cocoa-native-replay.json", receipt)
        checked[arm] = inputs
        print(f'{arm}: verified native outline IoU for {receipt["targets"]} targets', flush=True)


def check_interval_receipt(study):
    receipt = json.loads((study / "full-reference-interval-replay.json").read_bytes())
    control = study.parent / "surface-annotated-20261112-freshrefs"
    for arm in ("baseline", "surface", "shape", "combined"):
        run = (control if arm in ("baseline", "surface") else study / "progressive-l40s") / arm
        if receipt["execution_sha256"][arm] != sha(run / "execution.json"):
            raise ValueError(f"Reference intervals use a different {arm} execution")
    for arm in ("shape", "combined"):
        summary = json.loads((study / f"{arm}-full-summary.json").read_bytes())
        pairs = sha(study / f"{arm}-full-pairs.json")
        if pairs != receipt["pairs_sha256"][arm] or pairs != summary["pairs_sha256"]:
            raise ValueError(f"Reference intervals use different {arm} target pairs")
        for control_arm in ("baseline", "surface"):
            for subset in ("same-reference", "same-scoring-crop"):
                result = receipt["estimates"][f"{arm}-vs-{control_arm}-{subset}"]
                if result["n"] != receipt["groups"][subset]:
                    raise ValueError(f"Reference interval count differs for {subset}")
                check_estimate(result, summary["metrics"][f"{control_arm}-{subset}"]["LPIPS"])


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--watch",action="store_true")
    args=parser.parse_args()
    checked_outlines = {}
    checked_exports = {}
    while True:
        try:
            state=collect()
        except (subprocess.SubprocessError,ValueError,OSError) as exc:
            print(datetime.now(timezone.utc),str(exc),flush=True)
            if not args.watch:
                raise
        else:
            print(state["observed_at"],state["counts"],flush=True)
            if args.watch:
                for arm, job in state["export_jobs_by_arm"].items():
                    if state["jobs"].get(job, [""])[0] != "COMPLETED":
                        continue
                    names = (f"progressive-l40s/{arm}-full-metadata.tar.gz", f"progressive-l40s/{arm}-export-{job}.log",
                             f"progressive-l40s/{arm}/execution.json", f"{arm}-full-pairs.json", f"{arm}-full-summary.json")
                    digests = tuple(state["hashes"][name] for name in names)
                    if digests != checked_exports.get(arm):
                        receipt = verify_collected(OUT, arm, job)
                        write_json(OUT / f"{arm}-full-metadata-replay.json", receipt)
                        checked_exports[arm] = digests
                        print(f'{arm}: verified native scores for {receipt["targets"]} targets', flush=True)
                try:
                    check_closed_outlines(state["hashes"], checked_outlines)
                except subprocess.SubprocessError as exc:
                    print(f"Native outline transfer failed: {exc}", flush=True)
                    time.sleep(180)
                    continue
            final_jobs = [state["full_report_job"], *state.get("export_jobs", [])]
            interval_job = state.get("interval_job")
            if interval_job:
                final_jobs.append(interval_job)
            if not args.watch:
                return
            if all(state["jobs"].get(job, [""])[0] == "COMPLETED" for job in final_jobs):
                if interval_job:
                    check_interval_receipt(OUT)
                try:
                    replay_full_slide_tables()
                except subprocess.SubprocessError as exc:
                    print(f"Slurm accounting lookup failed: {exc}", flush=True)
                    time.sleep(180)
                    continue
                return
            failed_states = {"FAILED", "TIMEOUT", "OUT_OF_MEMORY", "CANCELLED", "NODE_FAIL", "PREEMPTED", "BOOT_FAIL", "DEADLINE"}
            failed = [f"{job} ({row[0]})" for job, row in state["jobs"].items()
                      if job in state["current_job_ids"] and job not in state.get("recovering", [])
                      and row[0].split()[0] in failed_states]
            if failed:
                raise SystemExit(f"Slurm reported a terminal failure for {', '.join(failed)}. Inspect status.json and the study logs.")
        time.sleep(180)


if __name__=="__main__":
    main()


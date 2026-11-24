"""Inventory a timed-out Shape run before the next evaluator opens its output."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from amodal_benchmark.dataset import read_official, sha256, write_json
from amodal_benchmark.parallel import output_lock


def read(path):
    return json.loads(path.read_bytes())


def file_inventory(directory):
    return {path.relative_to(directory).as_posix(): sha256(path)
            for path in sorted(directory.rglob("*")) if path.is_file()}


def target_inventory(run, official_ids):
    completed, partial = {}, {}
    targets = run / "targets"
    for directory in sorted(targets.glob("*/*")):
        if not directory.is_dir():
            continue
        tid = directory.relative_to(targets).as_posix()
        if tid not in official_ids:
            raise ValueError(f"Unexpected target directory {tid}")
        result = directory / "result.json"
        if result.is_file():
            if read(result)["id"] != tid:
                raise ValueError(f"Result identity differs for {tid}")
            completed[tid] = sha256(result)
        else:
            partial[tid] = file_inventory(directory)
    return completed, partial


def verify_timeout_snapshots(run, execution):
    """Compare each pre-retry record with the completed full execution."""
    counts = {}
    updated = set(execution["new_or_updated_target_ids"])
    for path in sorted(run.parent.glob(f"{run.name}-timeout-*.json")):
        receipt = read(path)
        job = receipt["job"]
        if (path.name != f"{run.name}-timeout-{job}.json" or receipt["state"] != "TIMEOUT"
                or receipt["arm"] != run.name or receipt["run_config_sha256"] != sha256(run / "run_config.json")):
            raise ValueError(f"Timeout inventory differs from the completed run: {path.name}")
        for tid, digest in receipt["completed"].items():
            if tid in updated or execution["result_artifacts"].get(tid) != digest:
                raise ValueError(f"Full continuation changed the pre-timeout record for {tid} from job {job}")
        if "cached_score_sha256" in receipt:
            scores = receipt["cached_score_sha256"]
            if len(scores) != receipt["cached_scores"] or not scores.keys() <= receipt["completed"].keys():
                raise ValueError(f"Cached score inventory differs from the completed targets for job {job}")
            for tid, digest in scores.items():
                score = run / "targets" / tid / "metrics.json"
                if not score.is_file() or sha256(score) != digest:
                    raise ValueError(f"Full continuation changed the pre-timeout score for {tid} from job {job}")
        for tid, files in receipt["partial"].items():
            archived = path.with_suffix("") / tid
            if not archived.is_dir() or file_inventory(archived) != files:
                raise ValueError(f"Archived partial target differs for {tid} from job {job}")
        counts[job] = len(receipt["completed"])
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arm", choices=["shape", "combined"])
    parser.add_argument("job", type=int)
    args = parser.parse_args()
    job = str(args.job)
    study = ROOT / "output/shape-20261121/progressive-l40s"
    entries = [line.split()[:3] for line in (study / "submissions.tsv").read_text().splitlines()]
    if [job, args.arm, "full"] not in entries:
        raise ValueError("Job is outside the recorded full study")
    raw = subprocess.check_output(["sacct", "-X", "-nP", "-j", job, "--format=JobIDRaw,State%32"], text=True)
    state = dict(line.split("|") for line in raw.splitlines())[job].split()[0]
    if state != "TIMEOUT":
        print(f"Job {job} is {state}; timeout inventory skipped.")
        return

    run = study / args.arm
    receipt = study / f"{args.arm}-timeout-{job}.json"
    with output_lock(run):
        execution = read(run / "execution.json")
        env = run / "provenance" / execution["id"] / "environment.json"
        if (sha256(env) != execution["environment_sha256"]
                or read(env)["execution_environment"]["SLURM_JOB_ID"] != job):
            raise ValueError("The active output belongs to a different execution")
        rows, _ = read_official(ROOT / "datasets/open_world_amodal")
        ids = [row["id"] for row in rows]
        if len(ids) != 2565 or execution["target_ids"] != ids:
            raise ValueError("Timeout run does not cover the frozen full target inventory")
        completed, partial = target_inventory(run, set(ids))
        cohort = read(study / f"{args.arm}-cohort" / "execution.json")["result_artifacts"]
        if len(cohort) != 204 or any(completed.get(tid) != digest for tid, digest in cohort.items()):
            raise ValueError("Pre-retry records differ from the frozen cohort")
        score_hashes = {}
        for tid in completed:
            score = run / "targets" / tid / "metrics.json"
            if score.is_file():
                score_hashes[tid] = sha256(score)
        inventory = dict(arm=args.arm, job=job, state=state, execution_id=execution["id"],
                         run_config_sha256=sha256(run / "run_config.json"), completed=completed,
                         cached_scores=len(score_hashes), cached_score_sha256=score_hashes)
        if receipt.exists():
            saved = read(receipt)
            original_partial = saved.pop("partial")
            # Preserve receipts created before score counts or score hashes were recorded.
            saved.setdefault("cached_scores", inventory["cached_scores"])
            saved.setdefault("cached_score_sha256", score_hashes)
            if (saved != inventory or not original_partial.keys() <= set(ids) - completed.keys()
                    or any(original_partial.get(tid) != files for tid, files in partial.items())):
                raise ValueError("Output changed since the timeout inventory was written")
        else:
            original_partial = partial
            write_json(receipt, dict(inventory, partial=partial))

        # One rename per target is atomic on the study filesystem. A second call
        # can finish an interrupted archive using the same immutable inventory.
        for tid, files in original_partial.items():
            source = run / "targets" / tid
            archived = receipt.with_suffix("") / tid
            if archived.exists():
                if source.exists() or not archived.is_dir() or file_inventory(archived) != files:
                    raise ValueError(f"Partial target archive conflicts with the inventory for {tid}")
            else:
                if tid not in partial:
                    raise ValueError(f"Partial target disappeared before archival for {tid}")
                archived.parent.mkdir(parents=True, exist_ok=True)
                source.rename(archived)
        print(f"{receipt.name}: {len(completed)} completed, {len(original_partial)} partial targets archived, "
              f"SHA-256 {sha256(receipt)}")


if __name__ == "__main__":
    main()

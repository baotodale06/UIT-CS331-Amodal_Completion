"""Follow the frozen continuation chain and export only its completed audit."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if (ROOT / "amodal_benchmark").is_dir():
    sys.path.insert(0, str(ROOT))

from amodal_benchmark.dataset import sha256, write_json
from amodal_benchmark.parallel import output_lock
from scripts.continue_shape_v2 import accounting, submit, verify_frozen_source
from scripts.export_shape_metadata import export


def verify_postprocessing(study):
    receipt = json.loads((study / "postprocessing-source-manifest.json").read_bytes())
    if receipt["base_evaluation_manifest_sha256"] != sha256(study / "evaluation-source-manifest.json"):
        raise ValueError("Postprocessing was prepared against a different evaluation snapshot")
    for name, expected in receipt["files"].items():
        if sha256(study / "postprocessing-source" / name) != expected:
            raise ValueError("Postprocessing source changed: " + name)


def finalize(study, job_id, controller_id, method="shape-v2"):
    if method not in ("shape-v2", "shape-calibrated"):
        raise ValueError("Unknown shape finalization method")
    study = Path(study).resolve()
    destination = study / "finalization-control"
    destination.mkdir(parents=True, exist_ok=True)
    with output_lock(destination):
        path = destination / (job_id + ".json")
        if path.exists():
            return json.loads(path.read_bytes())
        if (destination / (job_id + "-submission.json")).exists():
            raise ValueError("Uncertain previous submission: inspect Slurm before retrying")
        controller = accounting(controller_id)
        if controller["state"] != "COMPLETED" or controller["exit_code"] != "0:0":
            raise ValueError("Continuation controller did not complete successfully")
        events = json.loads((study / "continuations.json").read_bytes())["events"]
        matched = [event for event in events if event["previous"]["job_id"] == job_id]
        if len(matched) != 1:
            raise ValueError("Expected exactly one continuation decision for the benchmark job")
        event = matched[0]
        result = dict(at=datetime.now(timezone.utc).isoformat(), job_id=job_id,
                      controller_id=controller_id, continuation=event,
                      source_sha256=sha256(Path(__file__)))
        if event["action"] == "resumed":
            next_job, next_controller = event["next_job_id"], event["controller_job_id"]
            command = ["sbatch", "--parsable", "--partition=debug-cpu", "--cpus-per-task=2",
                       "--mem=8G", "--time=00:30:00", "--job-name=" + method + "-export",
                       "--dependency=afterany:" + next_controller,
                       "--output=" + str(study / "finalization-%j.log"),
                       "scripts/run_" + method.replace("-", "_") + "_finalization.sh", "--study", str(study),
                       "--job-id", next_job, "--controller-id", next_controller]
            # Persist the exact command before submission; an uncertain submission
            # requires review instead of silently submitting a duplicate.
            write_json(destination / (job_id + "-submission.json"), dict(command=command))
            result.update(action="scheduled", command=command,
                          follow_job_id=submit(command, study / "operational-source"))
        elif event["action"] == "finished":
            previous = accounting(job_id)
            if previous["state"] != "COMPLETED" or previous["exit_code"] != "0:0":
                raise ValueError("Export requires a successfully completed benchmark job")
            verify_frozen_source(study)
            verify_postprocessing(study)
            audit = json.loads((study / "full-audit.json").read_bytes())
            if method == "shape-calibrated":
                freeze = json.loads((study / "policy-freeze.json").read_bytes())
                if freeze["shape_policy"]["method"] != "shape-calibrated-probability":
                    raise ValueError("Calibrated finalization requires the calibrated frozen policy")
                if audit.get("selector_links", {}).get("verified") is not True:
                    raise ValueError("Calibrated target-to-decision bindings were not verified")
            execution = json.loads((study / "benchmark/execution.json").read_bytes())
            if (audit.get("verified") is not True or audit.get("targets") != 2565
                    or audit["freeze_sha256"] != sha256(study / "policy-freeze.json")
                    or execution.get("status") != "complete"
                    or len(execution["target_ids"]) != 2565
                    or not (study / "registry.csv").is_file()):
                raise ValueError("Full frozen audit, closed execution and registration required")
            archive = export(study / "benchmark", study / "benchmark-metadata.tar.gz")
            write_json(study / "metadata-export.json", dict(**archive,
                freeze_sha256=sha256(study / "policy-freeze.json"),
                audit_sha256=sha256(study / "full-audit.json"),
                postprocessing_manifest_sha256=sha256(study / "postprocessing-source-manifest.json")))
            result.update(action="exported", archive=archive,
                          four_metric_gate=audit["gate"]["eligible_for_success_claim"])
        elif event["action"] == "needs_review":
            result.update(action="needs_review", reason=event["reason"])
        else:
            raise ValueError("Unknown continuation decision")
        write_json(path, result)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--controller-id", required=True)
    args = parser.parse_args()
    if not args.job_id.isdigit() or not args.controller_id.isdigit():
        raise ValueError("Numeric Slurm job IDs required")
    print(json.dumps(finalize(args.study, args.job_id, args.controller_id)), flush=True)


if __name__ == "__main__":
    main()

"""Measure Shape LPIPS on the completed Visual Genome source using saved predictions."""
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from amodal_benchmark.dataset import read_official, sha256, write_json
from amodal_benchmark.metrics import MetricModels
from amodal_benchmark.protocol import failure_metrics
from amodal_benchmark.references import reference_paths
from scripts.audit_surface_progressive_result import estimate
from scripts.compare_shape_progressive import execution_scores


def main():
    import cv2
    import torch
    study = ROOT / "output/shape-20261121/progressive-l40s"
    run = study / "shape"
    early_path = study / "shape-vg-ssim.json"
    coverage_path = study / "shape-vg-inference-coverage.json"
    early = json.loads(early_path.read_bytes())
    if (sha256(early_path) != "20f6d153315a672e337807f595c3418898dec9b11b0264a499624804e78c1af4"
            or sha256(coverage_path) != early["coverage_sha256"]):
        raise ValueError("Different native VG inventory or CPU SSIM inputs")
    coverage = json.loads(coverage_path.read_bytes())
    cfg = json.loads((run / "metric_config.json").read_bytes())
    if cfg != early["metric_config"] or cfg["alignment"] != "released":
        raise ValueError("LPIPS must use the archived released scoring protocol")
    official = [r for r in read_official(ROOT / "datasets/open_world_amodal")[0] if r["source"] == "VG"]
    ids = [r["id"] for r in official]
    if len(ids) != 1294 or ids != [r["id"] for r in early["rows"]] or ids != [r["id"] for r in coverage["rows"]]:
        raise ValueError("The replay must cover the official VG order")
    cohort = study / "shape-cohort"
    retained = execution_scores(cohort, json.loads((cohort / "execution.json").read_bytes()))
    if sha256(cohort / "per_target.csv") != early["cohort_csv_sha256"]:
        raise ValueError("The retained scores changed")
    controls = {}
    historical = ROOT / "output/surface-annotated-20261112-freshrefs"
    for arm in ("baseline", "surface"):
        directory = historical / arm
        execution = directory / "execution.json"
        if (sha256(execution) != coverage["historical_execution_sha256"][arm]
                or sha256(directory / "per_target.csv") != early["control_csv_sha256"][arm]):
            raise ValueError(f"The historical {arm} scores changed")
        controls[arm] = execution_scores(directory, json.loads(execution.read_bytes()))
    torch.set_num_threads(int(os.environ["SLURM_CPUS_PER_TASK"]))
    cv2.setNumThreads(1)
    model = MetricModels(device="cpu")
    scored = {}
    retained_errors = []
    # Check the CPU implementation against all 109 existing L40S rows before measuring fresh targets.
    cases = sorted(zip(official, coverage["rows"], early["rows"]), key=lambda case: not case[1]["retained"])
    for target, snapshot, previous in cases:
        tid = target["id"]
        folder = run / "targets" / tid
        native = folder / "result.json"
        if sha256(native) != snapshot["result_sha256"]:
            raise ValueError(f"Native target changed: {tid}")
        rec = json.loads(native.read_bytes())
        status = rec["status"]
        if (rec["id"] != tid or status != previous["status"] or status != snapshot["shape"]
                or snapshot["retained"] != (tid in retained)):
            raise ValueError(f"Different target, status or cohort membership: {tid}")
        if status == "success":
            crop = reference_paths(run / "visible_references", target)["crop"]
            image = folder / "completion.png"
            if (sha256(crop) != previous["crop_sha256"] or sha256(image) != previous["completion_sha256"]
                    or rec["completion_sha256"] != previous["completion_sha256"]
                    or rec["evaluation_reference_artifacts"]["crop"] != previous["crop_sha256"]):
                raise ValueError(f"Scoring inputs changed: {tid}")
            metrics = model.score(crop, image, target["label"], rec["geometry"], alignment=cfg["alignment"])
        elif status in {"not_detected", "ambiguous_target", "no_completion"}:
            metrics = failure_metrics()
        else:
            raise ValueError(f"Unexpected VG target status: {tid} {status}")
        if not math.isclose(metrics["SSIM"], previous["SSIM"], rel_tol=0., abs_tol=1e-12):
            raise ValueError(f"Released score preprocessing differs from the native SSIM replay: {tid}")
        lpips = metrics["LPIPS"]
        if not math.isfinite(lpips):
            raise ValueError(f"Nonfinite LPIPS for {tid}")
        if tid in retained:
            error = abs(lpips - retained[tid]["LPIPS"])
            if status != retained[tid]["status"] or error > 1e-5:
                raise ValueError(f"CPU and retained L40S scores disagree for {tid} by {error}")
            if status == "success":
                retained_errors.append(error)
        scored[tid] = dict(id=tid, status=status, LPIPS=lpips)
        if len(scored) % 200 == 0:
            print(f"{len(scored)} of {len(ids)} scored", flush=True)
    if len(retained_errors) != 48 or sum(tid in retained for tid in ids) != 109:
        raise ValueError("Incomplete retained L40S validation")
    comparisons = {}
    for arm, control in controls.items():
        pairs = [dict(source="VG", image_identity="VG/" + r["original_id"],
                      baseline=control[r["id"]], surface=scored[r["id"]]) for r in official]
        comparisons[arm] = estimate(pairs, "LPIPS")
    receipt = dict(scope="CPU scoring of saved L40S Shape predictions for the complete VG source",
                   job=os.environ["SLURM_JOB_ID"], scorer_sha256=sha256(Path(__file__)),
                   metric_source_sha256=sha256(ROOT / "amodal_benchmark/metrics.py"),
                   ssim_sha256=sha256(early_path), coverage_sha256=sha256(coverage_path),
                   cohort_csv_sha256=early["cohort_csv_sha256"], control_csv_sha256=early["control_csv_sha256"],
                   metric_config=cfg, retained_successes=len(retained_errors), retained_max_error=max(retained_errors),
                   cpu_gpu_absolute_tolerance=1e-5, comparisons=comparisons, rows=[scored[tid] for tid in ids])
    write_json(study / "shape-vg-lpips.json", receipt)
    print(json.dumps(comparisons), flush=True)


if __name__ == "__main__":
    main()

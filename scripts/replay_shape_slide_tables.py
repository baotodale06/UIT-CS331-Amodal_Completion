"""Recompute the Week 8 full benchmark and severity tables from the collected runs."""
import csv
import json
import math
from pathlib import Path
from statistics import fmean
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from amodal_benchmark.dataset import read_official, sha256, write_json
from scripts.collect_surface_job_evidence import HOST, TERMINAL, ssh
from scripts.compare_shape_progressive import execution_scores

STUDY = ROOT / "output/shape-20261121"
CONTROL = ROOT / "output/surface-annotated-20261112-freshrefs"
ORIGINAL_CHART = ROOT / "output/week8-research-story-20261116/research/severity-joined.csv"
METHODS = {"baseline": "PMC", "shape": "Shape", "surface": "Surface", "combined": "Shape + Surface"}


def read(path):
    return json.loads(path.read_bytes())


def chart_groups(path):
    groups = [[] for _ in range(5)]
    with path.open(encoding="utf8", newline="") as stream:
        for row in csv.DictReader(stream):
            ratio = float(row["occlusion_ratio"])
            if not 0 <= ratio <= 1:
                raise ValueError(f"Invalid annotated occlusion for {row['id']}")
            groups[min(int(ratio * 5), 4)].append(row["id"])
    if len(set().union(*groups)) != 321 or list(map(len, groups)) != [203, 54, 39, 21, 4]:
        raise ValueError("The chart target inventory differs from the published Week 8 data")
    return groups


def method_values(scores, outlines, groups):
    if set(outlines) != set().union(*groups):
        raise ValueError("Native outline scores differ from the chart inventory")
    rows = []
    for group in groups:
        for tid in group:
            if (scores[tid]["status"] == "success") != outlines[tid]["success"]:
                raise ValueError(f"Score table and native outline disagree on coverage for {tid}")
        rows.append(dict(lpips=fmean(scores[tid]["LPIPS"] for tid in group),
                         saved=sum(outlines[tid]["success"] for tid in group),
                         amodal_iou=fmean(outlines[tid]["amodal_iou"] for tid in group)))
    return dict(LPIPS=fmean(r["LPIPS"] for r in scores.values()),
                SSIM=fmean(r["SSIM"] for r in scores.values()),
                saved=sum(r["status"] == "success" for r in scores.values()), bins=rows)


def allocated_hours(job_ids, accounting):
    seconds = 0
    for job in job_ids:
        state, elapsed, resources = accounting[job]
        if state not in TERMINAL:
            raise ValueError(f"Allocation {job} is still {state}")
        if state == "CANCELLED" and elapsed == 0 and not resources:
            continue
        if "gres/gpu=4" not in resources.split(","):
            raise ValueError(f"Missing four-GPU allocation for {job}")
        seconds += elapsed
    return seconds * 4 / 3600


def match_report(values, groups, report, exported):
    section = report.split("### 2,565 natural targets", 1)[1]
    table, heading, severity = section.partition(" · COCOA occlusion severity\n")
    if not heading:
        raise ValueError("Missing COCOA occlusion severity section")
    severity = severity.split("\n## ", 1)[0]
    if len(exported) != len(groups):
        raise ValueError("Unexpected number of exported severity bins")
    benchmark_header = "| Method | GPU | K per pass | Failure-inclusive LPIPS ↓ | SSIM ↑ | Saved | GPU-hours allocated |"
    if table.splitlines().count(benchmark_header) != 1:
        raise ValueError("Full benchmark columns differ from the requested metrics")
    severity_header = "| Occlusion | Targets | PMC · Quadro | Shape · L40S | Surface · Quadro | Shape + Surface · L40S |"
    parts = severity.split(severity_header)
    if len(parts) != 3:
        raise ValueError("COCOA severity method columns differ from the four benchmark methods")
    # In each table the first line after the header is the Markdown separator.
    tables = [part.strip().split("\n\n", 1)[0].splitlines()[1:] for part in parts[1:]]
    if any(len(rows) != len(groups) for rows in tables):
        raise ValueError("Unexpected number of rows in the COCOA severity tables")
    for arm, name in METHODS.items():
        result = values[arm]
        device = "L40S" if arm in ("shape", "combined") else "Quadro RTX 6000 (historical)"
        expected = [name, device, "1" if arm == "baseline" else "8", f'{result["LPIPS"]:.6f}',
                    f'{result["SSIM"]:.6f}', str(result["saved"]), f'{result["gpu_hours"]:.3f}']
        matches = [[s.strip() for s in line.strip("|").split("|")] for line in table.splitlines()
                   if line.startswith(f"| {name} | {device} |")]
        if len(matches) != 1:
            raise ValueError(f"Expected one full results row for {name}")
        matches[0][5] = matches[0][5].replace(",", "")
        if matches[0] != expected:
            raise ValueError(f"Full benchmark values differ for {name}")
    for index, (group, csv_row) in enumerate(zip(groups, exported)):
        label = f"{index * 20}–{(index + 1) * 20}%"
        if csv_row["bin"] != label or int(csv_row["n"]) != len(group):
            raise ValueError(f"COCOA severity inventory differs for {label}")
        lpips_cells = [label, str(len(group))]
        geometry_cells = lpips_cells.copy()
        for arm in METHODS:
            row = values[arm]["bins"][index]
            if (int(csv_row[f"{arm}_saved"]) != row["saved"]
                    or not math.isclose(float(csv_row[arm]), row["lpips"], rel_tol=0, abs_tol=1e-12)
                    or not math.isclose(float(csv_row[f"{arm}_amodal_iou"]), row["amodal_iou"],
                                        rel_tol=0, abs_tol=1e-12)):
                raise ValueError(f"COCOA severity CSV differs for {arm} in {label}")
            lpips_cells.append(f'{row["lpips"]:.4f} ({row["saved"]} saved)')
            geometry_cells.append(f'{row["amodal_iou"]:.4f}')
        for cells, rows in zip((lpips_cells, geometry_cells), tables):
            if rows[index] != "| " + " | ".join(cells) + " |":
                raise ValueError(f"COCOA severity report differs for {label}")


def replay_full_slide_tables():
    directory = STUDY / "progressive-l40s"
    original = read(STUDY / "slide-15-cohort-and-controls.json")
    jobs = {arm: next(row["jobs"] for row in original["rows"]
                      if row["scope"] == "full" and row["method"] == METHODS[arm])
            for arm in ("baseline", "surface")}
    jobs.update(shape=[], combined=[])
    for line in (directory / "submissions.tsv").read_text(encoding="utf8").splitlines():
        job, arm, stage, _ = line.split("\t", 3)
        if arm in ("shape", "combined") and stage in ("cohort", "full"):
            jobs[arm].append(job)
    ids = [row["id"] for row in read_official(ROOT / "datasets/open_world_amodal")[0]]
    groups = chart_groups(ORIGINAL_CHART)
    historical = read(STUDY / "cocoa-historical-controls.json")
    runs = {}
    for arm in METHODS:
        current = arm in ("shape", "combined")
        run = (directory if current else CONTROL) / arm
        execution = read(run / "execution.json")
        if execution["status"] != "complete" or execution["target_ids"] != ids or len(ids) != 2565:
            raise ValueError(f"{arm} needs a closed, full execution in official target order")
        scores = execution_scores(run, execution)
        if list(scores) != ids:
            raise ValueError(f"Incomplete or reordered scores for {arm}")
        if current:
            native = read(STUDY / f"{arm}-full-cocoa-native-replay.json")
            source = native["execution_sha256"]
            if native["method"] != arm or native["manifest_sha256"] != historical["manifest_sha256"]:
                raise ValueError(f"Different native outline evidence for {arm}")
            outlines = {r["id"]: dict(success=r["status"] == "success", amodal_iou=r["amodal_iou"])
                        for r in native["rows"]}
        else:
            source = historical["sources"][arm]["execution_sha256"]
            outlines = {r["id"]: r[arm] for r in historical["targets"]}
        if source != sha256(run / "execution.json"):
            raise ValueError(f"Scores and outline evidence came from different {arm} runs")
        runs[arm] = dict(method_values(scores, outlines, groups), execution_sha256=source,
                         scores_sha256=execution["artifacts"]["per_target.csv"], jobs=jobs[arm])
    job_ids = [job for arm in METHODS for job in jobs[arm]]
    if len(job_ids) != len(set(job_ids)) or not all(job.isdecimal() for job in job_ids):
        raise ValueError("Invalid allocation IDs in the study ledgers")
    output = ssh(HOST, "sacct -XnP -j " + ",".join(job_ids)
                 + " --format=JobIDRaw,State%32,ElapsedRaw,AllocTRES", capture_output=True,
                 text=True, timeout=60).stdout
    accounting = {}
    for line in output.splitlines():
        job, state, elapsed, resources = line.split("|")
        if job in accounting:
            raise ValueError(f"Duplicate Slurm accounting for {job}")
        accounting[job] = (state.split()[0], int(elapsed), resources)
    if set(accounting) != set(job_ids):
        raise ValueError("Slurm returned a different set of allocations")
    for arm in METHODS:
        runs[arm]["gpu_hours"] = allocated_hours(jobs[arm], accounting)
    report = STUDY / "REPORT.md"
    numeric = STUDY / "slide-16.csv"
    with numeric.open(encoding="utf8", newline="") as stream:
        match_report(runs, groups, report.read_text(encoding="utf8"), list(csv.DictReader(stream)))
    receipt = dict(targets=len(ids), group_sizes=list(map(len, groups)), methods=runs,
                   accounting=accounting, report_sha256=sha256(report), csv_sha256=sha256(numeric),
                   original_chart_join_sha256=sha256(ORIGINAL_CHART))
    write_json(STUDY / "full-slide-table-replay.json", receipt)
    print("Verified four full benchmark rows and five COCOA severity bins across four methods.")


if __name__ == "__main__":
    replay_full_slide_tables()

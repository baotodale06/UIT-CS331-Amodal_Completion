"""Publish the four-metric shape-only result only after full audited recovery."""
import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from amodal_benchmark.dataset import sha256, write_json
from scripts.assess_shape_v2 import METRICS
from scripts.verify_shape_v2_metadata import equal


def evidence(study):
    read = lambda path: json.loads(path.read_bytes())
    final = read(study/"final-recovery.json")
    snapshot = (study/final["snapshot"]).resolve()
    if not snapshot.is_relative_to(study.resolve()/"recovery"):
        raise ValueError("Final snapshot must be a dated recovery under this study")
    replay_path = study/"full-metadata-replay.json"
    if sha256(replay_path) != final["replay_sha256"]:
        raise ValueError("Full replay differs from final recovery receipt")
    replay = read(replay_path)
    if replay.get("verified") is not True or replay.get("targets") != 2565:
        raise ValueError("A verified 2,565-target replay is required")
    audit_path, freeze_path = snapshot/"full-audit.json", snapshot/"policy-freeze.json"
    if (sha256(audit_path) != replay["native_audit_sha256"]
            or sha256(freeze_path) != replay["freeze_sha256"]):
        raise ValueError("Native audit or frozen policy differs from the local replay")
    audit, freeze = read(audit_path), read(freeze_path)
    for name in ("gate", "full", "unseen"):
        equal(audit[name], replay[name], "report "+name)
    gate = replay["gate"]
    if (gate["targets"] != 2565 or gate["required_relative_gain_percent"] != 3.
            or set(gate["metrics"]) != set(METRICS)
            or gate["metrics"]["CLIP"]["control_n"] != 1394):
        raise ValueError("Report requires the original four-metric full-benchmark contract")
    expected = gate["clip_coverage_not_reduced"] and all(
        row["relative_gain_percent"] >= 3. for row in gate["metrics"].values())
    if bool(expected) != gate["eligible_for_success_claim"]:
        raise ValueError("Report gate contradicts its metric or coverage evidence")
    calibrated = freeze["shape_policy"].get("method") == "shape-calibrated-probability"
    prior_geometry = replay["prior_geometry"] if calibrated else {}
    for name in (() if calibrated else ("validation-native.json", "holdout-native.json")):
        geometry = read(snapshot/name)
        equal(geometry["shape_policy"], freeze["shape_policy"], name+"/policy")
        prior_geometry[geometry["split"]] = geometry["aggregate"]
    if audit["geometry"]["objects"] != 321 or audit["unseen_targets"] != 2017:
        raise ValueError("Frozen native geometry or historical unseen coverage changed")
    cost = read(snapshot/"allocation-cost.json")
    if cost["final"] is not True or final["allocation_final"] is not True:
        raise ValueError("Final reporting requires terminal allocation accounting")
    if (sha256(snapshot/"registry.csv") != replay["registration"]["registry_sha256"]
            or sha256(study/"benchmark-metadata/per_target.csv") != replay["method_scores_sha256"]
            or final["archive_sha256"] != read(snapshot/"metadata-export.json")["sha256"]):
        raise ValueError("Registration, scores or archive receipt changed after replay")
    return dict(snapshot=snapshot, replay=replay, audit=audit, freeze=freeze, cost=cost,
                prior_geometry=prior_geometry)


def number(value, digits=4):
    return "unavailable" if value is None else f"{value:.{digits}f}"


def interval(row):
    return "["+", ".join(number(value, 2) for value in row["relative_ci_percent"])+"]"


def draw_gains(replay, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True, constrained_layout=True)
    for axis, (subset, title) in zip(axes, (("full", "Full benchmark: 2,565 targets"),
                                         ("unseen", "Historical unseen: 2,017 targets"))):
        for index, metric in enumerate(METRICS):
            row = replay[subset][metric]
            point = row["relative_gain_percent"]
            low, high = row["relative_ci_percent"]
            if low is not None and high is not None:
                axis.plot([low, high], [index, index], color="#1e596b", linewidth=2)
            axis.scatter(point, index, color="#1e596b", s=40, zorder=3)
        axis.axvline(0, color="#777777", linewidth=1)
        axis.axvline(3, color="#b14b35", linestyle="--", linewidth=1.5, label="3% target")
        axis.set_yticks(range(4), ["CLIP", "LPIPS", "VGG similarity", "SSIM"])
        axis.set_title(title)
        axis.set_xlabel("Relative improvement (%)")
        axis.grid(axis="x", alpha=.15)
        axis.legend(loc="best", frameon=False)
    axes[0].invert_yaxis()
    figure.suptitle("Shape-aware alone vs original PMC\nPositive = better; nominal 95% image-cluster intervals", fontsize=12)
    figure.savefig(output/"metric-gains.png", dpi=180)
    figure.savefig(output/"metric-gains.svg")
    plt.close(figure)


def build(study, output):
    data = evidence(study)
    replay, audit, cost = (data[key] for key in ("replay", "audit", "cost"))
    calibrated = replay.get("method") == "shape-calibrated-probability"
    label = "Calibrated shape" if calibrated else "Shape v2"
    gate = replay["gate"]
    if output.exists():
        raise FileExistsError("Preserve existing published report: "+str(output))
    output.mkdir(parents=True)
    rows = []
    for subset in ("full", "unseen"):
        for metric in METRICS:
            row = replay[subset][metric]
            rows.append(dict(subset=subset, metric=metric, control=row["control"], method=row["method"],
                control_n=row["control_n"], method_n=row["method_n"], gain_percent=row["relative_gain_percent"],
                ci_low_percent=row["relative_ci_percent"][0], ci_high_percent=row["relative_ci_percent"][1],
                full_point_gate=gate["metrics"][metric]["pass_point_threshold"] if subset == "full" else ""))
    with (output/"metrics.csv").open("w", newline="", encoding="utf8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    passed = gate["eligible_for_success_claim"]
    verdict = ("All four full-benchmark point estimates reach the 3% target, with CLIP coverage preserved."
               if passed else "The requested 3% improvement on all four full-benchmark metrics was not achieved.")
    lines = ["# " + label + ": complete benchmark result", "", verdict, "",
        f"All 2,565 targets are included: **{replay['saved']:,} saved outputs** and **{replay['failures']:,} failures**. "
        "The original PMC control saved 1,394 outputs (1,171 failures). "
        "LPIPS, VGG Feature Similarity and SSIM retain failure penalties. "
        "CLIP uses each arm's successful-output denominator and must cover at least 1,394 outputs.", "",
        "This is an iterated benchmark. The nominal image-cluster bootstrap intervals below do not correct "
        "for repeated research attempts. The 3% point threshold is separate from statistical uncertainty.", ""]
    for subset, title in (("full", "Full benchmark"), ("unseen", "Historical unseen subset (2,017 targets)")):
        lines += ["## "+title, "", f"| Metric | Original PMC | {label} | Relative gain | 95% interval | Control / method n | 3% full gate |",
                  "| --- | ---: | ---: | ---: | --- | ---: | --- |"]
        for metric in METRICS:
            row = replay[subset][metric]
            point_gate = ("pass" if gate["metrics"][metric]["pass_point_threshold"] else "fail") if subset == "full" else "not primary"
            lines.append(f"| {metric} | {number(row['control'])} | {number(row['method'])} | "
                f"{number(row['relative_gain_percent'], 2)}% | {interval(row)}% | "
                f"{row['control_n']} / {row['method_n']} | {point_gate} |")
        lines.append("")
    lines += [f"CLIP coverage gate: **{'pass' if gate['clip_coverage_not_reduced'] else 'fail'}**.", "",
              "![Full and unseen relative improvements](metric-gains.png)", "", "## Native amodal geometry", "",
              "These overlap measurements evaluate shape and are separate from the four appearance metrics. "
              "Benchmark geometry includes failed outputs and evaluates the frozen 321-object COCOA inventory.", "",
              f"| Cohort | Metric | Original PMC | {label} | Relative gain | 95% interval | Control / method n |",
              "| --- | --- | ---: | ---: | ---: | --- | ---: |"]
    for cohort in ("full", "historical_unseen"):
        for metric in ("full_iou", "hidden_iou"):
            row = audit["geometry"][cohort][metric]
            lines.append(f"| {cohort} | {metric} | {number(row['control'])} | {number(row['method'])} | "
                f"{number(row['relative_gain_percent'], 2)}% | {interval(row)}% | {row['control_n']} / {row['method_n']} |")
    lines += ["", "The same native benchmark geometry is reported by the frozen occlusion strata:", "",
              f"| Occlusion stratum | Objects | Metric | Original PMC | {label} | Relative gain | 95% interval | Control / method n |",
              "| --- | ---: | --- | ---: | ---: | ---: | --- | ---: |"]
    for stratum, values in sorted(audit["geometry"]["occlusion_strata"].items()):
        for metric in ("full_iou", "hidden_iou"):
            if metric not in values:
                continue
            row = values[metric]
            lines.append(f"| {stratum} | {values['objects']} | {metric} | {number(row['control'])} | "
                f"{number(row['method'])} | {number(row['relative_gain_percent'], 2)}% | "
                f"{interval(row)}% | {row['control_n']} / {row['method_n']} |")
    prior_note = ("The correction was selected using previously observed validation data. The reserved holdout "
                  "and benchmark were also observed in earlier attempts; these are not untouched confirmation. "
                  "Validation uses native SAM support, whereas this reserved geometry diagnostic uses annotated "
                  "visible support. Their numbers describe different input conditions." if calibrated else
                  "The reserved holdout was read only after policy freezing.")
    lines += ["", "The separately trained shape prior was evaluated in native image coordinates, including truth "
              "outside its visible-derived crop. These development/holdout numbers average by image; "
              "they are not full-benchmark gains. " + prior_note, "",
              "| Prior split | Objects / images | Full IoU | Visible-only full IoU | Hidden IoU | Images with hidden truth |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for split in ("validation", "holdout"):
        row = data["prior_geometry"][split]
        lines.append(f"| {split} | {row['objects']} / {row['images']} | {number(row['full_iou'])} | "
                     f"{number(row.get('visible_only_full_iou'))} | {number(row['hidden_iou'])} | {row['hidden_iou_images']} |")
    pools = audit["selector"]
    if "transitions" in audit:
        lines += ["", "## Output and failure transitions", "", "| Original PMC status | Shape status | Targets |",
                  "| --- | --- | ---: |"]
        for row in audit["transitions"]:
            lines.append(f"| {row['control']} | {row['method']} | {row['targets']} |")
    lines += ["", f"Native selection replay: {pools['pools']:,} pools, {pools['candidates']:,} candidates, "
              f"{pools['fallbacks']:,} fallbacks. The policy uses shape scores alone; no Surface ranker or RGB correction.", "",
              "## Allocation accounting", "", "| Stage | Allocated GPU-hours |", "| --- | ---: |"]
    for role, hours in cost["reserved_gpu_hours_by_role"].items():
        lines.append(f"| {role} | {hours:.6f} |")
    lines += [f"| Total tracked Slurm allocations | {cost['total_reserved_gpu_hours']:.6f} |", "",
        cost.get("scope", ""), "",
        "Time includes failed and interrupted allocations; CPU controllers consume zero GPU-hours. "
        "The historical control used 24.596667 allocated GPU-hours across jobs 39486/39488, including reused cohort work. "
        "Training and smoke overhead are reported separately from the full benchmark.", "",
        "## Evidence", "", f"- Frozen policy SHA-256: `{replay['freeze_sha256']}`.",
        f"- Local full metadata replay SHA-256: `{sha256(study/'full-metadata-replay.json')}`.",
        f"- Native full audit SHA-256: `{replay['native_audit_sha256']}`.",
        f"- Registered execution: `{replay['method_execution']}`.",
        f"- Final snapshot: `{data['snapshot'].relative_to(study).as_posix()}`.",
        "- Local replay checks score-cache inputs, failure penalties, references, decisions, registration and statistics. "
        "Full native images and neural scoring remain the cluster audit's evidence; the local replay does not recompute neural metrics.", ""]
    draw_gains(replay, output)
    (output/"REPORT.md").write_text("\n".join(lines), encoding="utf8")
    write_json(output/"report-receipt.json", dict(full_four_metric_goal_passed=passed,
        builder_sha256=sha256(Path(__file__)), replay_sha256=sha256(study/"full-metadata-replay.json"),
        snapshot=data["snapshot"].relative_to(study).as_posix(),
        files={path.name:sha256(path) for path in sorted(output.iterdir()) if path.is_file()}))
    return dict(report=str(output/"REPORT.md"), full_four_metric_goal_passed=passed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=ROOT/"output/shape-v2-slurm-20261123")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    print(json.dumps(build(args.study.resolve(), (args.output or args.study/"report").resolve()), indent=2))


if __name__ == "__main__":
    main()

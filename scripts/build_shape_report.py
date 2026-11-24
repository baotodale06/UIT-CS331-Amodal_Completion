"""Write report-ready results from the verified controlled panel and completed progressive audits."""
import argparse
import csv
import hashlib
import io
import json
from collections import Counter
from math import fsum, isclose
from pathlib import Path
from statistics import fmean
import subprocess
import sys
import tarfile

import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.analyze_surface_followup import paired_interval
from scripts.audit_surface_progressive_result import METRICS, estimate

OUT=ROOT/"output/shape-20261121"


def read(name):
    return json.loads((OUT/name).read_text(encoding="utf8"))


def severe_plate_section(manifest_digest):
    """Measure the new methods on the plate highlighted by the historical severity analysis."""
    from amodal_benchmark.dataset import sha256

    target = "COCOA/COCO_train2014_000000060172_plate"
    geometry = OUT / "evaluation_only/cocoa"
    lines = []
    for arm, name in (("shape", "Shape"), ("combined", "Shape + Surface")):
        directory = Path("progressive-l40s") / arm / "targets" / target
        result = OUT / directory / "result.json"
        if not result.is_file():
            continue
        record = json.loads(result.read_bytes())
        if record["id"] != target:
            raise ValueError("Native result belongs to a different COCOA object")
        if record["status"] != "success":
            lines.extend(["", f'{name} (L40S) không lưu được kết quả cho chiếc đĩa này và nhận IoU bằng 0. '
                          f'[Kết quả của {name}]({directory.as_posix()}/result.json)'])
            continue
        manifest = geometry / "manifest.json"
        if sha256(manifest) != manifest_digest:
            raise ValueError("Plate annotations differ from the historical COCOA evaluation")
        label = next(r for r in json.loads(manifest.read_bytes())["rows"] if r["id"] == target)
        annotation = geometry / label["case"]
        completion = OUT / directory / "completion.png"
        if sha256(annotation) != label["sha256"] or sha256(completion) != record["completion_sha256"]:
            raise ValueError("Plate outline or completion differs from its recorded hash")
        with Image.open(completion) as native, Image.open(annotation) as encoded:
            if native.mode != "RGBA" or encoded.mode != "L" or list(encoded.size) != record["geometry"]["original_size"]:
                raise ValueError("Plate masks do not match the recorded image format or orientation")
            top, left = record["geometry"]["offset_rc"]
            canvas = Image.new("L", encoded.size)
            canvas.paste(native.getchannel("A"), (left, top))
            mask = np.asarray(canvas) > 0
            flags = np.asarray(encoded)
            outline = (flags & 1) != 0
            hidden = outline & ((flags & 2) == 0)
        hidden_area = np.count_nonzero(hidden)
        if not hidden_area:
            raise ValueError("No hidden area in the severe plate annotation")
        hidden_overlap = np.count_nonzero(mask & hidden)
        iou = f'{np.count_nonzero(mask & outline) / np.count_nonzero(mask | outline):.4f}'.replace(".", ",")
        percent = f'{100 * hidden_overlap / hidden_area:.2f}'.replace(".", ",")
        counts = [f'{count:,}'.replace(",", ".") for count in (hidden_overlap, hidden_area)]
        lines.extend(["", f'Trên chiếc đĩa ở ảnh COCO train 060172, {name} (L40S) đạt IoU {iou} với '
                      f'đường bao toàn vật thể. Trong phần đĩa được chú thích là bị che, mặt nạ kết quả '
                      f'phủ {counts[0]} trên {counts[1]} pixel ({percent}%). '
                      f'[Mặt nạ của {name}]({directory.as_posix()}/completion.png) · '
                      f'[Chú thích COCOA](evaluation_only/cocoa/{label["case"]})'])
        trace = OUT / "shape-plate-trace.json"
        if arm == "shape" and trace.is_file():
            replay = json.loads(trace.read_bytes())
            figure = OUT / "shape-plate-first-pass.png"
            if (replay["target"] != target or replay["record_sha256"] != sha256(result)
                    or replay["manifest_sha256"] != manifest_digest or replay["annotation_sha256"] != label["sha256"]
                    or replay["hidden_pixels"] != hidden_area or replay["figure_sha256"] != sha256(figure)
                    or [row["decision_sha256"] for row in replay["passes"]] !=
                    [row["decision_sha256"] for row in record["diagnostics"]["shape_iterations"]]):
                raise ValueError("Plate selection figure or trace differs from the scored case")
            first = replay["passes"][0]
            chosen = first["candidates"][first["selected"]]
            best = first["candidates"][first["highest_iou_index"]]
            def vi(value, precision):
                return f"{value:.{precision}f}".replace(".", ",")
            lines.extend(["", f'Ở lượt đầu của ca này, toàn bộ vùng mà mặt nạ dự đoán có thể phủ '
                          f'chỉ chứa {vi(100*first["available"]["hidden_recall"], 2)}% phần đĩa bị che. '
                          f'Mặt nạ U-Net thực tế phủ {vi(100*first["prior"]["hidden_recall"], 2)}%. '
                          f'Trong {len(first["candidates"])} ứng viên, Shape chọn ảnh số {first["selected"]+1}, '
                          f'có IoU {vi(chosen["iou"], 3)} với đường bao toàn đĩa. '
                          f'Ảnh số {first["highest_iou_index"]+1} đạt IoU {vi(best["iou"], 3)} '
                          f'và phủ {vi(100*best["hidden_recall"], 1)}% phần đĩa bị che. '
                          'Ứng viên này có trong cùng lượt sinh. Đường bao COCOA chỉ dùng để chấm sau khi chọn. '
                          '[Điểm của từng ứng viên qua các lượt](shape-plate-trace.json)',
                          "", "![Ảnh gốc, mặt nạ dự đoán và hai ứng viên ở lượt đầu](shape-plate-first-pass.png)"])
    return lines


def vg_ssim_paragraph(replay, surface):
    rows = replay["rows"]
    if ([r["id"] for r in rows] != [tid for tid in surface if tid.startswith("VG/")]
            or len(rows) != replay["targets"]):
        raise ValueError("Shape and Surface must cover the same Visual Genome targets in order")
    shared = [r for r in rows if r["status"] == surface[r["id"]]["status"] == "success"]
    means = {arm: f"{value:.4f}".replace(".", ",") for arm, value in replay["means"].items()}
    count = f'{len(rows):,}'.replace(",", ".")
    paragraph = (f'Riêng {count} đối tượng Visual Genome, SSIM trung bình là '
                 f'{means["shape"]} cho Shape (L40S), {means["baseline"]} cho PMC và '
                 f'{means["surface"]} cho Surface (cả hai dùng kết quả Quadro RTX 6000 đã đăng ký). '
                 'SSIM cao hơn là tốt hơn. Mỗi ca không lưu được kết quả nhận điểm 0.')
    if shared:
        shape_mean = f'{fmean(r["SSIM"] for r in shared):.4f}'.replace(".", ",")
        surface_mean = f'{fmean(surface[r["id"]]["SSIM"] for r in shared):.4f}'.replace(".", ",")
        paragraph += (f' Trên {len(shared)} đối tượng mà cả Shape và Surface đều lưu được kết quả, '
                      f'SSIM trung bình là {shape_mean} cho Shape và {surface_mean} cho Surface.')
    return (paragraph + ' [Điểm Shape theo từng đối tượng](progressive-l40s/shape-vg-ssim.json) · '
            '[Điểm Surface](../surface-annotated-20261112-freshrefs/surface/per_target.csv)')


def vg_lpips_paragraph(replay, ssim, closed=None):
    early = replay["rows"]
    ids = [(r["id"], r["status"]) for r in early]
    if (ids != [(r["id"], r["status"]) for r in ssim["rows"]]
            or replay["control_csv_sha256"] != ssim["control_csv_sha256"]):
        raise ValueError("The two VG metrics use different targets, statuses or historical controls")
    comparisons = replay["comparisons"]
    if not isclose(fmean(r["LPIPS"] for r in early), comparisons["baseline"]["method"], rel_tol=0., abs_tol=1e-12):
        raise ValueError("Early VG mean LPIPS differs from the individual scores")
    source = ("CPU chấm các ảnh Shape đã sinh trên L40S. "
              "[Điểm LPIPS trên CPU](progressive-l40s/shape-vg-lpips.json)")
    if closed is not None:
        rows = [r for r in closed if r["source"] == "VG"]
        if ids != [(r["id"], r["status"]["shape"]) for r in rows]:
            raise ValueError("The closed VG scores differ from the early target inventory")
        comparisons = {}
        for arm in ("baseline", "surface"):
            pairs = [dict(source=r["source"], image_identity=r["image_identity"], baseline=r[arm], surface=r["shape"])
                     for r in rows]
            comparisons[arm] = estimate(pairs, "LPIPS")
        maximum = max(abs(r["shape"]["LPIPS"] - provisional["LPIPS"]) for r, provisional in zip(rows, early))
        gap = f'{maximum:.2g}'.replace(".", ",")
        source = (f'Điểm Shape lấy từ lượt chạy L40S đã xác minh. Khi chấm cùng các đầu ra, chênh lệch '
                  f'LPIPS tuyệt đối lớn nhất so với lần chấm sớm trên CPU là {gap}. '
                  '[Điểm chính thức](shape-full-pairs.json) · '
                  '[Điểm CPU](progressive-l40s/shape-vg-lpips.json)')
    pmc, surface = comparisons["baseline"], comparisons["surface"]
    if (pmc["n"] != len(ids) or surface["n"] != len(ids) or pmc["images"] != surface["images"]
            or not isclose(pmc["method"], surface["method"], rel_tol=0., abs_tol=1e-12)):
        raise ValueError("Different Shape denominator or mean across VG comparisons")
    def number(value, precision):
        return f'{value:.{precision}f}'.replace(".", ",")
    count = f'{len(ids):,}'.replace(",", ".")
    text = (f'Trên {count} đối tượng Visual Genome, LPIPS trung bình là '
            f'{number(pmc["method"], 4)} cho Shape, {number(pmc["baseline"], 4)} cho PMC '
            f'và {number(surface["baseline"], 4)} cho Surface. Hai đối chứng dùng kết quả Quadro đã đăng ký. '
            'LPIPS thấp hơn là tốt hơn. Mỗi ca thất bại nhận điểm 1.')
    for name, values in (("PMC", pmc), ("Surface", surface)):
        low, high = values["relative_ci_percent"]
        text += (f' Chênh lệch tương đối của Shape so với {name} là {number(-values["relative_gain_percent"], 2)}% '
                 f'(KTC 95% từ {number(-high, 2)}% đến {number(-low, 2)}%).')
    return text + ' ' + source


def cocoa_hidden_appearance(cocoa, dream):
    """Compare the raw RGB of the four candidates used in the COCOA shape experiment."""
    if cocoa["cases_sha256"] != dream["cases_sha256"]:
        raise ValueError("COCOA outlines and hidden RGB scores use different test images")
    rows = cocoa["rows"]
    if len({r["image"] for r in rows}) != len(rows):
        raise ValueError("Expected one controlled target per COCOA image")
    distances = {r["id"]: r for r in dream["rows"]}
    paired = {"LPIPS": [], "DreamSim": []}
    for row in rows:
        tid = row["id"]
        raw = (OUT / "controlled" / row["case_id"] / "result.json").read_bytes()
        checksum = hashlib.sha256(raw).hexdigest()
        if checksum != row["controlled_result_sha256"] or checksum != distances[tid]["decision_sha256"]:
            raise ValueError(f"Geometry and appearance use different candidate selections for {tid}")
        decision = json.loads(raw)
        slot = decision["indices"]["shape4"]
        if (decision["target_id"] != tid or not 0 <= slot < 4
                or row["amodal_iou"]["shape4"] != row["candidate_amodal_iou"][slot]):
            raise ValueError(f"Four-candidate selection differs from the scored silhouette for {tid}")
        lpips = [candidate["hidden_spatial_lpips"] for candidate in decision["candidate_metrics"][:4]]
        for name, pool in (("LPIPS", lpips), ("DreamSim", distances[tid]["candidate_distances"][:4])):
            if len(pool) != 4:
                raise ValueError(f"Incomplete {name} candidate scores for {tid}")
            paired[name].append((fmean(pool), pool[slot]))
    stats = {}
    for name, pairs in paired.items():
        control, selected = zip(*pairs)
        stats[name] = paired_interval(control, selected, ["COCOA"] * len(rows))
    return stats


def outline_status(encoded, crop):
    full = encoded.point(lambda value: 255 * (value & 1))
    visible = encoded.point(lambda value: 255 * ((value >> 1) & 1))
    bounds = full.getbbox()
    if bounds is None:
        raise ValueError("Empty COCOA amodal outline")
    if visible.tobytes() != full.tobytes():
        return "visible mask differs"
    left, top, right, bottom = bounds
    if left == 0 or top == 0 or right == encoded.width or bottom == encoded.height:
        return "touches image border"
    if left < crop[0] or top < crop[1] or right > crop[2] or bottom > crop[3]:
        return "clipped by crop"
    return "eligible"


def controlled_hidden_pixels(annotation, crop, occluder):
    labels = np.asarray(annotation.crop(crop).resize(occluder.size, Image.Resampling.NEAREST))
    inside = (labels & 1) != 0
    visible = ((labels & 2) != 0) & (np.asarray(occluder) == 0)
    total = int(inside.sum())
    if total == 0:
        raise ValueError("COCOA outline is empty inside the model input crop")
    hidden = int((inside & ~visible).sum())
    return hidden, total


def controlled_visibility_section(cocoa, archive):
    eligible = []
    counts = Counter()
    severity = [[] for _ in range(5)]
    difficult = []
    source = ROOT/"output/surface-confirmation128-20261110"
    case_bytes = (source/"cases.json").read_bytes()
    if hashlib.sha256(case_bytes).hexdigest() != cocoa["cases_sha256"]:
        raise ValueError("Controlled severity analysis and polygon audit use different cases")
    cases = {case["case_id"]: case for case in json.loads(case_bytes)}
    with tarfile.open(archive, "r:gz") as bundle:
        prefix = "evaluation_only/cocoa/"
        manifest = bundle.extractfile(prefix + "manifest.json").read()
        if hashlib.sha256(manifest).hexdigest() != cocoa["geometry_manifest_sha256"]:
            raise ValueError("Visibility screen and controlled audit use different COCOA annotations")
        annotations = {r["id"]: r for r in json.loads(manifest)["rows"]}
        for row in cocoa["rows"]:
            item = annotations[row["id"]]
            if item["ratio"] != row["original_occlusion"]:
                raise ValueError(f'Original occlusion differs for {row["id"]}')
            raw = bundle.extractfile(prefix + item["case"]).read()
            if hashlib.sha256(raw).hexdigest() != item["sha256"]:
                raise ValueError(f'COCOA mask differs for {row["id"]}')
            case = cases[row["case_id"]]
            if (case["target_id"], case["crop_box"]) != (row["id"], row["crop_box"]):
                raise ValueError(f'Controlled crop differs for {row["id"]}')
            occluder = (source/"inputs"/row["case_id"]/"occlusion.png").read_bytes()
            if hashlib.sha256(occluder).hexdigest() != case["input_hashes"]["occlusion.png"]:
                raise ValueError(f'Artificial occlusion mask differs for {row["id"]}')
            with Image.open(io.BytesIO(raw)) as mask, Image.open(io.BytesIO(occluder)) as added:
                hidden, total = controlled_hidden_pixels(mask, row["crop_box"], added)
                severity[min(4, 5 * hidden // total)].append(row["amodal_iou"])
                status = ("already occluded" if item["ratio"] > 0
                          else outline_status(mask.convert("L"), row["crop_box"]))
            if 0.5 <= hidden / total <= 0.8:
                difficult.append((row["amodal_iou"], status))
            if status == "already occluded":
                continue
            counts[status] += 1
            if status == "eligible":
                eligible.append(row)
    lines = ["", f'Of the {cocoa["matched"]} matched objects, {sum(counts.values())} had zero '
             f'annotated occlusion before the artificial mask was applied. Among those, '
             f'{counts["touches image border"]} had an outline touching the original image border. '
             'We required identical visible and amodal annotations, an outline clear of '
             'the image border, and an input crop containing the outline. '
             f'{len(eligible)} objects met these conditions. '
             'The table gives original-frame amodal IoU for each eligible object.',
             "", "| COCOA target | PMC seed 11 | Uniform K=4 | Shape K=4 | Polygon oracle K=4 | Uniform K=8 | Shape K=8 | Surface K=8 | Combined K=8 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in eligible:
        scores = row["amodal_iou"]
        values = [scores["first"], scores["uniform4"], scores["shape4"], max(row["candidate_amodal_iou"][:4]),
                  scores["uniform8"], scores["shape8"], scores["surface8"], scores["combined8"]]
        cells = [f"{value:.4f}" for value in values]
        lines.append("| " + " | ".join([row["id"].split("/", 1)[-1], *cells]) + " |")
    lines.extend(["", "The sample is too small to estimate how Shape performs on fully visible objects in general. "
                  "[COCOA annotation masks](geometry.tar.gz) · [Frozen candidate scores](controlled-cocoa.json)"])
    lines.extend(["", "#### Occlusion in the controlled COCOA crops", "",
                  "For each input crop, we counted annotated amodal pixels that were already hidden or fell "
                  "under the added occluder. The denominator is the annotated object area inside the crop. "
                  "The scores below are mean amodal IoU in the original image frame. Each method uses the "
                  "same four candidates per image.", "",
                  "| Hidden in the input crop | Images | Uniform K=4 | Shape K=4 |",
                  "|---|---:|---:|---:|"])
    for index, group in enumerate(severity):
        means = [f'{fmean(scores[method] for scores in group):.4f}' if group else "—"
                 for method in ("uniform4", "shape4")]
        lines.append(f"| {index * 20}–{(index + 1) * 20}% | {len(group)} | {' | '.join(means)} |")
    lines.extend(["", f'Only {len(severity[3]) + len(severity[4])} images had at least 60% of the annotated '
                  'object hidden in the crop. The groups are descriptive and combine natural and artificial occlusion. '
                  '[Controlled crop and occluder records](../surface-confirmation128-20261110/cases.json) · '
                  '[Original-frame IoU scores](controlled-cocoa.json)'])
    if difficult:
        screens = Counter(status for _, status in difficult)
        uniform, shape = (f'{fmean(scores[method] for scores, _ in difficult):.4f}'.replace(".", ",")
                          for method in ("uniform4", "shape4"))
        wins = sum(scores["shape4"] > scores["uniform4"] for scores, _ in difficult)
        lines.extend(["", f'Trong {len(difficult)} ảnh COCOA có từ 50% đến 80% diện tích vật thể bị che '
                      f'trong khung đầu vào, IoU trung bình với đường bao toàn vật thể là {uniform} khi '
                      f'chọn đều từ bốn ứng viên và {shape} khi dùng Shape. Shape vượt mức trung bình '
                      f'của bốn ứng viên ở {wins}/{len(difficult)} ảnh. Trước khi che nhân tạo, '
                      f'{screens["already occluded"]} vật thể đã bị che một phần và '
                      f'{screens["touches image border"]} vật thể chạm biên ảnh. Số ca trong nhóm này '
                      f'đáp ứng đủ tiêu chí nhìn thấy toàn bộ vật thể và nằm trọn trong khung là '
                      f'{screens["eligible"]}/{len(difficult)}.'])
    return lines


def allocation(stage, arms=("baseline", "shape", "surface", "combined")):
    jobs=OUT/"progressive-l40s/submissions.tsv"
    stages = ("cohort", "full") if stage == "full" else ("cohort",)
    with jobs.open(encoding="utf8") as stream:
        entries = {row[0]: (row[1], row[2]) for row in csv.reader(stream, delimiter="\t")
                   if row[1] in arms and row[2] in stages}
    raw = subprocess.check_output(["sacct", "-X", "-nP", "-j", ",".join(entries),
                                   "--format=JobIDRaw,State%32,ElapsedRaw,AllocTRES"], text=True)
    hours = dict.fromkeys(arms, 0.)
    completed = set()
    seen = set()
    for line in raw.splitlines():
        job, state, seconds, resources = line.split("|")
        arm, job_stage = entries[job]
        if job in seen:
            raise ValueError(f"Duplicate Slurm accounting record for job {job}")
        seen.add(job)
        state = state.split()[0]
        if state == "CANCELLED" and int(seconds) == 0 and not resources:
            continue
        if state not in {"COMPLETED", "TIMEOUT", "FAILED", "OUT_OF_MEMORY", "CANCELLED", "NODE_FAIL", "PREEMPTED"}:
            raise ValueError(f"Allocation for job {job} is still open ({state})")
        if "gres/gpu:nvidia_l40s=4" not in resources.split(","):
            raise ValueError(f"Missing four-L40S allocation for job {job}")
        hours[arm] += 4 * int(seconds) / 3600
        if state == "COMPLETED":
            completed.add((arm, job_stage))
    if seen != set(entries) or completed != {(arm, s) for arm in arms for s in stages}:
        raise ValueError(f"Incomplete Slurm accounting for the {stage} comparison")
    return hours


def severity_table(records, annotations, arms, metric="LPIPS", *, include_saved=False):
    """Reproduce the five Week 8 bins using the independently linked COCOA annotations."""
    rows = {row["id"]: row for row in records}
    if len(rows) != len(records):
        raise ValueError("Duplicate target in paired results")
    if len({item["id"] for item in annotations}) != len(annotations):
        raise ValueError("Duplicate COCOA annotation in severity groups")
    bins = [0., .2, .4, .6, .8, 1.00001]
    result = []
    for i, (lower, upper) in enumerate(zip(bins, bins[1:])):
        group = [rows[item["id"]] for item in annotations if lower <= item["ratio"] < upper]
        if not group:
            raise ValueError("Missing COCOA occlusion bin")
        entry = {"bin": f"{i*20}–{(i+1)*20}%", "n": len(group)}
        for arm in arms:
            entry[arm] = fmean(row[arm][metric] for row in group)
            if include_saved:
                entry[f"{arm}_saved"] = sum(row["status"][arm] == "success" for row in group)
        result.append(entry)
    if [row["n"] for row in result] != [203, 54, 39, 21, 4]:
        raise ValueError("COCOA target inventory differs from the severity chart")
    return result


def audited_pairs(stage, arm, summary):
    suffix = "-vs-pmc" if stage == "cohort" else ""
    raw = (OUT/f"{arm}-{stage}{suffix}-pairs.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != summary["pairs_sha256"]:
        raise ValueError(f"Paired evidence differs from the {arm} audit")
    rows = json.loads(raw)
    if len(rows) != summary["targets"]:
        raise ValueError(f"Unexpected {arm} target count")
    return rows


def archived_surface_cohort_row(directory, paired):
    raw = (directory/"surface-cohort/per_target.csv").read_bytes()
    if hashlib.sha256(raw).hexdigest() != paired["surface_scores_sha256"]:
        raise ValueError("Historical Surface scores differ from the cohort comparison")
    rows = list(csv.DictReader(io.StringIO(raw.decode("utf8"))))
    if len(rows) != paired["all"]["n"] or len(rows) != 204:
        raise ValueError("Incomplete historical Surface cohort")
    lpips = fmean(float(row["LPIPS"]) for row in rows)
    if not isclose(lpips, paired["all"]["baseline"], rel_tol=0., abs_tol=1e-12):
        raise ValueError("Historical Surface cohort mean differs from the paired audit")
    ssim = fmean(float(row["SSIM"]) for row in rows)
    saved = sum(row["status"] == "success" for row in rows)
    cost = json.loads((directory/"report/cohort-compute.json").read_bytes())["surface-cohort"]
    if (cost["job_id"], cost["state"]) != ("39487", "COMPLETED"):
        raise ValueError("Unexpected historical Surface cohort allocation")
    return (f'| Surface | Quadro RTX 6000 (historical) | 8 | {lpips:.6f} | {ssim:.6f} | '
            f'{saved} | {cost["reserved_gpu_hours"]:.3f} |')


def exact_reference_lpips(result, method, control, *, crops_only=False):
    low, high = result["relative_ci_percent"]
    matched = "cropped RGBA scoring images" if crops_only else "raw SAM masks, full RGBA images and crops"
    return (f'On the {result["n"]:,} targets from {result["images"]:,} images with byte-identical '
            f'{matched}, mean LPIPS was {result["method"]:.6f} for {method} on L40S and '
            f'{result["baseline"]:.6f} for {control} on Quadro. The relative gain for the L40S run '
            f'was {result["relative_gain_percent"]:+.3f}% (nominal 95% interval {low:+.3f}% to {high:+.3f}%).')


def cohort_scoring_crop_table(pairs, references, controls):
    by_id = {row["id"]: row for row in references}
    ids = [row["id"] for row in pairs]
    if len(ids) != len(set(ids)) or ids != list(by_id) or len(by_id) != len(references):
        raise ValueError("Scoring reference audit differs from the paired cohort inventory")
    matched = []
    for row in pairs:
        ref = by_id[row["id"]]
        if ref["current_artifacts"] != row["method_reference"] or ref["input_sha256"] != row["input_sha256"]:
            raise ValueError(f'Scoring reference mismatch for {row["id"]}')
        if ref["historical_artifacts"]["crop"] == row["method_reference"]["crop"]:
            matched.append(row)
    lines = ["", "#### Cross-device comparison with identical LPIPS scoring crops", "",
             "The released evaluator reads the cropped RGBA image. A raw SAM mask can differ by a few pixels "
             "while erosion leaves that image identical. Failed completions retain their LPIPS penalty. "
             "Positive gains favor the L40S method.", "",
             "| L40S method | Quadro control | Targets | L40S LPIPS | Quadro LPIPS | Relative gain | Nominal 95% interval |",
             "|---|---|---:|---:|---:|---:|---:|"]
    labels = {"baseline": "PMC", "shape": "Shape", "surface": "Surface", "combined": "Shape + Surface"}
    for arm, control in (("baseline", "baseline"), ("shape", "surface"), ("combined", "surface")):
        scores = controls[control]
        if ids != list(scores):
            raise ValueError(f"Historical {control} scores differ from the paired cohort inventory")
        contrast = []
        for row in matched:
            old = scores[row["id"]]
            if int(old["seed"]) != row["seed"] or old["source"] != row["source"]:
                raise ValueError(f'Historical seed or source differs for {row["id"]}')
            contrast.append(dict(source=row["source"], image_identity=row["image_identity"],
                                 baseline=old, surface=row[arm]))
        result = estimate(contrast, "LPIPS")
        lo, hi = result["relative_ci_percent"]
        lines.append(f'| {labels[arm]} | {labels[control]} | {result["n"]} | {result["method"]:.6f} | '
                     f'{result["baseline"]:.6f} | {result["relative_gain_percent"]:+.3f}% | [{lo:+.3f}%, {hi:+.3f}%] |')
    return lines


def appearance_table(estimates):
    lines = ["", "#### Shape and Shape + Surface on the same L40S inputs", "",
             "LPIPS, VGG feature similarity and SSIM include every target with the frozen failure penalties. "
             "CLIP uses targets with a score from both methods. Positive relative gains favor Shape + Surface. "
             "Each interval uses 5,000 paired bootstrap draws of images within each data source.", "",
             "| Metric | Paired targets | Shape | Shape + Surface | Relative gain | Nominal 95% interval |",
             "|---|---:|---:|---:|---:|---:|"]
    for metric, label in (("LPIPS", "LPIPS ↓"), ("Feature Similarity", "VGG feature similarity ↑"),
                          ("SSIM", "SSIM ↑"), ("CLIP", "CLIP ↑")):
        result = estimates[metric]
        if result is None:
            lines.append(f"| {label} | 0 | — | — | — | — |")
            continue
        point = result["relative_gain_percent"]
        low, high = result["relative_ci_percent"]
        gain = f"{point:+.3f}%" if point is not None else "—"
        interval = f"[{low:+.3f}%, {high:+.3f}%]" if low is not None else "—"
        lines.append(f'| {label} | {result["n"]:,} | {result["baseline"]:.6f} | {result["method"]:.6f} | '
                     f'{gain} | {interval} |')
    return lines


def surface_cohort_geometry(pairs, arm, historical_targets):
    archived = {row["id"]: row for row in historical_targets}
    comparison = []
    for row in pairs:
        control = archived.get(row["id"])
        if (control is not None) != ("amodal_iou" in row[arm]):
            raise ValueError(f'COCOA polygon membership differs for {row["id"]}')
        if control is None:
            continue
        if (control["source"], control["image_identity"]) != (row["source"], row["image_identity"]):
            raise ValueError(f'COCOA image identity differs for {row["id"]}')
        comparison.append(dict(source=row["source"], image_identity=row["image_identity"],
                               baseline=control["surface"], surface=row[arm]))
    return estimate(comparison, "amodal_iou")


def coverage_contributions(rows, arm, overall_gain, *, comparator="baseline"):
    groups = {(True, True): [], (False, True): [], (True, False): [], (False, False): []}
    for row in rows:
        saved = row["status"][comparator] == "success", row["status"][arm] == "success"
        groups[saved].append(row[comparator]["LPIPS"] - row[arm]["LPIPS"])
    result = {key: (len(values), fsum(values) / len(rows)) for key, values in groups.items()}
    if not isclose(fsum(gain for _, gain in result.values()), overall_gain, rel_tol=0, abs_tol=1e-12):
        raise ValueError("Coverage contributions differ from the audited all-target LPIPS difference")
    return result


def coverage_row(rows, arm, comparator, gain, labels, *, cross_device=False):
    groups = coverage_contributions(rows, arm, gain, comparator=comparator)
    cells = [f"{count:,} ({value:+.6f})" for count, value in groups.values()]
    control_name = labels[comparator] + (" (Quadro, cross-device)" if cross_device else "")
    return "| " + " | ".join([labels[arm], control_name, *cells]) + " |"


def matched_shape_rows(stage, summaries):
    shape = audited_pairs(stage, "shape", summaries["shape"])
    combined = audited_pairs(stage, "combined", summaries["combined"])
    ids = [r["id"] for r in shape]
    if len(ids) != len(set(ids)) or ids != [r["id"] for r in combined]:
        raise ValueError("Shape and combined require the same ordered target inventory")
    keys = ["source", "image_identity", "input_sha256", "seed", "method_reference",
            "reference_equal", "gt_bin", *summaries["shape"]["controls"]]
    if summaries["shape"].get("historical_controls"):
        keys.append("historical_scoring_crop_sha256")
    for row, match in zip(shape, combined):
        for key in keys:
            if row[key] != match[key]:
                raise ValueError(f"L40S comparison differs on {row['id']} / {key}")
        for control in summaries["shape"]["controls"]:
            if row["status"][control] != match["status"][control]:
                raise ValueError(f"Control status differs on {row['id']} / {control}")
        row["combined"] = match["combined"]
        row["status"]["combined"] = match["status"]["combined"]
    counts = summaries["combined"].get("first_pass_comparison")
    if counts is not None:
        actual = Counter(row["first_pass_comparison"] for row in combined)
        if dict(actual) != counts:
            raise ValueError("First-pass counts differ from the combined target records")
    return shape


def source_table(paired, historical):
    device = "Quadro" if historical else "L40S"
    comparison = " Both comparisons against PMC are cross-device." if historical else ""
    lines = ["", "#### Results by data source", "",
             f"Saved counts are in PMC ({device}), Shape (L40S), and Shape + Surface (L40S) order.{comparison} "
             "Each gain compares mean LPIPS across all targets from that source, including failures. "
             "Positive values favor the first method named in the column. The bracketed 95% intervals "
             "use paired image resampling and are descriptive and unadjusted for multiple comparisons.", "",
             "| Source | Targets | Saved (PMC / Shape / combined) | Shape vs PMC | Combined vs PMC | Combined vs Shape |",
             "|---|---:|---:|---:|---:|---:|"]
    for source in sorted({row["source"] for row in paired}):
        group = [row for row in paired if row["source"] == source]
        saved = [sum(row["status"][arm] == "success" for row in group)
                 for arm in ("baseline", "shape", "combined")]
        cells = [source, f"{len(group):,}", " / ".join(str(n) for n in saved)]
        for control, method in (("baseline", "shape"), ("baseline", "combined"), ("shape", "combined")):
            contrast = [dict(source=source, image_identity=row["image_identity"],
                             baseline=row[control], surface=row[method]) for row in group]
            result = estimate(contrast, "LPIPS")
            gain = result["relative_gain_percent"]
            lo, hi = result["relative_ci_percent"]
            point = f"{gain:+.2f}%" if gain is not None else "undefined"
            interval = f"[{lo:+.2f}%, {hi:+.2f}%]" if lo is not None else "[undefined]"
            cells.append(f"{point} {interval}")
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def render_severity(bins):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    arms = ("baseline", "shape", "surface", "combined")
    with (OUT/"slide-16.csv").open("w", newline="", encoding="utf8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(bins[0]))
        writer.writeheader()
        writer.writerows(bins)
    fig, ax = plt.subplots(figsize=(9.6, 4.8), constrained_layout=True)
    for arm, label, marker, color in zip(arms, ("PMC · Quadro", "Shape · L40S", "Surface · Quadro", "Shape + Surface · L40S"),
                                          ("o", "^", "s", "D"), ("#0067a8", "#a24917", "#168777", "#7140a0")):
        ax.plot(range(5), [row[arm] for row in bins], marker=marker, color=color, label=label, linewidth=2)
    ax.set_xticks(range(5))
    ax.set_xticklabels([f'{row["bin"]}\nn = {row["n"]}' for row in bins])
    ax.set_ylabel("LPIPS trung bình ↓ (tính cả thất bại)")
    ax.set_xlabel("Tỷ lệ che khuất theo chú thích COCOA")
    ax.legend(ncol=4, loc="lower center", bbox_to_anchor=(.5, 1), frameon=False)
    ax.grid(axis="y", alpha=.2)
    fig.savefig(OUT/"slide-16.png", dpi=190, facecolor="white")
    plt.close(fig)


def severity_section(records, available):
    labels = {"baseline": "PMC · Quadro", "shape": "Shape · L40S", "surface": "Surface · Quadro",
              "combined": "Shape + Surface · L40S"}
    arms = [arm for arm in labels if arm in available]
    annotations = read("evaluation_only/cocoa/manifest.json")["rows"]
    bins = severity_table(records, annotations, arms, include_saved=True)
    for entry, geometry in zip(bins, severity_table(records, annotations, arms, "amodal_iou")):
        entry.update({f"{arm}_amodal_iou": geometry[arm] for arm in arms})
    headings = ["Occlusion", "Targets", *(labels[arm] for arm in arms)]
    header = "| " + " | ".join(headings) + " |"
    separator = "|---|" + "---:|" * (1 + len(arms))
    lines = ["", "### Slide 20 · COCOA occlusion severity", "",
             "Each cell gives mean LPIPS across every target in the group, followed by the number of saved completions.",
             "", header, separator]
    for row in bins:
        cells = [f'{row[arm]:.4f} ({row[f"{arm}_saved"]} saved)' for arm in arms]
        lines.append("| " + " | ".join([row["bin"], str(row["n"]), *cells]) + " |")
    if len(arms) == len(labels):
        render_severity(bins)
        lines.extend(["", "![Mean LPIPS by annotated occlusion severity](slide-16.png)", "",
                      "[Slide 20 numeric values](slide-16.csv)"])
    else:
        pending = ", ".join(name for arm, name in labels.items() if arm not in arms)
        lines.extend(["", f"Full results pending verification for {pending}."])
    lines.extend(["", "Mean amodal IoU uses the same five groups and assigns zero to failed completions. Higher is better.",
                  "", header, separator])
    for row in bins:
        cells = [f'{row[f"{arm}_amodal_iou"]:.4f}' for arm in arms]
        lines.append("| " + " | ".join([row["bin"], str(row["n"]), *cells]) + " |")
    return lines


def unseen_geometry_section(rows, summaries):
    """Report silhouette recovery on the polygon-annotated part of the frozen unseen split."""
    polygons = [row for row in rows if "amodal_iou" in row["baseline"]]
    direct = estimate(polygons, "amodal_iou")
    if direct is None or (direct["n"], direct["images"]) != (211, 202):
        raise ValueError("Unexpected COCOA polygon inventory on unseen images")
    controls = {}
    for arm, column in (("shape", "baseline"), ("combined", "method")):
        for control in ("baseline", "surface"):
            audited = summaries[arm]["metrics"][f"{control}-unseen"]["amodal_iou"]
            if ((audited["n"], audited["images"]) != (direct["n"], direct["images"])
                    or not isclose(audited["method"], direct[column], rel_tol=0., abs_tol=1e-12)):
                raise ValueError(f"Unseen COCOA geometry differs from the {arm} audit")
            if control in controls and controls[control] != audited["baseline"]:
                raise ValueError(f"Historical {control} scores differ between unseen COCOA comparisons")
            controls[control] = audited["baseline"]
    low, high = direct["absolute_ci"]
    def number(value, spec=".4f"):
        return format(value, spec).replace(".", ",")

    return (f'Trên {direct["n"]} đối tượng có đường bao COCOA thuộc {direct["images"]} ảnh trong tập chưa thấy, '
            f'IoU trung bình là {number(controls["baseline"])} với PMC và {number(controls["surface"])} với Surface '
            f'trên Quadro. Trên L40S, Shape đạt {number(direct["baseline"])} và Shape + Surface đạt '
            f'{number(direct["method"])}. Chênh lệch giữa Shape + Surface và Shape là '
            f'{number(direct["absolute_gain"], "+.4f")} (KTC 95% từ {number(low, "+.4f")} đến {number(high, "+.4f")}). '
            'Các ca không lưu được kết quả nhận IoU bằng 0.')


def full_conclusion_vi(lpips, unseen, geometry, hours):
    """Summarize the direct L40S comparison for the Vietnamese Week 8 slides."""
    def number(value, places):
        return f"{value:.{places}f}".replace(".", ",")

    if (lpips["n"], unseen["n"], geometry["n"]) != (2565, 2017, 321):
        raise ValueError("Vietnamese full conclusion requires the three full-study denominators")
    # Express percent change relative to Shape. The audit reports gain for a lower LPIPS.
    change = -lpips["relative_gain_percent"]
    low, high = (-value for value in reversed(lpips["relative_ci_percent"]))
    result = ["", f'Trên 2.565 đối tượng tự nhiên, LPIPS trung bình là {number(lpips["baseline"], 4)} '
              f'với Shape và {number(lpips["method"], 4)} với Shape + Surface, tính cả các ca thất bại. '
              f'Khi thêm Surface vào Shape, LPIPS thay đổi {number(change, 2)}% '
              f'(KTC 95% từ {number(low, 2)}% đến {number(high, 2)}%). '
              'Giá trị âm biểu thị LPIPS giảm. Hai phương pháp dùng cùng ảnh và mặt nạ tham chiếu trên L40S.',
              "", f'Trên 2.017 đối tượng thuộc tập chưa thấy, LPIPS lần lượt là '
              f'{number(unseen["baseline"], 4)} với Shape và {number(unseen["method"], 4)} với Shape + Surface. '
              f'Trên 321 đối tượng COCOA có chú thích hình dạng đầy đủ, IoU trung bình lần lượt là '
              f'{number(geometry["baseline"], 4)} và {number(geometry["method"], 4)}. '
              'Mỗi ca không lưu được kết quả nhận IoU bằng 0.']
    if "shape" in hours and "combined" in hours:
        result.extend(["", f'Shape sử dụng {number(hours["shape"], 2)} giờ-GPU L40S và Shape + Surface '
                       f'sử dụng {number(hours["combined"], 2)} giờ-GPU L40S. '
                       'Số liệu PMC và Surface ở trang chiếu 19 đến từ các lượt chạy trước trên Quadro RTX 6000.'])
    return result


def historical_control_table_vi(summaries, labels):
    rows = []
    hidden_rows = []
    for arm, summary in summaries.items():
        if not summary["historical_controls"]:
            continue
        for control in summary["controls"]:
            estimates = summary["metrics"][control + "-all"]
            cells = [labels[arm], labels[control]]
            for metric, count in (("LPIPS", 2565), ("amodal_iou", 321)):
                result = estimates[metric]
                if result["n"] != count:
                    raise ValueError(f"Unexpected {metric} target count for {arm} versus {control}")
                low, high = result["absolute_ci"]
                cells.append(f'{result["absolute_gain"]:+.4f} ({low:+.4f} đến {high:+.4f})'.replace(".", ","))
            rows.append("| " + " | ".join(cells) + " |")
            hidden = estimates.get("hidden_iou")
            if hidden is not None:
                if (hidden["n"], hidden["images"]) != (230, 220):
                    raise ValueError("Unexpected hidden-region COCOA target or image count")
                low, high = hidden["absolute_ci"]
                values = (f'{hidden["method"]:.4f} | {hidden["baseline"]:.4f} | '
                          f'{hidden["absolute_gain"]:+.4f} ({low:+.4f} đến {high:+.4f})').replace(".", ",")
                hidden_rows.append(f'| {labels[arm]} | {labels[control]} | {values} |')
    if not rows:
        return []
    lines = ["", "### So sánh với PMC và Surface trên ảnh tự nhiên", "",
            "Giá trị dương biểu thị phương pháp trên L40S có LPIPS thấp hơn hoặc IoU cao hơn đối chứng. "
            "Số trong ngoặc là khoảng tin cậy 95% danh nghĩa của chênh lệch ghép cặp theo ảnh. "
            "Mỗi ca thất bại nhận LPIPS bằng 1 và IoU bằng 0 nếu có đường bao COCOA.", "",
            "| Phương pháp (L40S) | Đối chứng (Quadro RTX 6000) | Mức giảm LPIPS (2.565 đối tượng) | "
            "Mức tăng IoU (321 đối tượng COCOA) |", "|---|---|---:|---:|", *rows]
    if hidden_rows:
        lines.extend(["", "Trong 321 đối tượng COCOA, có 230 đối tượng thuộc 220 ảnh có ít nhất một "
                      "pixel bị che theo chú thích. Phân tích bổ sung dưới đây loại phần nhìn thấy theo "
                      "COCOA khỏi cả mặt nạ dự đoán và mặt nạ chuẩn trước khi tính IoU. "
                      "Mỗi ca thất bại nhận 0. Khoảng tin cậy 95% lấy mẫu lại theo ảnh.", "",
                      "| Phương pháp (L40S) | Đối chứng (Quadro) | IoU phần bị che, L40S | "
                      "IoU phần bị che, Quadro | Chênh lệch (KTC 95% danh nghĩa) |",
                      "|---|---|---:|---:|---:|", *hidden_rows])
    return lines


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--accounting",action="store_true")
    args=parser.parse_args()
    data=read("controlled-summary.json")
    audit=read("controlled-audit.json")
    assert audit["passed"] and audit["candidates"]==1024
    comp=data["summary"]
    lines=["# Shape-Guided PMC · Week 8 report update","",
           "On the same 128 artificial occlusions used in the Week 8 presentation, choosing the candidate whose silhouette best matched the predicted shape raised mean mask IoU from 0.7142 to 0.7856. Object LPIPS fell from 0.1803 to 0.1532. The mean paired IoU gain was 0.0714 (95% interval 0.0566 to 0.0873).",
           "", "Page numbers follow the 24-slide CS331 presentation checked on 22 November 2026. [Week 8 slides](https://www.canva.com/d/j8Vzc325LOrZKtc)",
           "", "Trang 13–15 và 18 có thể dùng phương pháp Shape và quy tắc kết hợp đã triển khai. "
           "Hai ví dụ chiếc ô và xe buýt ở trang 21 đã có ảnh cho cả bốn phương pháp. "
           "Trang 22 có kết quả chọn ảnh trên cùng tám ứng viên cho từng ảnh trong tập kiểm tra 128 ảnh. "
           "Bảng 2.565 đối tượng ở trang 19 và biểu đồ theo mức che khuất ở trang 20 "
           "dùng kết quả đã xác minh của từng lượt chạy đầy đủ.",
           "","## Slide 22 · Equal-budget ablation","",
           "Trên 128 ảnh che nhân tạo, mỗi bộ chọn K=8 dùng cùng tám ứng viên PMC cho từng ảnh. "
           "So với chọn ngẫu nhiên đều, Shape nâng IoU với mặt nạ SAM trước khi che từ 0,7142 lên 0,7856 "
           "và giảm LPIPS vật thể từ 0,1803 xuống 0,1532. Hàng chọn ngẫu nhiên lấy trung bình cả tám kết quả. "
           "Hàng oracle dùng mặt nạ trước khi che để chọn ứng viên có IoU cao nhất khi đánh giá.","",
           "| Selector | K | Mask IoU ↑ | Hidden IoU ↑ | Object LPIPS ↓ | Hidden spatial LPIPS ↓ | Object DreamSim ↓ | Hidden RGB MAE ↓ |",
           "|---|---:|---:|---:|---:|---:|---:|---:|"]
    names=[("first","PMC, seed 11",1),("uniform8","PMC, uniform",8),("shape8","Shape",8),
           ("surface8","Surface",8),("combined8","Shape + Surface",8),("oracle8","Oracle shape",8),
           ("uniform4","PMC, uniform",4),("shape4","Shape",4),
           ("shape_weighted4","Shape with visible recall and leakage terms",4),("oracle4","Oracle shape",4),
           ("shape_weighted8","Shape with visible recall and leakage terms",8)]
    for key,label,k in names:
        metrics=comp["uniform4" if k==4 else "uniform8"][key]
        vals=[metrics[m]["method"] for m in ("iou","hidden_iou","object_lpips","hidden_spatial_lpips","dreamsim","hidden_core_mae")]
        lines.append("| "+label+" | "+str(k)+" | "+" | ".join(f"{v:.4f}" for v in vals)+" |")
    added_shape = comp["surface8"]["combined8"]
    iou, hidden_iou = added_shape["iou"], added_shape["hidden_iou"]
    lpips, cosine = added_shape["object_lpips"], added_shape["clip_text_cosine"]
    lines.extend(["", f'Adding the Shape rank to Surface on the same eight candidates increased mask IoU by '
                  f'{iou["absolute_gain"]:.4f} (nominal 95% interval {iou["absolute_ci"][0]:.4f} to '
                  f'{iou["absolute_ci"][1]:.4f}) and hidden-region IoU by {hidden_iou["absolute_gain"]:.4f} '
                  f'({hidden_iou["absolute_ci"][0]:.4f} to {hidden_iou["absolute_ci"][1]:.4f}). '
                  f'Object LPIPS fell by {lpips["relative_gain_percent"]:.2f}% '
                  f'({lpips["relative_ci_percent"][0]:.2f}% to {lpips["relative_ci_percent"][1]:.2f}%). '
                  f'CLIP text cosine also fell, from {cosine["baseline"]:.4f} to {cosine["method"]:.4f}. '
                  'All four comparisons use the selected raw RGB images and their candidate masks. '
                  '[Paired ablation estimates](controlled-comparisons.csv)'])
    four = comp["uniform4"]["shape4"]
    lo, hi = four["iou"]["absolute_ci"]
    lines+=["", f'The chat proposal specifies four candidates. On the same 128 images, Shape gained {four["iou"]["absolute_gain"]:.4f} mask IoU over uniform K=4 (nominal 95% interval {lo:.4f} to {hi:.4f}). Hidden-region LPIPS fell from {four["hidden_spatial_lpips"]["baseline"]:.4f} to {four["hidden_spatial_lpips"]["method"]:.4f}. The K=4 oracle row shows the maximum silhouette IoU available in those four candidates.']
    if (OUT/"controlled-visible-ablation.json").is_file():
        visible = read("controlled-visible-ablation.json")
        if (visible["cases_sha256"] != read("training.json")["confirmation_sha256"] or
                visible["cocoa_sha256"] != hashlib.sha256((OUT/"controlled-cocoa.json").read_bytes()).hexdigest()):
            raise ValueError("Visible-mask comparison uses different cases or COCOA scores")
        lines.extend(["", "**What the predicted hidden silhouette adds.** In this post hoc comparison, "
                      "the control selects the candidate with the highest IoU against the visible mask alone. "
                      "Shape adds its predicted hidden pixels to that mask. Both rules see the same candidates, "
                      "exclude empty masks and break ties by seed order. Each cell gives the paired change "
                      "and nominal 95% interval. The first three metrics use all 128 images. The last uses "
                      "the 32 images with independent COCOA outlines.", "",
                      "| K | Changed selections (of 128) | Mask IoU gain | Hidden IoU gain | Object LPIPS reduction | COCOA amodal IoU gain |",
                      "|---:|---:|---:|---:|---:|---:|"])
        for k in ("4", "8"):
            table = visible["comparisons"][k]
            cells = [k, str(table["changed"])]
            for metric in ("iou", "hidden_iou", "object_lpips", "amodal_iou"):
                stat = table[metric]
                if metric != "amodal_iou" and not isclose(stat["method"], comp["uniform"+k]["shape"+k][metric]["method"],
                                                          rel_tol=0., abs_tol=1e-12):
                    raise ValueError(f"Shape mean differs from the frozen comparison for K={k} / {metric}")
                lo, hi = stat["absolute_ci"]
                cells.append(f'{stat["absolute_gain"]:+.4f} [{lo:+.4f}, {hi:+.4f}]')
            lines.append("| " + " | ".join(cells) + " |")
        hidden = visible["comparisons"]["8"]["hidden_iou"]
        lines.extend(["", f'With eight candidates, mean IoU inside the artificial occluder rose from '
                      f'{hidden["baseline"]:.4f} using the visible mask alone to {hidden["method"]:.4f} '
                      'using the learned prior. [Selections, native mask digests and paired estimates]'
                      '(controlled-visible-ablation.json)'])
    dream = read("hidden-dreamsim.json") if (OUT/"hidden-dreamsim.json").is_file() else None
    if dream:
        if len(dream["rows"]) != 128 or len({r["id"] for r in dream["rows"]}) != 128:
            raise ValueError("Hidden-region DreamSim requires all 128 frozen test objects")
        lines+=["", "A supplemental DreamSim measurement retains the known hidden-core RGB and fills all other pixels with gray in both images. The CPU evaluation uses the frozen candidates and selections.", "",
                "| Selection | K | Uniform hidden DreamSim ↓ | Selected hidden DreamSim ↓ | Absolute reduction (nominal 95% interval) |",
                "|---|---:|---:|---:|---:|"]
        for key,label,k in (("shape4","Shape",4),("shape8","Shape",8),("surface8","Surface",8),("combined8","Shape + Surface",8)):
            stat=dream["summary"][key]
            lo,hi=stat["absolute_ci"]
            lines.append(f'| {label} | {k} | {stat["baseline"]:.4f} | {stat["method"]:.4f} | {stat["absolute_gain"]:+.4f} [{lo:+.4f}, {hi:+.4f}] |')
        lines+=["", "[Hidden-core scores for every candidate and paired intervals](hidden-dreamsim.json)"]
    lines+=["","Shape reduced object LPIPS by 15.07% against uniform selection (95% interval 12.10% to 18.11%). Against Surface, Shape gained 0.0172 mask IoU (0.0031 to 0.0316) and reduced object LPIPS by 4.22% (0.43% to 8.02%). Surface had lower hidden RGB MAE and higher text cosine. The combined rule had the highest hidden IoU among the three deployable K=8 selectors. Its 0.0006 lead over Shape is a point estimate.",
           "","Each interval uses 5,000 paired, source-stratified image bootstrap draws and is nominal. The two sources contribute 64 images each. All 128 images and all eight seeds enter the comparison. Nine candidate masks were empty. Every pool retained an eligible candidate."]
    cocoa = read("controlled-cocoa.json") if (OUT/"controlled-cocoa.json").is_file() else None
    if cocoa:
        scores = cocoa["comparisons"]
        sam_iou = fmean(r["pre_occlusion_sam_iou"] for r in cocoa["rows"])
        lines.extend(["", "### Controlled images with COCOA amodal outlines", "",
                      f'Independent full-object outlines were available for {cocoa["matched"]} of the 128 controlled images. '
                      f'In {cocoa["previously_occluded"]} of these, COCOA marked part of the object as hidden before '
                      f'the artificial occlusion was added. The original pre-occlusion SAM masks, mapped back from '
                      f'the crops, had mean IoU {sam_iou:.4f} with the COCOA outlines.', "",
                      'This post hoc comparison scores the saved candidate masks against those outlines across the '
                      'original image frame, including annotated pixels outside each crop. Each row uses the same '
                      'candidates for the selector and its uniform control.', "",
                      '| Selector | K | Uniform IoU | Selected IoU | Paired difference (nominal 95% interval) |',
                      '|---|---:|---:|---:|---:|'])
        for key, label in (("shape4", "Shape"), ("shape8", "Shape"), ("surface8", "Surface"), ("combined8", "Shape + Surface")):
            measured = scores[key]
            lo, hi = measured["absolute_ci"]
            lines.append(f'| {label} | {key[-1]} | {measured["baseline"]:.4f} | {measured["method"]:.4f} | '
                         f'{measured["absolute_gain"]:+.4f} [{lo:+.4f}, {hi:+.4f}] |')
        contrasts = []
        control = [row["amodal_iou"]["surface8"] for row in cocoa["rows"]]
        for method, label in (("shape8", "Shape"), ("combined8", "Shape + Surface")):
            selected = [row["amodal_iou"][method] for row in cocoa["rows"]]
            paired = paired_interval(control, selected, ["COCOA"] * len(control), higher=True)
            low, high = paired["absolute_ci"]
            contrasts.append(f'{paired["absolute_gain"]:+.4f} for {label} ({low:+.4f} to {high:+.4f})')
        lines.extend(["", 'On the same eight candidates per image, the paired differences in mean amodal IoU '
                      f'versus Surface were {" and ".join(contrasts)}. Parentheses give nominal 95% intervals.'])
        lines.extend(["", "[Matched targets, candidate scores and annotation provenance](controlled-cocoa.json)"])
        first = fmean(row["amodal_iou"]["first"] for row in cocoa["rows"])
        oracle = fmean(max(row["candidate_amodal_iou"][:4]) for row in cocoa["rows"])
        lines.extend(["", f'For the four-candidate experiment in the proposal, PMC seed 11 reached '
                      f'{first:.4f} mean amodal IoU on these {cocoa["matched"]} images. Uniform selection '
                      f'scored {scores["shape4"]["baseline"]:.4f} and Shape scored '
                      f'{scores["shape4"]["method"]:.4f}. Choosing the highest-IoU candidate separately '
                      f'for each image gave {oracle:.4f}. This oracle reads the COCOA outline during '
                      'evaluation and measures the best silhouette present in each four-candidate pool.'])
        if dream:
            appearance = cocoa_hidden_appearance(cocoa, dream)
            lines.extend(["", f'Trên cùng {cocoa["matched"]} ảnh COCOA, mỗi ảnh dùng bốn ứng viên. '
                          'Cột đối chứng lấy trung bình bốn ứng viên của từng ảnh. Hai chỉ số đo độ sai khác '
                          'với màu và họa tiết gốc ở phần vật thể bị che nhân tạo. Giá trị thấp hơn là tốt hơn.',
                          "", "| Chỉ số | Đối chứng K=4 | Shape K=4 | Mức giảm (KTC 95% danh nghĩa) |",
                          "|---|---:|---:|---:|"])
            for name, values in appearance.items():
                lo, hi = values["absolute_ci"]
                lines.append((f'| {name} | {values["baseline"]:.4f} | {values["method"]:.4f} | '
                              f'{values["absolute_gain"]:+.4f} ({lo:+.4f} đến {hi:+.4f}) |').replace(".", ","))
            lines.extend(["", "Shape tăng IoU với đường bao toàn vật thể và giảm DreamSim trên phần che nhân tạo. "
                          "Mức giảm LPIPS còn chưa chắc chắn vì khoảng tin cậy bao gồm 0. "
                          "[Đường bao và điểm IoU](controlled-cocoa.json) · "
                          "[Điểm DreamSim theo ứng viên](hidden-dreamsim.json)"])
        lines.extend(controlled_visibility_section(cocoa, OUT/"geometry.tar.gz"))
    lines += ["","## Slides 13–15 and 18 · Geometry signal and integration","",
           "We implemented the five-channel U-Net and IoU selector from section 2 of the saved discussion, *Người 1 — Shape-Guided PMC*. The predicted mask scored 0.8122 mean IoU against the 128 pre-occlusion masks and 0.5716 within the hidden region.",
           "", "**Trang 15 · Ba quyết định của Shape**", "",
           "- **Tín hiệu hình học.** U-Net nhận ảnh RGB đã che, mặt nạ phần thấy của vật thể và mặt nạ vùng che. Mô hình giữ phần thấy và chỉ dự đoán thêm trong vùng che.",
           "- **Tích hợp vào PMC.** Ở mỗi lượt, Shape chọn ảnh có mặt nạ vật thể không rỗng đạt IoU cao nhất với mặt nạ dự đoán. PMC dùng ảnh đã chọn cho lượt sau nếu vật thể vẫn bị che.",
           "- **Đánh giá.** Trên 128 ảnh che nhân tạo, so sánh các bộ chọn trên cùng tám ứng viên bằng IoU với mặt nạ trước khi che và LPIPS với ảnh vật thể gốc. Với ảnh tự nhiên, đo IoU trên 321 đối tượng có đường bao toàn vật thể của COCOA.",
           "","The development oracle gained 0.0721 mean IoU over uniform K=4 on the 183 historical Surface development objects. We then trained only the auxiliary U-Net using those objects, rotated and flipped crops, 20–80% artificial occlusion, 40 epochs and BCE + Dice + 0.2 visible-retention loss. Image identities and source hashes are disjoint from the 128 confirmation images. The PMC generator and the Surface ranker stayed frozen.",
           "", "**Trang 18 · Quy tắc chọn chung.** Trong mỗi nhóm ứng viên hợp lệ, Shape xếp IoU từ cao xuống thấp, còn Surface xếp điểm mất mát từ thấp lên cao. Hai tiêu chí có trọng số bằng nhau. Chọn ứng viên có tổng thứ hạng nhỏ nhất. Khi tổng hạng bằng nhau, lấy seed đứng trước. Nếu cả nhóm không có ứng viên hợp lệ, dùng ảnh từ seed đầu tiên.",
           "", "The 128-image experiment returns the selected raw RGB. Both the four-seed source comparison and the eight-seed slide comparison use these same rules. The weighted variant uses the chat’s 0.60 IoU + 0.25 visible recall − 0.15 leakage formula as a secondary comparison.",
           "","Shape and Surface chose the same candidate on 23 of 128 images. On the 105 disagreements, the combined rule chose Shape’s candidate 34 times, Surface’s 45 times and a third candidate 26 times. Seed order decided 36 ties for the best combined rank. On these 105 images, mean mask IoU was 0.7963 for Shape, 0.7754 for Surface and 0.7928 for combined. Mean hidden-core RGB error was 0.1699, 0.1607 and 0.1629, respectively. These descriptive averages use the saved candidates from the controlled experiment.",
           "","The progressive implementation projects the original observed image, visible support, initial occluders and outpaint area into the current crop on every pass. It uses the Surface seed derivation and target-specific DINO/SAM resegmentation. Shape returns raw RGB. The combined progressive arm applies the frozen Surface RGB correction and native visible-pixel restoration to its jointly selected candidate. The released occlusion check determines whether to continue.",
           "","![U-Net prior and output silhouettes on an umbrella and white bowl](shape-silhouettes.png)","",
           "Shape beat the expected uniform IoU on 107 of 128 images and lost on 21. The umbrella comes from slide 21. The white bowl is the largest IoU loss versus uniform selection, chosen after evaluation to illustrate a failure. Its U-Net prior reached IoU 0.764. Shape selected a candidate with IoU 0.807 from a pool averaging 0.899, while seed 11 reached 0.939.",
           "","## Slide 21 · Matched qualitative examples","",
           "Ở ảnh chiếc ô, Shape chọn seed 44. So với PMC dùng seed 11, IoU với mặt nạ trước khi che tăng từ "
           "0,446 lên 0,839 và LPIPS vật thể giảm từ 0,388 xuống 0,087. Shape + Surface chọn cùng ảnh với Surface "
           "(seed 33). [Kết quả từng ứng viên của ảnh chiếc ô](controlled/VG-020-2beacd8b40/result.json)","",
           "Ở ảnh xe buýt, Shape và Shape + Surface đều chọn seed 11 nên cho cùng kết quả với PMC. "
           "Surface chọn seed 55, có IoU cao hơn (0,951 so với 0,939) và LPIPS vật thể cao hơn "
           "(0,174 so với 0,166). LPIPS thấp hơn biểu thị ảnh gần với vật thể gốc hơn. "
           "[Kết quả từng ứng viên của ảnh xe buýt](controlled/VG-007-23ffa2bcf0/result.json)","",
           "![Umbrella and bus with measured IoU and object LPIPS](slide-17.png)","",
           "## Slides 19–20 · Natural-image evaluation","",
           "The 128-image experiment hides about 35% of an originally visible SAM region. Its shape target is the silhouette before that artificial occlusion. Naturally hidden geometry, heavy occlusion and full progressive compute cost require separate measurements. Direct comparison with the CVPR paper requires its 3,000-case COCO/Open Images test set.",
           "", "For natural images, LPIPS and SSIM compare the saved completion with a crop of the original photograph showing the object through its eroded SAM mask. "
           "The evaluator places both on white backgrounds, so a correctly reconstructed hidden part can still differ from white in the reference. "
           "A failed completion receives LPIPS 1 and SSIM 0. [Appearance scoring protocol](../../OPEN_WORLD_REPRODUCTION.md)",
           "","Trên 204 đối tượng, nhóm dùng lại kết quả PMC đã chạy trên L40S. Trên 2.565 đối tượng, "
           "hai đối chứng PMC và Surface dùng kết quả đã đăng ký trên Quadro RTX 6000. "
           "PMC dùng một ứng viên mỗi bước, còn Surface, Shape và Shape + Surface dùng tám. "
           "Hai phương pháp trên L40S dùng chung mặt nạ tham chiếu, danh sách nhãn, seed, điểm phạt và quy tắc dừng. "
           "Chênh lệch so với Quadro còn có thể phụ thuộc vào phần cứng, phần mềm lấy mẫu và khác biệt ở một số mặt nạ tham chiếu. "
           "Tập 2.017 đối tượng chưa thấy loại các ảnh từng dùng trong thí nghiệm có kiểm soát.",
           "","A separate geometry measure compares final masks with external COCOA amodal polygons for 321 uniquely matched targets. We map the output into the original image using its saved offset, score inside the annotated frame and assign IoU zero when completion fails. Occlusion-bin comparisons are descriptive. The severe 80–100% bin includes four targets in the full inventory."]
    if (OUT/"unseen-image-audit.json").is_file():
        audit = read("unseen-image-audit.json")
        overlaps = len(audit["cross_study_flagged_targets"])
        duplicate_pairs = [f"{a} and {b}" for a, b in audit["within_subset_identical_pairs_across_ids"]]
        duplicate_note = ("Within the unseen subset, identical decoded pixels occur under "
                          + ", ".join(duplicate_pairs) + ". ") if duplicate_pairs else ""
        lines.extend(["", f'The full benchmark includes {audit["full_training_overlap_targets"]} targets '
                      'from images used to train Shape. For the frozen unseen analysis, we compared '
                      f'{audit["source_files"]:,} source files from {audit["image_identities"]:,} image IDs '
                      f'with all {audit["previous_images"]} earlier study images, including the '
                      f'{audit["training_images"]} Shape training images. Identity and file-hash checks found '
                      f'{len(audit["shared_identities"])} shared IDs and {len(audit["shared_file_hashes"])} shared '
                      f'file hashes. Decoded pixels and aligned thumbnails flagged {overlaps} unseen targets '
                      'across studies. The screen can miss crops and transformations. '
                      f'{duplicate_note}'
                      'The bootstrap groups targets by the original image IDs. '
                      '[Source-image overlap audit](unseen-image-audit.json)'])
    if (OUT/"reference-drift-cohort.json").is_file():
        drift = read("reference-drift-cohort.json")
        matching_crops = sum(row["historical_artifacts"]["crop"] == row["current_artifacts"]["crop"]
                             for row in drift["records"])
        lines.extend(["", f'The native reference audit found {drift["changed_masks"]} of {drift["targets"]} '
                      f'Quadro/L40S cohort masks differed across {drift["total_changed_mask_pixels"]} pixels in total. '
                      f'Crop geometry changed on {drift["changed_geometry"]} targets. The cropped RGBA image used '
                      f'for appearance scoring matched byte for byte on {matching_crops} targets. '
                      '[Reference audit](reference-drift-cohort.json)'])
    if (OUT/"pmc-cohort-cross-device.json").is_file():
        comparison = read("pmc-cohort-cross-device.json")
        overall = comparison["results"]["all"]["LPIPS"]
        exact = comparison["results"]["identical_references"]["LPIPS"]
        counts = comparison["coverage"]
        saved_quadro = counts["both_saved"] + counts.get("quadro_only", 0)
        saved_l40s = counts["both_saved"] + counts.get("l40s_only", 0)
        lo, hi = overall["relative_ci_percent"]
        lines.extend(["", f'On the {overall["n"]} frozen targets, the two existing PMC runs scored '
                      f'{overall["baseline"]:.4f} failure-inclusive LPIPS on Quadro ({saved_quadro} saved) and '
                      f'{overall["method"]:.4f} on L40S ({saved_l40s} saved). The estimated L40S relative gain '
                      f'was {overall["relative_gain_percent"]:+.2f}% (nominal 95% interval {lo:+.2f}% to {hi:+.2f}%). '
                      f'{exact_reference_lpips(exact, "PMC", "PMC")} '
                      'The comparison includes all runtime differences between these two runs. '
                      '[PMC cohort comparison](pmc-cohort-cross-device.json)'])
        lines.extend(["", "One COCOA dog (train image 108113) accounts for 97.1% of the observed mean "
                      "LPIPS gap between the two PMC runs. Both initially selected the same dog bounding box "
                      "at a detector score of 0.36. Their raw masks differed by one pixel. Their LPIPS scoring crops matched byte for byte. Quadro saved "
                      "a completion with LPIPS 0.208432. On L40S, completion segmentation returned an empty "
                      "mask array on the first iteration, so the target received the failure score of 1.0. "
                      "It contributed 0.003880 to the 0.003997 mean difference across 204 targets. The cause "
                      "of the empty mask array remains undetermined. "
                      "[Quadro trace](../surface-annotated-20261112-freshrefs/baseline-cohort/targets/COCOA/COCO_train2014_000000108113_dog/result.json) "
                      "· [L40S trace](progressive-l40s/baseline-cohort/targets/COCOA/COCO_train2014_000000108113_dog/result.json) "
                      "· [Mask comparison](reference-drift-cohort.json)"])
    if (OUT/"first-pass-cross-device-cohort.json").is_file():
        pool = read("first-pass-cross-device-cohort.json")["summary"]
        lines.extend(["", f'On {pool["targets"]} cohort targets, historical Surface on Quadro and Shape on L40S '
                      'had identical scoring references, first-pass crops, visible masks and seeds. '
                      f'Of the {pool["candidates"]} corresponding raw candidate pairs, '
                      f'{pool["identical_rgb"]} had identical RGB pixels and {pool["identical_masks"]} had identical '
                      'candidate masks. The mean absolute difference across RGB channel values was '
                      f'{pool["mean_absolute_rgb_difference_8bit"]:.3f} on a 0–255 scale. These differences '
                      'arose before candidate selection. The two runs also used different sampling wrappers, '
                      'so this comparison cannot isolate a GPU effect. '
                      '[First-pass pixel comparisons](first-pass-cross-device-cohort.json)'])
    polygons = read("cocoa-historical-controls.json") if (OUT/"cocoa-historical-controls.json").is_file() else None
    if polygons:
        geometry = polygons["all"]
        lo, hi = geometry["absolute_ci"]
        lines.extend(["", f'Across {geometry["n"]} matched COCOA targets, historical PMC and Surface scored '
                      f'{geometry["baseline"]:.4f} and {geometry["method"]:.4f} mean amodal IoU. '
                      f'The paired difference was {geometry["absolute_gain"]:+.4f} (nominal 95% interval '
                      f'{lo:+.4f} to {hi:+.4f}). The CPU audit placed each saved completion’s alpha mask '
                      'in the original image and assigned zero IoU to failed completions. '
                      '[Per-target geometry and severity groups](cocoa-historical-controls.json)'])
        lines.extend(["", "In the 80–100% occlusion group, both historical methods completed the same "
                      "two of four targets. The plate in COCO train image 060172 accounts for 0.2042 "
                      "of the 0.2075 decrease in mean amodal IoU from PMC to Surface. Its individual IoU "
                      "fell from 0.9809 to 0.1641. The other saved target was an elephant. "
                      "[Four target records](cocoa-historical-controls.json)"])
        lines.extend(severe_plate_section(polygons["manifest_sha256"]))
    header = ["| Method | GPU | K per pass | Failure-inclusive LPIPS ↓ | SSIM ↑ | Saved | GPU-hours allocated |",
              "|---|---|---:|---:|---:|---:|---:|"]
    historical_hours = dict(baseline=24.597, surface=116.406)
    full_conclusion = []
    for stage,size in [("cohort",204),("full",2565)]:
        summaries={m:read(m+"-"+stage+"-summary.json") for m in ["shape","combined"] if (OUT/(m+"-"+stage+"-summary.json")).exists()}
        for arm in ("shape", "combined"):
            early_name = f"{arm}-{stage}-vs-pmc-summary.json"
            if arm not in summaries and (OUT/early_name).is_file():
                summaries[arm] = read(early_name)
        lines+=["",f"### {size:,} natural targets",""]
        if not summaries:
            if stage == "full" and polygons:
                lines.extend(header)
                for arm, name, k in (("baseline","PMC",1),("surface","Surface",8)):
                    prior = polygons["sources"][arm]["overall"]
                    lines.append(f'| {name} | Quadro RTX 6000 (historical) | {k} | {prior["LPIPS"]:.6f} | '
                                 f'{prior["SSIM"]:.6f} | {prior["successes"]:,} | {historical_hours[arm]:.3f} |')
                lines.extend(["", "[Historical verification and Slurm costs](../surface-annotated-full-review-20261114/REPORT.md)", ""])
            lines.append("Shape and Shape + Surface scores are pending verification.")
            continue
        controls = ["baseline"]
        surface_reference = next((r for r in summaries.values() if "surface-all" in r["metrics"]), None)
        if surface_reference:
            controls.append("surface")
        arms = [*controls, *summaries]
        historical = stage == "full"
        cost=allocation(stage, list(summaries) if historical else arms) if args.accounting else {}
        if historical:
            cost.update(historical_hours)
        lines.extend(header)
        reference=next(iter(summaries.values()))
        labels = {"baseline":"PMC", "surface":"Surface", "shape":"Shape", "combined":"Shape + Surface"}
        for arm in arms:
            result=surface_reference if arm=="surface" else summaries.get(arm,reference)
            metric=result["metrics"][("surface" if arm=="surface" else "baseline")+"-all"]
            column="baseline" if arm in ("baseline","surface") else "method"
            hours=f'{cost[arm]:.3f}' if arm in cost else "—"
            gpu="Quadro RTX 6000 (historical)" if historical and arm in controls else "L40S"
            k=1 if arm=="baseline" else 8
            lines.append(f'| {labels[arm]} | {gpu} | {k} | {metric["LPIPS"][column]:.6f} | {metric["SSIM"][column]:.6f} | {result["coverage"][arm].get("success",0)} | {hours} |')
        if stage == "cohort" and not surface_reference and reference.get("historical_surface_cohort"):
            archived = reference["historical_surface_cohort"]
            directory = ROOT/"output/surface-annotated-20261112-freshrefs"
            lines.extend([archived_surface_cohort_row(directory, archived), "",
                          "The Surface row reuses job 39487 on the same 204 targets. Comparisons with the "
                          "L40S rows are cross-device. Raw masks, full RGBA images and crops were all byte-identical on "
                          f'{archived["identical_references"]["n"]} targets. '
                          '[Historical Surface scores](../surface-annotated-20261112-freshrefs/surface-cohort/per_target.csv) · '
                          '[Surface allocation](../surface-annotated-20261112-freshrefs/report/cohort-compute.json)'])
        if cost:
            lines+=["", "GPU-hours use elapsed Slurm allocation time on four GPUs of the stated type. Full totals include reused cohort work and interrupted allocations. [Historical cost accounting](../surface-annotated-full-review-20261114/REPORT.md)"]
        for arm,result in summaries.items():
            table=result["metrics"]["baseline-all"]
            for comparator,label in [("baseline","PMC"),("surface","Surface")]:
                if comparator+"-all" not in result["metrics"]:
                    continue
                diff=result["metrics"][comparator+"-all"]["LPIPS"]
                lo,hi=diff["relative_ci_percent"]
                comparison=f"historical {label} (cross-device)" if historical else label
                lines+=["",f'{arm.capitalize()} relative LPIPS difference versus {comparison} was {diff["relative_gain_percent"]:+.3f}% (nominal 95% interval {lo:+.3f}% to {hi:+.3f}%, positive favors {arm}).']
                if historical:
                    same=result["metrics"][comparator+"-same-reference"]["LPIPS"]
                    lines.append(exact_reference_lpips(same, labels[arm], label))
                    crop = result["metrics"][comparator+"-same-scoring-crop"]["LPIPS"]
                    if crop["n"] != same["n"]:
                        lines.append(exact_reference_lpips(crop, labels[arm], label, crops_only=True))
            historical_cohort = result.get("historical_surface_cohort")
            if historical_cohort:
                contrast = historical_cohort["all"]
                exact = historical_cohort["identical_references"]
                lo, hi = contrast["relative_ci_percent"]
                lines.extend(["", f'At eight seeds per pass, {arm} scored {contrast["method"]:.6f} LPIPS on L40S '
                              f'and the published Surface cohort scored {contrast["baseline"]:.6f} on Quadro. '
                              f'The cross-device relative gain was {contrast["relative_gain_percent"]:+.3f}% '
                              f'(nominal 95% interval {lo:+.3f}% to {hi:+.3f}%). '
                              f'{exact_reference_lpips(exact, labels[arm], "Surface")}'])
            geo=table.get("amodal_iou")
            if geo:
                lo, hi = geo["absolute_ci"]
                lines.extend(["", f'On {geo["n"]} COCOA targets from {geo["images"]} images, '
                              f'{labels[arm]} scored {geo["method"]:.4f} mean amodal IoU versus '
                              f'{geo["baseline"]:.4f} for PMC. The absolute difference was '
                              f'{geo["absolute_gain"]:+.4f} (nominal 95% interval {lo:+.4f} to {hi:+.4f}).'])
                surface_geo = None
                if result.get("historical_controls"):
                    surface_geo = result["metrics"]["surface-all"]["amodal_iou"]
                elif historical_cohort and polygons:
                    surface_geo = surface_cohort_geometry(audited_pairs(stage, arm, result), arm, polygons["targets"])
                if surface_geo:
                    if ((surface_geo["n"], surface_geo["images"]) != (geo["n"], geo["images"])
                            or not isclose(surface_geo["method"], geo["method"], rel_tol=0., abs_tol=1e-12)):
                        raise ValueError(f"Historical Surface and PMC comparisons use different {arm} geometry")
                    low, high = surface_geo["absolute_ci"]
                    difference = f'{surface_geo["absolute_gain"]:+.4f} (KTC 95% từ {low:+.4f} đến {high:+.4f})'
                    means = [f'{surface_geo[key]:.4f}'.replace(".", ",") for key in ("method", "baseline")]
                    lines.extend(["", f'Trên {geo["n"]} đối tượng COCOA thuộc {geo["images"]} ảnh, '
                                  f'IoU trung bình của {labels[arm]} trên L40S là {means[0]} và của Surface '
                                  f'trên Quadro RTX 6000 là {means[1]}. Chênh lệch giữa hai lượt chạy là '
                                  f'{difference.replace(".", ",")}. Cả hai dùng tám ứng viên mỗi bước. '
                                  'Mỗi ca không lưu được kết quả nhận IoU bằng 0. '
                                  '[Điểm Surface theo từng đối tượng](cocoa-historical-controls.json)'])
            pools = result["pools"]
            lines.extend(["", f'{labels[arm]} recorded {pools:,} eight-candidate passes '
                          f'({8 * pools:,} candidate images). In {result["fallbacks"]:,} of those passes, '
                          'no candidate qualified and the selector used the first seed.'])
            evidence = f"{arm}-{stage}" + ("-vs-pmc" if stage == "cohort" else "")
            lines.extend(["", f'[Paired target scores]({evidence}-pairs.json) · [Estimates and intervals]({evidence}-summary.json)'])
            if stage=="full":
                reuse = result.get("cohort_reuse")
                if reuse:
                    lines.extend(["", f'The full {arm} execution preserved all {reuse["preserved_target_records"]} '
                                  'target records from its verified cohort snapshot byte for byte.'])
                for comparator in result["controls"]:
                    unseen = result["metrics"][f"{comparator}-unseen"]["LPIPS"]
                    if (unseen["n"], unseen["images"]) != (2017, 1900):
                        raise ValueError("Unexpected target or image count for the unseen comparison")
                    lo, hi = unseen["relative_ci_percent"]
                    lines.extend(["", f'On the 2,017 unseen targets, {labels[arm]} (L40S) scored '
                                  f'{unseen["method"]:.6f} LPIPS versus {unseen["baseline"]:.6f} for '
                                  f'{labels[comparator]} (Quadro). The cross-device relative gain was '
                                  f'{unseen["relative_gain_percent"]:+.3f}% (nominal 95% interval '
                                  f'{lo:+.3f}% to {hi:+.3f}%).'])
        lines.extend(["", "Each cell gives the number of targets and their contribution to the mean LPIPS "
                      f"difference across all {size:,} targets. Positive values favor the method in the first column.", "",
                      "| Method | Comparator | Both saved | Only method saved | Only comparator saved | Both failed |",
                      "|---|---|---:|---:|---:|---:|"])
        for arm, summary in summaries.items():
            rows = audited_pairs(stage, arm, summary)
            for comparator in summary["controls"]:
                gain = summary["metrics"][f"{comparator}-all"]["LPIPS"]["absolute_gain"]
                lines.append(coverage_row(rows, arm, comparator, gain, labels, cross_device=historical))
        if set(summaries) == {"shape","combined"}:
            paired = matched_shape_rows(stage, summaries)
            table = [dict(id=r["id"], source=r["source"], image_identity=r["image_identity"],
                          baseline=r["shape"], surface=r["combined"]) for r in paired]
            appearance = {metric: estimate(table, metric) for metric in METRICS}
            lpips = appearance["LPIPS"]
            lines.append(coverage_row(paired, "combined", "shape", lpips["absolute_gain"], labels))
            lines.extend(appearance_table(appearance))
            lines.append("")
            if "combined" in cost and "shape" in cost:
                allocation_ratio = cost["combined"] / cost["shape"]
                lines.append(f'Shape + Surface used {allocation_ratio:.3f} times the L40S GPU-hours of Shape '
                             f'across these {size:,} targets. The two methods ran '
                             f'{summaries["combined"]["pools"]:,} and {summaries["shape"]["pools"]:,} '
                             'eight-candidate sampling passes, respectively.')
            shared = [row for row in paired if row["status"]["shape"] == row["status"]["combined"] == "success"]
            if shared:
                lines.append(f'Among the {len(shared):,} targets where both methods saved outputs, '
                             f'mean LPIPS was {fmean(row["shape"]["LPIPS"] for row in shared):.6f} for Shape '
                             f'and {fmean(row["combined"]["LPIPS"] for row in shared):.6f} for Shape + Surface.')
            if stage == "cohort" and (OUT/"cohort-local-replay.json").is_file():
                replay = read("cohort-local-replay.json")
                for arm, summary in summaries.items():
                    if replay["pairs_sha256"][f"{arm}-cohort-vs-pmc-pairs.json"] != summary["pairs_sha256"]:
                        raise ValueError("Local cohort replay used different paired records")
                for metric, result in appearance.items():
                    independent = replay["same_device_combined_vs_shape"][metric]
                    for key in ("n", "images", "baseline", "method", "relative_gain_percent",
                                "absolute_ci", "relative_ci_percent"):
                        if not np.allclose(result[key], independent[key], rtol=0., atol=1e-10):
                            raise ValueError(f"Local {metric} cohort estimate differs on {key}")
                lines.extend(["", "An independent local calculation reproduced the paired intervals from "
                              "5,000 image-cluster bootstrap samples stratified by source. "
                              "[Local cohort replay](cohort-local-replay.json)"])
            counts = summaries["combined"].get("first_pass_comparison")
            if counts:
                changed = counts.get("changed_slot", 0)
                common = changed + counts.get("same_slot", 0)
                lines.extend(["", f'On the first pass, {common} targets had identical seeds, crops, saved prior '
                              f'inputs, predicted masks and all eight candidate image/mask pairs in both methods. '
                              f'The joint rule selected a different candidate on {changed} of these targets. '
                              f'Neither method sampled a first candidate pool on {counts.get("neither_sampled",0)} '
                              f'of {size:,} targets. Subsequent passes use each method’s selected image.'])
                mismatched = counts.get("different_inputs_or_pool", 0)
                shape_only = counts.get("shape_only", 0)
                combined_only = counts.get("combined_only", 0)
                if common + counts.get("neither_sampled", 0) + mismatched + shape_only + combined_only != size:
                    raise ValueError("First-pass outcomes do not cover the target inventory")
                if mismatched or shape_only or combined_only:
                    lines.append(f'Both methods sampled with different inputs or candidates on {mismatched:,} '
                                 f'targets. Only Shape sampled on {shape_only:,}, and only Shape + Surface sampled '
                                 f'on {combined_only:,}.')
            geom = estimate([r for r in table if "amodal_iou" in r["baseline"]], "amodal_iou")
            if geom:
                lo, hi = geom["absolute_ci"]
                lines.append(f'Across the {geom["n"]} COCOA polygon matches, combined minus Shape amodal IoU was {geom["absolute_gain"]:+.4f} (nominal 95% interval {lo:+.4f} to {hi:+.4f}).')
            if stage == "cohort":
                bed = next(row for row in paired if row["id"] == "COCOA/COCO_train2014_000000009199_bed")
                lpips_values = [f'{bed[arm]["LPIPS"]:.4f}'.replace(".", ",") for arm in ("shape", "combined")]
                iou_values = [f'{bed[arm]["amodal_iou"]:.4f}'.replace(".", ",") for arm in ("shape", "combined")]
                lines.extend(["", f"Với chiếc giường trong ảnh COCO 009199, LPIPS giảm từ {lpips_values[0]} "
                              f"(Shape) xuống {lpips_values[1]} (Shape + Surface). IoU với đường bao toàn bộ "
                              f"chiếc giường cũng giảm, từ {iou_values[0]} xuống {iou_values[1]}. "
                              "[Điểm Shape](shape-cohort-vs-pmc-pairs.json) · "
                              "[Điểm Shape + Surface](combined-cohort-vs-pmc-pairs.json)"])
                lines.extend(["", "Of the 22 natural cohort targets with COCOA outlines, only one had at least "
                              "60% annotated occlusion. The plate in COCO train image 024086 was 82.4% hidden. "
                              "PMC, Shape and Shape + Surface all rejected its visible mask as too small before "
                              "generating any candidates. This cohort therefore provides no comparison of "
                              "candidate selection at that severity. "
                              "[Occlusion measurements](cocoa-historical-controls.json) · "
                              "[Shape scores and stopping reasons](progressive-l40s/shape-cohort/per_target.csv) · "
                              "[Combined scores and stopping reasons](progressive-l40s/combined-cohort/per_target.csv)"])
            lines.extend(source_table(paired, historical))
            if stage == "cohort" and (OUT/"reference-drift-cohort.json").is_file():
                from scripts.compare_shape_progressive import execution_scores
                controls_dir = ROOT/"output/surface-annotated-20261112-freshrefs"
                score_tables = {}
                for control in ("baseline", "surface"):
                    run = controls_dir/(control + "-cohort")
                    if json.loads((run/"metric_config.json").read_bytes())["alignment"] != "released":
                        raise ValueError("Scoring-crop comparison requires released metric alignment")
                    score_tables[control] = execution_scores(run, json.loads((run/"execution.json").read_bytes()))
                lines.extend(cohort_scoring_crop_table(paired, drift["records"], score_tables))
            if stage == "full":
                from amodal_benchmark.cohort import select_manifest_rows
                inventory = ROOT/"output/surface-annotated-20261112-design/unseen-inventory.json"
                if hashlib.sha256(inventory.read_bytes()).hexdigest() != read("unseen-image-audit.json")["inventory_sha256"]:
                    raise ValueError("Unseen target inventory differs from the source-image audit")
                unseen_rows = select_manifest_rows(table, inventory)
                unseen = estimate(unseen_rows, "LPIPS")
                for column, arm in (("baseline", "shape"), ("method", "combined")):
                    audited = summaries[arm]["metrics"]["baseline-unseen"]["LPIPS"]
                    if ((unseen["n"], unseen["images"]) != (audited["n"], audited["images"])
                            or not isclose(unseen[column], audited["method"], rel_tol=0., abs_tol=1e-12)):
                        raise ValueError(f"Unseen LPIPS differs from the {arm} audit")
                lo, hi = unseen["relative_ci_percent"]
                full_conclusion = full_conclusion_vi(lpips, unseen, geom, cost)
                lines.extend(["", f'On the same {unseen["n"]:,} unseen targets, Shape scored '
                              f'{unseen["baseline"]:.6f} LPIPS and Shape + Surface scored {unseen["method"]:.6f}. '
                              f'The relative gain for the combined method on L40S was '
                              f'{unseen["relative_gain_percent"]:+.3f}% (nominal 95% interval {lo:+.3f}% to {hi:+.3f}%).'])
                lines.extend(["", unseen_geometry_section(unseen_rows, summaries)])
        elif historical:
            arm, summary = next(iter(summaries.items()))
            paired = audited_pairs(stage, arm, summary)
        if historical:
            lines.extend(severity_section(paired, arms))
            full_conclusion.extend(historical_control_table_vi(summaries, labels))
    vg_path = OUT / "progressive-l40s/shape-vg-ssim.json"
    if vg_path.is_file():
        from scripts.compare_shape_progressive import execution_scores
        vg = json.loads(vg_path.read_bytes())
        surface_run = ROOT / "output/surface-annotated-20261112-freshrefs/surface"
        execution = json.loads((surface_run / "execution.json").read_bytes())
        if execution["artifacts"]["per_target.csv"] != vg["control_csv_sha256"]["surface"]:
            raise ValueError("Visual Genome SSIM comparison uses a different Surface execution score table")
        lines.extend(["", "### SSIM trên Visual Genome", "",
                      vg_ssim_paragraph(vg, execution_scores(surface_run, execution))])
        lpips_path = OUT / "progressive-l40s/shape-vg-lpips.json"
        if lpips_path.is_file():
            replay = json.loads(lpips_path.read_bytes())
            if replay["ssim_sha256"] != hashlib.sha256(vg_path.read_bytes()).hexdigest():
                raise ValueError("The VG LPIPS replay uses different scoring inputs")
            summary = OUT / "shape-full-summary.json"
            closed = audited_pairs("full", "shape", json.loads(summary.read_bytes())) if summary.is_file() else None
            lines.extend(["", "### LPIPS trên Visual Genome", "", vg_lpips_paragraph(replay, vg, closed)])
    lines+=["","## Slides 23–24 · Limitations and conclusions","","Trên 128 ảnh che nhân tạo, khi giữ nguyên tám ứng viên cho mỗi ảnh, chọn theo hình dáng nâng IoU trung bình từ 0,7142 lên 0,7856 và giảm LPIPS vật thể từ 0,1803 xuống 0,1532. Với chiếc bát trắng, mặt nạ dự đoán dẫn đến việc chọn một ứng viên có IoU thấp hơn mức trung bình của nhóm. Trong dữ liệu che khuất tự nhiên, nhóm che 80–100% chỉ có bốn đối tượng COCOA có chú thích để đối chiếu."]
    if cocoa:
        labels = read("training-mask-audit.json")
        training = (OUT/"training.json").read_bytes()
        if (labels["training_sha256"] != hashlib.sha256(training).hexdigest()
                or labels["geometry_manifest_sha256"] != cocoa["geometry_manifest_sha256"]):
            raise ValueError("Training label audit uses different training or COCOA evidence")
        visible_iou = f'{labels["mean_visible_iou"]:.5f}'.replace(".", ",")
        amodal_iou = f'{labels["mean_amodal_iou"]:.5f}'.replace(".", ",")
        screen = labels["eligibility"]
        lines.extend(["", f'Trong {labels["verified_training_objects"]} vật thể dùng để huấn luyện U-Net, '
                      f'{labels["matched_cocoa_objects"]} vật thể có đường bao COCOA độc lập khớp đúng mã. '
                      f'Trước khi che nhân tạo, {screen["already occluded"]} vật thể đã bị che một phần và '
                      f'{screen["touches image border"]} vật thể khác chạm biên ảnh. Chỉ một ô tô và một con mèo '
                      'đáp ứng đủ tiêu chí nhìn thấy toàn bộ vật thể và nằm trọn trong khung. '
                      'Khi chiếu nhãn SAM huấn luyện 256 × 256 về khung ảnh gốc, IoU trung bình trên '
                      f'{labels["matched_cocoa_objects"]} vật thể là {visible_iou} với phần nhìn thấy và '
                      f'{amodal_iou} với toàn bộ vật thể. Trên tập con này, nhãn huấn luyện phản ánh '
                      'phần nhìn thấy tốt hơn toàn bộ hình dáng. '
                      '[Đối chiếu nhãn huấn luyện theo từng vật thể](training-mask-audit.json)'])
        lines.extend(["", f'Trong {cocoa["matched"]} ảnh thí nghiệm che nhân tạo có đường bao toàn vật thể của COCOA, '
                      f'{cocoa["previously_occluded"]} vật thể đã bị che một phần trong ảnh gốc. '
                      'Vì vậy, điểm IoU trên cả 128 ảnh đo mức khôi phục mặt nạ SAM ban đầu. '
                      'Trong các ca có đường bao COCOA, chỉ chiếc bát và con mèo đáp ứng đủ tiêu chí thấy toàn bộ '
                      'vật thể trước khi che, không chạm biên ảnh và nằm trọn trong khung đầu vào. '
                      'Mặt nạ nhân tạo che 24,43% diện tích chiếc bát và 34,93% diện tích con mèo. '
                      'Cả hai đều dưới ngưỡng che khuất 50% của nhóm khó. So với chọn đều từ tám ứng viên, '
                      'Shape làm giảm IoU ở chiếc bát và tăng IoU ở con mèo.'])
    cohort_scores = OUT/"shape-cohort-vs-pmc-pairs.json"
    if cohort_scores.is_file():
        rows = json.loads(cohort_scores.read_bytes())
        polygons = [r for r in rows if "amodal_iou" in r["shape"]]
        matched = sum(r["status"]["baseline"] == r["status"]["shape"] == "success" for r in polygons)
        baseline_saved = sum(r["status"]["baseline"] == "success" for r in polygons)
        shape_saved = sum(r["status"]["shape"] == "success" for r in polygons)
        lines.extend(["", "Trên 204 đối tượng tự nhiên, Shape có LPIPS 0,6974 và PMC có LPIPS 0,6893 "
                      "(thấp hơn là tốt hơn). Khoảng tin cậy 95% của chênh lệch bao gồm 0. "
                      f'Trong {len(polygons)} đối tượng COCOA có mặt nạ chuẩn, PMC lưu được {baseline_saved} '
                      f'kết quả, Shape lưu được {shape_saved}, trong đó {matched} đối tượng có kết quả từ cả hai. '
                      "Với tập con này, IoU trung bình tăng từ 0,2302 lên 0,2631."])
        if (OUT/"cohort-cocoa-hidden-iou.json").is_file():
            hidden = read("cohort-cocoa-hidden-iou.json")
            native = OUT / "cohort-cocoa-native-replay.json"
            if hashlib.sha256(native.read_bytes()).hexdigest() != hidden["native_replay_sha256"]:
                raise ValueError("Hidden-region scores and whole-object scores use different cohort masks")
            vs_pmc = hidden["comparisons"]["shape-vs-baseline"]
            vs_surface = hidden["comparisons"]["shape-vs-surface"]
            def decimal_vi(value):
                return f"{value:.4f}".replace(".", ",")
            lo, hi = vs_pmc["absolute_ci"]
            surface_lo, surface_hi = vs_surface["absolute_ci"]
            lines.extend(["", f'Phân tích bổ sung trên {hidden["eligible"]} vật thể có pixel bị che theo COCOA '
                          f'({hidden["images"]} ảnh) đo IoU sau khi bỏ vùng nhìn thấy theo chú thích khỏi cả '
                          'mặt nạ dự đoán và mặt nạ chuẩn. Ca không lưu được kết quả nhận 0. '
                          f'PMC và Shape trên L40S lần lượt đạt {decimal_vi(vs_pmc["baseline"])} và '
                          f'{decimal_vi(vs_pmc["method"])}. Chênh lệch là {decimal_vi(vs_pmc["absolute_gain"])} '
                          f'(KTC 95% danh nghĩa từ {decimal_vi(lo)} đến {decimal_vi(hi)}). '
                          f'Surface trên Quadro đạt {decimal_vi(vs_surface["baseline"])}. '
                          f'KTC 95% của chênh lệch Shape trừ Surface trải từ {decimal_vi(surface_lo)} '
                          f'đến {decimal_vi(surface_hi)}. '
                          '[Điểm vùng bị che theo từng vật thể](cohort-cocoa-hidden-iou.json)'])
        stages = read("progressive-l40s/shape-cohort-verification.json")["statistics"]["Overall"]["failure_stage_counts"]
        lines.extend(["", f'Trên toàn bộ 204 đối tượng, bước chọn mục tiêu thất bại ở {stages["target_selection"]} '
                      f'trường hợp, bộ lọc loại {stages["query_filter"]} mặt nạ quá nhỏ và PMC không phát hiện '
                      f'che khuất ở {stages["occlusion_check"]} trường hợp. Cả ba nhóm đều dừng trước khi tạo '
                      'ứng viên để Shape chấm điểm. '
                      '[Kết quả kiểm tra từng giai đoạn](progressive-l40s/shape-cohort-verification.json)'])
        if (OUT/"natural-cohort-coverage.png").is_file():
            lines.extend(["", "![Hai trường hợp PMC và Shape khác nhau về khả năng lưu kết quả](natural-cohort-coverage.png)", "",
                          "These are the only two targets where PMC and Shape differed in whether they saved a completion. "
                          "The dog accounts for 0.0214 of the 0.0329 mean IoU gain on the 22 COCOA targets. Its SAM "
                          "reference includes the doughnut-shaped object beside the dog, and Shape retained it. "
                          "The Shape mask has IoU 0.972 with SAM and 0.472 with the COCOA dog polygon. Within the "
                          "annotated frame, 75,532 of 144,902 predicted foreground pixels (52.1%) fall outside the dog polygon. "
                          "For the banana, PMC saved a completion and Shape’s final detector missed the object. "
                          "Both rows show outputs in original-image coordinates."])
            if (OUT/"combined-cohort-vs-pmc-summary.json").is_file():
                lines.extend(["", "Trên cùng 204 đối tượng và cùng mặt nạ tham chiếu, khi thêm điểm Surface "
                              "và hiệu chỉnh RGB vào Shape, LPIPS giảm từ 0,6974 xuống 0,6838 "
                              "(giảm 1,94%, KTC 95% từ 0,81% đến 3,57%). Phương pháp kết hợp lưu được "
                              "đầu ra cho cả hai ca trong hình. Ở ca con chó, mặt nạ chủ yếu giữ vật hình "
                              "bánh vòng bên cạnh. IoU với chú thích COCOA của con chó là 0,194, so với "
                              "0,472 của Shape."])
                if (OUT/"cohort-cocoa-native-replay.json").is_file():
                    outline_replay = read("cohort-cocoa-native-replay.json")
                    difference = outline_replay["combined_minus_shape"]
                    share = f'{100 * difference["dog_fraction_of_net_difference"]:.1f}'.replace(".", ",")
                    lines.extend(["", f'Shape và Shape + Surface lưu kết quả cho cùng '
                                  f'{difference["shared_successes"]} trong {outline_replay["targets"]} '
                                  f'đối tượng có đường bao COCOA. Riêng ca con chó chiếm {share}% '
                                  'mức giảm IoU trung bình quan sát được khi thêm Surface vào Shape. '
                                  'Khoảng tin cậy 95% của chênh lệch IoU trên tập này bao gồm 0. '
                                  '[Đối chiếu từng mặt nạ với đường bao COCOA](cohort-cocoa-native-replay.json)'])
    lines.extend(full_conclusion)
    lines+=["","## Evidence","",
            "The cached RGB and masks come from the original Surface confirmation job 37843 on Quadro. Four NVIDIA L40S GPUs ran the development oracle and U-Net training in job 44902 (3 min 48 s including GPU validation) and the controlled inference in job 44903 (35 s). Those times exclude the original candidate generation. CPU audit 44926 replayed the native IoUs, hidden-core RGB MAE and selector choices for all 1,024 candidates. LPIPS, CLIP and whole-object DreamSim come from the original per-candidate evaluations. CPU job 44951 measured the supplemental hidden-core DreamSim on the same saved RGB images in 7 min 56 s.",
            "","[Paired estimates and intervals](controlled-comparisons.csv) · [Native-image audit](controlled-audit.json) · [Training receipt](training.json) · [Experiment contract](../../configs/research/shape-20261121.json)",""]
    lines.extend(["The progressive cost table charges each method for its cohort and continuation. "
                  "Separate L40S allocations used 0.253 GPU-hours for the development oracle, validation and "
                  "U-Net training, 0.039 for the controlled inference, and 0.120 for the first Shape "
                  "attempt that stopped on a reference mismatch. The completed 204-target PMC control "
                  "used 1.404 GPU-hours. Slurm recorded zero allocation for both canceled Surface jobs "
                  "and the canceled full PMC job. [Slurm allocation records](progressive-l40s/setup-accounting.psv)", ""])
    (OUT/"REPORT.md").write_text("\n".join(lines),encoding="utf8")


if __name__=="__main__":
    main()


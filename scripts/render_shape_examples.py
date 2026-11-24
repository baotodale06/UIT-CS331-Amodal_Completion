"""Render Shape examples from the frozen controlled and natural cohorts."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.build_shape_report import audited_pairs, matched_shape_rows

STUDY = ROOT / "output/shape-20261121"
POOLS = ROOT / "output/surface-confirmation128-20261110"
COLORS = {"agreement": "#087d76", "missed": "#e69f00", "extra": "#7854a0"}


def draw_mask(ax, mask, truth):
    from matplotlib.colors import to_rgb
    colors = np.ones((*mask.shape, 3))
    colors[mask & truth] = to_rgb(COLORS["agreement"])
    colors[~mask & truth] = to_rgb(COLORS["missed"])
    colors[mask & ~truth] = to_rgb(COLORS["extra"])
    ax.imshow(colors, interpolation="nearest")
    iou = np.count_nonzero(mask & truth) / np.count_nonzero(mask | truth)
    ax.set_xlabel(f"IoU {iou:.3f}", fontsize=13)
    return iou


def load_mask(path):
    with Image.open(path) as image:
        return np.array(image.convert("L")) > 0


def render_natural_transitions():
    summaries = {arm: json.loads(path.read_bytes()) for arm in ("shape", "combined")
                 if (path := STUDY/f"{arm}-cohort-vs-pmc-summary.json").is_file()}
    pairs = (matched_shape_rows("cohort", summaries) if "combined" in summaries
             else audited_pairs("cohort", "shape", summaries["shape"]))
    assert len(pairs) == 204
    methods = ["baseline", *summaries]
    rows = sorted((r for r in pairs if len({r["status"][m] == "success" for m in methods}) > 1),
                  key=lambda r:r["id"])
    geometry = STUDY/"evaluation_only/cocoa"
    annotation = {r["id"]:r for r in json.loads((geometry/"manifest.json").read_bytes())["rows"]}
    out = STUDY/"progressive-l40s"
    fig, axes = plt.subplots(len(rows), len(methods)+1, figsize=(3.85*(len(methods)+1), 3.65*len(rows)), squeeze=False)
    for ax, title in zip(axes[0], ("Ảnh và mặt nạ tham chiếu", "PMC · L40S", "Shape · L40S", "Shape + Surface · L40S")):
        ax.set_title(title, fontsize=15, pad=9)
    for panels, row in zip(axes, rows):
        original = panels[0]
        for method, axis in zip(methods, panels[1:]):
            run = out/(method+"-cohort")
            folder = run/"targets"/row["id"]
            record = json.loads((folder/"result.json").read_bytes())
            assert record["status"] == row["status"][method] and record["input_sha256"] == row["input_sha256"]
            assert record["evaluation_reference_artifacts"] == row["method_reference"]
            if method == "baseline":
                source = ROOT/"datasets/open_world_amodal"/record["image"]
                assert hashlib.sha256(source.read_bytes()).hexdigest() == row["input_sha256"]
                with Image.open(source) as image:
                    original.imshow(image.convert("RGB"))
                    size = image.size
                mask_path = run/"visible_references/masks"/record["reference"]
                assert hashlib.sha256(mask_path.read_bytes()).hexdigest() == row["method_reference"]["mask"]
                original.contour(load_mask(mask_path), levels=[.5], colors=["#00dcff"], linewidths=1.7)
                item = annotation.get(row["id"])
                if item:
                    path = geometry/item["case"]
                    assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"]
                    with Image.open(path) as mask:
                        truth = (np.asarray(mask) & 1) > 0
                    original.contour(truth, levels=[.5], colors=["#ffc64b"], linewidths=1.9, linestyles="dashed")
                name = {"dog":"Chó", "banana":"Chuối"}.get(record["label"], record["label"])
                original.set_xlabel(f'{name} · {row["source"]}', fontsize=13)
            frame = Image.new("RGB", size, "white")
            if record["status"] == "success":
                path = folder/"completion.png"
                assert hashlib.sha256(path.read_bytes()).hexdigest() == record["completion_sha256"]
                row_offset, col_offset = record["geometry"]["offset_rc"]
                assert list(size) == record["geometry"]["original_size"]
                with Image.open(path) as image:
                    frame.paste(image.convert("RGB"), (col_offset,row_offset), image.getchannel("A"))
            else:
                assert record["failure_reason"] == "completion_not_detected"
                axis.text(.5, .5, "Không phát hiện vật thể\nsau khi hoàn thiện", transform=axis.transAxes,
                          ha="center", va="center", fontsize=14, color="#444444")
            axis.imshow(frame)
            scores = row[method]
            label = f'LPIPS {scores["LPIPS"]:.3f}'
            if "amodal_iou" in scores:
                label += f'  ·  IoU {scores["amodal_iou"]:.3f}'
            axis.set_xlabel(label, fontsize=13)
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#c5cbd1")
    fig.legend(handles=[Line2D([],[],color="#008aa2",lw=2,label="Vùng nhìn thấy theo SAM"),
                        Line2D([],[],color="#a76a00",lw=2,ls="--",label="Đường bao toàn vật thể theo COCOA")],
               loc="lower center", ncol=2, frameon=False, fontsize=12)
    fig.subplots_adjust(left=.03, right=.98, top=.92, bottom=.12, hspace=.28, wspace=.10)
    fig.savefig(STUDY/"natural-cohort-coverage.png", dpi=160, facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--natural-cohort", action="store_true")
    if parser.parse_args().natural_cohort:
        render_natural_transitions()
        return
    rows = [json.loads(p.read_text()) for p in sorted((STUDY/"controlled").glob("*/result.json"))]
    if len(rows) != 128:
        raise ValueError("Expected the complete slide 18 cohort")
    umbrella = next(row for row in rows if row["target_id"] == "VG/2316897_sunumbrella")
    worst = min(rows, key=lambda row: row["metrics"]["shape8"]["iou"] - row["metrics"]["uniform8"]["iou"])
    fig, axes = plt.subplots(2, 5, figsize=(12.6, 6.45))
    for axis, title in zip(axes[0], ("Ảnh đã che", "Mặt nạ trước khi che", "Dự đoán của U-Net", "PMC · seed 11", "Shape · chọn từ 8")):
        axis.set_title(title, fontsize=13, pad=9)
    for axes_row, row in zip(axes, (umbrella, worst)):
        case = row["case_id"]
        gt = load_mask(POOLS/"evaluation_only"/case/"full.png")
        with Image.open(POOLS/"inputs"/case/"observed.png") as image:
            axes_row[0].imshow(np.array(image.convert("RGB")))
        axes_row[0].set_ylabel(row["label"], fontsize=13)
        axes_row[1].imshow(gt, cmap="Greys", vmin=0, vmax=1, interpolation="nearest")
        prior = load_mask(STUDY/"controlled"/case/"prior.png")
        np.testing.assert_allclose(draw_mask(axes_row[2], prior, gt), row["prior_iou"])
        for axis, method in zip(axes_row[3:], ("first", "shape8")):
            seed = (1 + row["indices"][method])*11
            mask = load_mask(POOLS/"full"/"candidates"/case/f"seed-{seed}-mask.png")
            np.testing.assert_allclose(draw_mask(axis, mask, gt), row["metrics"][method]["iou"])
    for axis in axes.flat:
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_color("#d0d6d8")
    fig.legend([Patch(color=COLORS[key]) for key in ("agreement", "missed", "extra")],
               ["Trùng với mặt nạ gốc", "Thiếu", "Thừa"], loc="lower center", ncol=3, frameon=False, fontsize=13)
    fig.subplots_adjust(left=.055, right=.99, top=.93, bottom=.13, wspace=.11, hspace=.23)
    fig.savefig(STUDY/"shape-silhouettes.png", dpi=155, facecolor="white")
    plt.close(fig)
    print(umbrella["target_id"], worst["target_id"])


if __name__ == "__main__":
    main()

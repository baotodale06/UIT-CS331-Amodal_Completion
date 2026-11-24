"""Explain the six recorded Shape selections on the severely occluded COCOA plate."""
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from amodal_benchmark.dataset import sha256, write_json
from amodal_benchmark.shape import choose, score_masks
from scripts.audit_shape_controlled_cocoa import check_file, native_mask
from scripts.render_shape_examples import COLORS, draw_mask

STUDY = ROOT / "output/shape-20261121"
TARGET = "COCOA/COCO_train2014_000000060172_plate"


def measure(mask, outline, hidden):
    intersection = int(np.count_nonzero(mask & outline))
    union = int(np.count_nonzero(mask | outline))
    covered = int(np.count_nonzero(mask & hidden))
    return dict(intersection=intersection, union=union, iou=intersection / union,
                hidden_covered=covered, hidden_recall=covered / np.count_nonzero(hidden))


def render(source, box, directory, decision, original_masks, outline, first, path):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    selected, best = first["selected"], first["highest_iou_index"]
    left, top, right, bottom = box
    photos = [source]
    for index in (selected, best):
        name = f"candidate-{index}.png"
        check_file(directory / name, decision["artifacts"][name])
        with Image.open(directory / name) as im:
            frame = Image.new("RGB", source.size, "white")
            frame.paste(im.convert("RGB").resize((right-left, bottom-top), Image.Resampling.BILINEAR), (left, top))
            photos.append(frame)
    fig, axes = plt.subplots(2, 3, figsize=(10.8, 7.2))
    titles = ["Ảnh gốc", f"Shape chọn ứng viên {selected+1}", f"IoU cao nhất · ứng viên {best+1}"]
    masks = [original_masks["prior"], original_masks[f"mask-{selected}"], original_masks[f"mask-{best}"]]
    values = [first["prior"], first["candidates"][selected], first["candidates"][best]]
    for col, (title, photo, mask, result) in enumerate(zip(titles, photos, masks, values)):
        axes[0, col].imshow(photo)
        axes[0, col].set_title(title, fontsize=13)
        axis = axes[1, col]
        iou = draw_mask(axis, mask, outline)
        np.testing.assert_allclose(iou, result["iou"], rtol=0., atol=1e-12)
        label = "Mặt nạ dự đoán" if col == 0 else "Mặt nạ ứng viên"
        axis.set_xlabel((f'{label} · IoU {iou:.3f}\nPhủ {100*result["hidden_recall"]:.1f}% phần đĩa bị che')
                        .replace(".", ","), fontsize=11)
    for axis in axes.flat:
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_color("#ccd1d3")
    fig.legend(handles=[Patch(color=COLORS[k], label=label) for k, label in
                        (("agreement", "Trùng với đĩa"), ("missed", "Phần đĩa bị bỏ sót"),
                         ("extra", "Nằm ngoài đĩa"))],
               loc="lower center", ncol=3, frameon=False, fontsize=11)
    fig.subplots_adjust(left=.025, right=.975, bottom=.13, top=.95, hspace=.07, wspace=.045)
    fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight", pad_inches=.12)
    plt.close(fig)


def main():
    run = STUDY / "progressive-l40s/shape"
    folder = run / "targets" / TARGET
    record_path = folder / "result.json"
    record = json.loads(record_path.read_bytes())
    trace = record["diagnostics"]["shape_iterations"]
    if (record["id"] != TARGET or record["status"] != "success" or not trace
            or record["iterations"] != len(trace) or not record["iteration_limit_reached"]):
        raise ValueError("Expected the saved plate with a complete iteration trace at the stopping limit")
    geo = STUDY / "evaluation_only/cocoa"
    manifest = geo / "manifest.json"
    check_file(manifest, json.loads((STUDY / "cocoa-historical-controls.json").read_bytes())["manifest_sha256"])
    label = next(r for r in json.loads(manifest.read_bytes())["rows"] if r["id"] == TARGET)
    annotation = geo / label["case"]
    source_path = ROOT / "datasets/open_world_amodal" / record["image"]
    sam_path = run / "visible_references/masks" / record["reference"]
    for path, digest in ((annotation, label["sha256"]), (source_path, record["input_sha256"]),
                         (sam_path, record["evaluation_reference_artifacts"]["mask"])):
        check_file(path, digest)
    with Image.open(annotation) as encoded, Image.open(sam_path) as ref, Image.open(source_path) as photo:
        if encoded.size != ref.size or encoded.size != photo.size or list(photo.size) != record["geometry"]["original_size"]:
            raise ValueError("Annotation, observation and reference must share the original frame")
        flags, sam, source = np.array(encoded), ref.convert("L"), photo.convert("RGB")
    outline = (flags & 1) != 0
    hidden = outline & ((flags & 2) == 0)
    w, h = source.size
    left, top = 3*w - w//2, 3*h - h//2
    rows = []
    figure = STUDY / "shape-plate-first-pass.png"
    for iteration, entry in enumerate(trace):
        directory = folder / "shape" / f"iteration-{iteration:02d}"
        path = directory / "decision.json"
        check_file(path, entry["decision_sha256"])
        decision = json.loads(path.read_bytes())
        r0, r1, c0, c1 = decision["crop_rc"]
        box = (c0-left, r0-top, c1-left, r1-top)
        sampled, original = {}, {}
        for key in ["prior", "visible", "occlusion", *(f"mask-{i}" for i in range(8))]:
            name = key + ".png"
            check_file(directory / name, decision["artifacts"][name])
            with Image.open(directory / name) as image:
                sampled[key] = np.asarray(image) > 0
                original[key] = native_mask(image, box, source.size)
        size = sampled["visible"].shape[::-1]
        expected_visible = np.asarray(sam.crop(box).resize(size, Image.Resampling.NEAREST)) > 0
        if not np.array_equal(sampled["visible"], expected_visible):
            raise ValueError("Projected source SAM mask differs from the sampler")
        allowed = sampled["visible"] | sampled["occlusion"]
        if np.any(sampled["visible"] & ~sampled["prior"]) or np.any(sampled["prior"] & ~allowed):
            raise ValueError("Shape prior differs from its visible and occluder support")
        scores = score_masks(sampled["prior"], [sampled[f"mask-{i}"] for i in range(8)])
        selected, fallback = choose(scores)
        if (scores != decision["scores"] or decision["iteration"] != iteration
                or (selected, fallback) != (decision["index"], decision["fallback"])
                or (selected, fallback) != (entry["index"], entry["fallback"])):
            raise ValueError("Native masks contradict the recorded Shape choice")
        candidates = [measure(original[f"mask-{i}"], outline, hidden) for i in range(8)]
        best = max(range(len(candidates)), key=lambda i: candidates[i]["iou"])
        row = dict(iteration=iteration, box_xyxy=box, decision_sha256=entry["decision_sha256"],
                   selected=selected, highest_iou_index=best, shape_scores=scores, candidates=candidates,
                   prior=measure(original["prior"], outline, hidden),
                   available=measure(original["visible"] | original["occlusion"], outline, hidden))
        rows.append(row)
        if iteration == 0:
            render(source, box, directory, decision, original, outline, row, figure)
    write_json(STUDY / "shape-plate-trace.json", dict(target=TARGET, hidden_pixels=int(hidden.sum()),
               record_sha256=sha256(record_path), manifest_sha256=sha256(manifest), annotation_sha256=label["sha256"],
               script_sha256=sha256(Path(__file__)), figure_sha256=sha256(figure), passes=rows))


if __name__ == "__main__":
    main()

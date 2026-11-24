"""SAM-input training and native validation for an isolated shape-only prior."""
from collections import defaultdict
from pathlib import Path
import numpy as np
from PIL import Image
from amodal_benchmark.shape_v2 import observation_tensor, visible_box


def cropped_sample(rgb, truth, annotated_visible, support, box, size=256):
    observed = np.array(Image.fromarray(rgb).crop(box))
    input_mask = np.array(Image.fromarray(support).crop(box))
    resized = lambda value: np.array(Image.fromarray(value).crop(box).resize((size, size), Image.Resampling.NEAREST))
    return (observation_tensor(observed, input_mask, size=size),
            resized(truth).astype(np.float32)[None], resized(annotated_visible).astype(np.float32)[None])


class RobustData:
    def __init__(self, root, rows, sam_root, sam_rows, seed, size=256):
        self.root, self.rows, self.sam_root = Path(root), rows, Path(sam_root)
        self.sam_rows = {row["id"]:row for row in sam_rows}
        self.seed, self.size, self.epoch = seed, size, 0

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        import torch
        row = self.rows[index]
        with Image.open(self.root/row["image"]) as source:
            rgb = np.array(source.convert("RGB"))
        with Image.open(self.root/row["mask"]) as source:
            encoded = np.array(source)
        truth, annotated = (encoded & 1) > 0, (encoded & 2) > 0
        with Image.open(self.sam_root/self.sam_rows[row["id"]]["sam_mask"]) as source:
            sam = np.array(source) > 0
        if sam.shape != truth.shape:
            raise ValueError("SAM training mask frame differs")
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, index]))
        # Empty SAM inputs cannot define an inference crop. Keep these training
        # objects through the annotated branch; validation retains empty failures.
        support = sam.copy() if sam.any() and rng.random() < .75 else annotated.copy()
        if rng.random() < .5:
            rgb, truth, annotated, support = (value[:, ::-1].copy() for value in (rgb, truth, annotated, support))
        if rng.random() < .25:
            yy, xx = np.indices(support.shape)
            angle = rng.uniform(0, 2*np.pi)
            field = xx*np.cos(angle) + yy*np.sin(angle)
            extra = field >= np.quantile(field[support], rng.uniform(.2, .8))
            remainder = support & ~extra
            if remainder.sum() >= max(64, .1*support.sum()):
                support, annotated = remainder, annotated & ~extra
        box = visible_box(support, rng.uniform(1.5, 3.))
        x, y, visible_truth = cropped_sample(rgb, truth, annotated, support, box, self.size)
        return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(visible_truth)


def native_canvas(probability, box, shape):
    x0, y0, x1, y1 = box
    if probability.shape != (y1-y0, x1-x0) or not np.isfinite(probability).all():
        raise ValueError("Invalid native prior crop")
    height, width = shape
    left, top, right, bottom = max(0, x0), max(0, y0), min(width, x1), min(height, y1)
    canvas = np.zeros(shape, np.float32)
    canvas[top:bottom, left:right] = probability[top-y0:bottom-y0, left-x0:right-x0]
    return canvas


def geometry(prediction, truth, annotated):
    hidden, predicted_hidden = truth & ~annotated, prediction & ~annotated
    return dict(full_iou=float((prediction & truth).sum()/max(1, (prediction | truth).sum())),
        hidden_iou=float((predicted_hidden & hidden).sum()/max(1, (predicted_hidden | hidden).sum())) if hidden.any() else None)


def aggregate(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["image_id"]].append(row)
    result = dict(images=len(groups), objects=len(rows))
    for metric in ("full_iou", "hidden_iou"):
        values = [np.mean([row[metric] for row in group if row[metric] is not None])
                  for group in groups.values() if any(row[metric] is not None for row in group)]
        result[metric] = float(np.mean(values))
        result[metric+"_images"] = len(values)
    return result


def evaluate_native(model, data, selected, sam_root, device, maximum_images=None):
    """No SAM/annotated support restoration; scoring truth remains annotated."""
    import torch
    groups = defaultdict(list)
    for row in selected:
        groups[row["image_id"]].append(row)
    model.eval()
    records = []
    with torch.inference_mode():
        for image_index, (image_id, objects) in enumerate(sorted(groups.items())):
            if maximum_images is not None and image_index >= maximum_images:
                break
            with Image.open(data/objects[0]["image"]) as source:
                rgb = np.array(source.convert("RGB"))
            for row in objects:
                with Image.open(data/row["mask"]) as source:
                    encoded = np.array(source)
                truth, visible = (encoded & 1) > 0, (encoded & 2) > 0
                with Image.open(sam_root/"objects"/row["id"]/"sam-visible.png") as source:
                    support = np.array(source) > 0
                if support.shape != truth.shape:
                    raise ValueError("Native validation SAM frame differs")
                if support.any():
                    box = visible_box(support, 2.)
                    observed = np.array(Image.fromarray(rgb).crop(box))
                    cropped = np.array(Image.fromarray(support).crop(box))
                    x = torch.from_numpy(observation_tensor(observed, cropped)).unsqueeze(0).to(device)
                    logits = model(x)
                    p = torch.nn.functional.interpolate(logits, cropped.shape, mode="bilinear", align_corners=False).sigmoid()
                    prediction = native_canvas(p[0,0].cpu().numpy(), box, truth.shape) >= .5
                else:
                    prediction = np.zeros_like(truth)
                records.append(dict(id=row["id"], image_id=image_id, **geometry(prediction, truth, visible)))
    return records

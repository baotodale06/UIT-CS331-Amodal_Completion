"""Native validation for a separate calibrated-initialization training study."""
from collections import defaultdict
import hashlib
import io
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image

from amodal_benchmark.shape_calibrated_prior import original_prior
from scripts.shape_robust_data import aggregate, geometry
from scripts.shape_local_data import eligible, selection_floor


def calibration_summary(rows, reference, repeated):
    """Keep cross-device differences explicit; require exact same-device repeats."""
    if rows != repeated:
        raise ValueError('Initial native pixels must repeat on the training GPU')
    ids = [row['id'] for row in rows]
    expected = {row['id']: row for row in reference}
    if len(ids) != len(set(ids)) or not set(ids).issubset(expected):
        raise ValueError('Initial validation identity changed')
    local = aggregate(rows)
    parent = aggregate([expected[key] for key in ids])
    different = [key for key, row in zip(ids, rows)
                 if row['prediction_sha256'] != expected[key]['prediction_sha256']]
    return dict(training_gpu=local, parent_local_gpu=parent,
                selection_floor=selection_floor(local, parent),
                different_pixel_hash_objects=different, exact_training_gpu_repeat=True,
                cross_device_pixel_identity=not different,
                benchmark_admitted=False, benchmark_success=False)


def evaluate_native(model, data, selected, sam_root, device, maximum_images=None,
                    mask_archive=None):
    """The fixed -2 correction is applied after native logit interpolation."""
    groups = defaultdict(list)
    for row in selected:
        groups[row['image_id']].append(row)
    model.eval()
    records = []
    archive = zipfile.ZipFile(mask_archive, 'x', zipfile.ZIP_DEFLATED) if mask_archive else None
    try:
        for index, (image_id, objects) in enumerate(sorted(groups.items())):
            if maximum_images is not None and index >= maximum_images:
                break
            with Image.open(Path(data) / objects[0]['image']) as image:
                rgb = np.array(image.convert('RGB'))
            for row in objects:
                with Image.open(Path(data) / row['mask']) as image:
                    encoded = np.array(image)
                truth, visible = (encoded & 1) > 0, (encoded & 2) > 0
                with Image.open(Path(sam_root) / 'objects' / row['id'] / 'sam-visible.png') as image:
                    support = np.array(image) > 0
                if support.shape != truth.shape or rgb.shape[:2] != truth.shape:
                    raise ValueError('Native observation and annotation frames differ')
                probability, box, _, _ = original_prior(model, rgb, support, bias=-2.)
                prediction = probability >= .5
                if np.any(support & ~prediction):
                    raise ValueError('Observable support was removed')
                records.append(dict(id=row['id'], image_id=image_id,
                    prediction_sha256=hashlib.sha256(prediction.astype(np.uint8).tobytes()).hexdigest(),
                    **geometry(prediction, truth, visible)))
                if archive:
                    stream = io.BytesIO()
                    Image.fromarray(prediction.astype(np.uint8) * 255).save(stream, format='PNG')
                    archive.writestr(row['id'] + '.png', stream.getvalue())
        return sorted(records, key=lambda row: row['id'])
    finally:
        if archive:
            archive.close()

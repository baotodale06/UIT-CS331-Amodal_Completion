"""Tiền xử lý thuần NumPy: crop từ visible, không phụ thuộc annotations."""
import numpy as np


def visible_crop_bounds(visible, expand=2):
    rows, cols = np.where(visible)
    if not len(rows):
        raise ValueError('Visible mask rỗng')
    h, w = visible.shape
    cy, cx = (rows.min()+rows.max())/2, (cols.min()+cols.max())/2
    ry = max(1, (rows.max()-rows.min()+1)*expand/2)
    rx = max(1, (cols.max()-cols.min()+1)*expand/2)
    return max(0, int(cy-ry)), min(h, int(cy+ry)+1), max(0, int(cx-rx)), min(w, int(cx+rx)+1)

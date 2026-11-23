"""Metrics đánh giá độc lập; tuyệt đối không import module này từ scorer."""
import numpy as np


def mask_metrics(prediction, full_mask, visible):
    p, g, v = [np.asarray(a, bool) for a in (prediction, full_mask, visible)]
    if p.shape != g.shape or g.shape != v.shape:
        raise ValueError('Mask evaluation phải cùng hệ tọa độ ảnh gốc')
    if not g.any() or np.any(v & ~g):
        raise ValueError('Full GT phải không rỗng và chứa visible mask')
    union = int((p | g).sum())
    hidden_p, hidden_g = p & ~v, g & ~v
    hidden_union = int((hidden_p | hidden_g).sum())
    # Không có vùng hidden GT thì metric hidden không áp dụng, không ghi thành 1.
    return {'amodal_iou': float((p & g).sum()/union),
            'hidden_iou': float((hidden_p & hidden_g).sum()/max(hidden_union, 1)) if hidden_g.any() else None,
            'visible_recall': float((p & v).sum()/v.sum()) if v.any() else None,
            'dice': float(2*(p & g).sum()/max(int(p.sum()+g.sum()), 1))}


def hidden_rgb_metrics(prediction, reference, hidden):
    if prediction.shape != reference.shape or prediction.shape[:2] != hidden.shape:
        raise ValueError('RGB GT/prediction không khớp kích thước')
    if not hidden.any():
        return {'hidden_mae': None, 'hidden_psnr_db': None}
    delta = (prediction.astype(np.float64)-reference.astype(np.float64))[hidden]/255
    mse = float(np.mean(delta**2))
    # PSNR hoàn hảo ghi null và cờ riêng, tránh Infinity trong JSON chuẩn.
    return {'hidden_mae': float(np.abs(delta).mean()),
            'hidden_psnr_db': float(-10*np.log10(mse)) if mse > 0 else None,
            'perfect_hidden_rgb': mse == 0}


def aggregate(rows):
    names = ('amodal_iou', 'hidden_iou', 'dice', 'visible_recall', 'prior_iou',
             'hidden_mae', 'hidden_psnr_db', 'masked_lpips', 'seconds')
    return {name: {'mean': float(np.mean(values)) if values else None, 'count': len(values)}
            for name in names
            for values in [[row[name] for row in rows if row.get(name) is not None]]}

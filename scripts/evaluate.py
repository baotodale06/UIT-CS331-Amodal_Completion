#!/usr/bin/env python3
"""Đánh giá PMC trên manifest JSONL; --limit N hoặc --full, GT chỉ đọc sau inference."""
import argparse
import csv
import json
from pathlib import Path
import sys
import time
import uuid
import numpy as np
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from main import parse_args, preflight, QueryObject, process_object, save_mask
from pmc.backends import InferenceEngine, register_backends
from pmc.runtime import ModelManager, configure_logging, context, event, LOG
from pmc.shape_predictor import ShapePredictor, save_json
from pmc.metrics import mask_metrics, hidden_rgb_metrics, aggregate


def read_mask(path):
    with Image.open(path) as handle:
        return np.asarray(handle.convert('L')) > 0


def align_original(query, height, width):
    """Lấy vùng ảnh gốc từ canvas; không dịch mask để tối đa hóa IoU với GT."""
    canvas_h, canvas_w = query.query_mask_canvas.shape
    r0, c0 = canvas_h//2-height//2, canvas_w//2-width//2
    return (query.query_mask_canvas[r0:r0+height, c0:c0+width].astype(bool),
            query.img_canvas[r0:r0+height, c0:c0+width].astype(np.uint8))


def evaluate():
    parser = argparse.ArgumentParser(description=__doc__, epilog='Các cờ còn lại chuyển thẳng cho main.py, ví dụ --c2f_python, --device, --num_candidates.')
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--eval_output', type=Path, default=Path('evaluation_output'))
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--limit', type=int, default=5)
    group.add_argument('--full', action='store_true')
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--metrics-only', action='store_true', help='Đọc prediction_mask/prediction_rgb trong manifest, không load models')
    parser.add_argument('--lpips', action='store_true', help='LPIPS CPU trên RGB đã giữ nguyên ngoài vùng hidden')
    opts, remaining = parser.parse_known_args()
    if opts.limit < 1 or opts.start < 0:
        parser.error('limit phải dương, start không âm')
    args = parse_args(['--input_dir', str(opts.manifest.parent), *remaining])
    # Manifest đã xác định target: không loại object nhỏ bằng bộ lọc discovery của PMC.
    args.query_area_threshold = 0
    args.query_score_threshold = 0
    run_dir = opts.eval_output / ('eval_'+uuid.uuid4().hex[:12])
    run_dir.mkdir(parents=True)
    configure_logging(run_dir, args.log_level)
    save_json(run_dir/'config.json', {'evaluation': vars(opts), 'pipeline': vars(args)})
    rows, manager = [], None
    state = 'failed'
    base = opts.manifest.resolve().parent
    def asset(row, name):
        path = Path(row[name])
        return path if path.is_absolute() else base/path
    try:
        samples = [json.loads(line) for line in opts.manifest.read_text().splitlines() if line.strip()]
        samples = samples[opts.start:] if opts.full else samples[opts.start:opts.start+opts.limit]
        if not samples:
            raise ValueError('Manifest không có sample được chọn')
        if not opts.metrics_only:
            identity = preflight(args)
            manager = ModelManager(args.device, args.offload_policy)
            register_backends(manager, args)
            engine = InferenceEngine(manager, args)
            predictor = ShapePredictor(args, identity, manager)
        perceptual = None
        if opts.lpips:
            import lpips
            perceptual = lpips.LPIPS(net='alex').eval().cpu()
        for index, sample in enumerate(samples):
            folder = run_dir/f'sample_{index+opts.start:06d}'
            record = {'index': index+opts.start, 'id': str(sample.get('id', index)), 'status': 'pending', 'has_mask_gt': bool(sample.get('full_mask'))}
            started = time.perf_counter()
            with context(sample_id=record['id'], sample_index=record['index']):
                try:
                    # Đây là các quan sát được phép truyền vào inference, không có GT.
                    image = Image.open(asset(sample, 'image')).convert('RGB')
                    rgb = np.asarray(image)
                    visible = read_mask(asset(sample, 'visible_mask'))
                    if visible.shape != rgb.shape[:2] or not visible.any():
                        raise ValueError('Visible mask phải không rỗng và khớp ảnh')
                    prior = None
                    invalid = False
                    if opts.metrics_only:
                        folder.mkdir()
                        prediction = read_mask(asset(sample, 'prediction_mask'))
                        predicted_rgb = np.asarray(Image.open(asset(sample, 'prediction_rgb')).convert('RGB')) if sample.get('prediction_rgb') else None
                    else:
                        occluder = read_mask(asset(sample, 'occluder_mask'))
                        if occluder.shape != visible.shape:
                            raise ValueError('Occluder mask không khớp kích thước ảnh')
                        query = QueryObject(image, visible, 0, folder)
                        query.initial_occluder = occluder
                        category = sample['category']
                        process_object(query, image, [visible], [category], [1.0], engine, predictor, args, [category])
                        invalid = query.status == 'invalid_output'
                        if invalid:
                            prediction, predicted_rgb = np.zeros_like(visible), None
                        elif hasattr(query, 'query_mask_canvas'):
                            prediction, predicted_rgb = align_original(query, *visible.shape)
                        else:
                            prediction, predicted_rgb = visible.copy(), rgb.copy()
                        record['pipeline_status'] = query.status
                        if (folder/'shape_prior_f0.png').exists():
                            prior = read_mask(folder/'shape_prior_f0.png')
                    folder.mkdir(exist_ok=True)
                    image.save(folder/'input.png')
                    save_mask(folder/'visible.png', visible)
                    save_mask(folder/'prediction_mask.png', prediction)
                    if predicted_rgb is not None:
                        Image.fromarray(predicted_rgb).save(folder/'prediction_rgb.png')
                    # Selection hoàn tất. GT từ đây chỉ dùng metric/ảnh đối chiếu.
                    if sample.get('full_mask'):
                        full = read_mask(asset(sample, 'full_mask'))
                        record.update(mask_metrics(prediction, full, visible))
                        save_mask(folder/'gt_full.png', full)
                        if prior is not None:
                            record['prior_iou'] = mask_metrics(prior, full, visible)['amodal_iou']
                        if sample.get('gt_rgb') and predicted_rgb is not None:
                            reference = np.asarray(Image.open(asset(sample, 'gt_rgb')).convert('RGB'))
                            hidden = full & ~visible
                            record.update(hidden_rgb_metrics(predicted_rgb, reference, hidden))
                            Image.fromarray(reference).save(folder/'gt_rgb.png')
                            if perceptual is not None and hidden.any():
                                import torch
                                # Masked LPIPS proxy: ngoài hidden hai ảnh giống hệt nhau.
                                comparison = reference.copy()
                                comparison[hidden] = predicted_rgb[hidden]
                                tensors = [torch.from_numpy(a.copy()).permute(2,0,1)[None].float()/127.5-1 for a in (comparison, reference)]
                                with torch.inference_mode():
                                    record['masked_lpips'] = float(perceptual(*tensors).item())
                    overlay = rgb.copy()
                    if prediction.shape != visible.shape:
                        raise ValueError('Prediction không cùng hệ tọa độ ảnh gốc')
                    overlay[prediction] = (0.5*overlay[prediction]+np.array([0,127,0])).astype(np.uint8)
                    Image.fromarray(overlay).save(folder/'overlay.png')
                    panels = [rgb, overlay] + ([predicted_rgb] if predicted_rgb is not None else [])
                    Image.fromarray(np.concatenate(panels, axis=1)).save(folder/'comparison.png')
                    record['status'] = 'invalid_output' if invalid else 'ok'
                except Exception as exc:
                    record.update(status='failed', error=str(exc))
                    LOG.exception('evaluation_sample_failed')
                    # OOM không tiếp tục vì trạng thái CUDA có thể không còn đáng tin.
                    if manager is not None and isinstance(exc, manager.torch.cuda.OutOfMemoryError):
                        raise
                finally:
                    record['seconds'] = time.perf_counter()-started
                    rows.append(record)
                    folder.mkdir(exist_ok=True)
                    save_json(folder/'metrics.json', record)
                    event('evaluation_sample_result', **record)
                    save_json(run_dir/'per_sample.json', rows)
        state = 'completed_with_errors' if any(r['status'] == 'failed' for r in rows) else 'completed'
    finally:
        if manager:
            manager.close()
        summary = {'status': state, 'attempted': len(rows), 'successful': sum(r['status']=='ok' for r in rows),
                   'failed': sum(r['status']=='failed' for r in rows),
                   'invalid_output': sum(r['status']=='invalid_output' for r in rows),
                   'metrics': aggregate(rows),
                   'amodal_iou_with_failures_zero': float(np.mean([r.get('amodal_iou', 0) for r in rows if r['has_mask_gt']])) if any(r['has_mask_gt'] for r in rows) else None, 'output_dir': str(run_dir)}
        save_json(run_dir/'metrics_summary.json', summary)
        keys = sorted({key for row in rows for key in row})
        if keys:
            with (run_dir/'metrics.csv').open('w', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=keys)
                writer.writeheader()
                writer.writerows(rows)
        event('evaluation_summary', **summary)
    if state != 'completed':
        raise SystemExit(1)


if __name__ == '__main__':
    evaluate()

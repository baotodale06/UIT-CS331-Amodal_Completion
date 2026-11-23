"""Shape-Aware PMC triển khai theo A.4–A.7, inference tuần tự cho GPU 24 GB."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys
import uuid
import numpy as np
from PIL import Image, ImageFilter

from pmc.backends import InferenceEngine, register_backends
from pmc.geometry import (create_canvas, crop, aggregate_occluders, check_touch_boundary,
                          find_expand_bboxes_instaorder, compute_offset)
from pmc.runtime import LOG, ModelManager, configure_logging, context, event, stage
from pmc.scoring import ScoreContext, shape_score, target_mask, select_candidate
from pmc.shape_predictor import ShapePredictor, validate_c2f, save_json

ROOT = Path(__file__).resolve().parent


def parse_args(argv=None):
    """Đường dẫn model mặc định theo repo, không phụ thuộc thư mục gọi lệnh."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input_dir', type=Path)
    parser.add_argument('--img_filenames_txt', type=Path, help='Tùy chọn; nếu bỏ trống thì quét ảnh trong input_dir')
    parser.add_argument('--classes_txt', type=Path, default=ROOT / 'classes.txt')
    parser.add_argument('--output_dir', type=Path, default=ROOT / 'output')
    parser.add_argument('--gdino_config', type=Path, default=ROOT / 'Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py')
    parser.add_argument('--gdino_ckpt', type=Path, default=ROOT / 'Grounded-Segment-Anything/groundingdino_swint_ogc.pth')
    parser.add_argument('--sam_ckpt', type=Path, default=ROOT / 'Grounded-Segment-Anything/sam_vit_h_4b8939.pth')
    parser.add_argument('--instaorder_ckpt', type=Path, default=ROOT / 'InstaOrder/InstaOrder_ckpt/InstaOrder_InstaOrderNet_od.pth.tar')
    parser.add_argument('--lama_config_path', type=Path, default=ROOT / 'lama/big-lama/config.yaml')
    parser.add_argument('--lama_ckpt_path', type=Path, default=ROOT / 'lama/big-lama/models/best.ckpt')
    parser.add_argument('--c2f_source', type=Path, default=ROOT / 'third_party/c2f-seg')
    parser.add_argument('--c2f_python', type=Path, default=Path(sys.executable))
    parser.add_argument('--c2f_checkpoint', type=Path, default=ROOT / 'checkpoints/c2f/c2f_seg_COCOA.pth')
    parser.add_argument('--c2f_vq_checkpoint', type=Path, default=ROOT / 'checkpoints/c2f/vqgan_COCOA.pth')
    parser.add_argument('--c2f_config', type=Path, default=ROOT / 'checkpoints/c2f/c2f_seg_COCOA.yml')
    parser.add_argument('--c2f_vq_config', type=Path, default=ROOT / 'checkpoints/c2f/vqgan_COCOA.yml')
    parser.add_argument('--c2f_seed', type=int, default=42)
    parser.add_argument('--c2f_steps', type=int, default=3)
    parser.add_argument('--c2f_expand', type=float, default=2)
    parser.add_argument('--c2f_timeout', type=int, default=600)
    parser.add_argument('--shape_cache_dir', type=Path, default=ROOT / 'cache/shape')
    parser.add_argument('--shape_threshold', type=float, default=0.5)
    parser.add_argument('--shape_weights', type=float, nargs=3, default=[0.70, 0.20, 0.10], metavar=('IOU', 'VISIBLE', 'LEAK'))
    parser.add_argument('--min_visible_recall', type=float, default=0.3)
    parser.add_argument('--box_threshold', type=float, default=0.3)
    parser.add_argument('--text_threshold', type=float, default=0.25)
    parser.add_argument('--query_score_threshold', type=float, default=0.35)
    parser.add_argument('--query_area_threshold', type=float, default=0.02)
    parser.add_argument('--sam_box_batch_size', type=int, default=8)
    parser.add_argument('--candidate_seeds', type=int, nargs='+', default=[101, 203, 307, 409])
    parser.add_argument('--num_candidates', type=int, default=4)
    parser.add_argument('--max_iter_id', type=int, default=5, help='Số iteration tối đa cho mỗi object')
    parser.add_argument('--sd_target_size', type=int, default=512)
    parser.add_argument('--num_inference_steps', type=int, default=50)
    parser.add_argument('--mc_timestep', type=int, default=35)
    parser.add_argument('--standard_inpainting', action='store_true', help='Dùng SD inpainting thường thay mixed-context')
    parser.add_argument('--mc_clean_bkgd_img', type=Path, default=ROOT / 'images/gray_wallpaper.jpeg')
    parser.add_argument('--sd_model', default='stabilityai/stable-diffusion-2-inpainting')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--offload_policy', choices=['cpu', 'unload'], default='cpu', help='cpu giữ weights ở RAM; unload giải phóng rồi load lại')
    parser.add_argument('--save_interm', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--log_level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], default='INFO')
    parser.add_argument('--continue_on_error', action='store_true', help='Ghi lỗi object/image rồi chạy tiếp; OOM luôn dừng')
    parser.add_argument('--preflight_only', action='store_true', help='Kiểm tra cấu hình/assets/CUDA, không chạy dữ liệu')
    args = parser.parse_args(argv)
    if args.standard_inpainting:
        args.mc_timestep = None
    if not 1 <= args.num_candidates <= len(args.candidate_seeds):
        parser.error('num_candidates phải dương và không lớn hơn số candidate_seeds')
    if len(set(args.candidate_seeds)) != len(args.candidate_seeds) or any(s < 0 or s >= 2**63 for s in args.candidate_seeds):
        parser.error('candidate_seeds phải khác nhau, thuộc [0, 2**63)')
    if args.c2f_steps < 1 or args.c2f_expand < 1 or args.c2f_timeout < 1:
        parser.error('C2F steps/timeout phải dương, expand >=1')
    if args.max_iter_id < 1 or args.sam_box_batch_size < 1 or args.num_inference_steps < 1:
        parser.error('Số iteration, box batch và inference steps phải dương')
    if args.sd_target_size < 64 or args.sd_target_size % 8:
        parser.error('sd_target_size phải >=64 và chia hết cho 8')
    if args.mc_timestep is not None and not 0 <= args.mc_timestep < args.num_inference_steps:
        parser.error('mc_timestep phải thuộc [0, num_inference_steps)')
    if any(not 0 <= value <= 1 for value in (args.shape_threshold, args.min_visible_recall,
            args.box_threshold, args.text_threshold, args.query_score_threshold, args.query_area_threshold)):
        parser.error('Threshold phải thuộc [0,1]')
    if any(not np.isfinite(w) or w < 0 for w in args.shape_weights) or sum(args.shape_weights) <= 0:
        parser.error('shape_weights phải hữu hạn, không âm và tổng >0')
    if not args.preflight_only and args.input_dir is None:
        parser.error('Cần --input_dir khi chạy inference')
    for name, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, name, value.absolute())
    return args


class QueryObject:
    """Trạng thái object lưu hoàn toàn trên CPU; prior/visible gốc không bị cập nhật."""
    def __init__(self, image, mask, object_id, output_dir):
        self.img = np.asarray(image).copy()
        self.query_mask = mask.astype(np.uint8)
        self.mask_id = object_id
        self.output_img_dir = Path(output_dir)
        self.output_img_dir.mkdir(parents=True)
        self.iter_id = 0
        self.run_iter = True
        self.amodal_segmentation = None
        self.amodal_completion = None
        self.status = 'pending'


def save_mask(path, mask):
    Image.fromarray(np.asarray(mask, dtype=np.uint8) * 255).save(path)


def prepare_mixed_context_inputs(query_obj, engine, args, image, visible, editable):
    """Tách LaMa khỏi vòng seed; mọi seed dùng cùng background và mask chỉnh sửa."""
    if args.mc_timestep is None:
        size = (args.sd_target_size, args.sd_target_size)
        return {'image': Image.fromarray(image).resize(size, Image.Resampling.BICUBIC),
                'mask_image': Image.fromarray(editable * 255).resize(size, Image.Resampling.NEAREST)}
    with stage('prepare_mixed_context'):
        clean = Image.open(args.mc_clean_bkgd_img).convert('RGB').resize(
            (image.shape[1], image.shape[0]), Image.Resampling.BICUBIC)
        synthetic = image.copy()
        synthetic[visible == 0] = np.asarray(clean)[visible == 0]
        removed = engine.remove_object(image, (visible > 0) | (editable > 0))
        removed[visible > 0] = image[visible > 0]
        if args.save_interm:
            Image.fromarray(removed).save(query_obj.output_img_dir / f'background_{query_obj.iter_id:02d}.png')
        return {'image': Image.fromarray(synthetic), 'mask_image': Image.fromarray(editable * 255),
                'mixed_context': True, 'mixed_context_timestep': args.mc_timestep,
                'mixed_context_up_ft_indices': [2], 'query_mask_image': Image.fromarray(visible * 255),
                'object_removed_image': Image.fromarray(removed)}


def check_occlusion(image, current_visible, query_obj, bounds, engine, classes):
    """Chỉ gọi sau selection: segment lại rồi cập nhật canvas đúng một lần."""
    with stage('check_occlusion'):
        masks, _, _ = engine.segment([image], classes)[0]
        index, mask, reason = target_mask(masks, current_visible, engine.args.min_visible_recall)
        if reason:
            query_obj.run_iter = False
            query_obj.amodal_segmentation = None
            query_obj.status = 'invalid_output'
            event('occlusion_invalid_target', reason=reason)
            return
        occluders = engine.occluders(image, masks, index, find_expand_bboxes_instaorder(masks))
        occ = aggregate_occluders(mask, occluders, '', index, query_obj.iter_id)
        r0, r1, c0, c1 = bounds
        query_obj.img_canvas[r0:r1, c0:c1] = np.asarray(image)
        query_obj.query_mask_canvas[r0:r1, c0:c1] = mask
        query_obj.occ_mask_canvas[r0:r1, c0:c1] = occ
        query_obj.outpaint_mask_canvas[r0:r1, c0:c1] = 0
        query_obj.amodal_segmentation = mask.astype(np.uint8)
        query_obj.run_iter = bool(occ.any() or check_touch_boundary(mask))
        query_obj.status = 'remaining_occlusion' if query_obj.run_iter else 'completed'
        event('occlusion_result', continue_iteration=query_obj.run_iter, occluder_pixels=int(occ.sum()))


def run_iteration(query_obj, engine, args, classes):
    """K candidates chỉ được đọc để chấm điểm, không tự thay đổi trạng thái PMC."""
    with stage('iteration', iteration=query_obj.iter_id):
        inputs = [query_obj.img_canvas, query_obj.query_mask_canvas, query_obj.occ_mask_canvas,
                  query_obj.outpaint_mask_canvas, query_obj.shape_prior_canvas,
                  query_obj.original_visible_canvas, query_obj.prior_valid_canvas]
        outputs, r0, r1, c0, c1 = crop(inputs, query_obj.query_mask, query_obj.original_visible_canvas)
        image, current_visible, occ, outpaint, prior, original_visible, valid = outputs
        bounds = [r0, r1, c0, c1]
        editable = ((occ > 0) | (outpaint > 0)).astype(np.uint8)
        # MaxFilter 13 tương đương kernel 5x5 giãn 3 lần của PMC gốc.
        editable = np.asarray(Image.fromarray(editable * 255).filter(ImageFilter.MaxFilter(13))) > 0
        editable = editable.astype(np.uint8)
        generation_inputs = prepare_mixed_context_inputs(query_obj, engine, args, image, current_visible, editable)
        seeds = args.candidate_seeds[:args.num_candidates]
        candidates = engine.generate(generation_inputs, seeds, args.sd_target_size, query_obj.query_class)
        candidates = [candidate.resize((image.shape[1], image.shape[0]), Image.Resampling.BICUBIC)
                      for candidate in candidates]
        ctx = ScoreContext(prior.astype(bool), original_visible.astype(bool), valid.astype(bool))
        folder = query_obj.output_img_dir / f'iteration_{query_obj.iter_id:02d}'
        folder.mkdir()
        # Luôn lưu pool/điểm để tái hiện quyết định; save_interm chỉ điều khiển context phụ.
        for index, candidate in enumerate(candidates):
            candidate.save(folder / f'candidate_{index:02d}.png')
        save_mask(folder / 'shape_prior.png', prior)
        save_mask(folder / 'visible_anchor.png', original_visible)
        save_mask(folder / 'prior_valid.png', valid)
        save_mask(folder / 'editable.png', editable)
        with stage('candidate_segmentation'):
            segmentations = engine.segment(candidates, classes)
        records = []
        with stage('shape_scoring'):
            for index, (masks, _, _) in enumerate(segmentations):
                _, mask, reason = target_mask(masks, ctx.visible_mask, args.min_visible_recall)
                result = shape_score(mask, ctx, args.shape_weights)
                if reason:
                    result['reason'] = reason
                if mask is not None:
                    save_mask(folder / f'target_{index:02d}.png', mask)
                record = {'index': index, 'seed': seeds[index], 'valid': result['score'] is not None, **result}
                records.append(record)
                event('candidate_score', **record)
            selected, fallback = select_candidate(records)
            save_json(folder / 'selection.json', {'iteration': query_obj.iter_id, 'crop_bounds_rc': bounds,
                      'weights': args.shape_weights, 'candidates': records,
                      'selected_idx': selected, 'fallback_reason': fallback})
            event('candidate_selected', selected_idx=selected, seed=seeds[selected], fallback_reason=fallback)
        query_obj.amodal_completion = candidates[selected]
        query_obj.iter_id += 1
        check_occlusion(candidates[selected], current_visible, query_obj, bounds, engine, classes)


def process_object(query_obj, image, masks, names, scores, engine, predictor, args, classes):
    index = query_obj.mask_id
    visible = query_obj.query_mask
    query_obj.query_class = names[index]
    if scores[index] < args.query_score_threshold or visible.sum() < args.query_area_threshold * visible.size or names[index] not in classes:
        query_obj.status = 'skipped_invalid_query'
        event('object_skipped', reason=query_obj.status, predicted_class=names[index], confidence=scores[index])
        return
    if hasattr(query_obj, 'initial_occluder'):
        # Evaluation đưa observed occluder mask vào; đây không phải full-object GT.
        occ = np.asarray(query_obj.initial_occluder, dtype=np.uint8).copy()
        occ[visible > 0] = 0
    else:
        occluders = engine.occluders(image, masks, index, find_expand_bboxes_instaorder(masks))
        occ = aggregate_occluders(visible, occluders, names[index], index, 0).astype(np.uint8)
    if not occ.any() and not check_touch_boundary(visible):
        query_obj.status = 'skipped_no_occlusion'
        event('object_skipped', reason=query_obj.status)
        return
    prior = predictor.predict(np.asarray(image), visible, occ)
    save_mask(query_obj.output_img_dir / 'shape_prior_f0.png', prior)
    query_obj.img_canvas = create_canvas(query_obj.img, 6, 255)
    query_obj.query_mask_canvas = create_canvas(visible, 6, 0)
    query_obj.occ_mask_canvas = create_canvas(occ, 6, 0)
    query_obj.outpaint_mask_canvas = create_canvas(np.zeros_like(visible), 6, 1)
    query_obj.init_outpaint_mask_canvas = query_obj.outpaint_mask_canvas.copy()
    query_obj.original_visible_canvas = query_obj.query_mask_canvas.copy()
    query_obj.shape_prior_canvas = create_canvas(prior.astype(np.uint8), 6, 0)
    query_obj.prior_valid_canvas = create_canvas(np.ones_like(visible), 6, 0)
    while query_obj.run_iter and query_obj.iter_id < args.max_iter_id:
        run_iteration(query_obj, engine, args, classes)
    if query_obj.run_iter:
        query_obj.status = 'max_iterations_reached'
    if query_obj.amodal_segmentation is not None:
        query_obj.amodal_completion.save(query_obj.output_img_dir / 'amodal_completion.png')
        save_mask(query_obj.output_img_dir / 'amodal_segmentation.png', query_obj.amodal_segmentation)
        offsets = compute_offset(query_obj.query_mask_canvas, query_obj.init_outpaint_mask_canvas,
                                 query_obj.amodal_segmentation)
        save_json(query_obj.output_img_dir / 'offset.json', {'xy': offsets})


def preflight(args):
    """Dừng sớm với thông báo rõ, trước khi tải các model lớn lên RAM/GPU."""
    identity = validate_c2f(args)
    paths = [args.gdino_config, args.gdino_ckpt, args.sam_ckpt, args.instaorder_ckpt, args.classes_txt]
    if args.mc_timestep is not None:
        paths += [args.lama_config_path, args.lama_ckpt_path, args.mc_clean_bkgd_img]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f'Thiếu asset: {path}')
    import subprocess, os
    env = os.environ.copy()
    env.pop('PYTHONPATH', None)
    result = subprocess.run([str(args.c2f_python), '-m', 'pmc.c2f_worker',
                             '--check_environment', '--source', str(args.c2f_source)],
                            cwd=ROOT, env=env, capture_output=True, text=True, timeout=120)
    event('c2f_environment_check', stdout=result.stdout, stderr=result.stderr)
    if result.returncode:
        raise RuntimeError('Environment C2F chưa sẵn sàng: '+result.stderr[-2000:])
    import torch
    device = torch.device(args.device)
    if device.type != 'cuda' or not torch.cuda.is_available():
        raise RuntimeError('PMC/InstaOrder hiện yêu cầu CUDA; dùng --device cuda:0 trên server GPU')
    # InstaOrder gốc dùng .cuda() không chỉ rõ index nên đặt current device ở đây.
    torch.cuda.set_device(device)
    event('preflight_ok', torch_version=torch.__version__, python=platform.python_version(),
          gpu=torch.cuda.get_device_name(device), device=str(device), c2f=identity)
    return identity


def run_pipeline(args):
    # Run riêng không xóa output cũ, kể cả khi server chạy lại cùng ảnh.
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid.uuid4().hex[:8]
    directory = args.output_dir / run_id
    directory.mkdir(parents=True)
    configure_logging(directory, args.log_level)
    summary = {'output_dir': str(directory), 'run_id': run_id, 'status': 'running', 'objects': [], 'images': []}
    manager = None
    with context(run_id=run_id):
        try:
            save_json(directory / 'config.json', vars(args))
            with stage('preflight'):
                identity = preflight(args)
            if args.preflight_only:
                summary['status'] = 'preflight_ok'
                return
            manager = ModelManager(args.device, args.offload_policy)
            register_backends(manager, args)
            engine = InferenceEngine(manager, args)
            predictor = ShapePredictor(args, identity, manager)
            classes = [line.strip() for line in args.classes_txt.read_text().splitlines() if line.strip()]
            if not classes:
                raise ValueError('classes_txt rỗng')
            if args.img_filenames_txt:
                files = [args.input_dir / name.strip() for name in args.img_filenames_txt.read_text().splitlines() if name.strip()]
            else:
                files = sorted(path for path in args.input_dir.iterdir() if path.suffix.lower() in {'.png', '.jpg', '.jpeg', '.webp'})
            if not files:
                raise ValueError('Không có ảnh input để chạy')
            for image_index, path in enumerate(files):
                with context(image=str(path), image_index=image_index):
                    try:
                        with stage('input_segmentation'):
                            with Image.open(path) as handle:
                                image = handle.convert('RGB')
                            masks, names, scores = engine.segment([image], classes)[0]
                        if masks is None:
                            event('image_skipped', reason='no_detection')
                            summary['images'].append({'path': str(path), 'status': 'no_detection'})
                            continue
                        summary['images'].append({'path': str(path), 'status': 'segmented', 'objects': len(masks)})
                        for object_id in range(len(masks)):
                            folder = directory / f'image_{image_index:06d}' / f'object_{object_id:03d}'
                            query = QueryObject(image, masks[object_id], object_id, folder)
                            record = {'image': str(path), 'object_id': object_id, 'output': str(folder)}
                            with context(object_id=object_id):
                                try:
                                    with stage('object_pipeline'):
                                        process_object(query, image, masks, names, scores, engine, predictor, args, classes)
                                except Exception as exc:
                                    query.status = 'failed'
                                    record['error'] = str(exc)
                                    LOG.exception('object_failed')
                                    if isinstance(exc, manager.torch.cuda.OutOfMemoryError) or not args.continue_on_error:
                                        raise
                                finally:
                                    record.update(status=query.status, iterations=query.iter_id)
                                    summary['objects'].append(record)
                                    save_json(folder / 'result.json', record)
                                    save_json(directory / 'summary.json', summary)
                    except Exception as exc:
                        LOG.exception('image_failed')
                        summary['images'].append({'path': str(path), 'status': 'failed', 'error': str(exc)})
                        if isinstance(exc, manager.torch.cuda.OutOfMemoryError) or not args.continue_on_error:
                            raise
            failed = any(item['status'] == 'failed' for item in summary['objects'] + summary['images'])
            summary['status'] = 'completed_with_errors' if failed else 'completed'
        except Exception as exc:
            summary.update(status='failed', error=str(exc))
            LOG.exception('pipeline_failed')
            raise
        finally:
            if manager is not None:
                manager.close()
            save_json(directory / 'summary.json', summary)
            event('pipeline_finished', status=summary['status'], output=str(directory))
    return summary


if __name__ == '__main__':
    run_pipeline(parse_args())

"""Hình học canvas/crop kế thừa PMC, với bounds an toàn và mask không bị mutate."""
import math
import numpy as np


def find_mask_sides(mask, val=1):
    """Lấy biên theo thứ tự hàng/cột; không sửa mask đầu vào."""
    mask = (np.asarray(mask) > 0).astype(np.uint8)
    x_arr, y_arr = np.where(mask == val)
    if not len(x_arr):
        raise ValueError("Mask rỗng: không xác định được bounding box")
    x_min, x_max = min(x_arr), max(x_arr)
    y_min, y_max = min(y_arr), max(y_arr)
    return x_min, x_max, y_min, y_max


def expand_bbox(bboxes):
    """Mở rộng box theo quy tắc InstaOrder gốc."""
    new_bboxes = []
    for bbox in bboxes:
        centerx = bbox[0] + bbox[2] / 2.
        centery = bbox[1] + bbox[3] / 2.
        size = max([np.sqrt(bbox[2] * bbox[3] * 3.0), bbox[2] * 1.1, bbox[3] * 1.1])
        new_bbox = [int(centerx - size / 2.), int(centery - size / 2.), int(size), int(size)]
        new_bboxes.append(new_bbox)
    return np.array(new_bboxes)


def find_expand_bboxes_instaorder(masks):
    """Tính các box xywh từ instance masks để đưa vào InstaOrder."""
    bboxes = np.zeros((len(masks), 4))
    for i, mask in enumerate(masks):
        mask = (np.asarray(mask) > 0).astype(np.uint8)
        x_min_obj, x_max_obj, y_min_obj, y_max_obj = find_mask_sides(mask)
        w = y_max_obj - y_min_obj; h = x_max_obj - x_min_obj
        bboxes[i, 0] = y_min_obj
        bboxes[i, 1] = x_min_obj
        bboxes[i, 2] = w
        bboxes[i, 3] = h
    bboxes = expand_bbox(bboxes)
    return bboxes


def aggregate_occluders(query_mask, all_occluder_masks, query_class, mask_id, iter_id, save_interm=False, output_img_dir=None):
    """Hợp các occluder masks và loại vùng visible của target."""
    agg_occluder_mask = np.zeros((query_mask.shape))
    for occluder_mask in all_occluder_masks:
        agg_occluder_mask += occluder_mask
    agg_occluder_mask[agg_occluder_mask > 0] = 1

    # Ensure new occluders do not contain query mask
    query_occ_overlap = query_mask + agg_occluder_mask
    agg_occluder_mask[query_occ_overlap > 1] = 0  # Prevent occluder from containing query mask
    return agg_occluder_mask


def create_canvas(input_arr, size_multiplier, canvas_val):
    """Đặt ảnh/mask vào chính giữa canvas; giữ nguyên dtype để tiết kiệm RAM."""
    input_height, input_width = input_arr.shape[0], input_arr.shape[1]
    canvas_shape = list(input_arr.shape)
    canvas_shape[0] = int(input_height * size_multiplier)
    canvas_shape[1] = int(input_width * size_multiplier)
    canvas_shape = tuple(canvas_shape)

    assert canvas_val >= 0
    if canvas_val > 0:
        canvas = np.full(canvas_shape, canvas_val, dtype=input_arr.dtype)
    else:
        canvas = np.zeros(canvas_shape, dtype=input_arr.dtype)

    # Place input on canvas
    input_height_start = (canvas_shape[0] // 2) - (input_height // 2)
    input_height_end = input_height_start + input_height
    input_width_start = (canvas_shape[1] // 2) - (input_width // 2)
    input_width_end = input_width_start + input_width
    canvas[input_height_start : input_height_end, input_width_start : input_width_end] = input_arr
    return canvas


def check_touch_boundary(mask, gap_pixels=10):
    """Kiểm tra target có gần biên ảnh/crop hay không."""
    H, W = mask.shape[0], mask.shape[1]
    x_max_obj, x_min_obj, y_min_obj, y_max_obj = find_mask_sides(mask)

    sides_touched = set()
    if (x_max_obj >= H - gap_pixels):
        sides_touched.add("bottom")
    if (x_min_obj <= gap_pixels):
        sides_touched.add("top")
    if (y_max_obj >= W - gap_pixels):
        sides_touched.add("right")
    if (y_min_obj <= gap_pixels):
        sides_touched.add("left")
    return sides_touched


def find_crop_region(query_mask, query_mask_canvas, pad_pixels=60, crop_buffer=60):
    """Tính crop vuông theo PMC; x là hàng, y là cột trong code cũ."""
    query_mask_canvas_height, query_mask_canvas_width = query_mask_canvas.shape
    crop_x_min, crop_x_max, crop_y_min, crop_y_max = find_mask_sides(query_mask_canvas)
    crop_x_min = max(0, crop_x_min - crop_buffer)
    crop_x_max = min(query_mask_canvas_height, crop_x_max + crop_buffer)
    crop_y_min = max(0, crop_y_min - crop_buffer)
    crop_y_max = min(query_mask_canvas_width, crop_y_max + crop_buffer)

    # Conditional padding
    sides_touched = check_touch_boundary(query_mask)
    if "top" in sides_touched:
        crop_x_min -= pad_pixels
    if "bottom" in sides_touched:
        crop_x_max += pad_pixels
    if "left" in sides_touched:
        crop_y_min -= pad_pixels
    if "right" in sides_touched:
        crop_y_max += pad_pixels

    # Compute target cropped region size
    crop_height = crop_x_max - crop_x_min
    crop_width = crop_y_max - crop_y_min
    crop_target_size = max(crop_height, crop_width)

    # Update cropped region to square
    if crop_width < crop_target_size:
        crop_target_size_diff = (crop_target_size - crop_width) / 2
        crop_y_min -= math.floor(crop_target_size_diff)
        crop_y_max += math.ceil(crop_target_size_diff)
    if crop_height < crop_target_size:
        crop_target_size_diff = (crop_target_size - crop_height) / 2
        crop_x_min -= math.floor(crop_target_size_diff)
        crop_x_max += math.ceil(crop_target_size_diff)
    # Chặn bounds để NumPy không hiểu chỉ số âm thành vùng cuối canvas.
    # Dời cửa sổ vào trong canvas thay vì cắt mask/prior bằng quy tắc khác nhau.
    crop_target_size = min(crop_target_size, query_mask_canvas_height, query_mask_canvas_width)
    crop_x_min = min(max(crop_x_min, 0), query_mask_canvas_height - crop_target_size)
    crop_y_min = min(max(crop_y_min, 0), query_mask_canvas_width - crop_target_size)
    crop_x_max = crop_x_min + crop_target_size
    crop_y_max = crop_y_min + crop_target_size
    return crop_x_min, crop_x_max, crop_y_min, crop_y_max, crop_target_size


def crop(inputs, query_mask, query_mask_canvas):
    """Cắt tất cả ảnh/mask bằng cùng bounds, bảo toàn alignment."""
    crop_x_min, crop_x_max, crop_y_min, crop_y_max, crop_target_size = find_crop_region(query_mask, query_mask_canvas)
    return [input_arr[crop_x_min : crop_x_max, crop_y_min : crop_y_max].astype(np.uint8) for input_arr in inputs], crop_x_min, crop_x_max, crop_y_min, crop_y_max


def compute_offset(expanded_query_mask, init_outpainting_mask, amodal_segmentation):
    """Tính offset giữa ảnh hoàn thiện và vị trí trên ảnh ban đầu."""
    query_x_arr, query_y_arr = np.where(expanded_query_mask == 1)
    query_x_coord = min(query_x_arr); query_y_coord = min(query_y_arr)

    orig_x_arr, orig_y_arr = np.where(init_outpainting_mask == 0)
    orig_x_coord = min(orig_x_arr); orig_y_coord = min(orig_y_arr)

    amodal_seg_x_arr, amodal_seg_y_arr = np.where(amodal_segmentation == 1)
    amodal_x_coord = min(amodal_seg_x_arr); amodal_y_coord = min(amodal_seg_y_arr)

    x_offset = int(query_x_coord - amodal_x_coord - orig_x_coord)
    y_offset = int(query_y_coord - amodal_y_coord - orig_y_coord)
    return x_offset, y_offset

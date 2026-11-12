"""Call the pinned CS331 paper implementation, preserving its diffusion code."""
import ast
from functools import wraps
import hashlib
import inspect
import math
from pathlib import Path
import sys
import types

import numpy as np
from PIL import Image

UPSTREAM_COMMIT = "bfbfcdec581484586774d9ca53f5c603c97c725b"
UPSTREAM_MAIN_SHA256 = "eed6b9eefe6c8aff9ad3d14046c5df8c04b5bc79cfd8ef586e97e51456331c8e"
SD_MODEL = "sd2-community/stable-diffusion-2-inpainting"
SD_REVISION = "5f74973cbb64c8568780732c17f43eb269d63a0d"


def paper_touch_boundary(mask, gap_pixels=10):
    """CS331 Appendix A.1: any object pixel within ten pixels of an edge.

    The pinned release unpacks find_mask_sides' min/max coordinates in reverse.
    Keep the same inclusive comparisons, fixing only that coordinate mix-up.
    """
    rows, cols = np.where(np.asarray(mask) > 0)
    if not rows.size:
        return set()
    height, width = mask.shape
    return {side for side, touched in (
        ("top", rows.min() <= gap_pixels), ("bottom", rows.max() >= height - gap_pixels),
        ("left", cols.min() <= gap_pixels), ("right", cols.max() >= width - gap_pixels)) if touched}


def install_runtime_hooks(module, boundary_check="paper"):
    """Repair the boundary gate and observe the real upstream rejection paths.

    Each worker executes one target at a time. These hooks leave model calls,
    thresholds, successful returned values, and RNG state unchanged, except for
    the named boundary repair. Empty completion detections use upstream's None
    sentinel instead of reaching analyze_masks with no masks.
    """
    if boundary_check not in ("paper", "released"):
        raise ValueError("Boundary check must be paper or released")
    original_iteration = module.run_iteration
    original_valid = module.check_valid_query
    original_boundary = module.check_touch_boundary
    original_occlusion = module.check_occlusion
    valid_signature = inspect.signature(original_valid)
    active = None
    stage = None

    @wraps(original_valid)
    def valid(*args, **kwargs):
        result = original_valid(*args, **kwargs)
        if active is not None:
            bound = valid_signature.bind(*args, **kwargs)
            bound.apply_defaults()
            values = bound.arguments
            index = values["mask_id"]
            area = float(np.asarray(values["query_mask"]).astype(np.uint8).sum())
            fraction = area / (values["img"].shape[0] * values["img"].shape[1])
            score = float(values["pred_scores"][index])
            name = values["class_names"][index]
            active.update(query_mask_fraction=fraction, detector_score=score, detected_class=name)
            if result is None:
                active["stop_reason"] = (
                    "detector_score_below_threshold" if score < values["query_pred_score_thresh"] else
                    "query_mask_too_small" if fraction < values["query_mask_size_thresh"] else
                    "unknown_detected_class" if name not in values["classes"] else "invalid_query")
        return result

    boundary_function = paper_touch_boundary if boundary_check == "paper" else original_boundary

    @wraps(original_boundary)
    def boundary(*args, **kwargs):
        result = boundary_function(*args, **kwargs)
        if active is not None:
            key = "completion_boundary_sides" if stage == "completion" else "boundary_sides"
            active[key] = sorted(result)
        return result

    @wraps(original_occlusion)
    def occlusion(*args, **kwargs):
        nonlocal stage
        stage = "completion"
        # The expanded detector returns a valid (0,H,W) array when every query
        # is empty. Upstream checks only for None, then indexes masks[0]. Apply
        # its existing no-segmentation path at this boundary so initial scene
        # detection and the recorded detector/matcher identities stay intact.
        segment = module.segment

        @wraps(segment)
        def completion_segment(*segment_args, **segment_kwargs):
            rgb, masks, classes, scores = segment(*segment_args, **segment_kwargs)
            if masks is not None and len(masks) == 0:
                if active is not None:
                    active["completion_segmentation_failure"] = "empty_mask_array"
                masks = None
            return rgb, masks, classes, scores

        module.segment = completion_segment
        try:
            return original_occlusion(*args, **kwargs)
        finally:
            module.segment = segment

    @wraps(original_iteration)
    def iteration(query_obj, *args, **kwargs):
        nonlocal active, stage
        trace = {"iteration": int(query_obj.iter_id)}
        diagnostics = getattr(query_obj, "benchmark_diagnostics", None)
        if diagnostics is None:
            diagnostics = {"boundary_check": boundary_check, "iterations": []}
            query_obj.benchmark_diagnostics = diagnostics
        previous = active, stage
        active, stage = trace, "input"
        try:
            result = original_iteration(query_obj, *args, **kwargs)
            if "stop_reason" not in trace:
                mask = getattr(result, "amodal_segmentation", None)
                trace["stop_reason"] = (
                    "no_occlusion_detected" if result.iter_id == trace["iteration"] else
                    "completion_not_detected" if mask is None or not np.asarray(mask).any() else
                    "occlusion_remaining" if result.run_iter else "complete")
            diagnostics["iterations"].append(trace)
            diagnostics["stop_reason"] = trace["stop_reason"]
            return result
        finally:
            active, stage = previous

    module.check_valid_query = valid
    module.check_touch_boundary = boundary
    module.check_occlusion = occlusion
    module.run_iteration = iteration


def size_mixed_context_background(function):
    """Enlarge the backdrop only when a progressive crop exceeds its bounds.

    The released function indexes the fixed wallpaper using crop coordinates.
    Previously supported crops retain their original pixels and behavior.
    """
    signature = inspect.signature(function)

    @wraps(function)
    def sized(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        image = bound.arguments["sd_img"]
        background = bound.arguments["clean_bkgd_img"]
        height, width = image.shape[:2]
        bg_height, bg_width = background.shape[:2]
        needs_resize = False
        if bg_height < height or bg_width < width:
            # Preserve even oversized crops that previously worked because
            # their out-of-range pixels all belonged to the visible query.
            rows, cols = np.where(1 - bound.arguments["sd_modal_mask"] == 1)
            needs_resize = rows.size > 0 and (rows.max() >= bg_height or cols.max() >= bg_width)
        if needs_resize:
            scale = max(height / bg_height, width / bg_width)
            new_size = (max(width, math.ceil(bg_width * scale)),
                        max(height, math.ceil(bg_height * scale)))
            bound.arguments["clean_bkgd_img"] = np.asarray(
                Image.fromarray(background).resize(new_size, Image.Resampling.BICUBIC))
            print(f"Mixed Context background enlarged from {bg_width}x{bg_height} to "
                  f"{new_size[0]}x{new_size[1]} for {width}x{height} crop", flush=True)
        return function(*bound.args, **bound.kwargs)

    return sized


def load_upstream(repo, model, boundary_check="paper", resize_background=True):
    source = (Path(repo) / "main.py").read_bytes()
    if hashlib.sha256(source).hexdigest() != UPSTREAM_MAIN_SHA256:
        raise ValueError("Upstream main.py differs from the pinned CS331 implementation")
    tree = ast.parse(source, filename=str(Path(repo) / "main.py"))
    replacements = 0
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "from_pretrained" and node.args
                and isinstance(node.args[0], ast.Constant)
                and node.args[0].value == "stabilityai/stable-diffusion-2-inpainting"):
            node.args[0] = ast.copy_location(ast.Constant(model), node.args[0])
            if model == SD_MODEL:
                node.keywords.append(ast.keyword(arg="revision", value=ast.Constant(SD_REVISION)))
            replacements += 1
    if replacements != 2:
        raise ValueError("Expected two SD2 model-source arguments in pinned main.py")
    ast.fix_missing_locations(tree)
    module = types.ModuleType("pdmc_upstream")
    module.__file__ = str(Path(repo) / "main.py")
    sys.modules[module.__name__] = module
    exec(compile(tree, module.__file__, "exec"), module.__dict__)
    if resize_background:
        module.run_mixed_context_diffusion = size_mixed_context_background(module.run_mixed_context_diffusion)
    install_runtime_hooks(module, boundary_check)
    return module


def complete_target(upstream, models, pil, rgb, masks, classes, scores, target,
                    vocabulary, image_path, output, sam_path, gray_path, mc_timestep, max_iter):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    gdino, diffusion, ordering, lama = models
    query = upstream.QueryObject(str(image_path), rgb, pil, target,
                                 masks[target].copy(), str(output))
    while query.run_iter:
        query = upstream.run_iteration(
            query, str(output), masks, vocabulary, classes, scores, gdino,
            str(sam_path), ordering, diffusion, lama, mc_timestep, str(gray_path),
            save_interm=False)
        # Preserve the released main.py's iteration condition, including its
        # inclusive zero-based maximum (max_iter_id=5 permits six passes).
        if query.iter_id > max_iter:
            break
    mask = getattr(query, "amodal_segmentation", None)
    details = {"detected_class": classes[target], "detector_score": float(scores[target])}
    if hasattr(query, "benchmark_diagnostics"):
        details["diagnostics"] = query.benchmark_diagnostics
    if mask is None or not np.asarray(mask).any() or query.iter_id == 0:
        return {"status": "no_completion", "iterations": query.iter_id, **details,
                "failure_reason": details.get("diagnostics", {}).get("stop_reason", "unspecified_upstream_stop")}
    query.amodal_completion.save(output / "completion.jpg", quality=90)
    alpha = Image.fromarray((np.asarray(mask) > 0).astype(np.uint8) * 255, "L")
    alpha.save(output / "mask.png")
    # Match upstream's JPEG RGB output before adding its segmentation as alpha.
    with Image.open(output / "completion.jpg") as saved:
        rgba = saved.convert("RGBA")
    if rgba.size != alpha.size:
        raise ValueError("Upstream completion/mask dimensions differ")
    rgba.putalpha(alpha)
    rgba.save(output / "completion.png")
    offset = upstream.compute_offset(query.query_mask_canvas, query.init_outpaint_mask_canvas, mask)
    return {"status": "success", "iterations": query.iter_id,
            "geometry": {"original_size": list(pil.size), "offset_rc": [int(v) for v in offset]},
            **details,
            "iteration_limit_reached": bool(query.run_iter and query.iter_id > max_iter)}

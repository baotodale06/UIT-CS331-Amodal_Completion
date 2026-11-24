"""Shape-only or joint Shape/Surface selection inside each released PMC pass."""
from functools import wraps
import inspect
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .dataset import sha256, write_json
from .shape import choose, make_network, overlap, predict, resized, score_masks
from .surface import score_surface
from .surface_refinement import RefinementConfig, pool_surface_features, predict_rank, refine_surface
from .surface_ensemble import EnsembleConfig, refine_surface_pool
from .surface_runtime import progressive_seed, project_original_observation


def shape_contract(checkpoint, surface=None):
    training = json.loads(checkpoint.with_name("training.json").read_text())
    if sha256(checkpoint) != training["checkpoint_sha256"]:
        raise ValueError("Shape checkpoint differs from the training receipt")
    package = Path(__file__).resolve().parent
    if sha256(package/"shape.py") != training["implementations"]["amodal_benchmark/shape.py"]:
        raise ValueError("Shape implementation differs from training")
    return dict(method="combined" if surface else "shape", checkpoint_sha256=sha256(checkpoint),
                training_sha256=sha256(checkpoint.with_name("training.json")),
                implementation_sha256=sha256(Path(__file__)), shape_sha256=sha256(package/"shape.py"),
                seeds=list(range(11,89,11)), surface=surface,
                input="original image, original visible mask, initial occluders and outpaint in current crop",
                selection="prior IoU; combined uses equal negative IoU and frozen surface loss midranks",
                native_visible_restoration=bool(surface))


def joint_selection(observed, visible, images, masks, shape_scores, features, label, freeze):
    occlusion = ~visible
    surface = [score_surface(observed, visible, occlusion, im, mask) for im,mask in zip(images,masks)]
    pool = pool_surface_features(observed, visible, occlusion, images, masks)
    deep = features.extract(observed, visible, images, masks, label)
    descriptors = [{**a, **b, **c} for a,b,c in zip(surface,pool,deep)]
    surface_index, losses = predict_rank(descriptors,freeze["ranker"])
    index, fallback = choose(shape_scores,losses)
    config = RefinementConfig(**freeze["refinement"])
    corrected = [refine_surface(observed,visible,im,mask,config)[0] for im,mask in zip(images,masks)]
    corrected, details = refine_surface_pool(corrected,masks,visible,EnsembleConfig(**freeze["ensemble"]))
    return corrected[index], index, fallback, dict(surface_index=surface_index, surface_losses=losses,
                                                    observable_scores=descriptors, calibration=details)


def install_shape(upstream, models, sam_path, checkpoint, surface_freeze=None, features=None):
    import cv2
    import torch
    model = make_network().cuda().eval()
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    original_crop = upstream.crop
    original_mixed = upstream.run_mixed_context_diffusion
    original_check = upstream.check_occlusion
    segment = upstream.segment
    mixed_signature = inspect.signature(original_mixed)
    check_signature = inspect.signature(original_check)
    crop_rc = None

    @wraps(original_crop)
    def crop(*args, **kwargs):
        nonlocal crop_rc
        result = original_crop(*args, **kwargs)
        crop_rc = tuple(int(v) for v in result[1:])
        return result

    @wraps(original_mixed)
    def mixed(*args, **kwargs):
        bound = mixed_signature.bind(*args, **kwargs)
        bound.apply_defaults()
        query = bound.arguments["query_obj"]
        pipeline = bound.arguments["sd_inpaint_model"]
        label = bound.arguments["sd_prompt"]
        r0,r1,c0,c1 = crop_rc
        region = np.s_[r0:r1,c0:c1]
        rgb_native = np.asarray(query.init_img_canvas[region], np.uint8)
        visible_native = np.asarray(query.init_query_mask_canvas[region], bool)
        if visible_native.shape != bound.arguments["sd_occ_mask"].shape:
            raise ValueError("Original shape observation differs from sampler frame")
        occ_native = np.logical_or(query.init_occ_mask_canvas[region],query.init_outpaint_mask_canvas[region])
        occ_native = cv2.dilate(occ_native.astype(np.uint8),np.ones((5,5),np.uint8),iterations=3)
        target_seed = torch.initial_seed()
        seeds = [progressive_seed(target_seed,query.iter_id,slot) for slot in range(11,89,11)]

        @torch.inference_mode()
        def sample_pool(**arguments):
            images, masks = [], []
            for i,seed in enumerate(seeds):
                result = pipeline(**dict(arguments,generator=torch.Generator(device="cuda").manual_seed(seed)))
                image = result.images[0].convert("RGB")
                if i == 0:
                    first = result
                    observed, visible = project_original_observation(rgb_native,visible_native,image.size)
                    occlusion = resized(occ_native,image.size).astype(bool)
                    # Remove unknown RGB before resampling so it cannot bleed into the observed context.
                    neutral = rgb_native.copy()
                    neutral[occ_native.astype(bool) & ~visible_native] = 127
                    shape_input = resized(neutral,image.size,True)
                    prior = predict(model,shape_input,visible,occlusion)
                _, instances, _, _ = segment(models[0],upstream.run_sam,str(sam_path),image,
                                             upstream.transform_image(image),[label])
                mask = np.zeros_like(visible)
                if instances is not None and len(instances):
                    scores = [overlap(visible, instance) for instance in instances]
                    if max(scores) > 0:
                        mask = np.asarray(instances[int(np.argmax(scores))],bool)
                images.append(np.array(image))
                masks.append(mask)
            scores = score_masks(prior,masks)
            if surface_freeze:
                selected,index,fallback,details = joint_selection(observed,visible,images,masks,scores,
                                                                  features,label,surface_freeze)
            else:
                index,fallback = choose(scores)
                selected,details = images[index],{}
            folder = Path(query.output_img_dir)/"shape"/f"iteration-{query.iter_id:02d}"
            folder.mkdir(parents=True,exist_ok=True)
            Image.fromarray(shape_input).save(folder/"observed.png")
            for name,mask in [("prior",prior),("visible",visible),("occlusion",occlusion)]:
                Image.fromarray(mask.astype(np.uint8)*255).save(folder/(name+".png"))
            for i,(im,mask) in enumerate(zip(images,masks)):
                Image.fromarray(im).save(folder/f"candidate-{i}.png")
                Image.fromarray(mask.astype(np.uint8)*255).save(folder/f"mask-{i}.png")
            first.images[0] = Image.fromarray(selected)
            first.images[0].save(folder/"selected.png")
            write_json(folder/"decision.json",dict(iteration=int(query.iter_id),crop_rc=crop_rc,target_seed=target_seed,
                       seeds=seeds,scores=scores,index=index,fallback=fallback,**details,
                       artifacts={p.name:sha256(p) for p in folder.glob("*.png")}))
            query.benchmark_diagnostics.setdefault("shape_iterations",[]).append(
                dict(index=index,fallback=fallback,decision_sha256=sha256(folder/"decision.json")))
            return first
        bound.arguments["sd_inpaint_model"] = sample_pool
        return original_mixed(*bound.args,**bound.kwargs)

    @wraps(original_check)
    def check(*args, **kwargs):
        bound = check_signature.bind(*args,**kwargs)
        query = bound.arguments["query_obj"]
        r0,r1,c0,c1 = crop_rc
        region = np.s_[r0:r1,c0:c1]
        native = np.asarray(query.init_img_canvas[region],np.uint8)
        visible = np.asarray(query.init_query_mask_canvas[region],bool)
        completion = np.array(bound.arguments["amodal_completion"].convert("RGB"))
        completion[visible] = native[visible]
        query.amodal_completion = Image.fromarray(completion)
        bound.arguments["amodal_completion"] = query.amodal_completion
        return original_check(*bound.args,**bound.kwargs)

    upstream.crop = crop
    upstream.run_mixed_context_diffusion = mixed
    if surface_freeze:
        upstream.check_occlusion = check


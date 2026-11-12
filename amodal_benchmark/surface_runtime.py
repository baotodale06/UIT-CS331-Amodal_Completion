"""Observation-only Surface-PMC integration into the pinned progressive sampler."""
from functools import wraps
from contextlib import contextmanager
import hashlib
import inspect
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from .dataset import sha256, write_json
from .surface_features import ObservedSurfaceFeatures
from .surface_policy import apply_frozen_surface


def project_original_observation(rgb, visible, size):
    """Resize only measured target pixels, never mix unknown RGB into the anchor."""
    rgb, visible = np.asarray(rgb), np.asarray(visible, bool)
    if rgb.shape != (*visible.shape, 3):
        raise ValueError('Observation and visible support must share one frame')
    weight = np.array(Image.fromarray(visible.astype(np.float32)).resize(size, Image.Resampling.BILINEAR))
    channels = [np.array(Image.fromarray(rgb[..., c].astype(np.float32) * visible).resize(
        size, Image.Resampling.BILINEAR)) / np.maximum(weight, 1e-6) for c in range(3)]
    support = np.array(Image.fromarray(visible.astype(np.uint8) * 255).resize(
        size, Image.Resampling.NEAREST)) > 0
    observed = np.clip(np.rint(np.stack(channels, axis=-1)), 0, 255).astype(np.uint8)
    observed[~support] = 0
    return observed, support


def progressive_seed(target_seed, iteration, candidate_seed):
    payload = f'surface-progressive-v1:{target_seed}:{iteration}:{candidate_seed}'.encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], 'little') % (2**63 - 1)


def surface_contract(freeze_path, dreamsim_cache, development_review=False):
    freeze_path, dreamsim_cache = Path(freeze_path), Path(dreamsim_cache)
    freeze = json.loads(freeze_path.read_text())
    expected = ('frozen_before_fresh_test', 'provisional_development_review') if development_review else ('frozen_before_fresh_test',)
    if freeze['status'] not in expected:
        raise ValueError('Progressive evaluation requires a finalized method freeze')
    files = ('surface.py', 'surface_policy.py', 'surface_refinement.py', 'surface_ensemble.py',
             'surface_features.py', 'surface_runtime.py')
    implementations = {name: sha256(Path(__file__).parent / name) for name in files}
    if not development_review:
        for name, actual in implementations.items():
            if freeze.get('source_hashes', {}).get('amodal_benchmark/' + name) != actual:
                raise ValueError('Surface implementation differs from the pre-test freeze: ' + name)
    preload = [Path(value) for value in os.environ.get('LD_PRELOAD', '').split(':') if value]
    if any(not path.is_file() for path in preload):
        raise ValueError('Preloaded runtime libraries must resolve to captured files')
    return dict(schema=1, method='progressive-observation-anchored-surface-selection',
        freeze_sha256=sha256(freeze_path), frozen_method=freeze,
        implementation_sha256=implementations,
        dreamsim_manifest_sha256=sha256(dreamsim_cache / 'manifest.json'),
        preloaded_library_sha256={path.name: sha256(path) for path in preload},
        anchor='original observed target pixels in the current crop; normalized bilinear projection',
        native_visible_restoration='restore original measured RGB after upsampling, before final segmentation',
        generated_region='all predicted object pixels outside original visible support',
        seeds='sha256(surface-progressive-v1:target_seed:iteration:candidate_seed), first 8 little-endian bytes modulo 2**63-1',
        stopping='unchanged upstream occlusion check, resegmented after refinement',
        development_review=development_review)


@contextmanager
def isolated_dino_imports():
    """Keep DINO's unqualified imports separate from InstaOrder's utils package.

    Both released projects import a top-level ``utils``. Torch Hub prepends
    DINO's directory but Python otherwise reuses InstaOrder's cached module.
    Model functions retain their own module globals after the cache is restored.
    This runs during single-threaded worker initialization only.
    """
    def owned(name):
        return any(name == root or name.startswith(root + '.')
                   for root in ('utils', 'vision_transformer'))
    saved = {name: module for name, module in sys.modules.copy().items() if owned(name)}
    for name in saved:
        del sys.modules[name]
    try:
        yield
    finally:
        for name in list(sys.modules):
            if owned(name):
                del sys.modules[name]
        sys.modules.update(saved)


class SurfaceRuntimeFeatures:
    """The same pretrained observation features, without an evaluation interface."""
    def __init__(self, dreamsim_cache):
        import torch
        import lpips
        from dreamsim import dreamsim
        from transformers import CLIPModel, CLIPProcessor
        from .metrics import CLIP_MODEL, CLIP_REVISION
        self.torch = torch
        self.spatial = lpips.LPIPS(net='alex', spatial=True).eval().cuda()
        with isolated_dino_imports():
            self.dreamsim, self.dream_preprocess = dreamsim(pretrained=True, device='cuda', cache_dir=str(dreamsim_cache))
        self.dreamsim.eval()
        self.models = SimpleNamespace(
            clip=CLIPModel.from_pretrained(CLIP_MODEL, revision=CLIP_REVISION).eval().cuda(),
            processor=CLIPProcessor.from_pretrained(CLIP_MODEL, revision=CLIP_REVISION))
        self.extract = ObservedSurfaceFeatures(self)

    def tensor(self, rgb):
        return self.torch.from_numpy(rgb.copy()).permute(2, 0, 1)[None].float().cuda() / 127.5 - 1

    def dream_tensor(self, rgb):
        tensor = self.dream_preprocess(Image.fromarray(rgb))
        return (tensor.unsqueeze(0) if tensor.ndim == 3 else tensor).cuda()


def install_progressive_surface(upstream, models, sam_path, freeze, feature_models=None):
    """Rerank each diffusion pool, then let the released progressive check continue.

    Initial detections, target matching and reference generation are unchanged.
    The original observation is recovered using the exact crop coordinates, so
    later iterations never promote previously generated pixels to observations.
    """
    if getattr(upstream, '_surface_installed', False):
        raise ValueError('Surface runtime is already installed')
    original_iteration, original_crop = upstream.run_iteration, upstream.crop
    original_mixed = upstream.run_mixed_context_diffusion
    original_occlusion = upstream.check_occlusion
    # The annotated-vocabulary wrapper intentionally ignores its classes argument
    # for scene-level progressive checks. Candidate masks must retain the frozen
    # selector's target-only query even after that wrapper is installed per image.
    candidate_segment = upstream.segment
    iteration_signature = inspect.signature(original_iteration)
    mixed_signature = inspect.signature(original_mixed)
    occlusion_signature = inspect.signature(original_occlusion)
    active = None

    @wraps(original_crop)
    def crop(*args, **kwargs):
        result = original_crop(*args, **kwargs)
        if active is not None:
            _, r0, r1, c0, c1 = result
            query = active['query']
            anchor = np.asarray(query.init_img_canvas[r0:r1, c0:c1]).astype(np.uint8)
            visible = np.asarray(query.init_query_mask_canvas[r0:r1, c0:c1]) > 0
            if anchor.shape[:2] != result[0][0].shape[:2]:
                raise ValueError('Original observation and progressive crop are misaligned')
            active.update(anchor=anchor, visible=visible, crop_rc=[int(r0), int(r1), int(c0), int(c1)])
        return result

    @wraps(original_mixed)
    def mixed(*args, **kwargs):
        import torch
        if active is None or 'anchor' not in active:
            raise ValueError('Surface sampling requires an observed progressive crop')
        bound = mixed_signature.bind(*args, **kwargs)
        bound.apply_defaults()
        pipeline = bound.arguments['sd_inpaint_model']
        label = bound.arguments['sd_prompt']
        query = active['query']
        target_seed = torch.initial_seed()
        iteration = int(query.iter_id)
        candidate_seeds = [progressive_seed(target_seed, iteration, seed) for seed in freeze['candidate_seeds']]

        def sample_pool(**pipeline_arguments):
            images, masks, first = [], [], None
            with torch.inference_mode():
                for seed in candidate_seeds:
                    arguments = dict(pipeline_arguments, generator=torch.Generator(device='cuda').manual_seed(seed))
                    result = pipeline(**arguments)
                    if first is None:
                        first = result
                    image = result.images[0].convert('RGB')
                    if not images:
                        observed, visible = project_original_observation(active['anchor'], active['visible'], image.size)
                    if image.size != (observed.shape[1], observed.shape[0]):
                        raise ValueError('Candidate pool frames differ')
                    _, segments, _, _ = candidate_segment(models[0], upstream.run_sam, str(sam_path),
                        image, upstream.transform_image(image), [label])
                    predicted = np.zeros_like(visible)
                    if segments is not None and len(segments):
                        ious = [float((visible & mask).sum() / max(1, (visible | mask).sum())) for mask in segments]
                        best = int(np.argmax(ious))
                        if ious[best] > 0:
                            predicted = np.asarray(segments[best], bool)
                    images.append(np.array(image))
                    masks.append(predicted)
                features = feature_models.extract(observed, visible, images, masks, label) if feature_models else None
                selected, _, details = apply_frozen_surface(observed, visible, ~visible,
                    images, masks, freeze, deep_features=features)
            first.images[0] = Image.fromarray(selected)
            folder = Path(query.output_img_dir) / 'surface' / f'iteration-{iteration:02d}'
            folder.mkdir(parents=True, exist_ok=True)
            for index, image in enumerate(images):
                Image.fromarray(image).save(folder / f'candidate-{index}.png')
                Image.fromarray(masks[index].astype(np.uint8) * 255).save(folder / f'mask-{index}.png')
            first.images[0].save(folder / 'selected.png')
            Image.fromarray(observed).save(folder / 'observed.png')
            Image.fromarray(visible.astype(np.uint8) * 255).save(folder / 'visible.png')
            decision = dict(iteration=iteration, candidate_seeds=candidate_seeds,
                target_seed=target_seed, crop_rc=active['crop_rc'],
                inference_truth_access=False, **details)
            decision['artifacts'] = {path.name: sha256(path) for path in folder.glob('*.png')}
            write_json(folder / 'decision.json', decision)
            query.benchmark_diagnostics.setdefault('surface_iterations', []).append(dict(
                iteration=iteration, selected=details['index'], fallback=details['fallback'],
                decision_sha256=sha256(folder / 'decision.json')))
            return first

        bound.arguments['sd_inpaint_model'] = sample_pool
        return original_mixed(*bound.args, **bound.kwargs)

    @wraps(original_iteration)
    def iteration(*args, **kwargs):
        nonlocal active
        values = iteration_signature.bind(*args, **kwargs).arguments
        previous, active = active, {'query': values['query_obj']}
        try:
            return original_iteration(*args, **kwargs)
        finally:
            active = previous

    @wraps(original_occlusion)
    def occlusion(*args, **kwargs):
        bound = occlusion_signature.bind(*args, **kwargs)
        if active is not None and 'anchor' in active:
            image = np.array(bound.arguments['amodal_completion'].convert('RGB'))
            if image.shape != active['anchor'].shape:
                raise ValueError('Native output differs from its original-observation crop')
            image[active['visible']] = active['anchor'][active['visible']]
            restored = Image.fromarray(image)
            bound.arguments['amodal_completion'] = restored
            active['query'].amodal_completion = restored
            active['query'].benchmark_diagnostics['surface_native_visible_pixels'] = int(active['visible'].sum())
        return original_occlusion(*bound.args, **bound.kwargs)

    upstream.crop = crop
    upstream.run_mixed_context_diffusion = mixed
    upstream.run_iteration = iteration
    upstream.check_occlusion = occlusion
    upstream._surface_installed = True

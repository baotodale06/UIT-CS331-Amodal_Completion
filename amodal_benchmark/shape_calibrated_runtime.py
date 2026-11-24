"""Admitted, shape-only probability correction and unchanged candidate selection."""
from functools import wraps
import inspect
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .dataset import sha256, write_json
from .shape_calibrated_prior import candidate_policy, original_prior
from .shape_v2 import make_network, score_masks
from .shape_v2_runtime import SEED_SLOTS, choose, match_candidate, progressive_seed, project_prior

ROOT = Path(__file__).resolve().parents[1]
METHOD = "shape-calibrated-probability"
GPU_SOURCES = ("scripts/validate_shape_calibrated_prior.py", "scripts/validate_shape_external_adapter.py",
               "amodal_benchmark/shape_calibrated_prior.py", "amodal_benchmark/shape_v2.py")
REPLAY_SOURCE = "scripts/verify_shape_calibrated_admission.py"
EVIDENCE_FILES = ("local-contract.json", "local-result.json", "quadro-contract.json",
                  "quadro-result.json", "native-replay.json")


def read(path):
    return json.loads(Path(path).read_bytes())


def admission_sources():
    names = (*GPU_SOURCES, REPLAY_SOURCE, "scripts/package_shape_calibrated.py",
        "scripts/freeze_shape_probability_candidate.py", "amodal_benchmark/shape_calibrated_runtime.py",
        "amodal_benchmark/shape_v2_runtime.py", "evaluate_open_world.py")
    return {name: sha256(ROOT / name) for name in names}


def evidence_files(package):
    package = Path(package)
    return {name: sha256(package / name) for name in (*EVIDENCE_FILES,
        "candidate/candidate.json", "candidate/best.pt", "candidate/training.json",
        "candidate/diagnostic-replay.json", "candidate/selected-native.json", "candidate/selected-masks.zip")}


def verify_native_records(candidate, contract, result, family):
    """Check all exact-mask bindings, not only summary scores or a passed flag."""
    candidate = Path(candidate)
    expected = read(candidate / "selected-native.json")["rows"]
    rows = result.get("rows", [])
    if (len(expected) != 1264 or len({row["id"] for row in expected}) != 1264
            or len({row["image_id"] for row in expected}) != 217 or len(rows) != 1264
            or {row["id"] for row in rows} != {row["id"] for row in expected}):
        raise ValueError("Complete unique native replay inventory required")
    expected = {row["id"]: row for row in expected}
    required = dict(verified=True, complete=True, objects=1264, images=217, different_pixels=0,
        inference_gpu_validated=family == "quadro", benchmark_admitted=False, benchmark_success=False)
    if any(result.get(key) != value for key, value in required.items()):
        raise ValueError("Complete exact native replay required")
    required_contract = dict(device_family=family, torch="1.13.1", cuda="11.7", cuda_fft_round_trip=True,
        candidate_sha256=sha256(candidate / "candidate.json"), preflight=False, holdout_access=False,
        benchmark_access=False, benchmark_admitted=False, benchmark_success=False,
        sources={name: sha256(ROOT / name) for name in GPU_SOURCES})
    expected_gpu = "QUADRO RTX 6000" if family == "quadro" else "RTX 3060"
    if (any(contract.get(key) != value for key, value in required_contract.items())
            or expected_gpu not in contract.get("gpu", "").upper()):
        raise ValueError("Native GPU execution contract changed")
    for row in rows:
        reference = expected[row["id"]]
        if (row.get("image_id") != reference["image_id"] or row.get("different_pixels") != 0
                or row.get("expected_pixel_sha256") != reference["prediction_sha256"]
                or row.get("actual_pixel_sha256") != reference["prediction_sha256"]):
            raise ValueError("Native replay pixels or identities differ")


def verify_evidence(package):
    package = Path(package)
    policy = candidate_policy(package / "candidate")
    if policy.get("arithmetic") != dict(cudnn_deterministic=True, cudnn_benchmark=False,
            cudnn_allow_tf32=False, matmul_allow_tf32=False, autocast=False):
        raise ValueError("Frozen native arithmetic changed")
    for metric in ("full_iou", "hidden_iou"):
        values = [policy[name][metric] for name in ("native_baseline", "native_selected")]
        if (any(type(value) not in (int, float) or not np.isfinite(value) or not 0 <= value <= 1 for value in values)
                or values[1] < values[0]):
            raise ValueError("Both native geometry floors must be retained")
    for family in ("local", "quadro"):
        contract, result = (read(package / (family + suffix)) for suffix in ("-contract.json", "-result.json"))
        if result.get("contract_sha256") != sha256(package / (family + "-contract.json")):
            raise ValueError("Native replay contract binding changed")
        verify_native_records(package / "candidate", contract, result, family)
    replay = read(package / "native-replay.json")
    expected = dict(verified=True, objects=1264, images=217,
        candidate_sha256=sha256(package / "candidate/candidate.json"),
        local_result_sha256=sha256(package / "local-result.json"),
        quadro_result_sha256=sha256(package / "quadro-result.json"),
        selected_geometry=policy["native_selected"],
        replay_source_sha256=sha256(ROOT / REPLAY_SOURCE), benchmark_success=False)
    if any(replay.get(key) != value for key, value in expected.items()):
        raise ValueError("Independent selected-mask replay changed or is missing")
    return policy


def admitted_policy(package):
    package = Path(package)
    policy = verify_evidence(package)
    admission = read(package / "admission.json")
    if (admission.get("schema") != 1 or admission.get("method") != METHOD
            or admission.get("files") != evidence_files(package)
            or admission.get("sources") != admission_sources()
            or admission.get("benchmark_success") is not False
            or admission.get("checkpoint_sha256") != policy["checkpoint_sha256"]):
        raise ValueError("Calibrated admission changed; build a new isolated package")
    return policy


def shape_calibrated_contract(package):
    package = Path(package)
    policy = admitted_policy(package)
    return dict(schema=1, method=METHOD, checkpoint_sha256=policy["checkpoint_sha256"],
        candidate_sha256=sha256(package / "candidate/candidate.json"),
        admission_sha256=sha256(package / "admission.json"), files=evidence_files(package),
        implementations=admission_sources(), checkpoint_epoch=7, logit_bias=-2.,
        seeds=list(SEED_SLOTS), seed_rule="unchanged shape-v2 progressive_seed",
        observation="original visible RGB and SAM support only; unobserved RGB neutralized before resizing",
        crop="square visible box, padding 2.0; logits resized before fixed correction and sigmoid",
        projection="native probability into current PMC crop; bilinear resize; visible pixels set to one",
        selection="maximum expected mask IoU; exclude empty masks; first seed wins ties; slot 0 fallback",
        candidate_matching="highest IoU with original visible support; require positive overlap",
        arithmetic=policy["arithmetic"], surface=None, rgb_correction=False, validation_reused=True,
        interpretation="Validation-selected shape-only correction; benchmark improvement remains to be measured")


def load_prior(package):
    import torch
    package = Path(package)
    policy = admitted_policy(package)
    if (not torch.cuda.is_available() or torch.cuda.device_count() != 1
            or "QUADRO RTX 6000" not in torch.cuda.get_device_name().upper()
            or not str(torch.__version__).startswith("1.13.1") or torch.version.cuda != "11.7"):
        raise RuntimeError("Use one visible validated Quadro per benchmark worker")
    training = read(package / "candidate/training.json")
    saved = torch.load(package / "candidate/best.pt", map_location="cpu", weights_only=False)
    expected = next(row["validation"] for row in training["history"] if row["epoch"] == 7)
    if saved.get("epoch") != 7 or saved.get("contract") != training["contract"] or saved.get("validation") != expected:
        raise ValueError("Calibrated checkpoint payload changed")
    model = make_network(pretrained=False)
    model.load_state_dict(saved["model"], strict=True)
    return model.float().cuda().eval()


def install_shape_calibrated(upstream, models, sam_path, package):
    """Select among eight unchanged diffusion candidates; preserve all other gates."""
    import torch
    if any(getattr(upstream, name, False) for name in ("_shape_calibrated_installed", "_shape_constrained_installed",
            "_shape_v2_installed", "_surface_installed", "_shape_installed")):
        raise ValueError("Calibrated shape requires an unmodified shape-only selector")
    model = load_prior(package)
    original_iteration, original_crop = upstream.run_iteration, upstream.crop
    original_mixed, candidate_segment = upstream.run_mixed_context_diffusion, upstream.segment
    mixed_signature = inspect.signature(original_mixed)
    active = None

    @wraps(original_iteration)
    def iteration(query_obj, *args, **kwargs):
        nonlocal active
        if active is not None:
            raise ValueError("Nested calibrated shape execution")
        active = dict(query=query_obj, crop_rc=None)
        try:
            return original_iteration(query_obj, *args, **kwargs)
        finally:
            active = None

    @wraps(original_crop)
    def crop(*args, **kwargs):
        result = original_crop(*args, **kwargs)
        if active is not None:
            active["crop_rc"] = tuple(int(value) for value in result[1:])
        return result

    @wraps(original_mixed)
    def mixed(*args, **kwargs):
        bound = mixed_signature.bind(*args, **kwargs)
        bound.apply_defaults()
        query = bound.arguments["query_obj"]
        if active is None or active["query"] is not query or active["crop_rc"] is None:
            raise ValueError("Calibrated shape sampler has no matching progressive crop")
        crop_rc = active["crop_rc"]
        r0, r1, c0, c1 = crop_rc
        if (r1-r0, c1-c0) != bound.arguments["sd_occ_mask"].shape:
            raise ValueError("Calibrated prior and progressive crop frames differ")
        visible = np.asarray(query.init_query_mask_canvas, bool)
        canvas, box, observed, input_support = original_prior(model, query.init_img_canvas, visible)
        if observed is None:
            observed, input_support = np.full((1, 1, 3), 127, np.uint8), np.zeros((1, 1), bool)
        target_seed = torch.initial_seed()
        seeds = [progressive_seed(target_seed, query.iter_id, slot) for slot in SEED_SLOTS]
        pipeline, label = bound.arguments["sd_inpaint_model"], bound.arguments["sd_prompt"]

        @torch.inference_mode()
        def sample_pool(**arguments):
            images, masks = [], []
            for index, seed in enumerate(seeds):
                result = pipeline(**dict(arguments, generator=torch.Generator(device="cuda").manual_seed(seed)))
                if len(result.images) != 1:
                    raise ValueError("Calibrated shape expects one image per seed")
                image = result.images[0].convert("RGB")
                if index == 0:
                    first, size = result, image.size
                    prior, support = project_prior(canvas, visible, crop_rc, size)
                elif image.size != size:
                    raise ValueError("Candidate frames differ")
                _, instances, _, _ = candidate_segment(models[0], upstream.run_sam, str(sam_path), image,
                    upstream.transform_image(image), [label])
                images.append(np.array(image))
                masks.append(match_candidate(instances, support))
            scores = score_masks(prior, masks)
            index, fallback = choose(scores)
            folder = Path(query.output_img_dir) / METHOD / f"iteration-{query.iter_id:02d}"
            folder.mkdir(parents=True, exist_ok=True)
            Image.fromarray(observed).save(folder / "observed-input.png")
            Image.fromarray(input_support.astype(np.uint8)*255).save(folder / "input-visible.png")
            np.save(folder / "prior.npy", prior, allow_pickle=False)
            Image.fromarray(np.rint(prior*255).astype(np.uint8)).save(folder / "prior.png")
            Image.fromarray(support.astype(np.uint8)*255).save(folder / "visible.png")
            for slot, (image, mask) in enumerate(zip(images, masks)):
                Image.fromarray(image).save(folder / f"candidate-{slot}.png")
                Image.fromarray(mask.astype(np.uint8)*255).save(folder / f"mask-{slot}.png")
            first.images[0] = Image.fromarray(images[index])
            first.images[0].save(folder / "selected.png")
            write_json(folder / "decision.json", dict(schema=2, method=METHOD, logit_bias=-2., iteration=int(query.iter_id),
                crop_rc=crop_rc, prior_box_xyxy=box, target_seed=target_seed, seed_slots=list(SEED_SLOTS),
                seeds=seeds, scores=scores, index=index, fallback=fallback, inference_truth_access=False,
                rgb_correction=False, artifacts={p.name: sha256(p) for p in sorted(folder.iterdir())
                                                 if p.suffix in (".png", ".npy")}))
            query.benchmark_diagnostics.setdefault("shape_calibrated_iterations", []).append(dict(
                index=index, fallback=fallback, decision_sha256=sha256(folder / "decision.json")))
            return first

        bound.arguments["sd_inpaint_model"] = sample_pool
        return original_mixed(*bound.args, **bound.kwargs)

    upstream.run_iteration, upstream.crop = iteration, crop
    upstream.run_mixed_context_diffusion = mixed
    upstream._shape_calibrated_installed = True

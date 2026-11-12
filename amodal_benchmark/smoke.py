"""Synthetic, offline integration exercise of the actual evaluator and verifier."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image

from .dataset import SOURCES, target_record, write_json
from .verify import verify_artifacts


class _SyntheticPipeline:
    """Only model calls are substituted; selection, crops and artifacts stay real."""
    run_sam = None

    def load_models(self, *args):
        return [None] * 4

    def transform_image(self, image):
        return image

    def segment(self, model, sam, checkpoint, image, tensor, vocabulary):
        mask = np.zeros((1, image.height, image.width), dtype=bool)
        mask[0, 6:26, 6:26] = True
        return np.asarray(image).copy(), mask, ["cat"], [.9]

    def QueryObject(self, image_path, rgb, pil, target, mask, output):
        return SimpleNamespace(run_iter=True, iter_id=0, query_mask_canvas=mask,
                               init_outpaint_mask_canvas=mask)

    def run_iteration(self, query, *args, **kwargs):
        query.iter_id += 1
        query.run_iter = False
        color = tuple(random.randrange(30, 230) for _ in range(3))
        query.amodal_completion = Image.new("RGB", (32, 32), color)
        query.amodal_segmentation = query.query_mask_canvas
        return query

    def compute_offset(self, *args):
        return (0, 0)


class _SyntheticMetrics:
    def score(self, *args):
        return {"CLIP": 25., "LPIPS": .25, "Feature Similarity": .5, "SSIM": .75}

    def clip_score(self, *args):
        return 25.


def build_smoke_run(root):
    """Create a success and a detection failure, then test exact cache reuse."""
    import evaluate_open_world as evaluator
    root = Path(root).resolve()
    dataset, repo, output = root / "data", root / "synthetic-upstream", root / "run"
    rows = [target_record("VG", {"filename": f"{index}_{label}.jpg", "labels": [label]})
            for index, label in ((1, "cat"), (2, "dog"))]
    for row in rows:
        path = dataset / row["image"]
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (32, 32), "white").save(path)
    for name in ("main.py", "Grounded-Segment-Anything/groundingdino_swint_ogc.pth",
                 "Grounded-Segment-Anything/sam_vit_h_4b8939.pth",
                 "InstaOrder/InstaOrder_ckpt/InstaOrder_InstaOrderNet_od.pth.tar",
                 "lama/big-lama/models/best.ckpt"):
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("Synthetic CPU fixture, not model weights.\n", encoding="utf-8")
    (repo / "classes.txt").write_text("cat\ndog\n", encoding="utf-8")
    counts = {source: dict(zip(("images", "instances", "classes"), values[1:4]))
              for source, values in SOURCES.items()}
    fake_torch = SimpleNamespace(manual_seed=lambda seed: None,
        cuda=SimpleNamespace(manual_seed_all=lambda seed: None, empty_cache=lambda: None),
        backends=SimpleNamespace(cudnn=SimpleNamespace()))
    config = {
        "schema": 1, "project": {"slug": "synthetic-smoke", "task": "amodal-appearance-completion"},
        "evaluation_scope": "smoke",
        "dataset": {"root": str(dataset), "sources": list(SOURCES), "limit": 2, "allow_incomplete": False},
        "inference": {"pipeline_profile": "released-sam", "target_selection": "label-semantic",
                      "boundary_check": "released", "background_sizing": "auto", "detector_vocabulary": "released",
                      "seed": 0, "mc_timestep": 35, "max_iter": 5,
                      "sd_model": "sd2-community/stable-diffusion-2-inpainting"},
        "metrics": {"alignment": "released"}, "execution": {"workers": 1},
        "experiment": {"hypothesis": "Synthetic CPU artifact integration only; no quality estimate.",
                       "changed_factor": None},
    }
    write_json(root / "smoke-config.json", config)
    write_json(root / "synthetic-targets.json", rows)
    command = ["--config", str(root / "smoke-config.json"), "--output-dir", str(output), "--repo-dir", str(repo)]
    # Patch only within this call. No global sitecustomize hook, downloads, or GPU imports.
    with patch.dict(sys.modules, {"torch": fake_torch}), \
            patch.object(evaluator, "read_official", return_value=(rows, counts)), \
            patch.object(evaluator, "load_upstream", return_value=_SyntheticPipeline()), \
            patch("amodal_benchmark.metrics.MetricModels", _SyntheticMetrics), redirect_stdout(io.StringIO()):
        evaluator.main(command)
        saved = {row["id"]: (output / "targets" / row["id"] / "result.json").read_bytes() for row in rows}
        with patch.object(evaluator, "load_upstream", side_effect=AssertionError("Resume tried to load models")):
            evaluator.main(command)
        for identifier, data in saved.items():
            if (output / "targets" / identifier / "result.json").read_bytes() != data:
                raise AssertionError("Resume rewrote a completed result")
    report = verify_artifacts(output, dataset, rows, require_provenance=True)
    stat = report["statistics"]["Overall"]
    if (stat["successes"], stat["failures"], stat["CLIP_n"], stat["LPIPS_n"]) != (1, 1, 1, 2):
        raise AssertionError("Synthetic smoke lost success/failure denominators")
    if stat["LPIPS"] != .625:
        raise AssertionError("Synthetic smoke lost the failed-completion penalty")
    write_json(root / "smoke-report.json", {**report, "synthetic": True, "quality_estimate": False})
    return report


def smoke_check(output_dir=None):
    if output_dir is None:
        with tempfile.TemporaryDirectory(prefix="amodal-smoke-") as directory:
            report = build_smoke_run(directory)
        retained = "Temporary artifacts removed."
    else:
        root = Path(output_dir).resolve()
        root.mkdir(parents=True, exist_ok=False)
        report = build_smoke_run(root)
        retained = f"Debugging artifacts: {root}"
    print("CPU smoke passed: inference adapter, semantic matching, references, failure penalties, "
          "resume, provenance, and independent artifact verification.")
    print("Model calls and metric values are synthetic; this is not a model-quality or GPU test. " + retained)
    return report

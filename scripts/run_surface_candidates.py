#!/usr/bin/env python3
"""Generate shared one-step PMC pools from observation-only experiment inputs."""
import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback

import numpy as np
from PIL import Image

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from amodal_benchmark.dataset import sha256, write_json
from amodal_benchmark.inference import load_upstream, SD_MODEL, SD_REVISION, UPSTREAM_COMMIT
from amodal_benchmark.surface import score_surface, SurfaceConfig


class SeededPipeline:
    """Explicit generator reaches the pinned custom pipeline, including VAE draws."""
    def __init__(self, pipeline, seed, steps):
        self.pipeline, self.seed, self.steps = pipeline, seed, steps
        self.calls = 0

    def __call__(self, **kwargs):
        import torch
        self.calls += 1
        kwargs.update(generator=torch.Generator(device='cuda').manual_seed(self.seed),
                      num_inference_steps=self.steps)
        return self.pipeline(**kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment-dir', type=Path, required=True)
    parser.add_argument('--repo-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--worker-index', type=int, default=0)
    parser.add_argument('--worker-count', type=int, default=4)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--candidate-limit', type=int)
    args = parser.parse_args()
    experiment, repo, output = args.experiment_dir.resolve(), args.repo_dir.resolve(), args.output_dir.resolve()
    protocol = json.loads((experiment / 'protocol.json').read_text())
    cases = json.loads((experiment / 'generation_manifest.json').read_text())
    if args.limit:
        cases = cases[:args.limit]
    cases = cases[args.worker_index::args.worker_count]
    seeds = protocol['candidate_seeds'][:args.candidate_limit]
    output.mkdir(parents=True, exist_ok=True)
    contract = dict(protocol_sha256=sha256(experiment / 'protocol.json'),
                    generation_manifest_sha256=sha256(experiment / 'generation_manifest.json'),
                    seeds=seeds, limit=args.limit,
                    sources={str(p.relative_to(PROJECT)): sha256(p) for p in (
                        Path(__file__), PROJECT / 'amodal_benchmark/surface.py',
                        PROJECT / 'amodal_benchmark/inference.py', PROJECT / 'amodal_benchmark/appearance.py')})
    contract_path = output / f'contract-worker-{args.worker_index}.json'
    if contract_path.exists() and json.loads(contract_path.read_text()) != contract:
        raise ValueError('Changed generation contract: use a new output directory')
    write_json(contract_path, contract)
    import torch
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    os.chdir(repo)
    runtime = dict(started_at=datetime.now(timezone.utc).isoformat(),
                   job_id=os.environ.get('SLURM_JOB_ID'), worker=args.worker_index,
                   cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                   gpu=torch.cuda.get_device_name(0), torch=torch.__version__, cuda=torch.version.cuda,
                   python=sys.version, upstream_commit=UPSTREAM_COMMIT,
                   main_sha256=sha256(repo / 'main.py'), sd_model=SD_MODEL, sd_revision=SD_REVISION,
                   custom_pipeline_sha256=sha256(repo / 'diffusers/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_inpaint_mixed_context.py'),
                   packages=subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True).splitlines(),
                   explicit_generator=True, **contract)
    write_json(output / f'runtime-worker-{args.worker_index}.json', runtime)
    upstream = load_upstream(repo, SD_MODEL, boundary_check='released', resize_background=True)
    gsa = repo / 'Grounded-Segment-Anything'
    models = upstream.load_models(str(gsa / 'GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py'),
        str(gsa / 'groundingdino_swint_ogc.pth'), instaorder_ckpt=None,
        lama_config_path=str(repo / 'lama/big-lama/config.yaml'),
        lama_ckpt_path=str(repo / 'lama/big-lama/models/best.ckpt'), mc_timestep=protocol['mc_timestep'])
    gdino, diffusion, lama = models
    diffusion.set_progress_bar_config(disable=True)
    background = np.array(Image.open(repo / 'images/gray_wallpaper.jpeg').convert('RGB'))
    surface_config = SurfaceConfig(radius=protocol['boundary_radius'],
        min_region_pixels=protocol['min_region_pixels'], min_visible_recall=protocol['min_visible_recall'])
    for case in cases:
        folder = experiment / 'inputs' / case['case_id']
        for filename, expected in case['input_hashes'].items():
            if sha256(folder / filename) != expected:
                raise ValueError('Observation input hash mismatch')
        observed = np.array(Image.open(folder / 'observed.png').convert('RGB'))
        visible = np.array(Image.open(folder / 'visible.png').convert('L')) > 0
        occlusion = np.array(Image.open(folder / 'occlusion.png').convert('L')) > 0
        destination = output / 'candidates' / case['case_id']
        destination.mkdir(parents=True, exist_ok=True)
        for seed in seeds:
            record_path = destination / f'seed-{seed}.json'
            if record_path.exists():
                record = json.loads(record_path.read_text())
                for filename, digest in record.get('artifacts', {}).items():
                    if sha256(destination / filename) != digest:
                        raise ValueError('Cached candidate hash mismatch')
                print('RESUME', case['case_id'], seed, record['status'], flush=True)
                continue
            started = time.perf_counter()
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.cuda.reset_peak_memory_stats()
            record = dict(case_id=case['case_id'], seed=seed, source=case['source'], split=case['split'],
                          prompt=case['label'], input_hashes=case['input_hashes'], artifacts={})
            try:
                seeded = SeededPipeline(diffusion, seed, protocol['diffusion_steps'])
                with torch.inference_mode():
                    image = upstream.run_mixed_context_diffusion(None, seeded, lama, observed.copy(),
                        visible.astype(np.uint8), occlusion.astype(np.uint8), case['label'],
                        background, protocol['mc_timestep'], protocol['resolution'])
                if seeded.calls != 1:
                    raise ValueError('Explicit diffusion seed did not reach exactly one pipeline call')
                image = image.convert('RGB')
                if image.size != (protocol['resolution'],) * 2:
                    raise ValueError('Unexpected output frame; refuse implicit realignment')
                image.save(destination / f'seed-{seed}.png')
                record['artifacts'][f'seed-{seed}.png'] = sha256(destination / f'seed-{seed}.png')
                record['generation_complete'] = True
                record['generation_seconds'] = time.perf_counter() - started
                with torch.inference_mode():
                    _, masks, classes, scores = upstream.segment(gdino, upstream.run_sam,
                        str(gsa / 'sam_vit_h_4b8939.pth'), image, upstream.transform_image(image), [case['label']])
                predicted = np.zeros(visible.shape, bool)
                record['detection_count'] = 0 if masks is None else len(masks)
                if masks is not None and len(masks):
                    # The query text limits semantic scope; visible IoU identifies its instance.
                    ious = [float((visible & m).sum() / max(1, (visible | m).sum())) for m in masks]
                    best = int(np.argmax(ious))
                    if ious[best] > 0:
                        predicted = np.asarray(masks[best], bool)
                        record.update(matched_class=classes[best], detector_score=float(scores[best]),
                                      visible_iou=ious[best])
                Image.fromarray(predicted.astype(np.uint8) * 255).save(destination / f'seed-{seed}-mask.png')
                scoring_started = time.perf_counter()
                record['surface'] = score_surface(observed, visible, occlusion, np.array(image), predicted, surface_config)
                record['scoring_seconds'] = time.perf_counter() - scoring_started
                record['status'] = 'generated'
                record['artifacts'] = {p.name: sha256(p) for p in (
                    destination / f'seed-{seed}.png', destination / f'seed-{seed}-mask.png')}
            except Exception as error:
                record.update(status='error', error_type=type(error).__name__, error=str(error),
                              traceback=traceback.format_exc())
                record['failure_stage'] = 'segmentation_or_scoring' if record.get('generation_complete') else 'generation'
                print(record['traceback'], flush=True)
                # A saved RGB with segmentation failure still exists but is not silently
                # accepted as a valid scored candidate; evaluator retains error accounting.
            record['total_seconds'] = time.perf_counter() - started
            record['peak_allocated_bytes'] = int(torch.cuda.max_memory_allocated())
            write_json(record_path, record)
            print('CANDIDATE', case['case_id'], seed, record['status'],
                  round(record['total_seconds'], 2), record.get('surface', {}).get('valid'), flush=True)
            gc.collect()
            torch.cuda.empty_cache()
    runtime['finished_at'] = datetime.now(timezone.utc).isoformat()
    write_json(output / f'runtime-worker-{args.worker_index}.json', runtime)


if __name__ == '__main__':
    main()

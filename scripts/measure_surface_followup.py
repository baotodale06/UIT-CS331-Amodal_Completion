#!/usr/bin/env python3
"""Evaluate frozen surface choices with hidden and complete-object metrics."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from amodal_benchmark.dataset import sha256, write_json
from amodal_benchmark.metrics import MetricModels
from amodal_benchmark.surface import METHODS, select_candidate
from amodal_benchmark.surface_refinement import (RefinementConfig, predict_rank, refine_surface,
                                               pool_surface_features)


class FollowupMetrics:
    def __init__(self, dreamsim_cache=None):
        import torch
        import lpips
        self.torch = torch
        self.models = MetricModels('cuda')
        self.spatial = lpips.LPIPS(net='alex', spatial=True).eval().cuda()
        self.dreamsim = None
        if dreamsim_cache:
            from dreamsim import dreamsim
            self.dreamsim, self.dream_preprocess = dreamsim(pretrained=True, device='cuda', cache_dir=str(dreamsim_cache))
            self.dreamsim.eval()

    def tensor(self, rgb):
        return self.torch.from_numpy(rgb.copy()).permute(2, 0, 1)[None].float().cuda() / 127.5 - 1

    def dream_tensor(self, rgb):
        value = self.dream_preprocess(Image.fromarray(rgb))
        if value.ndim == 3:
            value = value.unsqueeze(0)
        if value.ndim != 4 or value.shape[0] != 1:
            raise ValueError('DreamSim preprocessing must return one BCHW image')
        return value.cuda()

    def reference(self, truth, full, hidden, label):
        torch = self.torch
        object_rgb = truth.copy()
        object_rgb[~full] = 0
        hidden_rgb = truth.copy()
        hidden_rgb[~hidden] = 127
        with torch.inference_mode():
            vgg = self.models.vgg(self.models.vgg_preprocess(object_rgb).unsqueeze(0).cuda())
            text = self.models.processor(text=[label], return_tensors='pt', padding=True).to('cuda')
            text = self.models.clip.get_text_features(**text)
            dream = self.dreamsim.embed(self.dream_tensor(object_rgb)) if self.dreamsim else None
        return dict(truth=truth, hidden=hidden, object_rgb=object_rgb, object_tensor=self.tensor(object_rgb),
                    hidden_tensor=self.tensor(hidden_rgb), vgg=vgg, text=text, dream=dream)

    def score(self, candidate, predicted, ref):
        import torch.nn.functional as F
        from skimage.metrics import structural_similarity
        torch = self.torch
        hidden = ref['hidden']
        obj = candidate.copy()
        obj[~predicted] = 0
        masked = candidate.copy()
        masked[~hidden] = 127
        with torch.inference_mode():
            hmap = self.spatial(ref['hidden_tensor'], self.tensor(masked))[0, 0]
            lpips = float(self.models.lpips(ref['object_tensor'], self.tensor(obj)).item())
            vgg = self.models.vgg(self.models.vgg_preprocess(obj).unsqueeze(0).cuda())
            feature = float(F.cosine_similarity(ref['vgg'], vgg).mean().item())
            inputs = self.models.processor(images=Image.fromarray(obj), return_tensors='pt').to('cuda')
            image = self.models.clip.get_image_features(**inputs)
            clip_cosine = float(F.cosine_similarity(image, ref['text']).item())
            dream = self.dreamsim.embed(self.dream_tensor(obj)) if self.dreamsim else None
            dream_distance = float((1 - F.cosine_similarity(dream, ref['dream'])).item()) if self.dreamsim else None
        return dict(hidden_core_mae=float(np.abs(candidate[hidden].astype(float) - ref['truth'][hidden]).mean() / 255),
            hidden_spatial_lpips=float(hmap[torch.from_numpy(hidden).cuda()].mean().item()),
            object_lpips=lpips, object_feature_similarity=feature, clip_text_cosine=clip_cosine,
            clip_text_logit=clip_cosine * float(self.models.clip.logit_scale.exp().item()),
            dreamsim=dream_distance,
            object_ssim=float(structural_similarity(ref['object_rgb'], obj, channel_axis=-1, data_range=255)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment-dir', type=Path, required=True)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--freeze', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--dreamsim-cache', type=Path)
    p.add_argument('--worker-index', type=int, default=0)
    p.add_argument('--worker-count', type=int, default=4)
    p.add_argument('--limit', type=int)
    p.add_argument('--development-review', action='store_true')
    p.add_argument('--features-dir', type=Path, help='Observation-only pretrained feature cache')
    p.add_argument('--extra-run-dir', type=Path, help='Additional seeds generated with the identical observations')
    a = p.parse_args()
    import torch
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    experiment, run, output = a.experiment_dir.resolve(), a.run_dir.resolve(), a.output_dir.resolve()
    freeze = json.loads(a.freeze.read_text())
    if freeze.get('status') != 'frozen_before_fresh_test' and not a.development_review:
        raise ValueError('Provisional policies are restricted to explicit development review')
    if not a.development_review:
        from amodal_benchmark.surface_policy import verify_frozen_surface_sources
        verify_frozen_surface_sources(freeze)
    config = RefinementConfig(**freeze['refinement'])
    seeds = freeze['candidate_seeds']
    cases = json.loads((experiment / 'cases.json').read_text())
    if not a.development_review:
        if set(c['image_identity'] for c in cases) & set(freeze['development']['image_identities']):
            raise ValueError('Fresh test overlaps development image identities')
        if set(c['source_sha256'] for c in cases) & set(freeze['development']['image_sha256']):
            raise ValueError('Fresh test overlaps development image bytes')
    if a.limit:
        cases = cases[:a.limit]
    cases = cases[a.worker_index::a.worker_count]
    output.mkdir(parents=True, exist_ok=True)
    runtime = dict(started_at=datetime.now(timezone.utc).isoformat(), job_id=os.environ.get('SLURM_JOB_ID'),
        worker=a.worker_index, gpu=torch.cuda.get_device_name(0), torch=torch.__version__,
        freeze_sha256=sha256(a.freeze), cases_sha256=sha256(experiment / 'cases.json'),
        source_sha256=sha256(Path(__file__)), refinement_sha256=sha256(ROOT / 'amodal_benchmark/surface_refinement.py'),
        dreamsim_manifest_sha256=sha256(a.dreamsim_cache / 'manifest.json') if a.dreamsim_cache else None,
        protocol=dict(hidden_mae='RGB L1 on fixed held-out core, predicted masks cannot restrict it',
            hidden_lpips='AlexNet spatial map on fixed hidden core; outside RGB=127',
            object_metrics='512x512 complete object on black, following PMC paper p.6; GT support versus predicted mask',
            clip='ViT-B/32 image versus label cosine and scaled logit',
            ssim='full fixed 512 frame, data_range=255',
            limitation='paired controlled study; not the original paper dataset or full progressive run'))
    write_json(output / f'runtime-worker-{a.worker_index}.json', runtime)
    models = FollowupMetrics(a.dreamsim_cache)
    for case in cases:
        destination = output / (case['case_id'] + '.json')
        if destination.exists():
            existing = json.loads(destination.read_text())
            if existing['freeze_sha256'] != runtime['freeze_sha256']:
                raise ValueError('Changed policy on resume')
            continue
        folder = run / 'candidates' / case['case_id']
        folders = [folder if (folder / f'seed-{seed}.json').exists() or not a.extra_run_dir else
                   a.extra_run_dir / 'candidates' / case['case_id'] for seed in seeds]
        records = [json.loads((d / f'seed-{seed}.json').read_text()) for seed, d in zip(seeds, folders)]
        scores = [r.get('surface', {}) for r in records]
        inputs = experiment / 'inputs' / case['case_id']
        for filename, expected in case['input_hashes'].items():
            if sha256(inputs / filename) != expected:
                raise ValueError('Observation identity mismatch')
        observed = np.array(Image.open(inputs / 'observed.png').convert('RGB'))
        visible = np.array(Image.open(inputs / 'visible.png').convert('L')) > 0
        occlusion = np.array(Image.open(inputs / 'occlusion.png').convert('L')) > 0
        candidates, masks = [], []
        for seed, record, candidate_folder in zip(seeds, records, folders):
            if record['seed'] != seed or record['input_hashes'] != case['input_hashes']:
                raise ValueError('Candidate seed or observation identity mismatch')
            for filename, expected in record['artifacts'].items():
                if sha256(candidate_folder / filename) != expected:
                    raise ValueError('Candidate checksum mismatch')
            rgb_path, mask_path = candidate_folder / f'seed-{seed}.png', candidate_folder / f'seed-{seed}-mask.png'
            candidates.append(np.array(Image.open(rgb_path).convert('RGB')) if rgb_path.exists() else np.zeros_like(observed))
            masks.append(np.array(Image.open(mask_path).convert('L')) > 0 if mask_path.exists() else np.zeros_like(visible))
        extra = pool_surface_features(observed, visible, occlusion, candidates, masks)
        scores = [dict(s, **f) for s, f in zip(scores, extra)]
        if freeze.get('pretrained_visible_features'):
            if not a.features_dir:
                raise ValueError('Frozen ranker requires observation-only deep features')
            features = json.loads((a.features_dir / (case['case_id'] + '.json')).read_text())
            contract = features['contract']
            if (contract['truth_access'] is not False or
                    contract['features_sha256'] != sha256(ROOT / 'amodal_benchmark/surface_features.py') or
                    contract['observation_manifest_sha256'] != sha256(experiment / 'generation_manifest.json')):
                raise ValueError('Pretrained feature provenance mismatch')
            if len(features['values']) != len(seeds) or contract.get('candidate_seeds', [11,22,33,44]) != seeds:
                raise ValueError('Unexpected candidate feature inventory')
            for seed, score, values, candidate_folder in zip(seeds, scores, features['values'], folders):
                if features['candidate_record_sha256'][f'seed-{seed}.json'] != sha256(candidate_folder / f'seed-{seed}.json'):
                    raise ValueError('Pretrained features belong to different candidates')
                score.update(values)
        selection = {'seed_11': 0,
            'local_surface': select_candidate(scores, METHODS['local_surface'], freeze['scales'])['index'],
            'visible_preservation': select_candidate(scores, {'visible_mae': 1})['index'],
            'consensus_only': select_candidate(scores, {'consensus_mae': 1})['index'],
            'trained_surface': predict_rank(scores, freeze['ranker'])[0]}
        valid_indices = [i for i,s in enumerate(scores) if s.get('valid')]
        if freeze.get('pretrained_visible_features'):
            selection['visible_deep_only'] = select_candidate(scores, {'visible_lpips': 1})['index']
            selection['semantic_only'] = select_candidate(
                [dict(s, negative_text=-s['candidate_text_cosine']) for s in scores], {'negative_text': 1})['index']
        calibrated = [refine_surface(observed, visible, candidate, mask, config)[0]
                      for candidate,mask in zip(candidates,masks)]
        corrected = calibrated
        if freeze.get('ensemble'):
            from amodal_benchmark.surface_ensemble import EnsembleConfig, refine_surface_pool
            corrected, _ = refine_surface_pool(calibrated,masks,visible,EnsembleConfig(**freeze['ensemble']))
        # Freeze observation-only choices to disk before opening evaluation RGB.
        write_json(output / 'decisions' / (case['case_id'] + '.json'), dict(case_id=case['case_id'],
            selection=selection, valid_indices=valid_indices, freeze_sha256=runtime['freeze_sha256'],
            observable_scores=scores,
            candidates={f'seed-{s}.json': sha256(d / f'seed-{s}.json') for s,d in zip(seeds,folders)}))
        evaluation = experiment / 'evaluation_only' / case['case_id']
        for filename, expected in case['evaluation_hashes'].items():
            if sha256(evaluation / filename) != expected:
                raise ValueError('Truth identity mismatch')
        truth = np.array(Image.open(evaluation / 'truth.png').convert('RGB'))
        hidden = np.array(Image.open(evaluation / 'hidden_core.png').convert('L')) > 0
        full = np.array(Image.open(evaluation / 'full.png').convert('L')) > 0
        ref = models.reference(truth, full, hidden, case['label'])
        candidate_metrics, refined_metrics, calibrated_metrics, artifacts = [], [], [], {}
        for index, (seed, record) in enumerate(zip(seeds, records)):
            if f'seed-{seed}.png' not in record['artifacts']:
                penalty = dict(hidden_core_mae=1., hidden_spatial_lpips=1., object_lpips=1.,
                    object_feature_similarity=0., clip_text_cosine=None, clip_text_logit=None,
                    dreamsim=1. if a.dreamsim_cache else None, object_ssim=0.)
                candidate_metrics.append(penalty)
                refined_metrics.append(penalty)
                calibrated_metrics.append(penalty)
                continue
            candidate, mask = candidates[index], masks[index]
            candidate_metrics.append(models.score(candidate, mask, ref))
            # Every raw candidate receives the same calibration; enables an
            # equal-budget calibrated-uniform control, separate from ranking.
            refined = corrected[index]
            refined_metrics.append(models.score(refined, mask, ref))
            if freeze.get('ensemble'):
                calibrated_metrics.append(models.score(calibrated[index],mask,ref))
            if index == selection['trained_surface']:
                artifact_dir = output / 'selected' / case['case_id']
                artifact_dir.mkdir(parents=True, exist_ok=True)
                Image.fromarray(refined).save(artifact_dir / 'surface.png')
                Image.fromarray(mask.astype(np.uint8)*255).save(artifact_dir / 'mask.png')
                artifacts = {p.name: sha256(p) for p in artifact_dir.iterdir()}
        write_json(destination, dict(case_id=case['case_id'], source=case['source'], label=case['label'],
            target_id=case['target_id'], freeze_sha256=runtime['freeze_sha256'],
            selection=selection, valid_indices=valid_indices, raw=candidate_metrics, refined=refined_metrics,
            calibrated=calibrated_metrics if freeze.get('ensemble') else refined_metrics,
            selected_artifacts=artifacts,
            candidate_seeds=seeds,
            generation_failures=sum(f'seed-{s}.png' not in r['artifacts'] for s, r in zip(seeds, records)),
            candidate_error_count=sum(r['status'] != 'generated' for r in records)))
        print('METRICS', case['case_id'], selection['trained_surface'], flush=True)
    runtime['finished_at'] = datetime.now(timezone.utc).isoformat()
    write_json(output / f'runtime-worker-{a.worker_index}.json', runtime)


if __name__ == '__main__':
    main()

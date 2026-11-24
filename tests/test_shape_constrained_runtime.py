"""Admission, unchanged candidate policy, and real evaluator integration on CPU."""
import copy
from contextlib import nullcontext, redirect_stderr
from io import StringIO
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from amodal_benchmark import shape_constrained_runtime as runtime
from amodal_benchmark.dataset import sha256, write_json
from evaluate_open_world import (parse_args, worker_command, validate_shape_constrained_scoring,
                                 validate_shape_v2_scoring)


def publish(folder, evidence):
    """Bind synthetic metadata; never create this fixture outside temporary tests."""
    receipt, best, native, result, replay = (evidence[name] for name in runtime.EVIDENCE_FILES)
    receipt['best_sha256'] = sha256(folder / 'best.pt')
    write_json(folder / 'best-validation-native.json', best)
    receipt['best_validation_sha256'] = sha256(folder / 'best-validation-native.json')
    write_json(folder / 'training.json', receipt)
    native.update(training_receipt_sha256=sha256(folder / 'training.json'), checkpoint_sha256=receipt['best_sha256'])
    write_json(folder / 'native-contract.json', native)
    result['contract_sha256'] = sha256(folder / 'native-contract.json')
    write_json(folder / 'native-result.json', result)
    replay.update(training_sha256=sha256(folder / 'training.json'), checkpoint_sha256=receipt['best_sha256'],
                  result_sha256=sha256(folder / 'native-result.json'))
    write_json(folder / 'native-replay.json', replay)
    write_json(folder / 'admission.json', dict(schema=1, method=runtime.METHOD,
        files={name: sha256(folder / name) for name in runtime.EVIDENCE_FILES},
        checkpoint_sha256=receipt['best_sha256'], selected_epoch=receipt['selected_epoch'],
        sources=runtime.admission_sources(), benchmark_success=False))


def receipt_fixture(folder):
    folder.mkdir(parents=True, exist_ok=True)
    checkpoint = folder / 'best.pt'
    checkpoint.write_bytes(b'SYNTHETIC TEST ONLY; not trained model weights')
    config = json.loads((runtime.ROOT / runtime.CONFIG).read_bytes())
    baseline, selected = dict(full_iou=.81, hidden_iou=.37), dict(full_iou=.82, hidden_iou=.39)
    contract = runtime.expected_training_contract(config)
    receipt = dict(schema=4, method=runtime.METHOD, architecture='resnet34_unet_5channel',
        platform='Slurm', complete=True, preflight=False, holdout_used=False, benchmark_used=False,
        epochs_completed=20, seed=config['seed'], selection=config['training']['selection'],
        geometry_policy=config['inference_geometry'], contract=contract,
        initial_validation=baseline, selected_epoch=2, new_checkpoint_eligible=True,
        history=[dict(epoch=epoch, validation=baseline if epoch == 1 else selected) for epoch in range(1,21)],
        runs=[dict(job_id='synthetic', world_size=4, expected_gpu='Quadro RTX 6000', gpu='Quadro RTX 6000',
            cuda_fft_round_trip=True, devices=[dict(rank=i, gpu='Quadro RTX 6000', cuda_fft_round_trip=True)
                                              for i in range(4)])])
    rows = [dict(id=f'test-{i}', image_id=i % 217) for i in range(1264)]
    best = dict(epoch=2, aggregate=selected, contract=contract, split='validation', benchmark_success=False, rows=rows)
    native = dict(schema=1, split='validation', objects=1264, images=217, selected_epoch=2,
        data_manifest_sha256=runtime.v2.TRAINING_DATA_SHA256,
        baseline_result_sha256=config['validation_diagnostic_result_sha256'],
        source_sha256=sha256(runtime.ROOT / runtime.NATIVE_SOURCE), geometry_policy=config['inference_geometry'],
        cuda_fft_round_trip=True, holdout_access=False, benchmark_access=False,
        job_id='synthetic', gpu='Quadro RTX 6000', torch='1.13.1+cu117', cuda='11.7')
    result = dict(complete=True, objects=1264, images=217, selected_epoch=2,
        max_selected_validation_difference=0., benchmark_success=False, rows=rows, geometry={'synthetic': True})
    replay = dict(verified=True, objects=1264, images=217, selected_epoch=2, new_checkpoint_eligible=True,
        replay_source_sha256=sha256(runtime.ROOT / runtime.REPLAY_SOURCE), geometry=result['geometry'], benchmark_success=False)
    evidence = dict(zip(runtime.EVIDENCE_FILES, (receipt, best, native, result, replay)))
    publish(folder, evidence)
    return checkpoint, evidence


class ConstrainedRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.base = ['--dataset-root', 'data', '--output-dir', 'out', '--repo-dir', 'repo',
                     '--pipeline-profile', 'released-sam']

    def test_cli_requires_isolated_shape_and_propagates_to_workers(self):
        flags = ['--shape-constrained-checkpoint', 'best.pt']
        args = parse_args(self.base + flags)
        command = worker_command(args)
        self.assertEqual(command[command.index(flags[0])+1], str(Path('best.pt').resolve()))
        for extra in (['--shape-v2-checkpoint', 'v2.pt'], ['--shape-checkpoint', 'legacy.pt'],
                      ['--surface-freeze', 'freeze.json'], ['--surface-dreamsim-cache', 'cache'],
                      ['--surface-development-review'], ['--reuse-results-from', 'old'],
                      ['--pipeline-profile', 'benchmark']):
            with self.subTest(extra=extra), redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                parse_args(self.base + flags + extra)

    def test_admission_rejects_incomplete_unverified_and_changed_native_evidence(self):
        with TemporaryDirectory() as temporary:
            folder = Path(temporary)
            checkpoint, evidence = receipt_fixture(folder)
            self.assertEqual(runtime.training_receipt(checkpoint)['selected_epoch'], 2)
            mutations = [('training.json', 'complete', False), ('training.json', 'preflight', True),
                ('training.json', 'holdout_used', True), ('training.json', 'benchmark_used', True),
                ('training.json', 'history', []), ('training.json', 'runs', []),
                ('training.json', 'selected_epoch', 3), ('training.json', 'new_checkpoint_eligible', False),
                ('native-result.json', 'max_selected_validation_difference', 1e-12),
                ('native-result.json', 'complete', False), ('native-result.json', 'rows', []),
                ('native-contract.json', 'source_sha256', 'changed'), ('native-contract.json', 'cuda_fft_round_trip', False),
                ('native-replay.json', 'verified', False), ('native-replay.json', 'replay_source_sha256', 'changed'),
                ('native-replay.json', 'selected_epoch', 3), ('native-replay.json', 'geometry', {})]
            for name, key, value in mutations:
                with self.subTest(name=name, key=key):
                    changed = copy.deepcopy(evidence)
                    changed[name][key] = value
                    publish(folder, changed)
                    with self.assertRaises(ValueError):
                        runtime.training_receipt(checkpoint)
            for key in runtime.expected_training_contract(json.loads((runtime.ROOT / runtime.CONFIG).read_bytes())):
                changed = copy.deepcopy(evidence)
                changed['training.json']['contract'][key] = 'changed'
                publish(folder, changed)
                with self.subTest(contract=key), self.assertRaisesRegex(ValueError, 'sources, data or configuration'):
                    runtime.training_receipt(checkpoint)
            publish(folder, evidence)
            checkpoint.write_bytes(b'changed weights')
            with self.assertRaisesRegex(ValueError, 'checkpoint bytes changed'):
                runtime.training_receipt(checkpoint)

    def test_epoch_zero_full_regression_and_stable_ties(self):
        with TemporaryDirectory() as temporary:
            checkpoint, evidence = receipt_fixture(Path(temporary))
            receipt = evidence['training.json']
            # Later tied epochs never displace the first eligible winner.
            self.assertEqual(runtime.selected_epoch(receipt)['epoch'], 2)
            for row in receipt['history']:
                row['validation'] = dict(full_iou=.80, hidden_iou=.99)
            self.assertEqual(runtime.selected_epoch(receipt)['epoch'], 0)
            publish(checkpoint.parent, evidence)
            with self.assertRaisesRegex(ValueError, 'Epoch zero'):
                runtime.training_receipt(checkpoint)
            receipt['history'][0]['validation']['hidden_iou'] = float('nan')
            with self.assertRaisesRegex(ValueError, 'finite and bounded'):
                runtime.selected_epoch(receipt)

    def test_scoring_without_weights_and_cross_method_substitution(self):
        with TemporaryDirectory() as temporary:
            checkpoint, evidence = receipt_fixture(Path(temporary))
            policy = runtime.shape_constrained_contract(checkpoint)
            args = parse_args(self.base + ['--score-only', '--shape-constrained-checkpoint', str(checkpoint)])
            validate_shape_constrained_scoring(args, dict(shape_policy=policy))
            with self.assertRaisesRegex(ValueError, 'differs from the saved'):
                validate_shape_constrained_scoring(args, dict(shape_policy={**policy, 'admission_sha256':'changed'}))
            with self.assertRaises(ValueError):
                validate_shape_v2_scoring(args, dict(shape_policy={'method':'shape-v2'}))
            checkpoint.unlink()
            args.shape_constrained_checkpoint = None
            with patch.object(runtime, 'training_receipt', side_effect=AssertionError('Must not load package')):
                validate_shape_constrained_scoring(args, dict(shape_policy=policy))
                args.shape_v2_checkpoint = Path('old.pt')
                with self.assertRaisesRegex(ValueError, 'cannot substitute'):
                    validate_shape_constrained_scoring(args, dict(shape_policy=policy))

    def test_prior_arithmetic_restores_every_flag_even_on_failure(self):
        cudnn = SimpleNamespace(deterministic=True, benchmark=True, allow_tf32=False)
        matmul = SimpleNamespace(allow_tf32=True)
        autocast_calls = []
        def autocast(device, enabled):
            autocast_calls.append((device, enabled))
            return nullcontext()
        torch = SimpleNamespace(backends=SimpleNamespace(cudnn=cudnn, cuda=SimpleNamespace(matmul=matmul)), autocast=autocast)
        with patch.dict(sys.modules, {'torch':torch}):
            with self.assertRaisesRegex(RuntimeError, 'synthetic'):
                with runtime.native_arithmetic():
                    self.assertEqual(vars(cudnn), dict(deterministic=False, benchmark=False, allow_tf32=True))
                    self.assertFalse(matmul.allow_tf32)
                    raise RuntimeError('synthetic')
        self.assertEqual(vars(cudnn), dict(deterministic=True, benchmark=True, allow_tf32=False))
        self.assertTrue(matmul.allow_tf32)
        self.assertEqual(autocast_calls, [('cuda', False)])

    def test_payload_epoch_validation_and_contract_checked_before_model_construction(self):
        with TemporaryDirectory() as temporary:
            checkpoint, evidence = receipt_fixture(Path(temporary))
            receipt = evidence['training.json']
            selected = runtime.selected_epoch(receipt)
            saved = dict(epoch=selected['epoch'], contract=receipt['contract'], validation=selected['validation'])
            torch = SimpleNamespace(__version__='1.13.1+cu117', load=lambda *args, **kwargs: saved)
            for key, value in (('epoch', 0), ('contract', {}), ('validation', {})):
                changed = {**saved, key:value}
                torch.load = lambda *args, **kwargs: changed
                with self.subTest(key=key), patch.dict(sys.modules, {'torch':torch}), \
                        patch.object(runtime, 'make_network', side_effect=AssertionError('Must not construct')):
                    with self.assertRaisesRegex(ValueError, 'payload differs'):
                        runtime.load_prior(checkpoint)

    def test_packaging_waits_for_completed_jobs_and_rejects_epoch_zero(self):
        from scripts.package_shape_constrained import package
        from scripts import collect_shape_constrained as collector
        with TemporaryDirectory() as temporary:
            study = Path(temporary)
            destination = study / 'admitted'
            status = dict(disposition='waiting', jobs={'synthetic':['PENDING','0:0']})
            write_json(study / 'collection-status.json', status)
            with patch.object(collector, 'STUDY', study), patch.object(collector, 'replay') as replay:
                with self.assertRaisesRegex(ValueError, 'completed recovery'):
                    package(study, destination)
                replay.assert_not_called()
                self.assertFalse(destination.exists())
                write_json(study / 'collection-status.json', dict(disposition='ready', jobs={'synthetic':['COMPLETED','0:0']}))
                replay.return_value = dict(verified=True, new_checkpoint_eligible=False)
                write_json(study / 'native-replay.json', replay.return_value)
                with self.assertRaisesRegex(ValueError, 'epoch zero'):
                    package(study, destination)
                self.assertFalse(destination.exists())

    def test_real_workers_resume_provenance_and_checkpoint_free_scoring(self):
        from test_parallel_evaluation import ParallelEvaluationTests
        from amodal_benchmark.verify import verify_run
        harness = ParallelEvaluationTests()
        harness.setUp()
        self.addCleanup(harness.doCleanups)
        checkpoint, evidence = receipt_fixture(harness.root / 'admission')
        shim = harness.root / 'runtime/sitecustomize.py'
        with shim.open('a') as stream:
            stream.write('''
from amodal_benchmark import shape_constrained_runtime
def constrained_install(upstream, models, sam_path, checkpoint):
    path = Path(os.environ["PDMC_TEST_TRACE"]) / f"constrained-{os.getpid()}.json"
    path.write_text(json.dumps(dict(checkpoint=str(checkpoint), gpu=os.environ.get("CUDA_VISIBLE_DEVICES"))))
shape_constrained_runtime.install_shape_constrained = constrained_install
if "--worker-index" in sys.argv and os.environ.get("PDMC_TEST_CONSTRAINED_DRIFT"):
    shape_constrained_runtime.SEED_SLOTS = (1, 2)
''')
        output = harness.root / 'constrained-output'
        flags = ['--pipeline-profile', 'released-sam', '--shape-constrained-checkpoint', str(checkpoint)]
        first = harness.evaluate(output, 2, limit=2, extra=flags, inference_only=False)
        self.assertEqual(first.returncode, 0, first.stdout + first.stderr)
        saved = (output / 'run_config.json').read_bytes()
        kept = {row['id']:(output / 'targets' / row['id'] / 'result.json').read_bytes() for row in harness.rows[:2]}
        resumed = harness.evaluate(output, 2, extra=flags, inference_only=False)
        self.assertEqual(resumed.returncode, 0, resumed.stdout + resumed.stderr)
        self.assertEqual((output / 'run_config.json').read_bytes(), saved)
        for identifier, data in kept.items():
            self.assertEqual((output / 'targets' / identifier / 'result.json').read_bytes(), data)
        installed = [json.loads(p.read_bytes()) for p in harness.trace.glob('constrained-*.json')]
        self.assertEqual({row['gpu'] for row in installed}, {'GPU-A','GPU-B'})
        self.assertTrue(all(row['checkpoint'] == str(checkpoint) for row in installed))
        self.assertEqual(verify_run(output, harness.dataset, require_provenance=True)['selected_targets'], 6)
        mismatch = harness.evaluate(harness.root / 'drift', 2, extra=flags,
                                    env={**harness.env, 'PDMC_TEST_CONSTRAINED_DRIFT':'1'})
        self.assertNotEqual(mismatch.returncode, 0)
        logs = list((harness.root / 'drift').glob('workers/*/worker-*.log'))
        self.assertTrue(any('differ from the coordinator' in p.read_text() for p in logs))
        checkpoint.unlink()
        for name in (*runtime.EVIDENCE_FILES, 'admission.json'):
            (checkpoint.parent / name).unlink()
        scored = harness.evaluate(output, 1, extra=['--pipeline-profile','released-sam','--score-only'], inference_only=False)
        self.assertEqual(scored.returncode, 0, scored.stdout + scored.stderr)
        self.assertEqual((output / 'run_config.json').read_bytes(), saved)
        self.assertEqual(verify_run(output, harness.dataset, require_provenance=True)['selected_targets'], 6)
        for source in output.glob('provenance/*/source'):
            for name in runtime.admission_sources():
                self.assertTrue((source / name).is_file(), name)

    def test_progressive_hooks_anchor_original_inputs_preserve_rgb_and_replay_decisions(self):
        with TemporaryDirectory() as temporary:
            seeds, anchors, outputs = [], [], []
            class Generator:
                def __init__(self, device):
                    self.device = device
                def manual_seed(self, seed):
                    self.seed = seed
                    return self
            torch = SimpleNamespace(initial_seed=lambda: 123, Generator=Generator,
                                    inference_mode=lambda: (lambda function: function))
            upstream = SimpleNamespace(run_sam=None, transform_image=lambda image: image)
            def crop(inputs):
                return [value[2:18, 2:22].copy() for value in inputs], 2, 18, 2, 22
            def pipeline(**kwargs):
                seeds.append(kwargs["generator"].seed)
                return SimpleNamespace(images=[Image.new("RGB", (24, 24), (64+(len(seeds)-1)%8, 80, 90))])
            def mixed(query_obj, sd_inpaint_model, sd_occ_mask, sd_prompt):
                return sd_inpaint_model(prompt=sd_prompt).images[0]
            def iteration(query_obj):
                images, *_ = upstream.crop([query_obj.img_canvas, query_obj.query_mask_canvas])
                result = upstream.run_mixed_context_diffusion(query_obj, pipeline, ~images[1], "chair")
                outputs.append(np.array(result))
                query_obj.img_canvas[:] = 255
                query_obj.query_mask_canvas[:] = True
                query_obj.iter_id += 1
                return query_obj
            def segment(*args):
                self.assertEqual(args[-1], ["chair"])
                image = args[3]
                # Second pass has no eligible masks and must fall back to slot 0.
                masks = np.ones((1, image.height, image.width), bool) if len(seeds) <= 8 else None
                return np.array(image), masks, ["chair"], [1.]
            def predict(model, observed, support):
                anchors.append((observed.copy(), support.copy()))
                return np.ones(support.shape, np.float32)
            original_check = lambda image: image
            upstream.crop, upstream.run_iteration, upstream.segment = crop, iteration, segment
            upstream.run_mixed_context_diffusion, upstream.check_occlusion = mixed, original_check
            visible = np.zeros((20, 24), bool)
            visible[5:15, 4:10] = True
            rgb = np.full((20, 24, 3), 100, np.uint8)
            query = SimpleNamespace(iter_id=0, output_img_dir=temporary, init_img_canvas=rgb,
                init_query_mask_canvas=visible, img_canvas=rgb.copy(), query_mask_canvas=visible.copy(),
                benchmark_diagnostics={})
            with patch.dict(sys.modules, {"torch": torch}), patch.object(runtime, "load_prior", return_value=object()), \
                    patch.object(runtime.v2, "predict_probability", side_effect=predict), \
                    patch.object(runtime, "native_arithmetic", side_effect=nullcontext):
                runtime.install_shape_constrained(upstream, [None], "sam", "synthetic.pt")
                upstream.segment = lambda *args: self.fail("Candidate used expanded scene detector")
                upstream.run_iteration(query)
                upstream.run_iteration(query)
                self.assertIs(upstream.check_occlusion, original_check)
                with self.assertRaises(ValueError):
                    runtime.install_shape_constrained(upstream, [None], "sam", "synthetic.pt")
                with self.assertRaisesRegex(ValueError, "no matching progressive crop"):
                    upstream.run_mixed_context_diffusion(query, pipeline, visible, "chair")
            self.assertEqual(seeds, [runtime.progressive_seed(123, iteration, slot)
                                    for iteration in (0, 1) for slot in runtime.SEED_SLOTS])
            np.testing.assert_array_equal(anchors[0][0], anchors[1][0])
            np.testing.assert_array_equal(anchors[0][1], anchors[1][1])
            self.assertEqual(len(query.benchmark_diagnostics["shape_constrained_iterations"]), 2)
            for number in (0, 1):
                folder = Path(temporary)/"shape-sam-constrained"/f"iteration-{number:02d}"
                decision = json.loads((folder/"decision.json").read_bytes())
                self.assertEqual(decision["fallback"], bool(number))
                self.assertEqual(decision["index"], 0)
                self.assertFalse(decision["inference_truth_access"])
                self.assertEqual((folder/"selected.png").read_bytes(), (folder/"candidate-0.png").read_bytes())
                for name, digest in decision["artifacts"].items():
                    self.assertEqual(sha256(folder/name), digest)
                prior = np.load(folder/"prior.npy", allow_pickle=False)
                masks = [np.array(Image.open(folder/f"mask-{i}.png")) > 0 for i in range(8)]
                self.assertEqual(runtime.score_masks(prior, masks), decision["scores"])
                np.testing.assert_array_equal(outputs[number], np.array(Image.open(folder/"selected.png")))


if __name__ == '__main__':
    unittest.main()

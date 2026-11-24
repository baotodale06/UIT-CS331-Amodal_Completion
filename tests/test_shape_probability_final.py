import copy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import zipfile

import numpy as np
from PIL import Image

from scripts.shape_probability_final_contract import (
    LOGS, POLICY_SHA256, CONFIG, ROOT, validate_parent, validate_checkpoint, variant_name,
    verify_sources,
)
from scripts.queue_shape_probability_final import decision
from scripts.verify_shape_probability_final import (
    binary, check_prediction, check_probability_scores, replay_masks, retained,
)


def png(values):
    buffer = io.BytesIO()
    Image.fromarray(values.astype(np.uint8)).save(buffer, format='PNG')
    return buffer.getvalue()


class FinalDiagnosticContractTests(unittest.TestCase):
    def setUp(self):
        self.policy = json.loads((ROOT / CONFIG).read_bytes())
        self.execution = dict(complete=True, commands=[dict(log=name, exit_code=0) for name in LOGS])
        self.training = dict(schema=8, method='shape-probability-workers4', platform='LocalWSL',
            complete=True, epochs_completed=20, holdout_used=False, benchmark_used=False,
            selected_epoch=0, new_checkpoint_eligible=False, best_sha256='best', contract={'seed': 7},
            history=[dict(epoch=epoch) for epoch in range(1, 21)])
        self.replay = dict(verified=True, benchmark_success=False,
            source_archive_sha256=self.policy['parent_source_archive_sha256'],
            selected_epoch=0, new_checkpoint_eligible=False, checkpoint_sha256='best')

    def test_fixed_policy_and_checkpoint_do_not_follow_selected_epoch(self):
        self.assertEqual(hashlib.sha256((ROOT / CONFIG).read_bytes()).hexdigest(), POLICY_SHA256)
        validate_parent(self.execution, self.training, self.replay, self.policy)
        checkpoint = dict(epoch=20, model={'weight': 1}, contract=self.training['contract'], history=self.training['history'])
        validate_checkpoint(checkpoint, self.training)
        for key, value in [('epoch', 0), ('contract', {}), ('history', []), ('model', {})]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_checkpoint(dict(checkpoint, **{key: value}), self.training)

    def test_rejects_partial_parent_stale_replay_and_policy_changes(self):
        cases = [('execution', 'complete', False), ('execution', 'commands', self.execution['commands'][:-1]),
                 ('training', 'epochs_completed', 19), ('training', 'holdout_used', True),
                 ('training', 'schema', 7), ('replay', 'checkpoint_sha256', 'stale'),
                 ('replay', 'verified', False), ('replay', 'selected_epoch', 2),
                 ('policy', 'fixed_epoch', 19), ('policy', 'logit_biases', [0, 1, 2])]
        for owner, key, value in cases:
            records = {name: copy.deepcopy(getattr(self, name)) for name in ('execution', 'training', 'replay', 'policy')}
            records[owner][key] = value
            with self.subTest(owner=owner, key=key), self.assertRaises(ValueError):
                validate_parent(**records)

    def test_queue_requires_real_process_exit_and_all_phases(self):
        self.assertEqual(decision(self.execution, {'disposition': 'verified'}, [502]), 'waiting_parent_processes')
        self.assertEqual(decision(self.execution, {'disposition': 'verified'}, []), 'ready')
        self.assertEqual(decision({'complete': False}, {}, []), 'needs_review')
        self.assertEqual(decision(self.execution, {'disposition': 'running_local'}, []), 'needs_review')
        self.execution['commands'][-1]['exit_code'] = 1
        self.assertEqual(decision(self.execution, {'disposition': 'verified'}, []), 'needs_review')

    def test_both_geometry_floors_are_required(self):
        summaries = {variant_name(0): dict(full_iou=.82, hidden_iou=.34),
                     variant_name(.5): dict(full_iou=.80, hidden_iou=.40),
                     variant_name(1): dict(full_iou=.82, hidden_iou=.38)}
        self.assertEqual(retained(summaries, dict(full_iou=.811, hidden_iou=.370), [0, .5, 1]), [1])

    def test_incomplete_source_manifest_fails_before_runtime(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            archive = folder / 'diagnostic-source.zip'
            with zipfile.ZipFile(archive, 'x') as bundle:
                bundle.writestr(CONFIG, (ROOT / CONFIG).read_bytes())
            manifest = folder / 'manifest.json'
            manifest.write_text(json.dumps(dict(files={CONFIG: POLICY_SHA256},
                zip_sha256=hashlib.sha256(archive.read_bytes()).hexdigest())), encoding='utf8')
            with self.assertRaisesRegex(ValueError, 'inventory incomplete'):
                verify_sources(manifest)

    def test_live_probe_failure_does_not_launch_diagnostic(self):
        from scripts import queue_shape_probability_final as queue
        with mock.patch.object(queue, 'validate_preparation'), mock.patch.object(queue, 'live_processes',
                side_effect=OSError('probe unavailable')), mock.patch.object(queue, 'STUDY') as study:
            study.__truediv__.return_value.exists.return_value = False
            with self.assertRaises(OSError):
                queue.run_chain()
            study.__truediv__.return_value.open.assert_not_called()


class FinalDiagnosticReplayTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data, self.sam = self.root / 'data', self.root / 'sam'
        self.data.mkdir()
        (self.sam / 'objects/a').mkdir(parents=True)
        self.mask = np.zeros((5, 5), dtype=bool)
        self.mask[2, 2] = True
        self.truth = self.mask.copy()
        self.truth[2, 3] = True
        self.record = dict(id='a', image_id='image', image='image.png', mask='truth.png')
        (self.data / 'image.png').write_bytes(png(np.zeros((5, 5, 3))))
        (self.data / 'truth.png').write_bytes(png(self.truth.astype(np.uint8) + 2 * self.mask))
        artifacts = {}
        for name in ('sam-visible.png', 'sam-prediction.png'):
            payload = png(self.mask * 255)
            (self.sam / 'objects/a' / name).write_bytes(payload)
            artifacts[name] = hashlib.sha256(payload).hexdigest()
        self.baseline = {'a': dict(self.record, artifacts=artifacts, sam_input=dict(full_iou=.5, hidden_iou=0.))}
        self.manifest = dict(artifacts={path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in self.data.iterdir()})
        values = dict(full_iou=.5, hidden_iou=0., brier=.04, nll=.1,
                      prediction_sha256=hashlib.sha256(self.mask.astype(np.uint8).tobytes()).hexdigest())
        self.biases = [0, .5, 1, 1.5, 2]
        self.keys = ['initial'] + [variant_name(bias) for bias in self.biases]
        summary = dict(objects=1, images=1, full_iou=.5, hidden_iou=0., full_iou_images=1, hidden_iou_images=1,
                       brier=.04, nll=.1, brier_images=1, nll_images=1)
        self.result = dict(rows=[dict(self.record, shape=[5, 5], variants={key: dict(values) for key in self.keys})],
                           summaries={key: dict(summary) for key in self.keys})
        self.raw = dict(rows=[dict(id='a', image_id='image', shape=[5, 5], box=[1, 1, 3, 3],
                                  **{key: values[key] for key in ('full_iou', 'hidden_iou', 'prediction_sha256')})],
                        aggregate={key: value for key, value in summary.items() if not key.startswith(('brier', 'nll'))})
        self.write_masks()
        with zipfile.ZipFile(self.root / 'final-zero-reference.zip', 'x') as bundle:
            bundle.writestr('a.png', png(self.mask * 255))

    def write_masks(self, corrupt=None, extra=False):
        with zipfile.ZipFile(self.root / 'masks.zip', 'w') as bundle:
            for key in self.keys:
                mask = self.mask.copy()
                if key == corrupt:
                    mask[2, 2] = False
                bundle.writestr(key + '/a.png', png(mask * 255))
            if extra:
                bundle.writestr('unexpected.png', png(self.mask * 255))

    def replay(self):
        return replay_masks(self.root, self.data, self.sam, [self.record], self.manifest,
                            self.baseline, self.result, self.raw, self.biases)

    def test_complete_native_replay(self):
        summaries, raw = self.replay()
        self.assertEqual(summaries['initial']['full_iou'], .5)
        self.assertEqual(raw['hidden_iou'], 0.)

    def test_corrupted_pixels_and_extra_archive_entry_fail(self):
        for corrupt, extra in [('final_bias_1', False), (None, True)]:
            self.write_masks(corrupt, extra)
            with self.subTest(corrupt=corrupt, extra=extra), self.assertRaises(ValueError):
                self.replay()

    def test_wrong_native_frame_and_nonbinary_pixels_fail(self):
        for payload in [png(np.zeros((4, 5))), png(np.ones((5, 5)))]:
            with self.assertRaises(ValueError):
                binary(payload, (5, 5))

    def test_false_summary_and_object_identity_fail(self):
        self.result['summaries']['initial']['full_iou'] = .99
        with self.assertRaises(ValueError):
            self.replay()
        self.result['summaries']['initial']['full_iou'] = .5
        self.result['rows'][0]['image_id'] = 'wrong'
        with self.assertRaises(ValueError):
            self.replay()

    def test_missing_support_rejected_even_with_matching_score_and_digest(self):
        empty = np.zeros((5, 5), dtype=bool)
        expected = dict(full_iou=0., hidden_iou=0., prediction_sha256=hashlib.sha256(empty.astype(np.uint8).tobytes()).hexdigest())
        with self.assertRaises(ValueError):
            check_prediction(empty, self.truth, self.mask, self.mask, expected)

    def test_probability_scores_reject_nan_and_invalid_support_coverage(self):
        for values in [dict(brier=float('nan'), nll=.1), dict(brier=1.01, nll=.1), dict(brier=.1, nll=100.)]:
            with self.assertRaises(ValueError):
                check_probability_scores(values, self.mask)
        with self.assertRaises(ValueError):
            check_probability_scores(dict(brier=0., nll=0.), np.ones((2, 2), dtype=bool))
        check_probability_scores(dict(brier=None, nll=None), np.ones((2, 2), dtype=bool))

    def test_nonmonotone_offsets_fail_even_when_row_geometry_is_consistent(self):
        larger = self.mask.copy()
        larger[1, 1] = True
        key = 'final_bias_0.5'
        values = self.result['rows'][0]['variants'][key]
        values.update(full_iou=1 / 3, hidden_iou=0.,
                      prediction_sha256=hashlib.sha256(larger.astype(np.uint8).tobytes()).hexdigest())
        with zipfile.ZipFile(self.root / 'masks.zip', 'w') as bundle:
            for name in self.keys:
                bundle.writestr(name + '/a.png', png((larger if name == key else self.mask) * 255))
        with self.assertRaisesRegex(ValueError, 'removed pixels'):
            self.replay()


if __name__ == '__main__':
    unittest.main()

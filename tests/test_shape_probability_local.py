"""A local fallback must not overlap live work or mistake failure for completion."""
import copy
import json
from pathlib import Path
import unittest
from tempfile import TemporaryDirectory
from unittest.mock import patch
import subprocess
from scripts import queue_shape_probability_local as queue
from scripts.queue_shape_probability_local import decision


class LocalProbabilityQueueTests(unittest.TestCase):
    def setUp(self):
        self.execution = dict(complete=True, commands=[dict(command=[label], exit_code=0)
            for label in ('test1', 'test2', 'test3', 'calibrate', 'preflight', 'train', 'native')])
        self.observed = dict(disposition='verified')

    def test_waits_for_live_process_even_if_receipt_says_complete(self):
        self.assertEqual(decision(self.execution, [dict(pid=23)], self.observed, 'QUEUED'), 'waiting_predecessor')

    def test_missing_handle_requires_terminal_verified_evidence(self):
        for change in ('incomplete', 'failed', 'missing_command', 'wrong_mode'):
            execution = copy.deepcopy(self.execution)
            if change == 'incomplete':
                execution['complete'] = False
            elif change == 'failed':
                execution['commands'][-1]['exit_code'] = 1
            elif change == 'missing_command':
                execution['commands'].pop()
            else:
                execution['commands'][-1]['command'] = ['train']
            with self.subTest(change=change):
                self.assertEqual(decision(execution, [], self.observed, 'QUEUED'), 'needs_review')
        self.assertEqual(decision(self.execution, [], {}, 'QUEUED'), 'waiting_predecessor_replay')

    def test_only_queued_kaggle_allows_local_launch(self):
        self.assertEqual(decision(self.execution, [], self.observed), 'check_kaggle')
        self.assertEqual(decision(self.execution, [], self.observed, 'QUEUED'), 'start_local')
        for state in ('RUNNING', 'COMPLETE'):
            self.assertEqual(decision(self.execution, [], self.observed, state), 'external_training_available')
        for state in ('ERROR', 'CANCELLED', 'UNKNOWN'):
            self.assertEqual(decision(self.execution, [], self.observed, state), 'needs_review')

    def test_observation_timeout_never_launches_training(self):
        with TemporaryDirectory() as temporary, patch.object(queue, 'STUDY', Path(temporary)), \
                patch.object(queue, 'poll', side_effect=subprocess.TimeoutExpired('wsl', 45)), \
                patch.object(queue, 'run_chain') as launch, patch('sys.argv', ['queue']):
            queue.main()
            launch.assert_not_called()
            self.assertFalse((Path(temporary) / 'launch-request.json').exists())
            status = json.loads((Path(temporary) / 'queue-last-error.json').read_bytes())
            self.assertEqual(status['disposition'], 'observation_error')

    def test_native_replay_keeps_both_floors_and_first_exact_tie(self):
        from scripts.verify_shape_probability_local import select_epoch
        training = dict(initial_validation=dict(full_iou=.81, hidden_iou=.37),
            original_quadro_validation=dict(full_iou=.82, hidden_iou=.38),
            selection_floor=dict(full_iou=.82, hidden_iou=.38), history=[
                dict(epoch=1, validation=dict(full_iou=.81, hidden_iou=.45)),
                dict(epoch=2, validation=dict(full_iou=.82, hidden_iou=.39)),
                dict(epoch=3, validation=dict(full_iou=.82, hidden_iou=.39))])
        self.assertEqual(select_epoch(training)['epoch'], 2)
        training['history'] = training['history'][:1]
        self.assertEqual(select_epoch(training)['epoch'], 0)
        training['selection_floor']['full_iou'] = .81
        with self.assertRaises(ValueError):
            select_epoch(training)

    def test_local_loss_contract_rejects_other_protocols_and_changed_batch(self):
        try:
            from scripts.train_shape_probability_local import validate_contract
        except ImportError:
            self.skipTest('Torch is optional in the CPU environment')
        root = Path(__file__).resolve().parents[1]
        config = json.loads((root / 'configs/research/shape-probability-local-20261123.json').read_bytes())
        self.assertEqual(validate_contract(config)['probability_loss_version'], 1)
        old = json.loads((root / 'configs/research/shape-local-constrained-20261123.json').read_bytes())
        with self.assertRaises(ValueError):
            validate_contract(old)
        for key, value in (('probability_loss_version', 0), ('world_size', 2), ('micro_batch_size', 16), ('epochs', 19)):
            changed = copy.deepcopy(config)
            changed['training'][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_contract(changed)


if __name__ == '__main__':
    unittest.main()

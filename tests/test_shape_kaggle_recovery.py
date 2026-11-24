"""Fail-closed recovery checks for versioned private Kaggle artifacts."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from scripts.collect_shape_kaggle import parse_status, recover, verify_sources


class KaggleRecoveryTests(unittest.TestCase):
    def test_terminal_state_requires_an_authoritative_known_status(self):
        self.assertEqual(parse_status('run has status "KernelWorkerStatus.COMPLETE"'), 'COMPLETE')
        for text in ('Permission denied', '', 'KernelWorkerStatus.UNKNOWN',
                     'KernelWorkerStatus.RUNNING KernelWorkerStatus.COMPLETE'):
            with self.assertRaises(ValueError):
                parse_status(text)

    def test_changed_input_bundle_is_rejected_before_replay(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root/'dataset').mkdir()
            (root/'dataset/bundle.json').write_text('{}')
            evidence = root/'recovered'
            evidence.mkdir()
            (evidence/'bundle.json').write_text('{"changed":true}')
            with patch('scripts.collect_shape_kaggle.STUDY',root):
                with self.assertRaises(ValueError):
                    verify_sources(evidence)

    def test_failed_run_is_preserved_without_attempting_training_replay(self):
        with TemporaryDirectory() as temporary:
            evidence = Path(temporary)/'failed'
            evidence.mkdir()
            (evidence/'training-preflight.log').write_text('Initial checkpoint does not replay')
            with patch('scripts.collect_shape_kaggle.subprocess.run'), patch('scripts.collect_shape_kaggle.replay') as replay:
                result = recover('owner/shape/2',evidence,'logical:v2','ERROR')
            replay.assert_not_called()
            self.assertFalse(result['verified'])
            self.assertEqual(result['disposition'],'needs_review')
            inventory = json.loads((Path(temporary)/'failed-inventory.json').read_bytes())
            self.assertIn('training-preflight.log',inventory['files'])


if __name__ == '__main__':
    unittest.main()

"""Execution identity and native evidence recovery for interactive Kaggle runs."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from scripts.collect_shape_kaggle_interactive import classify, extract_verified, digest, SESSION_ID, SOURCE_HASH
from scripts.export_shape_kaggle_interactive import terminal


class InteractiveCollectionTests(unittest.TestCase):
    def observation(self):
        return dict(session_id=SESSION_ID, execution=dict(session_id=SESSION_ID, source_sha256=SOURCE_HASH,
            wrapper_pid=118, child_pid=119, child_start_ticks='47854', complete=False,
            execution_mode='interactive', benchmark_success=False), processes=dict(
            wrapper_pid=dict(pid=118, state='S', start_ticks='47845'),
            child_pid=dict(pid=119, state='S', start_ticks='47854')))

    def test_live_child_requires_matching_start_ticks(self):
        observed = self.observation()
        self.assertEqual(classify(observed, dict(wrapper_pid=118)), 'running')
        observed['processes']['child_pid']['start_ticks'] = 'new-process'
        self.assertEqual(classify(observed, dict(wrapper_pid=118)), 'needs_review')

    def test_missing_child_does_not_imply_success(self):
        observed = self.observation()
        observed['processes'] = {}
        self.assertEqual(classify(observed, dict(wrapper_pid=118)), 'needs_review')

    def test_terminal_waits_for_process_exit(self):
        observed = self.observation()
        observed['execution'].update(complete=True, exit_code=0)
        self.assertEqual(classify(observed, dict(wrapper_pid=118)), 'finishing')
        observed['processes'] = dict(wrapper_pid=dict(pid=118, state='Z', start_ticks='47845'))
        self.assertEqual(classify(observed, dict(wrapper_pid=118)), 'terminal')

    def test_failed_execution_is_terminal_and_preserved(self):
        observed = self.observation()
        observed['execution'].update(complete=True, exit_code=1)
        observed['processes'] = {}
        self.assertEqual(classify(observed, dict(wrapper_pid=118)), 'terminal')

    def test_wrong_session_or_source_is_rejected(self):
        for field, value in [('session_id', 1), ('source_sha256', 'changed')]:
            observed = self.observation()
            observed['execution'][field] = value
            with self.assertRaises(ValueError):
                classify(observed, dict(wrapper_pid=118))

    def test_remote_export_rejects_live_child(self):
        receipt = self.observation()['execution'] | dict(complete=True, exit_code=0)
        with patch('scripts.export_shape_kaggle_interactive.process', return_value=dict(state='S',start_ticks='47854')):
            with self.assertRaises(ValueError):
                terminal(receipt)
        with patch('scripts.export_shape_kaggle_interactive.process', return_value=None):
            terminal(receipt)

    def export(self, root, name='training/result.json'):
        archive = root / 'outputs.zip'
        payload = json.dumps(dict(epoch=20)).encode()
        with zipfile.ZipFile(archive, 'w') as bundle:
            bundle.writestr(name, payload)
        receipt = dict(bytes=archive.stat().st_size, archive_sha256=digest(archive),
                       files={name: dict(bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())})
        return archive, receipt

    def test_verified_export_can_resume_without_overwriting(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive, receipt = self.export(root)
            destination = root / 'evidence'
            extract_verified(archive, receipt, destination)
            extract_verified(archive, receipt, destination)
            path = destination / 'training/result.json'
            path.write_text('modified')
            with self.assertRaisesRegex(ValueError, 'Existing recovered'):
                extract_verified(archive, receipt, destination)
            self.assertEqual(path.read_text(), 'modified')

    def test_corrupt_archive_and_file_hash_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive, receipt = self.export(root)
            bad = receipt | dict(archive_sha256='wrong')
            with self.assertRaisesRegex(ValueError, 'archive hash'):
                extract_verified(archive, bad, root/'evidence')
            receipt['files']['training/result.json']['sha256'] = 'wrong'
            with self.assertRaisesRegex(ValueError, 'file hash'):
                extract_verified(archive, receipt, root/'evidence')
            self.assertFalse((root/'evidence/training/result.json').exists())

    def test_path_escape_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive, receipt = self.export(root, '../escaped.json')
            with self.assertRaisesRegex(ValueError, 'Unsafe'):
                extract_verified(archive, receipt, root/'evidence')
            self.assertFalse((root/'escaped.json').exists())

    def test_unlisted_archive_member_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive, receipt = self.export(root)
            receipt['files'] = {}
            with self.assertRaisesRegex(ValueError, 'inventory'):
                extract_verified(archive, receipt, root/'evidence')


if __name__ == '__main__':
    unittest.main()

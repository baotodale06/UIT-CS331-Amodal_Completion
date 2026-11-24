"""Parallel evidence transfer must preserve byte identity and reject stale parts."""
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import recover_annotated_surface as recovery


class AnnotatedRecoveryTests(unittest.TestCase):
    def test_parallel_ranges_preserve_short_tail_and_resume_without_transfer(self):
        payload = bytes(range(256)) * 17001 + b'final short block'
        digest = hashlib.sha256(payload).hexdigest()

        def transfer(command, stdout, check):
            values = dict(part.split('=', 1) for part in shlex.split(command[-1])[1:])
            block, start, count = (int(values[key]) for key in ('bs', 'skip', 'count'))
            stdout.write(payload[start * block:(start + count) * block])
            return subprocess.CompletedProcess(command, 0)

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'evidence.partial'
            with patch.object(recovery.subprocess, 'run', side_effect=transfer) as calls:
                recovery.download_parallel('/immutable evidence/archive.gz', path, len(payload), digest)
                self.assertEqual(path.read_bytes(), payload)
                initial_calls = calls.call_count
                self.assertGreater(initial_calls, 1)
                recovery.download_parallel('/immutable evidence/archive.gz', path, len(payload), digest)
                self.assertEqual(calls.call_count, initial_calls)
            receipt = json.loads((path.parent / (path.name + '.chunks') / 'verification.json').read_text())
            self.assertTrue(receipt['verified'])
            self.assertEqual(sum(row['bytes'] for row in receipt['parts']), len(payload))

    def test_corrupt_bytes_never_receive_a_verification_receipt(self):
        payload = b'correct bytes'
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'evidence.partial'
            def corrupt(command, stdout, check):
                stdout.write(b'x' * len(payload))
            with patch.object(recovery.subprocess, 'run', side_effect=corrupt):
                with self.assertRaisesRegex(ValueError, 'independent remote hash'):
                    recovery.download_parallel('/archive', path, len(payload), hashlib.sha256(payload).hexdigest())
            self.assertFalse((path.parent / (path.name + '.chunks') / 'verification.json').exists())

    def test_parts_from_different_contract_are_rejected_before_transfer(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'evidence.partial'
            parts = path.parent / (path.name + '.chunks')
            parts.mkdir()
            (parts / 'contract.json').write_text('{}')
            with patch.object(recovery.subprocess, 'run') as calls:
                with self.assertRaisesRegex(ValueError, 'different archive'):
                    recovery.download_parallel('/archive', path, 1, 'wrong')
                calls.assert_not_called()


if __name__ == '__main__':
    unittest.main()

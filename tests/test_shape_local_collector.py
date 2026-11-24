"""Check observer states without altering or restarting a running trainer."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from scripts import collect_shape_local as collector


class LocalCollectorTests(unittest.TestCase):
    def test_single_observation_creates_lock_and_preserves_running_command(self):
        with TemporaryDirectory() as directory:
            study = Path(directory)
            execution = dict(complete=False,commands=[dict(command=['python','train'])])
            (study/'execution-v2.json').write_text(json.dumps(execution))
            with patch.object(collector,'STUDY',study),patch('sys.argv',['collect_shape_local.py']):
                collector.main()
            result = json.loads((study/'collection-status.json').read_bytes())
            self.assertEqual(result['disposition'],'waiting')
            self.assertEqual(result['last_command'],['python','train'])
            self.assertTrue((study/'collector-lock/.pdmc.lock').exists())
            self.assertEqual(json.loads((study/'execution-v2.json').read_bytes()),execution)

    def test_failed_command_is_terminal_without_retry(self):
        with TemporaryDirectory() as directory:
            study = Path(directory)
            execution = dict(complete=False,commands=[dict(command=['python','train'],exit_code=1)])
            (study/'execution-v2.json').write_text(json.dumps(execution))
            with patch.object(collector,'STUDY',study):
                self.assertEqual(collector.collect(),'failed')
            self.assertEqual(json.loads((study/'execution-v2.json').read_bytes()),execution)


if __name__=='__main__':
    unittest.main()

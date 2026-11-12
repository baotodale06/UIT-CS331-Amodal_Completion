"""Protect annotated Surface-PMC and outcome-independent cohort selection."""
import json
from pathlib import Path
import tempfile
import unittest

from amodal_benchmark.cohort import select_manifest_rows
from evaluate_open_world import parse_args, worker_command


class AnnotatedSurfaceTests(unittest.TestCase):
    def test_annotated_surface_is_forwarded_with_the_frozen_cohort(self):
        args = parse_args(['--dataset-root', 'data', '--output-dir', 'out', '--repo-dir', 'repo',
            '--pipeline-profile', 'released-sam', '--target-selection', 'label-semantic',
            '--detector-vocabulary', 'annotated', '--surface-freeze', 'freeze.json',
            '--surface-dreamsim-cache', 'cache', '--target-manifest', 'cohort.json'])
        command = worker_command(args)
        self.assertEqual(command[command.index('--detector-vocabulary') + 1], 'annotated')
        self.assertEqual(command[command.index('--target-manifest') + 1], str(Path('cohort.json').resolve()))
        self.assertIn('--surface-freeze', command)

    def test_cohort_rejects_unknown_duplicates_empty_and_reordered_targets(self):
        rows = [dict(id='VG/a'), dict(id='VG/b'), dict(id='COCOA/c')]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'cohort.json'
            for invalid in ([], ['VG/a', 'VG/a'], ['VG/unknown'], ['COCOA/c', 'VG/a']):
                path.write_text(json.dumps(dict(target_ids=invalid)))
                with self.assertRaises(ValueError):
                    select_manifest_rows(rows, path)
            path.write_text(json.dumps(dict(target_ids=['VG/a', 'COCOA/c'])))
            self.assertEqual(select_manifest_rows(rows, path), [rows[0], rows[2]])

    def test_limit_cannot_silently_shrink_a_frozen_cohort(self):
        with self.assertRaises(SystemExit):
            parse_args(['--dataset-root', 'data', '--output-dir', 'out',
                        '--target-manifest', 'cohort.json', '--limit', '8'])


if __name__ == '__main__':
    unittest.main()

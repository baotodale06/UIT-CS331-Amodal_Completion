"""Keep paired CLIP, changing coverage and failure penalties distinct."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import compare_surface_progressive as comparison
from amodal_benchmark.dataset import sha256, write_json
from amodal_benchmark.protocol import aggregate, failure_metrics


class SurfaceProgressiveComparisonTests(unittest.TestCase):
    def fixture(self, root, shared_success=True):
        rows = [dict(id='VG/' + name, source='VG', original_id=image)
                for name, image in (('a', 'one'), ('b', 'one'), ('c', 'two'))]
        metrics = [
            [dict(CLIP=10., LPIPS=.2, **{'Feature Similarity': .7, 'SSIM': .6}),
             dict(CLIP=20., LPIPS=.4, **{'Feature Similarity': .5, 'SSIM': .4}), None],
            [dict(CLIP=12., LPIPS=.1, **{'Feature Similarity': .8, 'SSIM': .7}), None,
             dict(CLIP=30., LPIPS=.3, **{'Feature Similarity': .6, 'SSIM': .5})],
        ]
        if not shared_success:
            metrics[1][0] = None
        reports = []
        for run_index, name in enumerate(('baseline', 'surface')):
            run = root / name
            config = dict(seed=0, checkpoints={'generator': 'immutable-generator'})
            if name == 'surface':
                config['surface_policy'] = dict(freeze_sha256='frozen',
                    frozen_method=dict(candidate_seeds=[11, 22, 33, 44, 55, 66, 77, 88]))
            write_json(run / 'run_config.json', config)
            write_json(run / 'metric_config.json', {'alignment': 'released'})
            write_json(run / 'execution.json', {'id': 'test-execution'})
            write_json(run / 'provenance/test-execution/environment.json', {'packages': []})
            aggregated_rows = []
            for row, values in zip(rows, metrics[run_index]):
                status = 'success' if values is not None else 'no_completion'
                record = dict(row, status=status, input_sha256=row['original_id'],
                    evaluation_reference_artifacts={'mask': row['id']}, seed=0)
                write_json(run / 'targets' / row['id'] / 'result.json', record)
                if values is not None:
                    write_json(run / 'targets' / row['id'] / 'metrics.json', {'metrics': values})
                aggregated_rows.append(dict(record, **(values or failure_metrics())))
            reports.append(dict(verified=True, full_evaluation=True,
                data_identity=dict(targets_sha256='targets', inputs_sha256='inputs', references_sha256='refs'),
                statistics=aggregate(aggregated_rows, detailed=True)))
        return rows, reports

    def run_comparison(self, root, rows, reports):
        argv = ['compare_surface_progressive.py', '--baseline', str(root / 'baseline'),
                '--surface', str(root / 'surface'), '--dataset-root', str(root / 'data'),
                '--output-dir', str(root / 'comparison'), '--registry', str(root / 'registry.csv')]
        with patch.object(sys, 'argv', argv), \
                patch.object(comparison, 'verify_run', side_effect=reports) as verify, \
                patch.object(comparison, 'read_official', return_value=(rows, {})), \
                patch.object(comparison, 'register_run') as register, redirect_stdout(io.StringIO()):
            comparison.main()
        for call in verify.call_args_list:
            self.assertEqual(call.kwargs, dict(require_full=True, require_provenance=True))
        self.assertEqual(register.call_count, 2)
        return json.loads((root / 'comparison/comparison.json').read_text())

    def test_paired_clip_excludes_changed_successes_but_overall_reports_them(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows, reports = self.fixture(root)
            result = self.run_comparison(root, rows, reports)
            self.assertEqual((result['shared_successes'], result['lost_successes'], result['gained_successes']),
                             (1, 1, 1))
            clip = result['metrics']['CLIP']
            self.assertEqual((clip['n'], clip['excluded_pairs']), (1, 2))
            self.assertAlmostEqual(clip['relative_gain_percent'], 20.)
            self.assertEqual(result['overall_metrics']['CLIP'],
                             dict(baseline=15., surface=21., baseline_n=2, surface_n=2))
            lpips = result['metrics']['LPIPS']
            self.assertEqual((lpips['n'], lpips['images']), (3, 2))
            self.assertAlmostEqual(lpips['baseline'], 1.6 / 3)
            self.assertAlmostEqual(lpips['method'], 1.4 / 3)
            self.assertAlmostEqual(lpips['relative_gain_percent'], 12.5)
            text = (root / 'comparison/REPORT.md').read_text()
            self.assertIn('| CLIP | 15.000000 | 2 | 21.000000 | 2 |', text)
            self.assertIn('| VG | 3 | 2 | 1 | 2 | 1 |', text)
            self.assertIn('descriptive, not a paired treatment effect', text)
            for name, expected in zip(('baseline', 'surface'), reports):
                self.assertEqual(json.loads((root / 'comparison' / (name + '-verification.json')).read_text()),
                                 expected)
            for name, expected in result['analysis_source_hashes'].items():
                self.assertEqual(sha256(root / 'comparison/analysis_source' / name), expected)

    def test_reference_mismatch_stops_before_reporting_or_registration(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows, reports = self.fixture(root)
            reports[1]['data_identity']['references_sha256'] = 'different-references'
            with self.assertRaisesRegex(ValueError, 'references_sha256'):
                self.run_comparison(root, rows, reports)
            self.assertFalse((root / 'comparison').exists())
            self.assertFalse((root / 'registry.csv').exists())

    def test_no_shared_successes_produces_unavailable_paired_clip(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows, reports = self.fixture(root, shared_success=False)
            result = self.run_comparison(root, rows, reports)
            self.assertEqual(result['shared_successes'], 0)
            self.assertEqual(result['metrics']['CLIP']['n'], 0)
            self.assertIsNone(result['metrics']['CLIP']['relative_gain_percent'])
            self.assertEqual(result['overall_metrics']['CLIP'],
                             dict(baseline=15., surface=30., baseline_n=2, surface_n=1))
            self.assertIn('| CLIP | unavailable | unavailable | unavailable | unavailable | 0 |',
                          (root / 'comparison/REPORT.md').read_text())

    def test_generator_mismatch_stops_before_reporting(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows, reports = self.fixture(root)
            config_path = root / 'surface/run_config.json'
            config = json.loads(config_path.read_text())
            config['checkpoints']['generator'] = 'different-generator'
            write_json(config_path, config)
            with self.assertRaisesRegex(ValueError, 'configurations differ'):
                self.run_comparison(root, rows, reports)
            self.assertFalse((root / 'comparison').exists())

    def test_cohort_requires_exact_inventory_and_cannot_register_as_full(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows, reports = self.fixture(root)
            manifest = root / 'cohort.json'
            write_json(manifest, dict(target_ids=[row['id'] for row in rows]))
            for name in ('baseline', 'surface'):
                write_json(root / name / 'execution.json', dict(id='test-execution',
                    target_ids=[row['id'] for row in rows]))
            argv = ['compare_surface_progressive.py', '--baseline', str(root / 'baseline'),
                    '--surface', str(root / 'surface'), '--dataset-root', str(root / 'data'),
                    '--output-dir', str(root / 'comparison'), '--target-manifest', str(manifest)]
            with patch.object(sys, 'argv', argv), \
                    patch.object(comparison, 'verify_run', side_effect=reports) as verify, \
                    patch.object(comparison, 'read_official', return_value=(rows, {})), \
                    patch.object(comparison, 'register_run') as register, redirect_stdout(io.StringIO()):
                comparison.main()
            self.assertFalse(register.called)
            self.assertTrue(all(call.kwargs == dict(require_full=False, require_provenance=True)
                                for call in verify.call_args_list))
            result = json.loads((root / 'comparison/comparison.json').read_text())
            self.assertEqual(result['evaluation_scope'], 'frozen research cohort')
            self.assertEqual(result['targets'], 3)
            self.assertEqual(result['research_cohort']['manifest_sha256'], sha256(manifest))
            with patch.object(sys, 'argv', argv + ['--registry', str(root / 'registry.csv')]):
                with self.assertRaises(SystemExit):
                    comparison.main()
            write_json(root / 'surface/execution.json', dict(id='test-execution', target_ids=['VG/a']))
            with patch.object(sys, 'argv', argv), patch.object(comparison, 'verify_run', side_effect=reports), \
                    patch.object(comparison, 'read_official', return_value=(rows, {})):
                with self.assertRaisesRegex(ValueError, 'inventory differs'):
                    comparison.main()


if __name__ == '__main__':
    unittest.main()

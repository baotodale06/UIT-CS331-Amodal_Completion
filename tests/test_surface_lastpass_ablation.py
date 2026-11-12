import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from measure_surface_lastpass_ablation import derive_raw_lastpass


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class SurfaceLastpassAblationTests(unittest.TestCase):
    def make_case(self, root):
        source = root/'source'
        pool = source/'surface/iteration-02'
        pool.mkdir(parents=True)
        image = np.full((14, 20, 4), [30, 60, 90, 255], dtype=np.uint8)
        image[:5, :8, 3] = 0
        Image.fromarray(image).save(source/'completion.png')
        Image.fromarray(image[..., 3]).save(source/'mask.png')
        Image.new('RGB', (12, 12), (120, 80, 40)).save(pool/'candidate-3.png')
        decision = dict(index=3, crop_rc=[4,18,7,27], artifacts={
            'candidate-3.png':digest(pool/'candidate-3.png')})
        (pool/'decision.json').write_text(json.dumps(decision))
        record = dict(status='success', completion_sha256=digest(source/'completion.png'),
            diagnostics=dict(surface_iterations=[dict(iteration=2, selected=3,
                decision_sha256=digest(pool/'decision.json'))]))
        return source, pool, record

    def test_raw_rgb_changes_but_final_alpha_size_and_selected_identity_are_fixed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, pool, record = self.make_case(root)
            path, details = derive_raw_lastpass(source, root/'derived', record)
            original = np.array(Image.open(source/'completion.png'))
            result = np.array(Image.open(path))
            self.assertEqual(result.shape, original.shape)
            np.testing.assert_array_equal(result[..., 3], original[..., 3])
            self.assertGreater(np.abs(result[..., :3].astype(float)-original[..., :3]).mean(), 20)
            self.assertEqual(details['selected_index'], 3)
            self.assertEqual(details['raw_candidate_sha256'], digest(pool/'candidate-3.png'))
            self.assertEqual(digest(source/'completion.png'), record['completion_sha256'])

    def test_corrupt_candidate_is_rejected_before_derivation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, pool, record = self.make_case(root)
            Image.new('RGB', (12,12), 'red').save(pool/'candidate-3.png')
            with self.assertRaisesRegex(ValueError, 'candidate checksum'):
                derive_raw_lastpass(source, root/'derived', record)
            self.assertFalse((root/'derived').exists())

    def test_changed_decision_link_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, pool, record = self.make_case(root)
            record['diagnostics']['surface_iterations'][-1]['decision_sha256'] = '0'*64
            with self.assertRaisesRegex(ValueError, 'Final decision'):
                derive_raw_lastpass(source, root/'derived', record)

    def test_unaligned_crop_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, pool, record = self.make_case(root)
            decision=json.loads((pool/'decision.json').read_text())
            decision['crop_rc'][1]+=1
            (pool/'decision.json').write_text(json.dumps(decision))
            record['diagnostics']['surface_iterations'][-1]['decision_sha256']=digest(pool/'decision.json')
            with self.assertRaisesRegex(ValueError, 'crop differs'):
                derive_raw_lastpass(source, root/'derived', record)


if __name__ == '__main__':
    unittest.main()

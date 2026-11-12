import json
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from prepare_surface_confirmation import confirmation_freeze
from finalize_surface_confirmation import check_cohort
from amodal_benchmark.surface_policy import apply_frozen_surface
from amodal_benchmark.surface_refinement import FEATURE_NAMES


class SurfaceConfirmationTests(unittest.TestCase):
    def test_unchanged_ranker_selects_exact_raw_rgb_even_on_visible_support(self):
        count = len(FEATURE_NAMES)
        parent = dict(candidate_seeds=[11, 22], pretrained_visible_features=False,
            ranker=dict(feature_names=FEATURE_NAMES, medians=[0]*count, center=[0]*count,
                        scale=[1]*count, coefficients=[1]*count, valid_only=True),
            refinement=dict(mode='affine', strength=1., preserve_visible=True),
            ensemble=dict(strength=.5), development={'images': 183})
        before = json.dumps(parent)
        freeze = confirmation_freeze(parent, [], 'parent-digest')
        yy, xx = np.indices((40, 48))
        observed = np.stack([100+xx, 80+yy, 110+xx//2], axis=-1).astype(np.uint8)
        visible = xx < 24
        candidates = [observed+5, observed+15]
        masks = [np.ones_like(visible), np.ones_like(visible)]
        rgb, mask, details = apply_frozen_surface(observed, visible, ~visible,
                                                candidates, masks, freeze)
        np.testing.assert_array_equal(rgb, candidates[details['index']])
        np.testing.assert_array_equal(mask, masks[details['index']])
        self.assertEqual(freeze['ranker'], parent['ranker'])
        self.assertEqual(json.dumps(parent), before)
        self.assertNotIn('ensemble', freeze)

    def test_confirmatory_cohort_rejects_any_previous_or_duplicated_image(self):
        freeze = {'confirmation': dict(expected_images=2, sources={'VG': 1, 'COCOA': 1},
            excluded_image_identities=['old'], excluded_image_sha256=['old-hash'])}
        cases = [dict(source='VG', image_identity='new1', source_sha256='hash1'),
                 dict(source='COCOA', image_identity='new2', source_sha256='hash2')]
        check_cohort(cases, freeze)
        for key, value in [('image_identity', 'old'), ('source_sha256', 'old-hash'),
                           ('source_sha256', 'hash1')]:
            changed = [dict(row) for row in cases]
            changed[1][key] = value
            with self.assertRaisesRegex(ValueError, 'overlap'):
                check_cohort(changed, freeze)
        with self.assertRaisesRegex(ValueError, 'Incomplete'):
            check_cohort(cases[:1], freeze)


if __name__ == '__main__':
    unittest.main()

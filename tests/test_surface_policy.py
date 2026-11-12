import unittest
from pathlib import Path
import tempfile

import numpy as np

from amodal_benchmark.surface_policy import apply_frozen_surface, verify_frozen_surface_sources
from amodal_benchmark.dataset import sha256
from amodal_benchmark.surface_refinement import FEATURE_NAMES


class SurfacePolicyTests(unittest.TestCase):
    def test_stale_deployment_cannot_open_final_test_and_paths_are_portable(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / 'amodal_benchmark' / 'method.py'
            source.parent.mkdir()
            source.write_text('frozen')
            freeze = dict(status='frozen_before_fresh_test',
                          source_hashes={'amodal_benchmark\\method.py': sha256(source)})
            verify_frozen_surface_sources(freeze, root)
            source.write_text('changed')
            with self.assertRaisesRegex(ValueError, 'differs from the deployment'):
                verify_frozen_surface_sources(freeze, root)

    def setUp(self):
        yy,xx=np.indices((40,48))
        self.visible=xx<24
        self.observed=np.stack([100+xx,80+yy,110+xx//2],axis=-1).astype(np.uint8)
        self.images=[self.observed+5,self.observed+15]
        self.masks=[np.ones_like(self.visible),np.ones_like(self.visible)]
        n=len(FEATURE_NAMES)
        self.freeze=dict(candidate_seeds=[11,22],pretrained_visible_features=False,
            refinement=dict(mode='affine',strength=1.),ranker=dict(feature_names=FEATURE_NAMES,
                medians=[0]*n,center=[0]*n,scale=[1]*n,coefficients=[1]*n,valid_only=True))

    def test_complete_policy_cannot_read_hidden_observation(self):
        poison=self.observed.copy()
        poison[~self.visible]=255
        first=apply_frozen_surface(self.observed,self.visible,~self.visible,self.images,self.masks,self.freeze)
        second=apply_frozen_surface(poison,self.visible,~self.visible,self.images,self.masks,self.freeze)
        np.testing.assert_array_equal(first[0],second[0])
        np.testing.assert_array_equal(first[1],second[1])
        self.assertEqual(first[2],second[2])

    def test_refuses_missing_required_features_or_changed_candidate_budget(self):
        deep=dict(self.freeze,pretrained_visible_features=True)
        with self.assertRaises(ValueError):
            apply_frozen_surface(self.observed,self.visible,~self.visible,self.images,self.masks,deep)
        with self.assertRaises(ValueError):
            apply_frozen_surface(self.observed,self.visible,~self.visible,self.images[:1],self.masks[:1],self.freeze)


if __name__=='__main__':
    unittest.main()

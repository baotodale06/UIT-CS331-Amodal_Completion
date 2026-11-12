import unittest
from importlib.util import find_spec

import numpy as np

from amodal_benchmark.surface_ensemble import EnsembleConfig, refine_surface_pool


@unittest.skipUnless(find_spec('scipy'), 'Optional research dependency scipy')
class SurfaceEnsembleTests(unittest.TestCase):
    def setUp(self):
        yy, xx = np.indices((48,64))
        self.visible = xx < 20
        base = np.stack([100+xx%5,120+yy%7,90+xx%3],axis=-1).astype(np.uint8)
        self.images = np.stack([base,base+10,base+20,base+30])
        self.images[:,self.visible] = base[self.visible]
        self.masks = np.ones(self.images.shape[:3],bool)

    def test_zero_strength_and_single_candidate_preserve_pixels(self):
        for images,masks,config in ((self.images,self.masks,EnsembleConfig(strength=0)),
                                   (self.images[:1],self.masks[:1],EnsembleConfig())):
            actual,_ = refine_surface_pool(images,masks,self.visible,config)
            np.testing.assert_array_equal(actual,images)

    def test_consensus_is_bounded_and_preserves_observation(self):
        actual,_ = refine_surface_pool(self.images,self.masks,self.visible,
                                       EnsembleConfig(strength=1,max_shift=8))
        np.testing.assert_array_equal(actual[:,self.visible],self.images[:,self.visible])
        self.assertLessEqual(np.abs(actual.astype(float)-self.images).max(),8)
        self.assertLess(actual[:,:,-10:].astype(float).std(0).mean(),
                        self.images[:,:,-10:].astype(float).std(0).mean())

    def test_outside_predicted_masks_cannot_influence_surface_consensus(self):
        masks = self.masks.copy()
        masks[0,:,35:] = False
        masks[1,:,50:] = False
        changed = self.images.copy()
        changed[~masks] = 255
        first,_ = refine_surface_pool(self.images,masks,self.visible)
        second,_ = refine_surface_pool(changed,masks,self.visible)
        np.testing.assert_array_equal(first[masks],second[masks])
        np.testing.assert_array_equal(first[~masks],self.images[~masks])

    def test_refuses_misaligned_or_unbounded_inputs(self):
        with self.assertRaises(ValueError):
            refine_surface_pool(self.images/255,self.masks,self.visible)
        with self.assertRaises(ValueError):
            refine_surface_pool(self.images,self.masks,self.visible,EnsembleConfig(strength=1.1))
        with self.assertRaises(ValueError):
            refine_surface_pool(self.images,self.masks[:,:,:20],self.visible)


if __name__ == '__main__':
    unittest.main()

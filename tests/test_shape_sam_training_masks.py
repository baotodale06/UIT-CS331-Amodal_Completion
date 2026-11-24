"""Training-mask preparation must keep disjoint development data separate."""
import unittest
import numpy as np
from scripts.prepare_shape_sam_training import shard_groups, training_groups, visible_prompt


class TrainingMaskTests(unittest.TestCase):
    def test_shards_cover_training_once_without_validation(self):
        manifest = dict(holdout_packaged=False, rows=[dict(id=str(i), image_id=i, split="train") for i in range(9)]
            + [dict(id="validation", image_id=100, split="validation")])
        groups = training_groups(manifest)
        assigned = [image for rank in range(4) for image,_ in shard_groups(groups, rank, 4)]
        self.assertEqual(sorted(assigned), list(range(9)))

    def test_cross_split_image_rejected(self):
        with self.assertRaisesRegex(ValueError, "crosses"):
            training_groups(dict(holdout_packaged=False, rows=[dict(image_id=1, split=split) for split in ("train", "validation")]))

    def test_holdout_package_rejected(self):
        with self.assertRaisesRegex(ValueError, "holdout-free"):
            training_groups(dict(holdout_packaged=False, rows=[dict(image_id=i, split=split)
                for i,split in enumerate(("train", "validation", "holdout"))]))

    def test_only_visible_extent_defines_prompt(self):
        self.assertEqual(visible_prompt(np.array([[0,1,1],[0,0,0]], bool)), [1,0,3,1])
        with self.assertRaises(ValueError):
            visible_prompt(np.zeros((2,2), bool))


if __name__ == "__main__":
    unittest.main()

"""Independent validation replay rejects changed masks and target eligibility."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from PIL import Image
from scripts.recover_shape_sam_validation import check_ids, geometry, mask, paired_summary
from scripts.prepare_shape_sam_training import shard_groups, training_groups, visible_prompt


class SamRecoveryTests(unittest.TestCase):
    def test_replay_does_not_repair_missing_visible_pixels(self):
        truth = np.array([[1, 1], [1, 0]], bool)
        visible = np.array([[1, 0], [1, 0]], bool)
        prediction = np.array([[0, 1], [0, 1]], bool)
        self.assertEqual(geometry(prediction, truth, visible), dict(full_iou=.25, hidden_iou=.5))

    def test_reject_nonbinary_and_resized_masks(self):
        with TemporaryDirectory() as work:
            path = Path(work)/"mask.png"
            Image.fromarray(np.array([[0, 127]], np.uint8)).save(path)
            with self.assertRaises(ValueError):
                mask(path, (1, 2))
            Image.fromarray(np.array([[0, 255]], np.uint8)).save(path)
            with self.assertRaises(ValueError):
                mask(path, (2, 1))

    def test_reject_missing_or_duplicate_objects(self):
        for rows in ([dict(id="a")], [dict(id="a"), dict(id="a")]):
            with self.assertRaises(ValueError):
                check_ids(rows, {"a", "b"})

    def test_fixed_truth_and_image_weighting(self):
        rows = [dict(image_id=i, annotated_input=dict(full_iou=a, hidden_iou=a),
            sam_input=dict(full_iou=b, hidden_iou=b)) for i,a,b in [(1,1.,0.),(1,1.,0.),(2,0.,1.)]]
        result = paired_summary(rows)
        self.assertEqual(result["full_iou"]["sam_minus_annotated"], 0.)
        rows[0]["sam_input"]["hidden_iou"] = None
        with self.assertRaises(ValueError):
            paired_summary(rows)

    def test_training_shards_exclude_validation_and_are_disjoint(self):
        manifest = dict(holdout_packaged=False, rows=[dict(id=str(i), image_id=i, split="train") for i in range(9)]
            + [dict(id="validation", image_id=100, split="validation")])
        groups = training_groups(manifest)
        shards = [shard_groups(groups, rank, 4) for rank in range(4)]
        self.assertEqual(sorted(image for shard in shards for image,_ in shard), list(range(9)))
        manifest["rows"].append(dict(id="overlap", image_id=1, split="validation"))
        with self.assertRaises(ValueError):
            training_groups(manifest)

    def test_prompt_uses_only_visible_pixels(self):
        self.assertEqual(visible_prompt(np.array([[0,1,1],[0,0,0]], bool)), [1,0,3,1])
        with self.assertRaises(ValueError):
            visible_prompt(np.zeros((2,2), bool))


if __name__ == "__main__":
    unittest.main()

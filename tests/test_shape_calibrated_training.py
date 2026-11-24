import unittest

from scripts.shape_calibrated_training import calibration_summary, eligible


def row(identity='a', full=.8, hidden=.4, pixels='first'):
    return dict(id=identity, image_id=1, full_iou=full, hidden_iou=hidden,
                prediction_sha256=pixels)


class CalibratedTrainingTests(unittest.TestCase):
    def test_cross_device_difference_is_disclosed_without_lowering_floors(self):
        measured = [row(full=.81, hidden=.39, pixels='different')]
        result = calibration_summary(measured, [row()], measured)
        self.assertEqual(result['different_pixel_hash_objects'], ['a'])
        self.assertFalse(result['cross_device_pixel_identity'])
        self.assertEqual(result['selection_floor'], dict(full_iou=.81, hidden_iou=.4))
        self.assertFalse(result['benchmark_admitted'])

    def test_equal_scores_do_not_hide_nondeterministic_pixels(self):
        with self.assertRaises(ValueError):
            calibration_summary([row()], [row()], [row(pixels='changed')])

    def test_duplicate_or_unknown_object_is_rejected(self):
        for rows in ([row(), row()], [row(identity='unknown')]):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                calibration_summary(rows, [row()], rows)

    def test_candidate_must_keep_both_floors_and_improve(self):
        floor = dict(full_iou=.81, hidden_iou=.4)
        self.assertFalse(eligible(row(full=.8, hidden=.5), floor, (.4, .81)))
        self.assertFalse(eligible(row(full=.82, hidden=.39), floor, (.4, .81)))
        self.assertFalse(eligible(row(full=.81, hidden=.4), floor, (.4, .81)))
        self.assertTrue(eligible(row(full=.81, hidden=.41), floor, (.4, .81)))


if __name__ == '__main__':
    unittest.main()

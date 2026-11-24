"""Exercise independent local selection and rejection of malformed mask evidence."""
import io
import unittest

import numpy as np
from PIL import Image

from scripts.verify_shape_local import binary_payload, select_epoch


class LocalReplayTests(unittest.TestCase):
    def test_original_floor_and_stable_tie_are_preserved(self):
        training = dict(initial_validation=dict(full_iou=.81,hidden_iou=.37),
            original_quadro_validation=dict(full_iou=.82,hidden_iou=.38),
            selection_floor=dict(full_iou=.82,hidden_iou=.38),history=[
                dict(epoch=1,validation=dict(full_iou=.81,hidden_iou=.45)),
                dict(epoch=2,validation=dict(full_iou=.82,hidden_iou=.39)),
                dict(epoch=3,validation=dict(full_iou=.82,hidden_iou=.39))])
        self.assertEqual(select_epoch(training)['epoch'],2)
        training['history'] = training['history'][:1]
        self.assertEqual(select_epoch(training)['epoch'],0)

    def test_nonfinite_geometry_and_lowered_floor_are_rejected(self):
        training = dict(initial_validation=dict(full_iou=.81,hidden_iou=.37),
            original_quadro_validation=dict(full_iou=.82,hidden_iou=.38),
            selection_floor=dict(full_iou=.82,hidden_iou=.38),history=[
                dict(epoch=1,validation=dict(full_iou=.83,hidden_iou=float('nan')))])
        with self.assertRaises(ValueError):
            select_epoch(training)
        training['history'] = []
        training['selection_floor']['full_iou'] = .81
        with self.assertRaises(ValueError):
            select_epoch(training)

    def test_mask_must_be_binary_and_in_native_frame(self):
        def payload(array):
            stream = io.BytesIO()
            Image.fromarray(np.array(array,dtype=np.uint8)).save(stream,format='PNG')
            return stream.getvalue()
        data = payload([[0,255],[255,0]])
        np.testing.assert_array_equal(binary_payload(data,[2,2]),[[False,True],[True,False]])
        with self.assertRaises(ValueError):
            binary_payload(data,[1,4])
        with self.assertRaises(ValueError):
            binary_payload(payload([[0,127],[255,0]]),[2,2])


if __name__=='__main__':
    unittest.main()

import unittest

import numpy as np
import pandas as pd

from scripts.matched_lesion import SIZE, perturb, select_regions, signed_rank


class MatchedLesionTests(unittest.TestCase):
    def test_translated_controls_preserve_shape_area_and_anatomy(self):
        left = np.zeros((SIZE, SIZE), bool)
        right = np.zeros_like(left)
        left[20:205, 15:106] = True
        right[20:205, 120:210] = True
        boxes = pd.DataFrame([dict(x_dis=100, y_dis=450, width_dis=75, height_dis=80)])
        region, reason = select_regions(boxes, (left, right))
        self.assertIsNone(reason)
        self.assertGreaterEqual(len(region['controls']), 3)
        for control in region['controls']:
            self.assertEqual(int(control.sum()), region['index_area'])
            self.assertTrue(np.all(left[control]))
            self.assertFalse(np.any(control & region['all_boxes']))

    def test_identity_and_locality(self):
        image = np.random.default_rng(3).random((SIZE, SIZE)).astype(np.float32)
        empty = np.zeros_like(image, bool)
        self.assertTrue(np.array_equal(perturb(image, empty, 10), image))
        region = empty.copy(); region[20:30, 20:30] = True
        altered = perturb(image, region, 10)
        self.assertTrue(np.array_equal(altered[~region], image[~region]))
        self.assertFalse(np.array_equal(altered[region], image[region]))

    def test_signed_rank_zero_and_direction(self):
        zero = signed_rank([0, 0, 0])
        self.assertEqual(zero['status'], 'sin_diferencias_no_nulas')
        self.assertIsNone(zero['p'])
        positive = signed_rank(np.arange(1, 18))
        self.assertEqual(positive['n'], 17)
        self.assertEqual(positive['zeros'], 0)
        self.assertEqual(positive['rank_biserial'], 1)
        self.assertLess(positive['p'], 0.05)


if __name__ == '__main__':
    unittest.main()

import unittest

import numpy as np

from scripts.occlusion_analysis import matched_lung_masks


class OcclusionControlTest(unittest.TestCase):
    def test_matched_masks_stay_inside_lung_with_exact_area(self):
        lung = np.zeros((224, 224), dtype=bool)
        lung[20:200, 20:200] = True
        for mask in matched_lung_masks(lung, count=4379, seed=23):
            self.assertEqual(int(mask.sum()), 4379)
            self.assertFalse((mask & ~lung).any())


if __name__ == "__main__":
    unittest.main()

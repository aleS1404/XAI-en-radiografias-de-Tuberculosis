import unittest

import numpy as np
import pandas as pd

from scripts.lesion_roi import boxes_to_mask, roi_scores


class LesionMetricsTest(unittest.TestCase):
    def test_uniform_map_has_unit_gain(self):
        box = pd.DataFrame([dict(x_dis=256, y_dis=256, width_dis=256, height_dis=256)])
        lesion = boxes_to_mask(box, 224)
        lung = np.zeros((224, 224), dtype=bool)
        lung[20:200, 20:200] = True
        result = roi_scores(np.ones((224, 224)), lesion, lung)
        self.assertAlmostEqual(result["gain_image"], 1.0)
        self.assertAlmostEqual(result["gain_lung"], 1.0)

    def test_point_and_gain_follow_a_focal_map(self):
        box = pd.DataFrame([dict(x_dis=256, y_dis=256, width_dis=256, height_dis=256)])
        lesion = boxes_to_mask(box, 224)
        lung = np.ones((224, 224), dtype=bool)
        heatmap = np.zeros((224, 224))
        heatmap[80, 80] = 1
        result = roi_scores(heatmap, lesion, lung)
        self.assertEqual(result["point_lung"], 1.0)
        self.assertGreater(result["gain_lung"], 1.0)

    def test_null_map_is_reported_as_missing(self):
        box = pd.DataFrame([dict(x_dis=256, y_dis=256, width_dis=256, height_dis=256)])
        lesion = boxes_to_mask(box, 224)
        result = roi_scores(np.zeros((224, 224)), lesion, np.ones((224, 224), dtype=bool))
        self.assertTrue(np.isnan(result["gain_lung"]))
        self.assertTrue(np.isnan(result["point_lung"]))


if __name__ == "__main__":
    unittest.main()

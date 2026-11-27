import unittest

import numpy as np

from scripts.analyze_performance import metrics, stratified_indices


class PerformanceAnalysisTest(unittest.TestCase):
    def test_confusion_and_auc_use_fixed_threshold(self):
        result = metrics([0, 0, 1, 1], [.10, .70, .40, .90], .50)
        self.assertEqual((result["tn"], result["fp"], result["fn"], result["tp"]), (1, 1, 1, 1))
        self.assertAlmostEqual(result["sensitivity"], .5)
        self.assertAlmostEqual(result["specificity"], .5)
        self.assertAlmostEqual(result["auc"], .75)

    def test_bootstrap_keeps_both_classes(self):
        y = np.array([0] * 80 + [1] * 58)
        rng = np.random.default_rng(41)
        for _ in range(100):
            sample = y[stratified_indices(y, rng)]
            self.assertEqual((sample == 0).sum(), 80)
            self.assertEqual((sample == 1).sum(), 58)


if __name__ == "__main__":
    unittest.main()

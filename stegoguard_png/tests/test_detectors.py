import unittest

import numpy as np

from stegoguard.analysis import (chi_square_pov, flat_region_test, rs_analysis, run_statistical, spa_analysis,
                                 ws_analysis)
from stegoguard.config import Config
from stegoguard.embedding import embed_random_rate

from .helpers import camera_png, hmi_png, image, thermal_png


class DetectorsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rng = np.random.default_rng(5)
        cls.covers = [image(camera_png(2)), image(thermal_png(3))]
        cls.hmi = image(hmi_png(4))
        cls.thr = Config().thresholds

    def test_estimators_track_embedding_rate(self):
        for img in self.covers:
            ch = list(img.channels.values())[0]
            for rate in (0.2, 0.5):
                st = embed_random_rate({"c": ch}, rate, "lsb_replacement", self.rng)["c"]
                for fn in (rs_analysis, spa_analysis, ws_analysis):
                    est = fn(st)["estimate"]
                    self.assertAlmostEqual(est, rate, delta=0.1, msg=f"{fn.__name__} rate={rate} est={est}")

    def test_clean_covers_low_estimate(self):
        for img in self.covers + [self.hmi]:
            r = run_statistical(img.channels, 255, self.thr, full=False)
            self.assertLess(r["effective_payload"], self.thr.payload_suspicious)
            self.assertLess(r["detection_probability"], 0.3)

    def test_chi_square_full_sequential(self):
        ch = list(self.covers[0].channels.values())[0]
        self.assertLess(chi_square_pov(ch)["p_value"], 0.5)
        full = embed_random_rate({"c": ch}, 1.0, "lsb_replacement", self.rng)["c"]
        self.assertGreater(chi_square_pov(full)["p_value"], 0.95)

    def test_frn_detects_lsb_matching_on_hmi(self):
        ch = self.hmi.channels["R"]
        self.assertLess(flat_region_test(ch)["estimate"], 0.01)
        for rate in (0.1, 1.0):
            st = embed_random_rate({"c": ch}, rate, "lsb_matching", self.rng)["c"]
            self.assertAlmostEqual(flat_region_test(st)["estimate"], rate, delta=0.1)

    def test_frn_not_applicable_on_noisy_images(self):
        for img in self.covers:
            self.assertFalse(flat_region_test(list(img.channels.values())[0])["applicable"])


if __name__ == "__main__":
    unittest.main()

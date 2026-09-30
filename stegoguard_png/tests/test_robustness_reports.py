import json
import os
import tempfile
import threading
import unittest

from stegoguard.analyzer import Analyzer
from stegoguard.attacks import assess_attacks
from stegoguard.cli import main
from stegoguard.config import Config
from stegoguard.database import History
from stegoguard.monitor import FolderMonitor
from stegoguard.report import write_reports
from stegoguard.robustness import assess_robustness
from stegoguard.samples import append_trailing

from .helpers import camera_png, hmi_png, image


class RobustnessTest(unittest.TestCase):
    def test_robustness_structure(self):
        cfg = Config()
        cfg.robustness.trials = 1
        cfg.robustness.rates = [0.01, 0.05, 0.2, 0.5, 1.0]
        r = assess_robustness(image(camera_png(3)), cfg)
        self.assertTrue(r["supported"])
        rep = r["methods"]["lsb_replacement"]
        self.assertLess(rep["safe_rate"], 0.2)        # LSB-замена выявляется при r >= 0.2
        self.assertGreater(r["methods"]["lsb_matching"]["stability_index"], rep["stability_index"])
        self.assertTrue(0 <= r["stability_index"] <= 100)
        psnrs = [p["psnr"] for p in rep["points"]]
        self.assertEqual(psnrs, sorted(psnrs, reverse=True))

    def test_hmi_is_low_stability(self):
        cfg = Config()
        cfg.robustness.trials = 1
        cfg.robustness.rates = [0.02, 0.05, 0.2, 0.5, 1.0]
        r = assess_robustness(image(hmi_png(3)), cfg)
        self.assertLess(r["stability_index"], 70)

    def test_attacks(self):
        r = assess_attacks(image(camera_png(4)))
        self.assertTrue(r["supported"])
        self.assertTrue(r["results"]["png_resave"]["survives"])
        self.assertTrue(r["results"]["jpeg_75"]["destroyed"])
        self.assertTrue(r["results"]["sanitize"]["destroyed"])


class ReportsCliMonitorTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = self.tmp.name
        self.clean = os.path.join(self.d, "in", "clean.png")
        self.bad = os.path.join(self.d, "in", "bad.png")
        os.makedirs(os.path.dirname(self.clean))
        data = camera_png(9, 160, 200)
        with open(self.clean, "wb") as fh:
            fh.write(data)
        with open(self.bad, "wb") as fh:
            fh.write(append_trailing(data, os.urandom(1000)))

    def tearDown(self):
        self.tmp.cleanup()

    def test_reports(self):
        res = Analyzer().analyze_many([self.clean, self.bad])
        out = os.path.join(self.d, "rep")
        w = write_reports(res, out)
        self.assertEqual(len(w["json"]), 2)
        self.assertTrue(os.path.exists(os.path.join(out, "index.html")))
        self.assertTrue(os.path.exists(os.path.join(out, "summary.csv")))
        with open(w["json"][1], encoding="utf-8") as fh:
            self.assertEqual(json.load(fh)["verdict"], "STEGO_DETECTED")

    def test_cli_exit_codes(self):
        self.assertEqual(main(["analyze", self.clean]), 0)
        self.assertEqual(main(["analyze", self.bad]), 2)
        self.assertEqual(main(["analyze", os.path.join(self.d, "nope.png")]), 3)

    def test_monitor_quarantine(self):
        q = os.path.join(self.d, "q")
        ok = os.path.join(self.d, "ok")
        hist = History(os.path.join(self.d, "h.db"))
        mon = FolderMonitor(Analyzer(), os.path.dirname(self.clean), interval=0.01, quarantine_dir=q,
                            clean_dir=ok, history=hist)
        mon.run(stop_event=threading.Event(), max_cycles=3)
        self.assertTrue(os.path.exists(os.path.join(q, "bad.png")))
        self.assertTrue(os.path.exists(os.path.join(ok, "clean.png")))
        self.assertEqual(sum(hist.stats().values()), 2)


if __name__ == "__main__":
    unittest.main()

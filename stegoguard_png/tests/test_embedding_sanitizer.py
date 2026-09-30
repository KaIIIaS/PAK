import os
import tempfile
import unittest

import numpy as np

from stegoguard.analyzer import Analyzer
from stegoguard.embedding import CapacityError, ExtractionError, capacity_bytes, embed_message, extract_message
from stegoguard.image_io import load_image
from stegoguard.metrics import psnr, ssim
from stegoguard.sanitizer import sanitize
from stegoguard.samples import append_trailing, insert_chunk_before_iend

from .helpers import camera_png, image


class EmbeddingTest(unittest.TestCase):
    def setUp(self):
        self.img = image(camera_png(7))

    def test_roundtrip_with_and_without_key(self):
        for key in (None, "секрет"):
            for method in ("lsb_replacement", "lsb_matching"):
                msg = os.urandom(500)
                st = embed_message(self.img, msg, key=key, method=method)
                self.assertEqual(extract_message(st, key=key), msg)
                for name in st.channels:
                    self.assertLessEqual(int(np.abs(st.channels[name] - self.img.channels[name]).max()), 1)

    def test_wrong_key(self):
        st = embed_message(self.img, b"hello", key="a")
        with self.assertRaises(ExtractionError):
            extract_message(st, key="b")

    def test_capacity(self):
        with self.assertRaises(CapacityError):
            embed_message(self.img, b"x" * (capacity_bytes(self.img) + 1))

    def test_metrics(self):
        a = self.img.channels["R"]
        self.assertEqual(psnr(a, a), float("inf"))
        self.assertAlmostEqual(ssim(a, a), 1.0, places=6)


class SanitizerAndAnalyzerTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def _write(self, name, data):
        p = os.path.join(self.dir, name)
        with open(p, "wb") as fh:
            fh.write(data)
        return p

    def test_full_pipeline(self):
        clean = camera_png(8)
        img = load_image(clean)
        stego = embed_message(img, os.urandom(capacity_bytes(img) // 3), key="k")
        p_clean = self._write("clean.png", clean)
        p_stego = os.path.join(self.dir, "stego.png")
        stego.save_png(p_stego)
        p_struct = self._write("struct.png", append_trailing(
            insert_chunk_before_iend(clean, b"prVt", os.urandom(2000)), b"MZ" + os.urandom(100)))

        an = Analyzer()
        self.assertEqual(an.analyze_file(p_clean)["verdict"], "CLEAN")
        self.assertEqual(an.analyze_file(p_stego)["verdict"], "STEGO_DETECTED")
        r = an.analyze_file(p_struct)
        self.assertEqual(r["verdict"], "STEGO_DETECTED")

        for src in (p_stego, p_struct):
            dst = os.path.join(self.dir, "san_" + os.path.basename(src))
            rep = sanitize(src, dst, seed=1)
            self.assertGreater(rep["psnr"], 48)
            after = an.analyze_file(dst)
            self.assertEqual(after["verdict"], "CLEAN", after["reasons"])
            with self.assertRaises(ExtractionError):
                extract_message(load_image(dst), key="k")

    def test_error_verdict_for_garbage(self):
        p = self._write("bad.png", b"not an image at all")
        r = Analyzer().analyze_file(p)
        self.assertIn(r["verdict"], ("STEGO_DETECTED", "SUSPICIOUS", "ERROR"))
        self.assertTrue(any("PNG" in x for x in r["reasons"]))


if __name__ == "__main__":
    unittest.main()

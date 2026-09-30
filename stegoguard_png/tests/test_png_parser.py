import os
import unittest
import zlib

from stegoguard.png_parser import expected_raw_size, parse_png, scan_signatures, shannon_entropy
from stegoguard.samples import append_trailing, corrupt_crc, inject_idat_extra, insert_chunk_before_iend

from .helpers import camera_png


def codes(st):
    return {f.code for f in st.findings}


class PNGParserTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.clean = camera_png()

    def test_clean_file_has_no_serious_findings(self):
        st = parse_png(self.clean)
        self.assertTrue(st.valid_signature)
        self.assertEqual(st.structural_risk, 0)
        self.assertEqual(st.decompressed_size, st.expected_size)

    def test_not_png(self):
        st = parse_png(b"PK\x03\x04" + b"\x00" * 100)
        self.assertIn("NOT_PNG", codes(st))
        self.assertIn("ZIP", st.findings[0].details)

    def test_trailing_data(self):
        st = parse_png(append_trailing(self.clean, b"PK\x03\x04" + os.urandom(500)))
        self.assertIn("TRAILING_DATA", codes(st))
        self.assertEqual(st.trailing_bytes, 504)
        self.assertEqual(st.max_severity, "critical")

    def test_zero_padding_is_low(self):
        st = parse_png(append_trailing(self.clean, b"\x00" * 4))
        self.assertIn("TRAILING_PADDING", codes(st))
        self.assertLess(st.structural_risk, 30)

    def test_crc_error(self):
        self.assertIn("CRC_MISMATCH", codes(parse_png(corrupt_crc(self.clean))))

    def test_private_chunk(self):
        st = parse_png(insert_chunk_before_iend(self.clean, b"stEg", os.urandom(3000)))
        self.assertIn("UNKNOWN_CHUNK", codes(st))

    def test_encoded_text(self):
        data = insert_chunk_before_iend(self.clean, b"zTXt",
                                        b"Comment\x00\x00" + zlib.compress(os.urandom(2000).hex().encode()))
        st = parse_png(data)
        self.assertIn("ENCODED_TEXT", codes(st))
        self.assertEqual(st.text_entries[0]["keyword"], "Comment")

    def test_embedded_file_in_text(self):
        data = insert_chunk_before_iend(self.clean, b"tEXt", b"Author\x00%PDF-1.7 hidden document")
        self.assertIn("EMBEDDED_FILE", codes(parse_png(data)))

    def test_idat_extra(self):
        self.assertIn("ZLIB_TRAILING", codes(parse_png(inject_idat_extra(self.clean, os.urandom(300)))))

    def test_expected_size_interlaced(self):
        self.assertEqual(expected_raw_size(8, 8, 8, 2, 0), 8 * (1 + 24))
        # Adam7 для 1x1: единственный проход 1
        self.assertEqual(expected_raw_size(1, 1, 8, 0, 1), 2)

    def test_entropy_and_signatures(self):
        self.assertAlmostEqual(shannon_entropy(bytes(range(256)) * 4), 8.0, places=5)
        self.assertEqual(shannon_entropy(b""), 0.0)
        self.assertTrue(any("ELF" in n for _, n in scan_signatures(b"xx\x7fELFyy")))


if __name__ == "__main__":
    unittest.main()

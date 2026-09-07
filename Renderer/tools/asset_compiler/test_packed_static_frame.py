"""Analytic tests for the offline octahedral direction decoder."""
import math
import struct
import unittest

from Renderer.tools.asset_compiler.packed_static_frame import decode_octahedral_snorm8


class PackedNormalTests(unittest.TestCase):
    def test_axes_and_negative_extreme(self):
        for packed,expected in (((0,0),(0,0,1)),((127,0),(1,0,0)),
                                ((-128,0),(-1,0,0)),((0,127),(0,1,0)),
                                ((0,-128),(0,-1,0)),((127,127),(0,0,-1))):
            with self.subTest(packed=packed):
                self.assertEqual(decode_octahedral_snorm8(struct.pack('<2b',*packed)),list(expected))

    def test_lower_hemisphere_fold_and_offset(self):
        n=decode_octahedral_snorm8(b'xx'+struct.pack('<2b',95,-95),2)
        length=math.sqrt(32*32+32*32+63*63)
        for actual,expected in zip(n,(32/length,-32/length,-63/length)):
            self.assertAlmostEqual(actual,expected)

    def test_all_encodings_finite_unit_length(self):
        for x in range(-128,128):
            for y in range(-128,128):
                n=decode_octahedral_snorm8(struct.pack('<2b',x,y))
                self.assertTrue(all(math.isfinite(v) for v in n))
                self.assertAlmostEqual(sum(v*v for v in n),1.)


if __name__=='__main__':unittest.main()

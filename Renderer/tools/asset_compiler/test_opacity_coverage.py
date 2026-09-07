"""Use independent DDS decoding to verify BC4-to-BC3 coverage preservation."""
import io
import unittest
from PIL import Image

from Renderer.tools.asset_compiler.opacity_coverage import bc4_blocks_to_bc3_alpha
from Renderer.tools.asset_compiler.c3x_asset_compiler import make_dds_dx10_header


class OpacityCoverageTests(unittest.TestCase):
    def test_both_endpoint_modes_and_all_indices(self):
        indices=sum((i%8)<<(3*i) for i in range(16)).to_bytes(6,'little')
        payload=b''.join(bytes(pair)+indices for pair in ((0,255),(255,0),(40,190),(190,40)))
        info={'width':8,'height':8,'mip_count':1}
        before=Image.open(io.BytesIO(make_dds_dx10_header({**info,'dxgi_format':80})+payload)).convert('L')
        after=Image.open(io.BytesIO(make_dds_dx10_header({**info,'dxgi_format':77})+bc4_blocks_to_bc3_alpha(payload))).convert('RGBA')
        self.assertEqual(before.tobytes(),after.getchannel('A').tobytes())
        self.assertEqual(after.convert('RGB').getextrema(),((255,255),(255,255),(255,255)))

    def test_reject_truncated_block(self):
        with self.assertRaises(ValueError):bc4_blocks_to_bc3_alpha(bytes(7))


if __name__=='__main__':unittest.main()

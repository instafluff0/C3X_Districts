"""Lossless source recovery and malformed-package guards, using local licensed inputs."""
import tempfile
from pathlib import Path
import struct
import unittest

from Renderer.tools.asset_compiler import wave_blp_extractor as waves

SOURCE = waves.DEFAULT_ASSETS / "Base/Platforms/Windows/BLPs/Wave.blp"


@unittest.skipUnless(SOURCE.is_file(), "Local source game is not installed")
class SourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.package, cls.layout, cls.textures, cls.delays = waves.decode(SOURCE)

    def test_complete_payload_round_trip(self):
        for texture in self.textures.values():
            dds = waves.rgba_dds(texture["width"], texture["height"], texture["mip_count"], texture["payload"])
            self.assertEqual(dds[148:], texture["payload"])
            self.assertEqual(struct.unpack_from("<I", dds, 128)[0], 28)
            self.assertEqual(len(dds)-148, texture["payload_bytes"])
            self.assertEqual(waves.digest(dds[148:]),texture["payload_sha256"])

    def test_every_crest_has_active_and_inactive_extent(self):
        count=self.layout["delays_per_page"]
        for page in range(self.layout["pages"]):
            values=struct.unpack_from(f"<{count}f",self.delays,page*count*4)
            self.assertTrue(any(0<value<1 for value in values))
            self.assertTrue(any(value>1e30 for value in values))
            self.assertTrue(all(0<=value<=1 or value==float.fromhex('0x1.fffffep+127') for value in values))

    def test_source_header_rejects_truncation(self):
        with tempfile.TemporaryDirectory() as directory:
            source=Path(directory)/"truncated.blp"
            source.write_bytes(SOURCE.read_bytes()[:-512])
            with self.assertRaises(ValueError):waves.decode(source)

    def test_crest_markers_track_the_spatial_peak(self):
        # The table locates the crest within each source row; it is not a
        # sequence of frame numbers. Keep this evidence separate from the
        # still-unrecovered source shader's use of that location.
        atlas=self.textures['crest']['payload'];active=exact=0
        for page in range(16):
            values=struct.unpack_from('<512f',self.delays,page*512*4)
            for row,marker in enumerate(values):
                if marker>1:continue
                offset=((page//8*512+row)*1024+page%8*128)*4
                samples=atlas[offset:offset+128*4:4]
                error=abs(samples.index(max(samples))-marker*128)
                self.assertLessEqual(error,7)
                active+=1;exact+=error==0
        self.assertGreater(exact/active,.90)

    def test_inactive_crest_markers_preserve_faint_art(self):
        # FLT_MAX is not a row opacity mask. Every page retains low-amplitude
        # feathering beyond its marked crest; discarding those rows cuts it off.
        atlas=self.textures['crest']['payload']
        for page in range(16):
            values=struct.unpack_from('<512f',self.delays,page*512*4)
            inactive=[]
            for row,delay in enumerate(values):
                if delay<1e30:continue
                offset=((page//8*512+row)*1024+page%8*128)*4
                inactive.extend(atlas[offset:offset+128*4:4])
            self.assertGreater(max(inactive),0)
            self.assertLessEqual(max(inactive),32)

    def test_texture_pointer_cannot_escape_graph(self):
        data=bytearray(SOURCE.read_bytes())
        entry=self.package.unique_allocation("CoastlineWaves::PackageEntry")
        offset,_=self.package.resolve(entry)
        struct.pack_into("<Q",data,self.package.package_file_offset+offset+0x58,0)
        with tempfile.TemporaryDirectory() as directory:
            source=Path(directory)/"corrupt.blp";source.write_bytes(data)
            with self.assertRaises(ValueError):waves.decode(source)

    def test_active_source_still_reproduces(self):
        second=waves.decode(SOURCE)
        self.assertEqual(second[3],self.delays)
        self.assertEqual(second[2],self.textures)


if __name__=="__main__":unittest.main()

"""Executable placement and optional source-pack contract."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from Renderer.tools.asset_compiler import build_wave_runtime as builder

class CoastalWaveTests(unittest.TestCase):
    def test_geometry(self):
        compiler=shutil.which('c++')
        if not compiler:self.skipTest('C++ compiler unavailable')
        with tempfile.TemporaryDirectory() as tmp:
            exe=Path(tmp)/'waves'
            subprocess.run([compiler,'-std=c++17','-O2',str(Path(__file__).with_suffix('.cpp')),'-o',str(exe)],check=True)
            subprocess.run([str(exe)],check=True)

    @unittest.skipUnless(builder.SOURCE.is_dir(), 'Local normalized art unavailable')
    def test_generic_pack(self):
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)
            builder.build(output)
            self.assertEqual((output/'crest.dds').read_bytes(),(builder.SOURCE/'crest.dds').read_bytes())
            self.assertEqual((output/'auxiliary.dds').read_bytes(),(builder.SOURCE/'auxiliary.dds').read_bytes())
            self.assertEqual((output/'delays.dds').read_bytes()[148:],(builder.SOURCE/'crest-delays.f32').read_bytes())
            self.assertEqual(len((output/'waves.bin').read_bytes()),16)

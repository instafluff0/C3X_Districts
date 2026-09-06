from pathlib import Path
import subprocess
import tempfile
import unittest

class VariedCoastTests(unittest.TestCase):
    def test_real_region_domain_and_world_continuity(self):
        source=Path(__file__).with_suffix('.cpp')
        with tempfile.TemporaryDirectory() as tmp:
            binary=Path(tmp)/'coast'
            subprocess.run(['clang++','-std=c++17','-O2',str(source),'-o',str(binary)],check=True)
            subprocess.run([str(binary)],check=True)

if __name__=='__main__':unittest.main()

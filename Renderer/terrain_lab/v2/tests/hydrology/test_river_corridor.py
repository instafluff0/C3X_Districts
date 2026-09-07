import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
V2 = ROOT / 'Renderer/terrain_lab/v2'


class RiverCanopyTests(unittest.TestCase):
    def test_stable_topology_and_relief_routing(self):
        sys.path.insert(0, str(V2 / 'app'))
        import real_map
        _, data = real_map.load_registry()
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            files = []
            for i, origin in enumerate(([76, 58], [78, 60])):
                path = tmp / f'{i}.csv'
                path.write_bytes(real_map.csv_bytes(data, dict(origin=origin, extent=[10, 10], halo=6)))
                files.append(str(path))
            binary = tmp / 'corridor'
            subprocess.run(['clang++', '-std=c++17', '-O2',
                            str(V2 / 'tests/hydrology/test_river_corridor.cpp'),
                            '-o', str(binary)], check=True, cwd=ROOT)
            profile=V2 / 'fixtures/beauty/source-river-pools-r1/pool_profiles.csv'
            subprocess.run([str(binary), *files, str(profile)], check=True, cwd=ROOT)

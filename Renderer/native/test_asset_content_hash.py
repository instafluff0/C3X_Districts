from pathlib import Path
import subprocess
import tempfile
import unittest


class AssetContentHash(unittest.TestCase):
    def test_upstream_vector_and_invalidation(self):
        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory) / "hash"
            subprocess.run(["c++", "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
                            str(Path(__file__).with_suffix(".cpp")), "-o", str(binary)], check=True)
            subprocess.run([str(binary)], check=True)

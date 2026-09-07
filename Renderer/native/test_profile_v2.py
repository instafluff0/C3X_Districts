"""Differential tests against the immutable implementation pickup package."""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import tempfile
import unittest


class ProfileV2Tests(unittest.TestCase):
    def test_pinned_reference_integrity(self):
        root = Path(__file__).resolve().parents[2]
        manifest = json.loads((Path(__file__).parent / 'profile_v2/provenance.json').read_text())
        for entry in manifest['files']:
            with self.subTest(path=entry['target']):
                self.assertEqual(hashlib.sha256((root / entry['target']).read_bytes()).hexdigest(), entry['sha256'])

    def test_production_queries_match_pinned_reference(self):
        compiler = shutil.which('c++') or shutil.which('g++')
        if compiler is None:
            self.skipTest('portable C++ compiler unavailable')
        with tempfile.TemporaryDirectory() as directory:
            exe = Path(directory) / 'profile_v2_test'
            subprocess.run([compiler, '-std=c++17', '-O2', str(Path(__file__).with_suffix('.cpp')),
                            '-o', str(exe)], check=True)
            subprocess.run([str(exe)], check=True)
            source = Path(__file__).parent / 'profile_v2/test_cliffs.cpp'
            subprocess.run([compiler, '-std=c++17', '-O2', str(source), '-o', str(exe)], check=True)
            subprocess.run([str(exe)], check=True)


if __name__ == '__main__':
    unittest.main()

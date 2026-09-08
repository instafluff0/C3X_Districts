"""Differential tests for the production render-core queries and source pins."""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import tempfile
import unittest


class RenderCoreTests(unittest.TestCase):
    def test_pinned_source_integrity(self):
        root = Path(__file__).resolve().parents[2]
        manifest = json.loads((Path(__file__).parent / 'render_core/source_provenance.json').read_text())
        for entry in manifest['files']:
            with self.subTest(path=entry['target']):
                self.assertEqual(hashlib.sha256((root / entry['target']).read_bytes()).hexdigest(), entry['sha256'])

    def test_production_queries_match_pinned_source(self):
        compiler = shutil.which('c++') or shutil.which('g++')
        if compiler is None:
            self.skipTest('portable C++ compiler unavailable')
        with tempfile.TemporaryDirectory() as directory:
            exe = Path(directory) / 'render_core_test'
            subprocess.run([compiler, '-std=c++17', '-O2', str(Path(__file__).with_suffix('.cpp')),
                            '-o', str(exe)], check=True)
            subprocess.run([str(exe)], check=True)
            source = Path(__file__).parent / 'render_core/test_cliffs.cpp'
            subprocess.run([compiler, '-std=c++17', '-O2', str(source), '-o', str(exe)], check=True)
            subprocess.run([str(exe)], check=True)


if __name__ == '__main__':
    unittest.main()

"""Differential tests for the production render-core queries and source pins."""
from pathlib import Path
import hashlib
import json
import unittest
from Renderer.native.native_cpp_test import run_cpp


class RenderCoreTests(unittest.TestCase):
    def test_pinned_source_integrity(self):
        root = Path(__file__).resolve().parents[2]
        manifest = json.loads((Path(__file__).parent / 'render_core/source_provenance.json').read_text())
        for entry in manifest['files']:
            with self.subTest(path=entry['target']):
                # Git's Windows checkout may use CRLF; the preserved source
                # pins describe canonical LF text. Keep the pinned digest.
                source = (root / entry['target']).read_bytes().replace(b'\r\n', b'\n')
                self.assertEqual(hashlib.sha256(source).hexdigest(), entry['sha256'])

    def test_production_queries_match_pinned_source(self):
        run_cpp('#include "Renderer/native/test_render_core.cpp"\n')
        run_cpp('#include "Renderer/native/render_core/test_cliffs.cpp"\n')


if __name__ == '__main__':
    unittest.main()

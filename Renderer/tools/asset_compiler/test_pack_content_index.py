import json
import tempfile
import unittest
from pathlib import Path

from Renderer.tools.asset_compiler.pack_content_index import build_content_addressed_bundle, validate_bundle


class PackContentIndexTests(unittest.TestCase):
    def test_bundle_deduplicates_identical_bytes_and_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            pack = Path(directory) / "pack"
            pack.mkdir()
            (pack / "manifest.json").write_text(json.dumps({"schema": "test"}), encoding="utf-8")
            (pack / "a.bin").write_bytes(b"same")
            (pack / "nested").mkdir()
            (pack / "nested/b.bin").write_bytes(b"same")
            first = build_content_addressed_bundle(pack)
            first_blob = (pack / "content_addressed/objects.bin").read_bytes()
            second = build_content_addressed_bundle(pack)
            self.assertEqual(first, second)
            self.assertEqual(first_blob, (pack / "content_addressed/objects.bin").read_bytes())
            self.assertEqual(2, first["summary"]["deduplicated_bytes"] // 2)
            self.assertEqual(first["summary"], validate_bundle(pack / "content_addressed"))


if __name__ == "__main__":
    unittest.main()

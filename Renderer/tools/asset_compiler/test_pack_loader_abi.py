from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from Renderer.tools.asset_compiler.pack_content_index import build_content_addressed_bundle
from Renderer.tools.asset_compiler.pack_loader_abi import BundleMount, RuntimePackSet, validate_spec


def _pack(root: Path, name: str, files: dict[str, bytes]) -> Path:
    pack = root / name
    pack.mkdir()
    (pack / "manifest.json").write_text(json.dumps({"schema": "fixture", "name": name}), encoding="utf-8")
    for relative, data in files.items():
        path = pack / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    build_content_addressed_bundle(pack)
    return pack / "content_addressed"


class PackLoaderAbiTests(unittest.TestCase):
    def test_partial_override_last_mount_wins_and_base_remains_visible(self) -> None:
        validate_spec()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = _pack(root, "base", {"textures/a.bin": b"base-a", "textures/shared.bin": b"same"})
            override = _pack(root, "override", {"textures/a.bin": b"override-a", "textures/b.bin": b"same"})
            packs = RuntimePackSet([BundleMount(base, "base"), BundleMount(override, "override")])
            self.assertEqual(b"override-a", packs.read("textures/a.bin"))
            self.assertEqual(b"same", packs.read("textures/shared.bin"))
            plan = packs.plan(["textures/shared.bin", "textures/b.bin"], 1024)
            self.assertEqual(1, plan["deduplicated_path_references"])
            self.assertEqual(1, len(packs.load(plan)))

    def test_missing_budget_and_corruption_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            bundle = _pack(Path(directory), "base", {"data.bin": b"123456"})
            packs = RuntimePackSet([BundleMount(bundle, "base")])
            with self.assertRaisesRegex(ValueError, "missing"):
                packs.read("missing.bin")
            with self.assertRaisesRegex(ValueError, "exceeds budget"):
                packs.plan(["data.bin"], 5)
            blob = bundle / "objects.bin"
            blob.write_bytes(b"X" + blob.read_bytes()[1:])
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                BundleMount(bundle, "corrupt")

    def test_unsafe_paths_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            bundle = _pack(Path(directory), "base", {"data.bin": b"ok"})
            packs = RuntimePackSet([BundleMount(bundle, "base")])
            with self.assertRaisesRegex(ValueError, "unsafe component"):
                packs.read("../data.bin")


if __name__ == "__main__":
    unittest.main()

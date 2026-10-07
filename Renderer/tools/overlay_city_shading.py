#!/usr/bin/env python3
"""Overlay the current Lab city shading onto a Renderer64 runtime shader pack.

Renderer64 compiles its shaders from a pinned pack, so Lab edits do not reach
the game by themselves. For every pack shader with a generated counterpart in
the checkout, this replaces one region and keeps every other byte, including
other features' overlays: the city material block, from the
`CityMaterialFrame` constant buffer through the city pixel entry. It carries

- the third material constant (`CityLight`) and the `Q8_CITY_LOOK`,
  `Q8_CITY_EMISSION_LOOK`, `Q8_CITY_LOOK_PALE` and `Q8_CITY_TIME` selections;
- the pack-selected readability response for lit bodies and windows;
- attached flame, smoke and night-light effects on camera-facing quads.

Every addition is identity for a pack whose look and effects are empty, so
earlier city packs render unchanged. Review with --dry-run first. The previous
pack is kept beside it, and `city-overlay.json` records the hashes.
"""
import argparse
import datetime
import difflib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from Renderer.tools.overlay_resource_shading import digest

START = "cbuffer CityMaterialFrame : register(b7)"
END = "\n// The cached 88-byte city vertex retains every consumed source channel."
MARKER = "#define Q8_CITY_TIME"


def region(text, start=START, end=END):
    begin = text.find(start)
    if begin < 0 or text.count(start) != 1:
        return None
    finish = text.find(end, begin + len(start))
    return None if finish < 0 else (begin, finish)


def city_overlay(pack_text, source_text, shown=None):
    """The pack text with the source's city material region, and whether it applied."""
    target, source = region(pack_text), region(source_text)
    if not target or not source:
        return pack_text, False
    old, new = pack_text[target[0]:target[1]], source_text[source[0]:source[1]]
    if old == new or MARKER not in new:
        return pack_text, False
    if shown is not None:
        shown.append("".join(difflib.unified_diff(old.splitlines(keepends=True), new.splitlines(keepends=True),
                                                  "city material (pack)", "city material (checkout)")))
    return pack_text[:target[0]] + new + pack_text[target[1]:], True


def overlay(pack, backup=None, dry_run=False):
    from Renderer.lab.preparation import require_current
    require_current(ROOT)
    pack = pack.resolve()
    changed = {}
    for path in sorted(pack.rglob("*.hlsl")):
        source = ROOT / path.relative_to(pack)
        if not source.is_file():
            continue
        text, shown = path.read_text(), []
        result, applied = city_overlay(text, source.read_text(), shown)
        if applied:
            changed[path] = result
            if dry_run:
                print(f"##### {path.relative_to(pack).as_posix()}\n" + "".join(shown))
    if not changed or dry_run:
        return None
    stamp = datetime.date.today().strftime("%Y%m%d")
    backup = (backup or pack.with_name(pack.name + "-before-cities-" + stamp)).resolve()
    if backup.exists():
        raise ValueError("Backup already exists; choose another: " + str(backup))
    shutil.copytree(pack, backup)
    name = lambda path: path.relative_to(pack).as_posix()
    receipt = pack / "city-overlay.json"
    record = {
        "previous": json.loads(receipt.read_text()) if receipt.exists() else None,
        "source_commit": subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                        capture_output=True, text=True).stdout.strip(),
        "purpose": "Lab city shading (readability look, window shoulder, attached flame/smoke/light "
                   "effects) overlaid on the active runtime pack; other shader code preserved",
        "base": backup.relative_to(ROOT).as_posix(),
        "regions": {name(path): ["city_material"] for path in changed},
        "inputs": {name(path): digest(path.read_bytes()) for path in changed},
    }
    for path, text in changed.items():
        path.write_text(text)
    record["outputs"] = {name(path): digest(path.read_bytes()) for path in changed}
    receipt.write_text(json.dumps(record, indent=2) + "\n")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pack", type=Path, default=ROOT / "Renderer/packs/Renderer64ResidentRuntime")
    parser.add_argument("--backup", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    record = overlay(args.pack, args.backup, args.dry_run)
    if record:
        print(json.dumps(record["regions"], indent=2))


if __name__ == "__main__":
    main()

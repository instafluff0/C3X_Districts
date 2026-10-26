#!/usr/bin/env python3
"""Overlay the site depth basis onto a Renderer64 runtime shader pack.

Renderer64 compiles its shaders from a pinned pack, so Lab edits do not reach
the game by themselves. In VSSharedFeature (render_core/rigid_feature.hlsl),
site pieces (goody huts, barbarian camps, tile buildings: slots 21-28, owner
.18-.188) now take the natural height-depth basis that bridges, tunnels and
farm pieces already use; on the feature basis raised natural ground hid their
lower parts. This inserts only those lines into every pack shader carrying
VSSharedFeature and keeps every other byte. Changed files are kept beside the
pack, and `site-depth-overlay.json` records the hashes.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[2]
SOURCE = "Renderer/native/render_core/rigid_feature.hlsl"
FARM = " bool farm=i.projection.z!=0 && farm_kit_material(i.placement_view.w);\n"
SITE = "bool site=i.projection.z!=0"
BASIS = re.compile(r" if\((bridge(?: \|\| \w+)*)\)\{float h=p\.world\.z\*112-2\.5;\n")


def digest(data):
    return hashlib.sha256(data if isinstance(data, bytes) else data.encode()).hexdigest()


def site_lines(source):
    """The source's site comment and condition, which follow its farm line."""
    start = source.index(FARM) + len(FARM)
    end = source.index(";\n", source.index(SITE, start)) + 2
    return source[start:end]


def apply(text, lines):
    """`text` with the site lines after its farm line and site in its basis test; None if absent or current."""
    if SITE in text or text.count(FARM) != 1:
        return None
    at = text.index(FARM) + len(FARM)
    found = BASIS.match(text, at)
    if not found:
        raise ValueError("VSSharedFeature's natural-basis test moved; update this overlay")
    test = found.group(0).replace(found.group(1), found.group(1) + " || site")
    return text[:at] + lines + test + text[found.end():]


def overlay(pack, backup=None, dry_run=False):
    pack = pack.resolve()
    lines = site_lines((ROOT / SOURCE).read_text())
    changed = {}
    for path in sorted(pack.rglob("*.hlsl")):
        text = path.read_text()
        if "VSSharedFeature" in text:
            new = apply(text, lines)
            if new is not None:
                changed[path] = new
    if not changed or dry_run:
        return {path.relative_to(pack).as_posix(): None for path in changed} if changed else None
    stamp = datetime.date.today().strftime("%Y%m%d")
    backup = (backup or pack.with_name(pack.name + "-before-site-depth-" + stamp)).resolve()
    if backup.exists():
        raise ValueError("Backup already exists; choose another: " + str(backup))
    name = lambda path: path.relative_to(pack).as_posix()
    for path in changed:
        (backup / name(path)).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, backup / name(path))
    record = {
        "source_commit": subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                        capture_output=True, text=True).stdout.strip(),
        "purpose": "Site pieces (slots 21-28, owner .18-.188) take the natural height-depth basis in "
                   "VSSharedFeature; other shader code preserved",
        "source": SOURCE,
        "base": backup.relative_to(ROOT).as_posix() if backup.is_relative_to(ROOT) else backup.name,
        "inputs": {name(path): digest(path.read_bytes()) for path in changed},
    }
    for path, text in changed.items():
        path.write_text(text)
    record["outputs"] = {name(path): digest(path.read_bytes()) for path in changed}
    (pack / "site-depth-overlay.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", type=Path, default=ROOT / "Renderer/packs/Renderer64ResidentRuntime")
    parser.add_argument("--backup", type=Path, help="Where to keep the changed files (default: dated sibling)")
    parser.add_argument("--dry-run", action="store_true", help="Report the shaders that would change")
    args = parser.parse_args()
    record = overlay(args.pack, args.backup, args.dry_run)
    print("Site depth already current" if record is None else
          json.dumps(sorted(record) if args.dry_run else
                     {"changed": sorted(record["outputs"]), "base": record["base"]}, indent=2))


if __name__ == "__main__":
    main()

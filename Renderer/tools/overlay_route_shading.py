#!/usr/bin/env python3
"""Overlay the current Lab route shading onto a Renderer64 runtime shader pack.

Renderer64 compiles its shaders from a pinned pack, so Lab edits do not reach
the game by themselves. This replaces only the route branch (roads and
railroads, surface kind 11) in every pack shader that carries the pack's
active copy of it. Everything else keeps its exact bytes. The previous pack is
kept beside it, and `route-overlay.json` records the hashes.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
SOURCE = "Renderer/native/city_fidelity/hydrology.hlsl"
START = "    if (input.panel > 0.5 && input.surface_kind > 10.5 && input.surface_kind < 11.5)\n    {"
END = "    if (roads_only > 0.5 || railroads_only > 0.5 || resources_only > 0.5 ||"


def digest(data):
    return hashlib.sha256(data if isinstance(data, bytes) else data.encode()).hexdigest()


def branch(text):
    if text.count(START) != 1 or END not in text[text.index(START):]:
        return None
    start = text.index(START)
    return start, text.index(END, start)


def overlay(pack, backup=None):
    from Renderer.lab.preparation import require_current
    require_current(ROOT)
    pack = pack.resolve()
    source = (ROOT / SOURCE).read_text()
    span = branch(source)
    if not span:
        raise ValueError("Generated route branch is missing: " + SOURCE)
    new = source[span[0]:span[1]]
    active = (pack / SOURCE).read_text()
    span = branch(active)
    if not span:
        raise ValueError("Pack lacks the active route branch: " + SOURCE)
    old = active[span[0]:span[1]]
    if old == new:
        return None
    changed = {}
    for path in sorted(pack.rglob("*.hlsl")):
        text = path.read_text()
        span = branch(text)
        if span and text[span[0]:span[1]] == old:
            changed[path] = text[:span[0]] + new + text[span[1]:]
    stamp = datetime.date.today().strftime("%Y%m%d")
    backup = (backup or pack.with_name(pack.name + "-before-routes-" + stamp)).resolve()
    if backup.exists():
        raise ValueError("Backup already exists; choose another: " + str(backup))
    shutil.copytree(pack, backup)
    name = lambda path: path.relative_to(pack).as_posix()
    record = {
        "source_commit": subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                        capture_output=True, text=True).stdout.strip(),
        "purpose": "Lab route shading (pattern roads) overlaid on the active runtime pack; other shader code preserved",
        "source": SOURCE,
        "base": backup.relative_to(ROOT).as_posix(),
        "old_branch_sha256": digest(old),
        "new_branch_sha256": digest(new),
        "inputs": {name(path): digest(path.read_bytes()) for path in changed},
    }
    for path, text in changed.items():
        path.write_text(text)
    record["outputs"] = {name(path): digest(path.read_bytes()) for path in changed}
    (pack / "route-overlay.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", type=Path, default=ROOT / "Renderer/packs/Renderer64ResidentRuntime")
    parser.add_argument("--backup", type=Path, help="Where to keep the previous pack (default: dated sibling)")
    args = parser.parse_args()
    record = overlay(args.pack, args.backup)
    print("Route branch already current" if record is None else
          json.dumps({"changed": sorted(record["outputs"]), "base": record["base"]}, indent=2))


if __name__ == "__main__":
    main()

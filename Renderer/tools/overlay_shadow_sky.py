#!/usr/bin/env python3
"""Overlay shadows that shade the sky fill onto a Renderer64 runtime pack.

Renderer64 compiles its shaders from a pinned pack, so source edits do not
reach the game by themselves. Every shadow should remove noon's share of
open-ground light at every hour (user, 2026-10-27); at night most light is sky
fill, so `EnvironmentState::shadow_strength` now runs to 2 and strength past 1
also shades the ambient term. This applies `render_core/generate_shaders.py`'s
`SHADOW_SKY` lines to every pack shader carrying `frame_illumination` and
keeps every other byte. Changed files are kept beside the pack, and
`shadow-sky-overlay.json` records the hashes.
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
from Renderer.native.render_core.generate_shaders import SHADOW_SKY, shadow_sky  # noqa: E402


def digest(data):
    return hashlib.sha256(data).hexdigest()


def overlay(pack, backup=None, dry_run=False):
    pack = pack.resolve()
    found = {}
    for path in sorted(pack.rglob("*.hlsl")):
        text = path.read_text()
        if "float3 frame_illumination" in text and SHADOW_SKY[0][1] not in text:
            found[path] = shadow_sky(text)
    name = lambda path: path.relative_to(pack).as_posix()
    if not found or dry_run:
        return sorted(name(path) for path in found)
    stamp = datetime.date.today().strftime("%Y%m%d")
    backup = (backup or pack.with_name(pack.name + "-before-shadow-sky-" + stamp)).resolve()
    if backup.exists():
        raise ValueError("Backup already exists; choose another: " + str(backup))
    for path in found:
        (backup / name(path)).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, backup / name(path))
    record = {
        "source_commit": subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                        capture_output=True, text=True).stdout.strip(),
        "purpose": "Shadow strength past 1 shades the sky fill in frame_illumination; cast-shadow alpha "
                   "reads it clamped; other shader code preserved",
        "source": "Renderer/native/render_core/generate_shaders.py SHADOW_SKY",
        "base": backup.relative_to(ROOT).as_posix() if backup.is_relative_to(ROOT) else backup.name,
        "inputs": {name(path): digest(path.read_bytes()) for path in found},
    }
    for path, text in found.items():
        path.write_text(text)
    record["outputs"] = {name(path): digest(path.read_bytes()) for path in found}
    (pack / "shadow-sky-overlay.json").write_text(json.dumps(record, indent=1) + "\n")
    return sorted(record["outputs"])


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("pack", type=Path)
    parser.add_argument("--backup", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    changed = overlay(args.pack, args.backup, args.dry_run)
    print(("would change" if args.dry_run else "changed"), len(changed), "files")
    for path in changed:
        print(" ", path)


if __name__ == "__main__":
    main()

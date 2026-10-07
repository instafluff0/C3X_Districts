#!/usr/bin/env python3
"""Overlay the accepted mountain shading onto a Renderer64 runtime shader pack.

Renderer64 compiles its shaders from a pinned pack, so Lab edits do not reach
the game by themselves. For every pack shader with a generated counterpart in
the checkout, this replaces only the mountain regions and keeps every other
byte, including other features' overlays:

- the accepted rock/snow material constants (`MTN_*`), inserted before shade();
- the mountain material body: Civ III's snow cap (material.y 2..2.5), finer
  rock texture, softer cracks, lighter rock and snow on gentler faces;
- the relief caster's vertex function and mountain clip, so mountains cast
  ground shadows again while hills keep theirs.

Whole regions are copied rather than diff hunks. Review with --dry-run first.
The previous pack is kept beside it, and `mountain-overlay.json` records the
hashes.
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

KNOBS = ("// Accepted 2026-10-07 (mountains Lab", "#define MTN_SNOW_SLOPE_HIGH 0.8\n#endif\n")
KNOBS_BEFORE = "Output shade(P input) {\n"
# (name, start marker, end marker, marker proving the source carries the change)
REGIONS = (
    ("mountain_material", "        float height = input.material.x;\n",
     "        specular_map = lerp(ground_specular, mountain_specular, rock_detail_coverage);\n", "snow_cap"),
    ("caster_vertex", "Pixel VS(Input i) {", "return o; }\n", "o.boundary=i.world.z"),
    ("caster_mountain_clip", "  if(!volcano_body)clip(", ";return i.depth;\n", "i.boundary-.012"),
)


def region(text, start, end):
    begin = text.find(start)
    if begin < 0 or text.count(start) != 1:
        return None
    finish = text.find(end, begin + len(start))
    return None if finish < 0 else (begin, finish + len(end))


def mountain_overlay(pack_text, source_text, shown=None):
    """The pack text with the source's mountain regions, and the regions applied."""
    result, applied = pack_text, []
    knobs = region(source_text, *KNOBS)
    if knobs and KNOBS[0] not in result and result.count(KNOBS_BEFORE) == 1:
        block = source_text[knobs[0]:knobs[1]]
        result = result.replace(KNOBS_BEFORE, block + KNOBS_BEFORE)
        applied.append("material_constants")
    for name, start, end, marker in REGIONS:
        target, source = region(result, start, end), region(source_text, start, end)
        if not target or not source:
            continue
        old, new = result[target[0]:target[1]], source_text[source[0]:source[1]]
        if old == new or marker not in new:
            continue
        result = result[:target[0]] + new + result[target[1]:]
        applied.append(name)
        if shown is not None:
            shown.append("".join(difflib.unified_diff(old.splitlines(keepends=True), new.splitlines(keepends=True),
                                                      name + " (pack)", name + " (checkout)")))
    return result, applied


def overlay(pack, backup=None, dry_run=False):
    from Renderer.lab.preparation import require_current
    require_current(ROOT)
    pack = pack.resolve()
    changed, regions = {}, {}
    for path in sorted(pack.rglob("*.hlsl")):
        if path.name not in ("mountain.hlsl", "source_caster.hlsl"):
            continue
        source = ROOT / path.relative_to(pack)
        if not source.is_file():
            continue
        text, shown = path.read_text(), []
        result, applied = mountain_overlay(text, source.read_text(), shown)
        if result != text:
            changed[path], regions[path] = result, applied
            if dry_run:
                print(f"##### {path.relative_to(pack).as_posix()}: {', '.join(applied)}\n" + "".join(shown))
    if not changed or dry_run:
        return None
    stamp = datetime.date.today().strftime("%Y%m%d")
    backup = (backup or pack.with_name(pack.name + "-before-mountains-" + stamp)).resolve()
    if backup.exists():
        raise ValueError("Backup already exists; choose another: " + str(backup))
    shutil.copytree(pack, backup)
    name = lambda path: path.relative_to(pack).as_posix()
    receipt = pack / "mountain-overlay.json"
    record = {
        "previous": json.loads(receipt.read_text()) if receipt.exists() else None,
        "source_commit": subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                        capture_output=True, text=True).stdout.strip(),
        "purpose": "Accepted mountain material (Civ III snow caps, rock detail) and relief caster "
                   "(mountain ground shadows) overlaid on the active runtime pack; other shader code preserved",
        "base": backup.relative_to(ROOT).as_posix(),
        "regions": {name(path): applied for path, applied in regions.items()},
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

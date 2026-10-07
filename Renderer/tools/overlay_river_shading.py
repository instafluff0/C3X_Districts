#!/usr/bin/env python3
"""Overlay the current Lab river shading onto a Renderer64 runtime shader pack.

Renderer64 compiles its shaders from a pinned pack, so Lab edits do not reach
the game by themselves. For every pack shader with a generated counterpart in
the checkout, this replaces only three river regions and keeps every other
byte, including other features' overlays:

- the `C3X_RIVER_NATURAL_DEPTH` define (natural height-depth basis for rivers);
- `translated_depth`, whose river branch uses that basis and a small bias;
- the `Q3_CONTINUOUS_RIVERS` branch of `q3_water_material` (ocean optics,
  stream lines, banks).

Whole regions are copied rather than diff hunks, so a river edit can never
be split around unrelated nearby code. Review with --dry-run first. The
previous pack is kept beside it, and `river-overlay.json` records the hashes.
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

DEFINE = "#define C3X_RIVER_NATURAL_DEPTH 1\n"
DEFINE_AFTER = "#define Q6_WORLD_SHADOWS 1\n"
# (name, start marker, end marker, marker proving the source carries the change)
REGIONS = (
    ("translated_depth", "float translated_depth(IntegratedVertexInput input, bool feature)\n", "\n}\n",
     "C3X_RIVER_NATURAL_DEPTH"),
    ("river_material", "#ifdef Q3_CONTINUOUS_RIVERS\n  float2 world=q3_source_world(input);",
     "#elif defined(Q3_STATIC_OPTICS_V2)", "stream_lines"),
)


def region(text, start, end):
    begin = text.find(start)
    if begin < 0 or text.count(start) != 1:
        return None
    finish = text.find(end, begin + len(start))
    return None if finish < 0 else (begin, finish + len(end))


def river_overlay(pack_text, source_text, shown=None):
    """The pack text with the source's river regions, and the regions applied."""
    result, applied = pack_text, []
    if DEFINE in source_text and DEFINE not in result and result.count(DEFINE_AFTER) == 1:
        result = result.replace(DEFINE_AFTER, DEFINE_AFTER + DEFINE)
        applied.append("define")
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
        source = ROOT / path.relative_to(pack)
        if not source.is_file():
            continue
        text, shown = path.read_text(), []
        result, applied = river_overlay(text, source.read_text(), shown)
        if result != text:
            changed[path], regions[path] = result, applied
            if dry_run:
                print(f"##### {path.relative_to(pack).as_posix()}: {', '.join(applied)}\n" + "".join(shown))
    if not changed or dry_run:
        return None
    stamp = datetime.date.today().strftime("%Y%m%d")
    backup = (backup or pack.with_name(pack.name + "-before-rivers-" + stamp)).resolve()
    if backup.exists():
        raise ValueError("Backup already exists; choose another: " + str(backup))
    shutil.copytree(pack, backup)
    name = lambda path: path.relative_to(pack).as_posix()
    receipt = pack / "river-overlay.json"
    record = {
        "previous": json.loads(receipt.read_text()) if receipt.exists() else None,
        "source_commit": subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                        capture_output=True, text=True).stdout.strip(),
        "purpose": "Lab river shading (natural depth basis, ocean optics, stream lines, banks) overlaid "
                   "on the active runtime pack; other shader code preserved",
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

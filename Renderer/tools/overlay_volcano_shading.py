#!/usr/bin/env python3
"""Overlay the accepted volcano shading onto a Renderer64 runtime shader pack.

Renderer64 compiles its shaders from a pinned pack, so Lab edits do not reach
the game by themselves. For each pack `mountain.hlsl` with a generated
counterpart in the checkout, this replaces only the volcano regions and keeps
every other byte, including other features' overlays:

- the volcano material block (Civ VI ash at element scale, the rise blend into
  the ground, crater pool and eruption lava);
- the relief rise helper and the volcano share of `mountain_material_weight`;
- the two albedo call sites, which now keep ground below the ash;
- the lava emission added to the lit colour.

Review with --dry-run first. The previous pack is kept beside it, and
`volcano-overlay.json` records the hashes.
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
from Renderer.tools.overlay_mountain_shading import region

MATERIAL_END = "\n#endif\n\nstruct V {"
WEIGHT = ("// Relief surfaces can contain captured volcanoes beside ordinary mountains.\n",
          "#else\n    return 1;\n#endif\n}\n")
RISE = "// Relief rise above the ground, in mountain units.\n"
OLD_CALL = "    albedo = volcano_albedo(albedo, input.volcano_owner, input.world.z);\n"
NEW_CALL = "    albedo = volcano_surface(albedo, ground_albedo, input.volcano_owner, mountain_rise);\n"
EMISSION = "#ifdef BEAUTY_VOLCANO_MATERIAL\n    radiance += volcano_emission(input.volcano_owner, mountain_rise);\n#endif\n"
EMISSION_BEFORE = "    // The relief patch owns its terrain as well as its rock."


def material_block(text):
    """The volcano material between its #ifdef and the vertex struct."""
    anchor = text.find("Texture2D VolcanoColor : register(t69);")
    opening = text.rfind("#ifdef BEAUTY_VOLCANO_MATERIAL\n", 0, anchor)
    finish = text.find(MATERIAL_END, anchor)
    if anchor < 0 or opening < 0 or finish < 0 or text.count("Texture2D VolcanoColor : register(t69);") != 1:
        return None
    return opening + len("#ifdef BEAUTY_VOLCANO_MATERIAL\n"), finish


def volcano_overlay(pack_text, source_text, shown=None):
    """The pack text with the source's volcano regions, and the regions applied."""
    result, applied = pack_text, []

    def replace(name, target, source):
        nonlocal result
        if not target or not source:
            return
        old, new = result[target[0]:target[1]], source_text[source[0]:source[1]]
        if old == new:
            return
        result = result[:target[0]] + new + result[target[1]:]
        applied.append(name)
        if shown is not None:
            shown.append("".join(difflib.unified_diff(old.splitlines(keepends=True), new.splitlines(keepends=True),
                                                      name + " (pack)", name + " (checkout)")))

    if "float3 volcano_surface(" in source_text:
        replace("volcano_material", material_block(result), material_block(source_text))
    weight = region(source_text, RISE, WEIGHT[1])
    if weight and RISE not in result:
        replace("relief_rise_weight", region(result, *WEIGHT), weight)
    if NEW_CALL in source_text and OLD_CALL in result:
        result = result.replace(OLD_CALL, NEW_CALL)
        applied.append("albedo_calls")
    if EMISSION in source_text and EMISSION not in result and result.count(EMISSION_BEFORE) == 1:
        result = result.replace(EMISSION_BEFORE, EMISSION + EMISSION_BEFORE)
        applied.append("lava_emission")
    return result, applied


def overlay(pack, backup=None, dry_run=False, source_root=ROOT):
    pack = pack.resolve()
    changed, regions = {}, {}
    for path in sorted(pack.rglob("mountain.hlsl")):
        source = source_root / path.relative_to(pack)
        if not source.is_file():
            continue
        text, shown = path.read_text(), []
        result, applied = volcano_overlay(text, source.read_text(), shown)
        if result != text:
            changed[path], regions[path] = result, applied
            if dry_run:
                print(f"##### {path.relative_to(pack).as_posix()}: {', '.join(applied)}\n" + "".join(shown))
    if not changed or dry_run:
        return None
    stamp = datetime.date.today().strftime("%Y%m%d")
    backup = (backup or pack.with_name(pack.name + "-before-volcanoes-" + stamp)).resolve()
    if backup.exists():
        raise ValueError("Backup already exists; choose another: " + str(backup))
    shutil.copytree(pack, backup)
    name = lambda path: path.relative_to(pack).as_posix()
    receipt = pack / "volcano-overlay.json"
    record = {
        "previous": json.loads(receipt.read_text()) if receipt.exists() else None,
        "source_commit": subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                        capture_output=True, text=True).stdout.strip(),
        "purpose": "Accepted volcano material (ash blend into ground, crater pool, eruption lava) "
                   "overlaid on the active runtime pack; other shader code preserved",
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
    parser.add_argument("--source-root", type=Path, default=ROOT,
                        help="tree holding the generated shaders (default: the checkout)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.source_root.resolve() == ROOT and not args.dry_run:
        from Renderer.lab.preparation import require_current
        require_current(ROOT)
    record = overlay(args.pack, args.backup, args.dry_run, args.source_root.resolve())
    if record:
        print(json.dumps(record["regions"], indent=2))


if __name__ == "__main__":
    main()

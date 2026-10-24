#!/usr/bin/env python3
"""Overlay the natural-wonder shading onto a Renderer64 runtime shader pack.

Renderer64 compiles its shaders from a pinned pack, so Lab edits do not reach the
game by themselves (docs/lab_material_integration.md). For each pack shader with a
generated counterpart in the checkout, this inserts only the natural-wonder regions
(docs/natural_wonder_rendering.md) and keeps every other byte, other overlays included:

- mountain.hlsl: the region colours of a wonder stamp (region kind in volcano_owner.w);
- the city effect layer (feature, city, rigid_feature, resource_body): the geyser,
  waterfall and spray effects, their kind dispatch, the plume's `keep` option and the
  effect-axis pixel scale;
- hydrology.hlsl: a water wonder's foam collar;
- every shader with the shared receiver query: a wonder lake or channel (marked
  shape_visibility.x 2) keeps a third of the cast shadow.

Seasons are not part of this overlay: reapply them afterwards with
Renderer/tools/season_shading.py. Review with --dry-run first. The previous pack is
kept beside it, and `natural-wonder-overlay.json` records the hashes.
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

# (name, first line of the region, its last line (None: up to the anchor), pack anchor,
# insert "before" or "after" the anchor)
INSERTIONS = (
    ("wonder_regions", "    // Natural wonders (docs/natural_wonder_rendering.md): where no volcano owns the\n",
     "        albedo = lerp(albedo, special, wonder_special);\n    }\n",
     "        albedo = lerp(albedo, graded, rock_albedo_coverage);\n    }\n", "after"),
    ("wonder_effects", "// Natural wonders (docs/natural_wonder_rendering.md): a geyser (94).", None,
     "float4 q8_city_effect(FeaturePixelInput p){", "before"),
    ("wonder_dispatch", " if(kind>3.5 && kind<4.5)return q8_effect_geyser(q,pixel,t,seed,strength,night);\n",
     " if(kind>8.5 && kind<9.5)return q8_effect_spray(q,pixel,t,seed,strength,night);\n",
     " if(kind>2.5){\n  // A volcano's plume (93)", "before"),
    ("wonder_collar", "    // Natural wonders: a water wonder's ribbon (base terrain 2 + seed) is a\n",
     "        return float4(lit*coverage,coverage);\n    }\n",
     "    float seed=input.base_terrain,seed2=input.real_terrain;\n", "before"),
    ("wonder_water_shadow", "  // Natural wonders (docs/natural_wonder_rendering.md): a wonder's lake or fjord channel\n",
     "  if(water && input.shape_visibility.x>1.5)visibility=lerp(1.0,visibility,.35);\n",
     "  return visibility;\n }\n#endif\n return legacy_shadow;\n}\nfloat3 q6_receiver_illumination(PixelInput input,", "before"),
)
# Lines the effects need changed in place: (name, pack text, checkout text).
REPLACEMENTS = (
    ("plume_keep", "float4 shape,float3 look){", "float4 shape,float3 look,bool keep=false){"),
    ("plume_clip", "  clip(alpha-.004);\n  float grey=", "  if(!keep)clip(alpha-.004);\n  float grey="),
    ("effect_pixel", " float2 pixel=float2(max(abs(ddx(p.uv.x)),1e-5),max(abs(ddy(p.uv.y)),1e-5));",
     " float2 pixel=float2(max(length(float2(ddx(p.uv.x),ddy(p.uv.x))),1e-5),\n"
     "                     max(length(float2(ddx(p.uv.y),ddy(p.uv.y))),1e-5));"),
)


def region(text, first, last, anchor):
    start = text.find(first)
    if start < 0 or text.count(first) != 1:
        return None
    if last is None:
        end = text.find(anchor, start)
        return text[start:end] if end >= 0 else None
    end = text.find(last, start)
    return text[start:end + len(last)] if end >= 0 else None


def wonder_overlay(pack_text, source_text, shown=None):
    """The pack text with the checkout's natural-wonder regions, and the regions applied."""
    result, applied = pack_text, []
    for name, old, new in REPLACEMENTS:
        if new in source_text and new not in result and result.count(old) == 1:
            result = result.replace(old, new)
            applied.append(name)
    for name, first, last, anchor, where in INSERTIONS:
        block = region(source_text, first, last, anchor)
        if block is None or first in result or result.count(anchor) != 1:
            continue
        at = result.index(anchor) + (len(anchor) if where == "after" else 0)
        result = result[:at] + block + result[at:]
        applied.append(name)
        if shown is not None:
            shown.append("".join(difflib.unified_diff([], block.splitlines(keepends=True), name, name + " (checkout)")))
    return result, applied


def overlay(pack, backup=None, dry_run=False, source_root=ROOT):
    pack = pack.resolve()
    changed, regions = {}, {}
    for path in sorted(pack.rglob("*.hlsl")):
        source = source_root / path.relative_to(pack)
        if not source.is_file():
            continue
        text, shown = path.read_text(), []
        result, applied = wonder_overlay(text, source.read_text(), shown)
        if result != text:
            changed[path], regions[path] = result, applied
            if dry_run:
                print(f"##### {path.relative_to(pack).as_posix()}: {', '.join(applied)}\n" + "".join(shown))
    if not changed or dry_run:
        return None
    stamp = datetime.date.today().strftime("%Y%m%d")
    backup = (backup or pack.with_name(pack.name + "-before-natural-wonders-" + stamp)).resolve()
    if backup.exists():
        raise ValueError("Backup already exists; choose another: " + str(backup))
    shutil.copytree(pack, backup)
    name = lambda path: path.relative_to(pack).as_posix()
    receipt = pack / "natural-wonder-overlay.json"
    record = {
        "previous": json.loads(receipt.read_text()) if receipt.exists() else None,
        "source_commit": subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                        capture_output=True, text=True).stdout.strip(),
        "purpose": "Natural-wonder region colours, attached effects and foam collars overlaid on the "
                   "active runtime pack; other shader code preserved",
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

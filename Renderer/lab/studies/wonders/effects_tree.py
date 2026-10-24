#!/usr/bin/env python3
"""Private native tree for the wonders' Civ VI effects (Lab only).

A wonder composition's fires, smoke and lamps (effects.py) are drawn in
Renderer64's city effect layer (material 90 + kind) at their anchors. The
production resource pack has no effect records, so this Lab tree reads them
from a text sidecar named by C3X_LAB_WONDER_EFFECTS, one line per effect:

    composition kind du dv lift width height seed strength

du, dv: world offsets from the tile's corner (column, row) the composition
bodies use; lift: height above the ground in body units (tiles of 150/.82).
The checkout is never written (mountains/private_tree.py).

    python3 Renderer/lab/studies/wonders/effects_tree.py prepare
    python3 Renderer/lab/studies/wonders/effects_tree.py build
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from Renderer.lab.studies.mountains import private_tree

NAME = "effects"
private_tree.OUT = ROOT / "Renderer/lab/out/wonders"

PATCHES = {
    # A wonder cell's composition (authority "wonder:<cell name>") stands on the tile
    # whose Lab resource composition has that name; cities never select one.
    "Renderer/native/city_fidelity/compiler.h": [
        ("    if(record.city_id<0)return nullptr;\n",
         "    // Lab only (studies/wonders/city_pack.py): a wonder cell's composition.\n"
         "    if(record.city_id<0){\n"
         "        if(record.resource_id<0)return nullptr;\n"
         "        std::string const wonder=\"wonder:\"+std::string(record.resource_name,\n"
         "            strnlen(record.resource_name,sizeof(record.resource_name)));\n"
         "        for(auto const&t:library.compositions)if(t.authority==wonder)return &t;\n"
         "        return nullptr;\n"
         "    }\n"),
        ("        if(t.culture==culture && t.era==era && t.size==size && t.capital==unsigned(capital) &&",
         "        if(t.authority.rfind(\"wonder:\",0)!=0 && t.culture==culture && t.era==era && t.size==size && t.capital==unsigned(capital) &&"),
        ("        if(t.culture!=culture || t.era!=era || t.size!=size ||",
         "        if(t.authority.rfind(\"wonder:\",0)==0 || t.culture!=culture || t.era!=era || t.size!=size ||")],
    # The Lab library is production plus the catalog's wonders; lamps (7) and blinkers (8)
    # are Lab effect kinds (kits.LAMP_EDIT).
    "Renderer/native/city_fidelity/runtime.h": [
        ("bytes.size()>128u*1024u*1024u", "bytes.size()>512u*1024u*1024u"),
        ("if(e.kind<0 || e.kind>=float(effect_kind_count) || e.kind!=std::floor(e.kind) ||",
         "if(e.kind<0 || (e.kind>=float(effect_kind_count) && e.kind!=7.f && e.kind!=8.f) || e.kind!=std::floor(e.kind) ||")],
    # The wonders' Civ VI textures join production's in the city texture budget.
    "Renderer/native/city_fidelity/gpu.h": [
        ("texture_bytes+bytes.size()>128u*1024u*1024u", "texture_bytes+bytes.size()>1024u*1024u*1024u")],
    # Resource tiles are prepared too: a wonder cell's composition is a city composition.
    "Renderer/native/object_preparation.h": [
        ("    if(!capture.road_mask && !capture.railroad_mask && capture.city_id<0 && !volcano &&",
         "    // Lab only (studies/wonders): a wonder cell's composition stands on a resource tile.\n"
         "    bool const lab_wonder=input.city_ready && capture.resource_id>=0;\n"
         "    if(!capture.road_mask && !capture.railroad_mask && capture.city_id<0 && !volcano && !lab_wonder &&")],
}


def prepare() -> None:
    root = private_tree.tree(NAME)
    private_tree.own(NAME, list(PATCHES))
    for relative in PATCHES:
        (root / relative).parent.mkdir(parents=True, exist_ok=True)
        (root / relative).write_bytes((ROOT / relative).read_bytes())
    private_tree.sync(NAME)
    for relative, edits in PATCHES.items():
        path = root / relative
        text = path.read_text()
        for old, new in edits:
            if text.count(old) != 1:
                raise SystemExit(f"{relative}: patch anchor not found once: {old[:70]!r}")
            text = text.replace(old, new)
        path.write_text(text)
    print("patched", len(PATCHES), "files in", root.relative_to(ROOT))


def binaries() -> tuple[Path, Path]:
    root = private_tree.tree(NAME)
    return root / "Renderer/native/build/candidate/C3XRenderer.dll", root / "Renderer/lab/.cache/native_preview.exe"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("action", choices=("prepare", "build"))
    args = parser.parse_args()
    if args.action == "prepare":
        prepare()
    else:
        private_tree.build(NAME)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

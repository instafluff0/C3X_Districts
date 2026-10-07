#!/usr/bin/env python3
"""Measure Civ III unit sprites into a generic size table for unit packs.

For every PRTO key named in PediaIcons.txt, the unit's INI DEFAULT animation is
read (first frame of each direction; palette indices below 224, i.e. without
shadow, smoke and transparency) and its median silhouette area, height and
width at 128-pixel tiles is recorded, with its lift: the median shadow row
below the lowest body pixel (positive for units drawn flying). Only numbers are
written; no art.

Search order follows Civ III: optional scenario roots first, then Conquests,
Play the World and the base game under the Civ III root. A scenario with its own
units can be measured by passing its folder with --scenario-root.

    python3 -m Renderer.tools.asset_compiler.civ3_unit_sprites
    python3 -m Renderer.tools.asset_compiler.civ3_unit_sprites --scenario-root PATH --output FILE

The Civ III root defaults to $C3X_CIV3_ROOT, else two levels above this checkout.
"""
from __future__ import annotations

import argparse
import configparser
import json
import os
from pathlib import Path

import numpy as np

from Renderer.tools.asset_compiler import civ3_flc

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = ROOT / "Renderer/inventory/civ3_unit_sprite_sizes.json"


def game_roots(civ3_root: Path, scenarios=()):
    return [Path(p) for p in scenarios] + [civ3_root / "Conquests", civ3_root / "civ3PTW", civ3_root]


def pedia_names(roots) -> dict[str, str]:
    names = {}
    for root in reversed(roots):  # later (higher priority) roots override
        text = root / "Text/PediaIcons.txt"
        if not text.is_file():
            continue
        lines = text.read_text(errors="replace").splitlines()
        for i, line in enumerate(lines[:-1]):
            if line.startswith("#ANIMNAME_PRTO_"):
                names[line[len("#ANIMNAME_"):].strip()] = lines[i + 1].strip()
    # Without an #ANIMNAME entry Civ III uses the unit's own folder name.
    for root in roots:
        folder = root / "Art/Units"
        if folder.is_dir():
            for unit in folder.iterdir():
                if unit.is_dir():
                    names.setdefault("PRTO_" + unit.name.replace(" ", "_"), unit.name)
    return names


def default_flc(roots, name: str) -> Path | None:
    for root in roots:
        folder = root / "Art/Units" / name
        if not folder.is_dir():
            continue
        files = {p.name.lower(): p for p in folder.iterdir()}
        ini = next((p for n, p in files.items() if n.endswith(".ini")), None)
        if ini is None:
            continue
        parser = configparser.ConfigParser(strict=False, interpolation=None)
        parser.optionxform = str
        parser.read_string(ini.read_text(errors="replace"))
        flc = parser.get("Animations", "DEFAULT", fallback="").strip()
        if flc and flc.lower() in files:
            return files[flc.lower()]
    return None


def measure(path: Path) -> dict:
    flc = civ3_flc.read(path)
    rows, lifts = [], []
    for direction in range(min(8, flc.directions)):
        frame = flc.direction_frame(direction)
        body = frame < 224
        if body.any():
            ys, xs = np.nonzero(body)
            rows.append((int(body.sum()), int(np.ptp(ys)) + 1, int(np.ptp(xs)) + 1))
            # Shadow palette 240-254: a flying sprite's shadow lies well below
            # its lowest body pixel; a grounded one's surrounds its base.
            shadow = np.nonzero((frame >= 240) & (frame < 255))[0]
            if len(shadow):
                lifts.append(float(np.median(shadow)) - int(ys.max()))
    area, height, width = (float(np.median(column)) for column in zip(*rows))
    return {"area": area, "height": height, "width": width,
            "lift": float(np.median(lifts)) if lifts else 0.0}


def build(civ3_root: Path, scenarios=(), output: Path = DEFAULT_OUTPUT) -> dict:
    roots = game_roots(civ3_root, scenarios)
    units = {}
    for prto, name in sorted(pedia_names(roots).items()):
        path = default_flc(roots, name)
        if path is not None:
            units[prto] = {"art": name, **measure(path)}
    if not units:
        raise ValueError("No Civ III unit sprites found; set C3X_CIV3_ROOT or --civ3-root")
    table = {"schema": 1, "tile_width": 128,
             "method": "DEFAULT FLC first frame per direction; palette indices < 224; median over directions; lift = shadow (240-254) median row - body bottom row",
             "units": units}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(table, indent=1, sort_keys=True) + "\n")
    return table


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--civ3-root", type=Path, default=Path(os.environ.get("C3X_CIV3_ROOT", ROOT.parents[1])))
    parser.add_argument("--scenario-root", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    table = build(args.civ3_root, args.scenario_root, args.output)
    print(f"{len(table['units'])} Civ III unit sprites -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

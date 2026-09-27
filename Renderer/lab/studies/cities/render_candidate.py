#!/usr/bin/env python3
"""Replay a disposable city pack on test.biq terrain and restore the Lab root."""

import argparse
import filecmp
import json
import os
import shutil
from pathlib import Path

from Renderer.lab.studies.cities.build_layouts import ERAS, STYLES
from Renderer.lab.studies.cities.test_biq_gallery import PACK, ROOT, render


WALL_PACK = ROOT / "Renderer/lab/out/cities/test-biq/root/Renderer/packs/CityAdjunctsNormalized/wall_runtime.bin"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-pack", type=Path, required=True)
    parser.add_argument("--candidate-wall-pack", type=Path)
    parser.add_argument("--source-pack", type=Path, required=True)
    parser.add_argument("--extra-source-pack", type=Path, action="append", default=[])
    parser.add_argument("--name-prefix", default="medieval-candidate")
    parser.add_argument("--culture", choices=tuple(s.lower().replace(" ", "_") for s in STYLES),
                        default="mediterranean")
    parser.add_argument("--era", choices=tuple(e.lower() for e in ERAS), default="medieval")
    parser.add_argument("--site", default="20,64", help="Captured test.biq tile column,row")
    parser.add_argument("--variant", action="append",
                        choices=("town-base", "town-capital", "town-both",
                                 "city-base", "city-walls", "city-capital", "city-both",
                                 "metro-base", "metro-capital", "metro-both"),
                        help="Render one variant; repeat for several (default: all)")
    args = parser.parse_args()
    site = tuple(int(value) for value in args.site.split(","))
    if len(site) != 2:
        parser.error("--site must be column,row")
    culture = tuple(s.lower().replace(" ", "_") for s in STYLES).index(args.culture)
    era = tuple(e.lower() for e in ERAS).index(args.era)
    for source_path in [args.source_pack, *args.extra_source_pack]:
        source = source_path.resolve()
        relative = source.relative_to(ROOT)
        target = ROOT / "Renderer/lab/out/cities/test-biq/root" / relative
        for file in source.rglob("*"):
            if not file.is_file():
                continue
            copy = target / file.relative_to(source)
            copy.parent.mkdir(parents=True, exist_ok=True)
            if copy.exists() and filecmp.cmp(file, copy, shallow=False):
                continue
            if copy.exists():
                copy.unlink()
            os.link(file, copy)
    candidate = (args.candidate_pack / "city.bin").read_bytes()
    original = PACK.read_bytes()
    wall_candidate = args.candidate_wall_pack.read_bytes() if args.candidate_wall_pack else None
    wall_original = WALL_PACK.read_bytes() if wall_candidate is not None else None
    records = []
    try:
        if wall_candidate is not None:
            temporary = WALL_PACK.with_suffix(".candidate.tmp")
            temporary.write_bytes(wall_candidate)
            os.replace(temporary, WALL_PACK)
        temporary = PACK.with_suffix(".candidate.tmp")
        temporary.write_bytes(candidate)
        os.replace(temporary, PACK)
        for suffix, size, capital, walls in (
            ("town-base", 0, False, False),
            ("town-capital", 0, True, False),
            ("town-both", 0, True, True),
            ("city-base", 1, False, False),
            ("city-walls", 1, False, True),
            ("city-capital", 1, True, False),
            ("city-both", 1, True, True),
            ("metro-base", 2, False, False),
            ("metro-capital", 2, True, False),
            ("metro-both", 2, True, True),
        ):
            if args.variant and suffix not in args.variant:
                continue
            records.append(render(culture, era, size, site,
                                  args.name_prefix + "-" + suffix,
                                  capital=capital, walls=walls))
    finally:
        temporary = PACK.with_suffix(".restore.tmp")
        temporary.write_bytes(original)
        os.replace(temporary, PACK)
        if wall_original is not None:
            temporary = WALL_PACK.with_suffix(".restore.tmp")
            temporary.write_bytes(wall_original)
            os.replace(temporary, WALL_PACK)
    report = args.candidate_pack.parent / (args.name_prefix + "-replay.json")
    report.write_text(json.dumps(records, indent=2) + "\n")
    print(report)


if __name__ == "__main__":
    main()

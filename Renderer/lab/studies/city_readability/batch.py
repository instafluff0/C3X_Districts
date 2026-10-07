#!/usr/bin/env python3
"""One review batch: a single candidate DLL renders both the production pack and
the readability candidate, so only the city pack differs between the labels.

    python3 Renderer/lab/studies/city_readability/batch.py PACK [--prefix NAME]
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from Renderer import renderer
from Renderer.lab.studies.city_readability import cases as city_cases
from Renderer.lab.studies.city_readability import study

LADDERS = tuple(f"city-ladder-{c}" for c in city_cases.CULTURES)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pack")
    parser.add_argument("--prefix", default="final")
    parser.add_argument("--dll", type=Path, help="reuse a pinned DLL instead of building")
    parser.add_argument("--skip", nargs="*", default=(), help="stages to skip: ladders gameplay night reduced closeups")
    args = parser.parse_args()
    dll = study.OUT / "dll" / f"{args.prefix}.dll"
    dll.parent.mkdir(parents=True, exist_ok=True)
    if args.dll:
        shutil.copy2(args.dll, dll)
    else:
        renderer.ensure_candidate([])
        shutil.copy2(ROOT / "Renderer/native/build/candidate/C3XRenderer.dll", dll)
    before, after = f"before-{args.prefix}", args.prefix
    for label, pack in ((before, ""), (after, args.pack)):
        if "ladders" not in args.skip:
            study.render(label, LADDERS + ("city-gameplay-industrial",), (128,), 12, pack, dll)
        if "night" not in args.skip:
            study.render(label, ("city-gameplay-industrial",), (128,), 0, pack, dll, suffix="-night")
        if "reduced" not in args.skip:
            study.render(label, ("city-gameplay-industrial",), (64,), 12, pack, dll)
    if "closeups" not in args.skip:
        study.closeups([before, after], ["-", args.pack], [str(dll), str(dll)])
    print("BATCH", before, after, renderer.checksum(dll))


if __name__ == "__main__":
    main()

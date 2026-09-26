#!/usr/bin/env python3
"""Replay a disposable city pack on test.biq terrain and restore the Lab root."""

import argparse
import json
import os
import shutil
from pathlib import Path

from Renderer.lab.studies.cities.test_biq_gallery import PACK, ROOT, render


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-pack", type=Path, required=True)
    parser.add_argument("--source-pack", type=Path, required=True)
    parser.add_argument("--extra-source-pack", type=Path, action="append", default=[])
    parser.add_argument("--name-prefix", default="medieval-candidate")
    args = parser.parse_args()
    for source_path in [args.source_pack, *args.extra_source_pack]:
        source = source_path.resolve()
        relative = source.relative_to(ROOT)
        target = ROOT / "Renderer/lab/out/cities/test-biq/root" / relative
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(source, target, copy_function=os.link)
    candidate = (args.candidate_pack / "city.bin").read_bytes()
    original = PACK.read_bytes()
    records = []
    try:
        temporary = PACK.with_suffix(".candidate.tmp")
        temporary.write_bytes(candidate)
        os.replace(temporary, PACK)
        for suffix, size, capital, walls in (
            ("town-base", 0, False, False),
            ("town-both", 0, True, True),
            ("city-both", 1, True, True),
            ("metro-both", 2, True, True),
        ):
            records.append(render(2, 1, size, (20, 64),
                                  args.name_prefix + "-" + suffix,
                                  capital=capital, walls=walls))
    finally:
        temporary = PACK.with_suffix(".restore.tmp")
        temporary.write_bytes(original)
        os.replace(temporary, PACK)
    report = args.candidate_pack.parent / (args.name_prefix + "-replay.json")
    report.write_text(json.dumps(records, indent=2) + "\n")
    print(report)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build the Lab sizing candidate pack from the Civ III size audit.

Each unit's uniform scale is changed so its idle silhouette area (ground-clipped,
median over eight directions) matches Civ III's own DEFAULT sprite times
`--factor`. Units without a Civ III sprite take the median foot-unit change.
Only bindings.json differs from the production pack; clips and textures are
APFS clones (independent copies, no extra disk). Production stays read-only.

    python3 Renderer/lab/studies/unit_readability/audit.py
    python3 Renderer/lab/studies/unit_readability/sizing.py [--factor 1.0]
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
PACKS = ROOT / "Renderer/packs"
AUDIT = ROOT / "Renderer/lab/out/unit-readability/size_audit.json"


def clone_tree(source: Path, output: Path) -> None:
    if output.exists():
        shutil.rmtree(output)
    # APFS clone; fall back to an ordinary copy on other file systems.
    if subprocess.run(["cp", "-cR", str(source), str(output)]).returncode:
        shutil.copytree(source, output)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", default="UnitAnimationFidelity")
    parser.add_argument("--output", default="UnitSizingLab")
    parser.add_argument("--factor", type=float, default=1.0, help="linear size relative to Civ III")
    args = parser.parse_args(argv)
    source, output = PACKS / args.source, PACKS / args.output
    if output.resolve() == source.resolve() or not args.output.endswith("Lab"):
        raise SystemExit("output must be a separate *Lab pack")
    audit = {r["binding"]: r for r in json.loads(AUDIT.read_text())}
    changes = {k: math.sqrt(r["civ3"]["area"] / r["ours"]["area"]) for k, r in audit.items() if r["civ3"]}
    foot = sorted(changes[k] for k, r in audit.items() if r["civ3"] and r["civ3"]["width"] < 36)
    fallback = foot[len(foot) // 2]
    clone_tree(source, output)
    bindings = json.loads((source / "bindings.json").read_text())
    report = {}
    for key, unit in bindings.items():
        if not isinstance(unit, dict) or "scale" not in unit:
            continue
        change = changes.get(key, fallback if audit.get(key) and "Leader" in audit[key]["prto"][0] else 1.0) * args.factor
        report[unit.get("key0", key)] = {"old": unit["scale"], "new": unit["scale"] * change, "change": change}
        unit["scale"] *= change
        unit["fit_policy"] = f"lab_civ3_sprite_area_x{args.factor:g}"
    (output / "bindings.json").write_text(json.dumps(bindings, indent=2, sort_keys=True) + "\n")
    (output / "sizing_lab.json").write_text(json.dumps({"source": args.source, "factor": args.factor,
                                                        "foot_fallback": fallback, "units": report}, indent=1))
    print(f"{output.name}: {len(report)} units, factor {args.factor}, foot fallback {fallback:.3f}")
    for name, row in sorted(report.items(), key=lambda item: item[1]["change"]):
        print(f"  {name[5:]:22} x{row['change']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

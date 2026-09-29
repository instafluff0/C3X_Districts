"""Refresh the carrier's source tangent frames in the local production frame pack."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from Renderer.lab.platform import ROOT
from Renderer.lab.studies.units.frame_probe import recover


STAGING = ROOT / "Renderer/lab/out/units/settler-carrier/frame-refresh"
FRAME_PACK = ROOT / "Renderer/packs/UnitFrameFidelity"
PREFIX = "unit/settler/"
EXPECTED = {PREFIX + role for role in ("backpack", "armor", "body", "head")}


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    recovered = recover(subjects=[("settler", "UNIT_SETTLER", 2)], output=STAGING)
    if set(recovered["components"]) != EXPECTED:
        raise ValueError("Carrier source component inventory changed")
    source_manifest = read(STAGING / "manifest.json")
    manifest = read(FRAME_PACK / "manifest.json")
    frames = read(FRAME_PACK / "frames.json")
    evidence = read(FRAME_PACK / "source-build.json")
    replacement = read(STAGING / "source-build.json")
    for catalog, field, source in ((manifest, "assets", source_manifest),
                                   (frames, "components", recovered)):
        catalog[field] = {key: value for key, value in catalog[field].items()
                          if not key.startswith(PREFIX)}
        catalog[field].update(source[field])
    evidence["components"] = [row for row in evidence["components"]
                              if not row["asset_id"].startswith(PREFIX)]
    evidence["components"].extend(replacement["components"])
    for path in STAGING.rglob("*"):
        if path.is_file() and path.name not in {"manifest.json", "frames.json", "source-build.json"}:
            destination = FRAME_PACK / path.relative_to(STAGING)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
    write(FRAME_PACK / "manifest.json", manifest)
    write(FRAME_PACK / "frames.json", frames)
    write(FRAME_PACK / "source-build.json", evidence)
    print("Refreshed four carrier tangent frames; other unit frames preserved")


if __name__ == "__main__":
    main()

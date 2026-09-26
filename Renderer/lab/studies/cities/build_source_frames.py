#!/usr/bin/env python3
"""Collect verified source frames for every palace used by the Lab sheets."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
STUDY = ROOT / "Renderer/lab/studies/cities"
OUTPUT = ROOT / "Renderer/lab/out/cities/source-frames"
PALACES = ROOT / "Renderer/packs/CityPalacesNormalized"
SOURCE = ROOT / "Renderer/packs/CityFidelitySources/current"


def build() -> Path:
    layouts = json.loads((STUDY / "layouts.json").read_text())
    manifest = json.loads((PALACES / "manifest.json").read_text())
    frames = {}
    for name in ("palace-normals.json", "asian-ancient-palace-normals.json"):
        frames.update(json.loads((SOURCE / name).read_text())["meshes"])
    asian = ROOT / "Renderer/lab/out/cities/upstream-ancientwood/source-frames.json"
    frames.update(json.loads(asian.read_text())["meshes"])
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for asset in sorted({design["palace"]["asset"] for design in layouts["designs"]}):
        landmark = json.loads((PALACES / manifest["assets"][asset]["landmark"]).read_text())
        geometry = [json.loads((PALACES / name).read_text())["asset_id"]
                    for name in landmark["components"]["geometry"]]
        if all(name in frames for name in geometry):
            continue
        output = OUTPUT / ("palace-" + asset.rsplit("/", 1)[-1] + ".json")
        if not output.exists():
            subprocess.run([sys.executable,
                            str(ROOT / "Renderer/lab/shared/cities/prepare_normals.py"),
                            "--palace", asset, "--include-frame", "--output", str(output)],
                           cwd=ROOT, check=True)
        recovered = json.loads(output.read_text())["meshes"]
        if any(name not in recovered and name not in frames for name in geometry):
            raise ValueError(f"incomplete palace source frame: {asset}")
        frames.update(recovered)
    output = OUTPUT / "all-city-source-frames.json"
    output.write_text(json.dumps({"schema": "c3x.lab.city_source_frames.v1",
                                  "meshes": frames}, separators=(",", ":")) + "\n")
    print(f"Wrote {len(frames)} verified source meshes to {output}")
    return output


if __name__ == "__main__":
    build()

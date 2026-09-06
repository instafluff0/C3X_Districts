#!/usr/bin/env python3
"""Write a minimal CivNexus6 model companion for model-aware clip sampling."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from Renderer.tools.asset_compiler import normalized_skin


def _number(value: float) -> str:
    if not math.isfinite(value):
        raise ValueError("CN6 companion contains a non-finite value")
    # CivNexus6's legacy CN6 parser accepts decimals but not exponent notation.
    if abs(value) < 5.0e-10:
        value = 0.0
    text = f"{value:.9f}".rstrip("0").rstrip(".")
    return text if text and text != "-0" else "0"


def write_model_companion(
    skeleton_path: Path, output: Path, units_per_tile: float = 100.0
) -> dict[str, object]:
    if not math.isfinite(units_per_tile) or units_per_tile <= 0.0:
        raise ValueError("units_per_tile must be positive and finite")
    skeleton = normalized_skin.load_skeleton(skeleton_path)
    lines = ["// C3X normalized skeleton companion", "skeleton"]
    for index, bone in enumerate(skeleton["bones"]):
        local = bone["local"]
        position = [value * units_per_tile for value in local["position"]]
        inverse_bind = list(bone["inverse_bind_matrix"])
        for component in (12, 13, 14):
            inverse_bind[component] *= units_per_tile
        values = position + local["orientation"] + inverse_bind
        name = bone["name"]
        if '"' in name or "\n" in name or "\r" in name:
            raise ValueError(f"bone {index} has a CN6-unsafe name")
        lines.append(
            f'{index} "{name}" {bone["parent"]} '
            + " ".join(_number(value) for value in values)
        )

    # CivNexus's CN6 importer requires a mesh. This three-vertex root-weighted
    # placeholder is deliberately unrelated to the runtime mesh; only the
    # reconstructed model/skeleton is used by Granny's animation sampler.
    lines.extend(["meshes:1", 'mesh:"C3X_Sampling_Companion"', "materials", '"ResourceMaterial"', "vertices"])
    for position in ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)):
        values = (
            list(position)
            + [0.0, 0.0, 1.0]  # normal
            + [1.0, 0.0, 0.0]  # tangent
            + [0.0, 1.0, 0.0]  # binormal
            + [0.0, 0.0] * 3  # uv0, uv1, uv2
            + [0] * 8  # skeleton joint indices
            + [255] + [0] * 7  # byte weights
        )
        lines.append(" ".join(_number(float(value)) for value in values))
    lines.extend(["triangles", "0 1 2 0", "end"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "schema": "c3x.granny_model_companion.v0",
        "skeleton": str(skeleton_path),
        "output": str(output),
        "bones": len(skeleton["bones"]),
        "units_per_tile": units_per_tile,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("skeleton", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--units-per-tile", type=float, default=100.0)
    args = parser.parse_args(argv)
    try:
        report = write_model_companion(args.skeleton, args.output, args.units_per_tile)
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(
        f"Wrote {report['output']} ({report['bones']} bones; "
        f"{report['units_per_tile']:g} units/tile)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

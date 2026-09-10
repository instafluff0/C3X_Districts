"""Summarize verified standalone workloads without claiming native presentation.

Optional paired comparisons require identical binaries, runtime inputs, cameras,
and quality settings. Only explicit cache-control switches may differ.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

from Renderer.native.compare_zoom_benchmark import pixels, run


def fields(line):
    return dict(re.findall(r"(\w+)=([^ ]+)", line))


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def distribution(values):
    values = sorted(values)
    if not values or any(not math.isfinite(v) or v < 0 for v in values):
        raise ValueError("Missing, negative or nonfinite measurements")
    return {"samples": len(values), "median_ms": statistics.median(values),
            "p95_ms": values[math.ceil(.95 * len(values)) - 1],
            "p99_ms": values[math.ceil(.99 * len(values)) - 1], "max_ms": values[-1],
            "over_33_4_ms": sum(v > 33.4 for v in values), "over_100_ms": sum(v > 100 for v in values),
            "hundred_sample_requirement_met": len(values) >= 100}


def inspect(directory):
    receipt = json.loads((directory / "inputs.json").read_text())
    completion = json.loads((directory / "evidence.json").read_text())
    if ((directory / "completion.txt").read_text().strip() != "0" or
            completion.get("returncode") != 0 or completion.get("invocation") != receipt.get("invocation") or
            completion.get("inputs_unchanged") is not True or completion.get("binaries_unchanged") is not True):
        raise ValueError("Unverified invocation or modified inputs/binaries")
    if set(receipt["binaries"]) != {"C3XRenderer.dll", "biq_preview.exe"}:
        raise ValueError("Incomplete binary identities")
    for name, expected in receipt["binaries"].items():
        if digest(directory / name) != expected:
            raise ValueError("Binary changed after verification")
    args = receipt["args"]
    env = receipt.get("environment", {})
    if (receipt.get("quality_mode", "current") != "current" or args.get("region_diagnostics") or args.get("reflection_ablation") or
            env.get("C3X_RENDERER_REGION_DIAGNOSTICS") == "1" or env.get("C3X_RENDERER_REFLECTION_CONTROL") == "1"):
        raise ValueError("Diagnostic ablation/logging is not performance evidence")
    lines = (directory / "benchmark.log").read_text().splitlines()
    if not lines or not lines[-1].startswith("BIQ ") or "0 fallback" not in lines[-1]:
        raise ValueError("Missing completed zero-fallback witness")
    scenario = args["scenario"]
    if scenario == "navigation" and args.get("resident"):
        run("resident warmup", directory, "navigation")
        prefix, end = "RESIDENT_NAV ", "RESIDENT_END status=pass"
        count = int(args["resident_steps"])
        image_names = [f"zoom.bmp.resident{i}.bmp" for i in range(count)]
    elif scenario == "distant":
        prefix, end = "DISTANT_NAV ", "DISTANT_END status=pass"
        count = int(args["distant_steps"])
        image_names = [f"zoom.bmp.distant{i}.bmp" for i in range(count)]
    elif scenario == "zoom":
        _, all_rows = run("zoom", directory, "zoom")
        prefix, end = "ZOOM cycle=", None
        count = len(all_rows)
        image_names = [f"zoom.bmp.z{r['width']}.bmp" for r in all_rows if r["cycle"] == "0"]
    else:
        raise ValueError("Select a resident pan, zoom, or distant sweep")
    rows = [fields(line) for line in lines if line.startswith(prefix)]
    if count < 1 or len(rows) != count or (end and lines.count(end) != 1) or any(r["result"] != "1" for r in rows):
        raise ValueError("Incomplete or failed measured sweep")
    if scenario != "zoom" and [int(r["step"]) for r in rows] != list(range(count)):
        raise ValueError("Reordered or duplicated samples")
    if scenario == "navigation":
        tile_height = int(args["tile_width"]) // 2
        for i, row in enumerate(rows):
            pixel_y = (i + 1) * tile_height if count == 14 else (i + 1) * 15 * tile_height // (count + 1)
            if (int(row["x"]), int(row["y"]), int(row.get("pixel_y", pixel_y))) != (35, 39 + pixel_y // tile_height * 2, pixel_y):
                raise ValueError("Unexpected resident camera sequence")
    if scenario == "distant" and len({(r["x"], r["y"]) for r in rows}) != count:
        raise ValueError("Distant sweep repeats destinations")
    images = {}
    for name in image_names:
        if "images" in completion and completion["images"].get(name) != digest(directory / name):
            raise ValueError("Image changed after verification")
        size, body = pixels(directory / name)
        if size != (int(args["width"]), -int(args["height"])):
            raise ValueError("Image dimensions differ from measured viewport")
        images[name] = hashlib.sha256(body).hexdigest()
    measured = rows[1:] if scenario == "zoom" else rows
    report = {"endpoint": "standalone capture plus completed render; no native presentation",
              "scenario": scenario, "viewport": [args["width"], args["height"]],
              "waves": args["waves"], "clock": "fixed replay clock; changing animation is a separate witness",
              "binaries": receipt["binaries"], "images": images,
              "image_receipt_verified": "images" in completion,
              "camera_requests": [{k: r[k] for k in ("cycle", "step", "x", "y", "pixel_y", "width") if k in r} for r in rows],
              "timing": {k: distribution([float(r[k]) for r in measured])
                         for k in ("ms", "capture_ms", "geometry_ms", "draw_ms", "readback_ms")
                         if all(k in r for r in measured)},
              "mesh_builds": sum(int(r["built"]) for r in measured),
              "upload_bytes": sum(int(r["upload_bytes"]) for r in measured) if all("upload_bytes" in r for r in measured) else None,
              "native_first_response_ms": None, "native_presented_frames": None,
              "map_prepared_before_sweep": False if scenario == "distant" else None}
    if scenario == "zoom":
        report["warm_zoom"] = distribution([float(r["ms"]) for r in rows if int(r["cycle"]) > 0])
        report["first_zoom_changes"] = distribution([float(r["ms"]) for r in rows[1:] if r["cycle"] == "0"])
    memory = [fields(l) for l in lines if l.startswith("CAMERA memory ")]
    if memory:
        report["sampled_address_space_not_peak"] = {k: min(int(r[k]) for r in memory)
                                                   for k in ("available_virtual", "largest_free_region")}
    trace = (directory / "renderer.log").read_text().splitlines()
    phases = [fields(l) for l in trace if "stage=navigation-phases " in l][-count:]
    if len(phases) == count:
        report["cpu_phases"] = {k: distribution([float(r[k]) for r in phases])
                                for k in phases[0] if k.endswith("_ms") and all(k in r for r in phases)}
        report["retained_cpu_bytes"] = {k: max(int(r[k]) for r in phases)
                                        for k in ("shadow_proof_bytes", "contributor_index_bytes") if k in phases[0]}
    return receipt, report


def compare(reference, candidate):
    before, old = inspect(reference)
    after, new = inspect(candidate)
    if before["binaries"] != after["binaries"] or before["inputs"] != after["inputs"]:
        raise ValueError("Paired evidence requires identical binaries and runtime inputs")
    controls = {"dependency_control", "index_control", "center_shore_control", "world_regions_control",
                "backdrop_control", "wave_control", "composition_casters_control"}
    ignored = controls | {"out", "binaries"}
    if {k: v for k, v in before["args"].items() if k not in ignored} != {k: v for k, v in after["args"].items() if k not in ignored}:
        raise ValueError("Paired cameras or quality settings differ")
    env_controls = {"C3X_RENDERER_REGION_DEPENDENCY_CONTROL", "C3X_RENDERER_REGION_INDEX_CONTROL",
                    "C3X_RENDERER_CENTER_SHORE_CONTROL", "C3X_RENDERER_WORLD_REGIONS_CONTROL",
                    "C3X_RENDERER_BACKDROP_REUSE_CONTROL", "C3X_RENDERER_WAVE_REUSE_CONTROL",
                    "C3X_RENDERER_COMPOSITION_CASTERS_CONTROL", "C3X_RENDERER_TRACE_FILE"}
    if ({k: v for k, v in before.get("environment", {}).items() if k not in env_controls} !=
            {k: v for k, v in after.get("environment", {}).items() if k not in env_controls} or
            old["camera_requests"] != new["camera_requests"]):
        raise ValueError("Paired environment or camera requests differ")
    return {"reference": old, "candidate": new, "all_images_exact": old["images"] == new["images"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.reference, args.candidate) if args.reference else inspect(args.candidate)[1]
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"report": str(args.out), "all_images_exact": result.get("all_images_exact"),
                      "timing": result.get("candidate", result)["timing"]["ms"]}))
    return 1 if result.get("all_images_exact") is False else 0


if __name__ == "__main__":
    raise SystemExit(main())

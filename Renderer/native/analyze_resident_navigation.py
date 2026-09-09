"""Verify a retained-camera sweep against independent cold-render pixels.

Read-only. This fourteen-view test does not certify arbitrary-map navigation
or native Civ III input-to-display latency. Synthetic unit tests are not runs.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import struct

from Renderer.native.compare_zoom_benchmark import pixels, run


def fields(line):
    return dict(re.findall(r"(\w+)=([^ ]+)", line))


def checksum(body):
    value = 14695981039346656037
    for byte in body:
        value = ((value ^ byte) * 1099511628211) & ((1 << 64) - 1)
    return value


def read_sweep(directory, mode):
    if (directory / "completion.txt").read_text().strip() != "0":
        raise ValueError("Missing successful process exit receipt")
    # Require the full warmup and its revisit comparisons, not a partial log.
    run(mode, directory, "navigation")
    lines = (directory / "benchmark.log").read_text().splitlines()
    starts = [fields(line) for line in lines if line.startswith("RESIDENT_BEGIN ")]
    ends = [line for line in lines if line.startswith("RESIDENT_END ")]
    rows = [fields(line) for line in lines if line.startswith("RESIDENT_NAV ")]
    images = [fields(line) for line in lines if line.startswith("RESIDENT_IMAGE ")]
    if len(starts) != 1 or ends != ["RESIDENT_END status=pass"] or len(rows) != 14 or len(images) != 14:
        raise ValueError("Incomplete or duplicate resident sweep")
    resident_lines = [line.split(" ", 1)[0] for line in lines if line.startswith("RESIDENT_")]
    if resident_lines != ["RESIDENT_BEGIN"] + [item for _ in range(14) for item in ("RESIDENT_NAV", "RESIDENT_IMAGE")] + ["RESIDENT_END"]:
        raise ValueError("Resident evidence is out of order")
    header = starts[0]
    if header["mode"] != mode or int(header["steps"]) != 14:
        raise ValueError("Expected independent cold reference and retained candidate")
    dimensions = (int(header["width"]), int(header["height"]))
    if min(dimensions) <= 0 or int(header["tile_width"]) <= 0:
        raise ValueError("Invalid viewport")
    samples = []
    for step, row in enumerate(rows):
        if (int(row["step"]), int(row["x"]), int(row["y"]), int(row["result"])) != (step, 35, 41 + step * 2, 1):
            raise ValueError("Unexpected camera sequence or failed draw")
        sample = {name: int(row[name]) for name in ("built", "reused", "upload_bytes")}
        sample.update({name: float(row[name]) for name in ("ms", "capture_ms", "geometry_ms", "draw_ms", "readback_ms")})
        if any(not math.isfinite(value) or value < 0 for value in sample.values()):
            raise ValueError("Negative/nonfinite counters or timings")
        if sample["ms"] <= 0 or sample["capture_ms"] > sample["ms"]:
            raise ValueError("Invalid end-to-end timing")
        try:
            size, body = pixels(directory / f"zoom.bmp.resident{step}.bmp")
        except struct.error as error:
            raise ValueError("Truncated resident image header") from error
        if size != (dimensions[0], -dimensions[1]):
            raise ValueError("Rendered image size differs from the measured viewport")
        image = images[step]
        if (int(image["step"]), int(image["saved"]), int(image["bytes"]), int(image["fnv64"])) != (step, 1, len(body), checksum(body)):
            raise ValueError("Image does not match the completed draw's pixel receipt")
        samples.append((sample, body))
    return header, samples


def analyze(reference, candidate, max_ms=100.0):
    if not math.isfinite(max_ms) or max_ms <= 0:
        raise ValueError("Latency target must be positive and finite")
    cold_header, cold = read_sweep(reference, "cold")
    warm_header, warm = read_sweep(candidate, "retained")
    for field in ("width", "height", "tile_width", "steps"):
        if cold_header[field] != warm_header[field]:
            raise ValueError("Reference and candidate camera definitions differ")
    cold_dll = hashlib.sha256((reference / "C3XRenderer.dll").read_bytes()).hexdigest()
    warm_dll = hashlib.sha256((candidate / "C3XRenderer.dll").read_bytes()).hexdigest()
    if cold_dll != warm_dll:
        raise ValueError("Residency proof requires the same renderer DLL")
    results = []
    for step, ((cold_row, expected), (warm_row, actual)) in enumerate(zip(cold, warm)):
        if not cold_row["built"] or not cold_row["upload_bytes"]:
            raise ValueError("Cold reference did not demonstrate fresh geometry preparation")
        results.append({"step": step, "x": 35, "y": 41 + step * 2, **warm_row,
                        "pixel_exact": actual == expected,
                        "image_sha256": hashlib.sha256(actual).hexdigest(),
                        "no_rebuild_or_upload": warm_row["built"] == 0 and warm_row["upload_bytes"] == 0})
    timings = sorted(row["ms"] for row in results)
    exact = all(row["pixel_exact"] for row in results)
    resident = all(row["no_rebuild_or_upload"] for row in results)
    responsive = max(timings) <= max_ms
    return {"scope": "14 new views between warmed regions; not arbitrary-map or live-game proof",
            "dll_sha256": warm_dll, "viewport": [int(warm_header["width"]), int(warm_header["height"])],
            "steps": results, "pixels_exact": exact, "no_rebuilds_or_uploads": resident,
            "latency_target_ms": max_ms, "latency_target_met": responsive,
            "median_ms": statistics.median(timings), "max_ms": max(timings),
            "pass": exact and resident and responsive}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--max-ms", type=float, default=100.0)
    args = parser.parse_args()
    try:
        result = analyze(args.reference, args.candidate, args.max_ms)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(2, f"Resident navigation evidence rejected: {error}\n")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["pass"] else 1)


if __name__ == "__main__":
    main()

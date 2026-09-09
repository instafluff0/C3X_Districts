"""Verify a retained-camera sweep against independent cold-render pixels.

Read-only. This bounded resident-view test does not certify arbitrary-map navigation
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
    if len(starts) != 1:
        raise ValueError("Incomplete or duplicate resident sweep")
    count = int(starts[0]["steps"])
    if not 14 <= count <= 1000 or ends != ["RESIDENT_END status=pass"] or len(rows) != count or len(images) != count:
        raise ValueError("Incomplete or duplicate resident sweep")
    resident_lines = [line.split(" ", 1)[0] for line in lines if line.startswith("RESIDENT_")]
    if resident_lines != ["RESIDENT_BEGIN"] + [item for _ in range(count) for item in ("RESIDENT_NAV", "RESIDENT_IMAGE")] + ["RESIDENT_END"]:
        raise ValueError("Resident evidence is out of order")
    header = starts[0]
    if header["mode"] != mode:
        raise ValueError("Expected independent cold reference and retained candidate")
    dimensions = (int(header["width"]), int(header["height"]))
    if min(dimensions) <= 0 or int(header["tile_width"]) < 2:
        raise ValueError("Invalid viewport")
    samples = []
    pattern = header.get("pattern", "tile-v1")
    tile_height = int(header["tile_width"]) // 2
    if (pattern == "tile-v1" and count != 14) or pattern not in ("tile-v1", "pixel-v1") or (pattern == "pixel-v1" and count >= 15*tile_height):
        raise ValueError("Unknown or repeated camera pattern")
    for step, row in enumerate(rows):
        pixel_y = (step+1)*tile_height if pattern == "tile-v1" else (step+1)*15*tile_height//(count+1)
        y = 39 + (pixel_y//tile_height)*2
        if pattern == "pixel-v1" and int(row["pixel_y"]) != pixel_y:
            raise ValueError("Unexpected pixel camera")
        if (int(row["step"]), int(row["x"]), int(row["y"]), int(row["result"])) != (step, 35, y, 1):
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
        sample.update(x=35, y=y, pixel_y=pixel_y)
        for name in ("reused_pixels", "draw_pixels", "cached_pixels"):
            if name in row:
                sample[name] = int(row[name])
                if sample[name] < 0:
                    raise ValueError("Negative pixel reuse counter")
        # Keep only a checked content digest, never a growing array of images.
        samples.append((sample, hashlib.sha256(body).hexdigest()))
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
    if cold_header.get("pattern", "tile-v1") != warm_header.get("pattern", "tile-v1"):
        raise ValueError("Reference and candidate camera patterns differ")
    evidence_present = [(p / "inputs.json").is_file() for p in (reference, candidate)]
    if any(evidence_present):
        if not all(evidence_present):
            raise ValueError("Both runs need matched runtime receipts")
        receipts = [json.loads((p / "inputs.json").read_text()) for p in (reference, candidate)]
        for receipt in receipts:
            if (receipt.get("args", {}).get("region_diagnostics") or
                receipt.get("environment", {}).get("C3X_RENDERER_REGION_DIAGNOSTICS") == "1"):
                raise ValueError("Per-region diagnostic logging cannot establish performance")
            if (receipt.get("quality_mode", "current") != "current" or
                receipt.get("args", {}).get("reflection_ablation") or
                receipt.get("environment", {}).get("C3X_RENDERER_REFLECTION_CONTROL") == "1"):
                raise ValueError("Diagnostic pass ablation cannot establish current-quality performance")
        for p, receipt in zip((reference, candidate), receipts):
            completion = json.loads((p / "evidence.json").read_text())
            if completion["invocation"] != receipt["invocation"] or completion["returncode"] != 0 or not completion["inputs_unchanged"]:
                raise ValueError("Invocation or runtime inputs were not verified")
            if set(receipt["binaries"])!={"C3XRenderer.dll", "biq_preview.exe"}:
                raise ValueError("Incomplete binary receipt")
            for name, expected in receipt["binaries"].items():
                if name not in ("C3XRenderer.dll", "biq_preview.exe") or hashlib.sha256((p / name).read_bytes()).hexdigest()!=expected:
                    raise ValueError("Recorded binary identity changed")
        if receipts[0]["inputs"] != receipts[1]["inputs"]:
            raise ValueError("Runtime inputs differ between runs")
        for name in ("scenario", "width", "height", "waves", "resident_steps", "tile_width", "region_size", "world_grid"):
            if receipts[0]["args"].get(name)!=receipts[1]["args"].get(name):
                raise ValueError("Scene or quality settings differ: " + name)
        for name in ("C3X_RENDERER_PREVIEW_SEASON", "C3X_RENDERER_PREVIEW_ANIMATION"):
            if receipts[0]["environment"].get(name)!=receipts[1]["environment"].get(name):
                raise ValueError("Captured environment differs: " + name)
        if bool(receipts[0]["args"].get("world_backdrops"))!=bool(receipts[1]["args"].get("world_backdrops")):
            raise ValueError("Animation raster grid differs")
        if bool(receipts[0]["args"].get("world_waves"))!=bool(receipts[1]["args"].get("world_waves")):
            raise ValueError("Wave projection differs")
        if bool(receipts[0]["args"].get("world_regions"))!=bool(receipts[1]["args"].get("world_regions")):
            raise ValueError("Completed-region raster recipe differs")
        if receipts[0]["binaries"].get("biq_preview.exe")!=receipts[1]["binaries"].get("biq_preview.exe"):
            raise ValueError("Benchmark executables differ")
    results = []
    for step, ((cold_row, expected), (warm_row, actual)) in enumerate(zip(cold, warm)):
        if not cold_row["built"] or not cold_row["upload_bytes"]:
            raise ValueError("Cold reference did not demonstrate fresh geometry preparation")
        results.append({"step": step, **warm_row,
                        "pixel_exact": actual == expected,
                        "image_sha256": actual,
                        "no_rebuild_or_upload": warm_row["built"] == 0 and warm_row["upload_bytes"] == 0})
    timings = sorted(row["ms"] for row in results)
    exact = all(row["pixel_exact"] for row in results)
    resident = all(row["no_rebuild_or_upload"] for row in results)
    responsive = max(timings) <= max_ms
    percentile = lambda p: timings[math.ceil(len(timings)*p)-1]
    return {"scope": f"{len(results)} new views between warmed regions; not arbitrary-map or live-game proof",
            "dll_sha256": warm_dll, "viewport": [int(warm_header["width"]), int(warm_header["height"])],
            "steps": results, "pixels_exact": exact, "no_rebuilds_or_uploads": resident,
            "latency_target_ms": max_ms, "latency_target_met": responsive,
            "median_ms": statistics.median(timings), "max_ms": max(timings),
            "p95_ms": percentile(.95), "p99_ms": percentile(.99),
            "discrete_sample_requirement_met": len(results) >= 100,
            "over_33_4_ms": sum(t > 33.4 for t in timings), "over_100_ms": sum(t > 100 for t in timings),
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

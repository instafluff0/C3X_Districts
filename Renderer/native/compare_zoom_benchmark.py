"""Compare isolated zoom runs, including pixels, timing, and exact DLL hashes."""
from pathlib import Path
import argparse
import hashlib
import json
import math
import re
import statistics
import struct

ROOT = Path(__file__).resolve().parent


def run(label, directory, scenario):
    prefix = "NAV" if scenario == "navigation" else "ZOOM"
    steps = 6 if scenario == "navigation" else 5
    lines = (directory / "benchmark.log").read_text().splitlines()
    rows = [dict(re.findall(r"(\w+)=([^ ]+)", line)) for line in lines if line.startswith(prefix + " cycle=")]
    cycles = len(rows) // steps
    if cycles < 2 or len(rows) != steps * cycles or any(row["result"] != "1" for row in rows):
        raise ValueError(f"{label}: incomplete or failed camera cycle")
    parity = [line for line in lines if line.startswith(prefix + " parity")]
    if len(parity) != steps * (cycles - 1) or any("status=pass" not in line for line in parity):
        raise ValueError(f"{label}: repeated-camera parity failed")
    if not lines[-1].startswith("BIQ ") or "0 fallback" not in lines[-1]:
        raise ValueError(f"{label}: missing successful completion marker")
    ids = range(6) if scenario == "navigation" else (128,112,96,80,64)
    field = "step" if scenario == "navigation" else "width"
    if [(int(row["cycle"]),int(row[field])) for row in rows] != [(cycle,step) for cycle in range(cycles) for step in ids]:
        raise ValueError(f"{label}: unexpected camera sequence")
    return directory, rows


def large_address_aware(directory):
    data = (directory / "biq_preview.exe").read_bytes()
    pe = struct.unpack_from("<I", data, 0x3c)[0]
    if data[:2] != b"MZ" or data[pe:pe+4] != b"PE\0\0":
        raise ValueError("Invalid preview executable")
    return bool(struct.unpack_from("<H", data, pe+22)[0] & 0x20)


def memory_samples(directory):
    rows = [dict(re.findall(r"(\w+)=(\d+)", line))
            for line in (directory / "benchmark.log").read_text().splitlines()
            if line.startswith("CAMERA memory ")]
    if not rows:
        return None  # Older witnesses did not sample memory; never report zero.
    return {"samples": len(rows),
            "minimum_available_virtual_bytes": min(int(row["available_virtual"]) for row in rows),
            "minimum_largest_free_region_bytes": min(int(row["largest_free_region"]) for row in rows),
            "total_virtual_bytes": min(int(row["total_virtual"]) for row in rows)}


def pixels(path):
    data = path.read_bytes()
    if data[:2] != b"BM":
        raise ValueError(f"Invalid BMP: {path.name}")
    offset = struct.unpack_from("<I", data, 10)[0]
    width, height = struct.unpack_from("<ii", data, 18)
    bits = struct.unpack_from("<H", data, 28)[0]
    if bits != 32 or width <= 0 or height == 0:
        raise ValueError("Benchmark requires 32-bit images")
    body = data[offset:offset + width * abs(height) * 4]
    if len(body) != width * abs(height) * 4:
        raise ValueError("Truncated benchmark image")
    return (width, height), body


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=("zoom", "navigation"), default="zoom")
    parser.add_argument("--baseline", type=Path, default=ROOT / "build/zoom-baseline")
    parser.add_argument("--candidate", type=Path, default=ROOT / "build/zoom-candidate")
    args = parser.parse_args()
    baseline, before = run("baseline", args.baseline, args.scenario)
    candidate, after = run("candidate", args.candidate, args.scenario)
    if len(before) != len(after):
        raise ValueError("Benchmark cycle counts differ")
    comparisons = []
    variants = [(f"z{width}", {"width": width}) for width in (128,112,96,80,64)] if args.scenario == "zoom" else [(f"nav{step}", {"step": step}) for step in range(6)]
    for suffix, identity in variants:
        size, old = pixels(baseline / f"zoom.bmp.{suffix}.bmp")
        other_size, new = pixels(candidate / f"zoom.bmp.{suffix}.bmp")
        if size != other_size or len(old) != len(new):
            raise ValueError("Image dimensions changed")
        changed = error = 0
        for offset in range(0, len(old), 4):
            deltas = [abs(old[offset + channel] - new[offset + channel]) for channel in range(4)]
            changed += max(deltas) > 2
            error += sum(deltas)
        passed = changed <= len(old) // 4000 and error <= len(old) // 100
        comparisons.append({**identity, "changed_pixels": changed, "absolute_error": error, "pass": passed})
    timings = []
    for b, a in zip(before, after):
        keys = ("cycle", "width") if args.scenario == "zoom" else ("cycle", "step", "x", "y", "width")
        if any(b[key] != a[key] for key in keys):
            raise ValueError("Benchmark sequences differ")
        timings.append({**{key: int(a[key]) for key in keys},
                        "before_ms": float(b["ms"]), "after_ms": float(a["ms"]),
                        "speedup": round(float(b["ms"]) / max(.001, float(a["ms"])), 2)})
    result = {"scenario": args.scenario,
              "baseline_large_address_aware": large_address_aware(baseline),
              "candidate_large_address_aware": large_address_aware(candidate),
              "baseline_memory": memory_samples(baseline),
              "candidate_memory": memory_samples(candidate),
              "candidate_device_recoveries": max(int(row["recoveries"]) for row in after) if all("recoveries" in row for row in after) else None,
              "baseline_sha256": hashlib.sha256((baseline / "C3XRenderer.dll").read_bytes()).hexdigest(),
              "candidate_sha256": hashlib.sha256((candidate / "C3XRenderer.dll").read_bytes()).hexdigest(),
              "pixels": comparisons, "timings": timings,
              "change_median_before_ms": statistics.median(row["before_ms"] for row in timings[1:]),
              "change_median_after_ms": statistics.median(row["after_ms"] for row in timings[1:]),
              "pass": all(row["pass"] for row in comparisons)}
    for label, rows in (("first_use", [row for row in timings[1:] if row["cycle"] == 0]),
                        ("revisit", [row for row in timings if row["cycle"] > 0])):
        result[label] = {"samples": len(rows), "before_median_ms": statistics.median(row["before_ms"] for row in rows),
                         "before_max_ms": max(row["before_ms"] for row in rows),
                         "before_p95_ms": sorted(row["before_ms"] for row in rows)[math.ceil(.95*len(rows))-1] if len(rows)>=30 else None,
                         "after_median_ms": statistics.median(row["after_ms"] for row in rows),
                         "after_max_ms": max(row["after_ms"] for row in rows),
                         "after_p95_ms": sorted(row["after_ms"] for row in rows)[math.ceil(.95*len(rows))-1] if len(rows)>=30 else None}
    (candidate / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

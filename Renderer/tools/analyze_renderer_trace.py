#!/usr/bin/env python3
"""Summarize bounded DLL traces or debugger-captured C3X renderer messages."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import re


def analyze(text: str) -> dict:
    stages: Counter = Counter()
    cache_paths: Counter = Counter()
    invalidations: Counter = Counter()
    timings: dict[str, list[float]] = defaultdict(list)
    totals: Counter = Counter()
    peak_buffer_bytes = 0
    for line in text.splitlines():
        if "[C3X renderer]" not in line:
            continue
        fields = dict(re.findall(r"([a-z_]+)=([^\s]+)", line))
        stage = fields.get("stage", "unknown")
        stages[stage] += 1
        for name, value in fields.items():
            if name.endswith("_ms") and not name.startswith("max_"):
                try:
                    sample = float(value)
                except ValueError:
                    continue
                if math.isfinite(sample) and sample >= 0:
                    timings[f"{stage}.{name}"].append(sample)
        # Composite duplicates frame counters; count DLL frame records once.
        if stage == "frame":
            cache_paths[fields.get("cache", "unknown")] += 1
            for reason in ("camera", "scene", "environment", "wrap", "content", "ownership", "device"):
                if fields.get(reason) == "1":
                    invalidations[reason] += 1
            for name in ("built", "reused", "evicted", "upload_bytes", "reused_pixels", "draw_pixels"):
                if fields.get(name, "").isdigit():
                    totals[name] += int(fields[name])
            if fields.get("gpu_bytes", "").isdigit():
                peak_buffer_bytes = max(peak_buffer_bytes, int(fields["gpu_bytes"]))
    distributions = {}
    for key, samples in sorted(timings.items()):
        samples.sort()
        distributions[key] = {
            "samples": len(samples),
            "p50_ms": samples[math.ceil(len(samples) * 0.50) - 1],
            "p95_ms": samples[math.ceil(len(samples) * 0.95) - 1],
            "max_ms": samples[-1],
        }
    return {
        "stages": dict(stages), "cache_paths": dict(cache_paths),
        "invalidations": dict(invalidations), "totals": dict(totals),
        "peak_gpu_buffer_bytes": peak_buffer_bytes, "timings": distributions,
        "note": "Readback includes pending GPU execution. Summary-mode traces are sampled; percentiles describe logged samples only.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    args = parser.parse_args()
    print(json.dumps(analyze(args.trace.read_text(encoding="utf-8", errors="replace")), indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Summarize bounded DLL traces or debugger-captured C3X renderer messages."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import re


def usage_summary(events: list[dict]) -> dict:
    """Correlate complete DLL calls; camera gestures are inferred, not input hooks."""
    sessions, current, pending, previous = [], {}, {}, {}
    samples, counts, objects = defaultdict(list), Counter(), Counter()
    unmatched_results = invalid = failed = 0
    for row in events:
        process = row.get("process", "unknown")
        stage = row.get("stage")
        if stage == "usage-session":
            identity = (process, row.get("qpc", "unknown"))
            current[process] = identity
            sessions.append({k: row[k] for k in ("process", "qpc", "qpc_frequency", "utc_unix_ms", "schema") if k in row})
            continue
        identity = current.get(process, (process, "start_not_captured"))
        if stage == "usage-settings":
            continue
        try:
            key = (identity, int(row["request"]))
            if stage == "usage-view":
                old = previous.get(identity)
                kind = "initial" if old is None else "stationary"
                if row.get("origin_valid") != "1":
                    kind = "unknown_view"
                elif old:
                    if any(row.get(k) != old.get(k) for k in ("world_width", "world_height", "wrap_x", "wrap_y", "world_revision")):
                        kind = "world_change"
                    elif any(row.get(k) != old.get(k) for k in ("tile_width", "tile_height")):
                        kind = "zoom"
                    elif any(row.get(k) != old.get(k) for k in ("target_width", "target_height")):
                        kind = "resize"
                    elif old.get("origin_valid") != "1":
                        kind = "unknown_view"
                    else:
                        dx = abs(int(row["origin_x"]) - int(old["origin_x"]))
                        dy = abs(int(row["origin_y"]) - int(old["origin_y"]))
                        if dx or dy:
                            kind = "distant_camera_change" if dx > int(row["target_width"])/2 or dy > int(row["target_height"])/2 else "nearby_camera_change"
                pending[key] = (row, kind)
                previous[identity] = row
                for name in ("visible", "cities", "roads", "railroads", "farms", "mines", "camps", "resources", "tile_units"):
                    objects[name] = max(objects[name], int(row.get(name, 0)))
            elif stage == "usage-result":
                before = pending.pop(key, None)
                if before is None:
                    unmatched_results += 1
                    continue
                view, kind = before
                elapsed = float(row["call_ms"])
                if not math.isfinite(elapsed) or elapsed < 0 or int(row["qpc"]) < int(view["qpc"]):
                    invalid += 1
                    continue
                if row.get("result") != "1":
                    failed += 1
                    continue
                samples[kind].append(elapsed)
                counts[kind] += 1
        except (KeyError, ValueError, OverflowError):
            invalid += 1
    workloads = {}
    for name, values in sorted(samples.items()):
        values.sort()
        workloads[name] = {"samples": len(values), "p50_ms": values[math.ceil(len(values)*.5)-1],
                           "p95_ms": values[math.ceil(len(values)*.95)-1], "p99_ms": values[math.ceil(len(values)*.99)-1],
                           "max_ms": values[-1], "hundred_sample_requirement_met": len(values) >= 100}
    return {"sessions": sessions, "workloads": workloads, "completed_calls": sum(counts.values()),
            "failed_calls": failed, "unmatched_begins": len(pending), "unmatched_results": unmatched_results,
            "invalid_records": invalid, "maximum_observed_tile_counts": dict(objects),
            "note": "DLL call latency includes queue/worker waits. Stationary means unchanged view, not proof of user inactivity. Camera changes are inferred; distant changes do not prove a minimap click. Tile-unit counts are not the number of unit bodies drawn. Missing starts/ends indicate incomplete capture. No native-presented-frame or input-to-display pass is inferred."}


def analyze(text: str) -> dict:
    stages: Counter = Counter()
    cache_paths: Counter = Counter()
    invalidations: Counter = Counter()
    timings: dict[str, list[float]] = defaultdict(list)
    totals: Counter = Counter()
    peak_buffer_bytes = 0
    usage_events = []
    unit_ids, unit_actions = set(), Counter()
    unit_hits = unit_misses = 0
    for line in text.splitlines():
        if "[C3X renderer]" not in line:
            continue
        fields = dict(re.findall(r"([a-z_]+)=([^\s]+)", line))
        stage = fields.get("stage", "unknown")
        stages[stage] += 1
        if stage.startswith("usage-"):
            usage_events.append(fields)
        if stage == "unit-body":
            if "id" in fields:
                unit_ids.add((fields.get("process"), fields["id"]))
            unit_actions[fields.get("action", "unknown")] += 1
            unit_hits += fields.get("cache_hit") == "1"
            unit_misses += fields.get("cache_hit") == "0"
            # The prefix also has an absolute `ms` timestamp. Only the duration
            # after the stage marker is a unit-call latency measurement.
            duration = re.search(r"stage=unit-body\b.*\bms=([^\s]+)", line)
            if duration:
                try:
                    elapsed = float(duration[1])
                    if math.isfinite(elapsed) and elapsed >= 0:
                        timings["unit-body.call_ms"].append(elapsed)
                except ValueError:
                    pass
        for name, value in fields.items():
            if name.endswith("_ms") and name not in ("utc_unix_ms", "animation_ms") and not name.startswith(("max_", "cumulative_")):
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
        "usage": usage_summary(usage_events),
        "unit_activity": {"distinct_logged_ids": len(unit_ids), "draws_by_action": dict(unit_actions),
                          "pose_hits": unit_hits, "pose_misses": unit_misses},
        "note": "Readback includes pending GPU execution. Summary-mode traces are sampled; percentiles describe logged samples only.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    args = parser.parse_args()
    print(json.dumps(analyze(args.trace.read_text(encoding="utf-8", errors="replace")), indent=2))


if __name__ == "__main__":
    main()

"""Summarize a Renderer64 diagnostic capture without claiming scanout latency."""
import argparse
from bisect import bisect_left, bisect_right
import csv
import json
import math
from pathlib import Path
import statistics


FAMILIES = {3: "scene", 7: "camera", 12: "visual", 13: "presentation",
            19: "native_bridge", 20: "unit_visual", 21: "unit_move",
            22: "unit_spawn", 23: "unit_state"}


def records(path):
    if path.is_file():
        with path.open(encoding="utf-8-sig") as stream:
            for line in stream:
                if line.strip():
                    yield json.loads(line)


def stats(values):
    values = sorted(values)
    if not values:
        return None
    return {"count": len(values), "median_ms": statistics.median(values),
            "p95_ms": values[math.ceil(len(values) * .95) - 1],
            "max_ms": values[-1], "over_100_ms": sum(value > 100 for value in values)}


def analyze(session):
    metadata = json.loads((session / "session.json").read_text(encoding="utf-8-sig"))
    if metadata.get("renderer_backend") != "Renderer64 direct surface":
        raise ValueError("Session was not captured with Renderer64")
    report_path = session / "inspection/report.json"
    report = json.loads(report_path.read_text(encoding="utf-8-sig")) if report_path.is_file() else {}
    frequency = report.get("frequency")
    origin = report.get("qpc_origin")
    family_data = {family: {"calls": 0, "errors": 0, "service": [], "input_ticks": []}
                   for family in FAMILIES}
    timeline_path = session / "inspection/timeline.jsonl"
    for event in records(timeline_path):
        family = event.get("family")
        if family not in family_data:
            continue
        item = family_data[family]
        item["calls"] += 1
        item["errors"] += event.get("result", 0) < 0
        begin, end = event.get("input_ticks"), event.get("result_ticks")
        if frequency and begin is not None and end is not None and end >= begin:
            item["service"].append(1000 * (end - begin) / frequency)
            if family in (7, 21) and event.get("result", 0) >= 0:
                item["input_ticks"].append(begin)
    families = {name: {"calls": family_data[family]["calls"],
                       "errors": family_data[family]["errors"],
                       "bridge_call_service": stats(family_data[family]["service"])}
                for family, name in FAMILIES.items()}

    present_path = session / "frames.csv"
    target = metadata.get("presentmon_target", "")
    helper_rows = []
    if present_path.is_file():
        with present_path.open(encoding="utf-8-sig", newline="") as stream:
            helper_rows = [row for row in csv.DictReader(stream)
                           if row.get("Application", "").lower() == target.lower()]

    def numeric(rows, field):
        values = []
        for row in rows:
            try:
                value = float(row.get(field, ""))
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                values.append(value)
        return values

    presentation = {"target": target, "rows": len(helper_rows),
                    "present_intervals": stats(numeric(helper_rows, "msBetweenPresents")),
                    "display_change_intervals": stats(numeric(helper_rows, "msBetweenDisplayChange")),
                    "present_api": stats(numeric(helper_rows, "msInPresentAPI")),
                    "gpu_active": stats(numeric(helper_rows, "msGPUActive"))}
    present_ticks = sorted(int(row["QPCTime"]) for row in helper_rows
                           if row.get("QPCTime", "").isdigit())
    if frequency and origin and present_ticks:
        recent_indices = {}
        for family in (7, 21):
            event_ticks = sorted(origin + tick for tick in family_data[family]["input_ticks"])
            indices = set()
            for index, row in enumerate(helper_rows):
                value = row.get("QPCTime", "")
                if not value.isdigit():
                    continue
                when = int(value)
                at = bisect_right(event_ticks, when) - 1
                if at >= 0 and when - event_ticks[at] <= frequency:
                    indices.add(index)
            recent_indices[family] = indices
            following = [helper_rows[index] for index in sorted(indices)]
            presentation[FAMILIES[family] + "_following_1s"] = {
                "rows": len(following),
                "present_intervals": stats(numeric(following, "msBetweenPresents")),
                "gpu_active": stats(numeric(following, "msGPUActive"))}
        outside = [row for index, row in enumerate(helper_rows)
                   if index not in recent_indices[7] and index not in recent_indices[21]]
        presentation["outside_camera_or_move"] = {
            "rows": len(outside),
            "present_intervals": stats(numeric(outside, "msBetweenPresents")),
            "gpu_active": stats(numeric(outside, "msGPUActive"))}
        for family in (7, 21):
            candidates = []
            for tick in family_data[family]["input_ticks"]:
                when = origin + tick
                at = bisect_left(present_ticks, when)
                if at < len(present_ticks):
                    candidates.append(1000 * (present_ticks[at] - when) / frequency)
            families[FAMILIES[family]]["next_helper_present_opportunity"] = stats(candidates)

    helper_memory = list(records(session / "renderer64-memory.jsonl"))
    window_events = list(records(session / "window/timeline.jsonl"))
    native_memory = [row for row in window_events if row.get("event") == "process_memory"]
    return {"scope": "diagnostic; bridge call service and next helper present do not prove correct displayed-frame latency",
            "capture_result": metadata.get("result"),
            "journal_complete": bool(report.get("complete") and report.get("verified_prefix")),
            "journal_calls": report.get("calls"), "families": families,
            "presentation": presentation,
            "helper_private_peak_mib": max((row.get("private_bytes", 0) for row in helper_memory), default=0) / 1048576,
            "civ3_private_peak_mib": max((row.get("private_bytes", 0) for row in native_memory), default=0) / 1048576,
            "civ3_min_free_mib": min((row.get("free_bytes", 0) for row in native_memory), default=0) / 1048576,
            "window_samples": sum("frame" in row for row in window_events),
            "missing": [name for name, available in (("journal", timeline_path.is_file()),
                       ("helper_presentmon", bool(helper_rows)),
                       ("helper_memory", bool(helper_memory)),
                       ("window", bool(native_memory))) if not available]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", type=Path)
    args = parser.parse_args()
    print(json.dumps(analyze(args.session), indent=2))

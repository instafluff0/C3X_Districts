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
    elif scenario == "idle":
        prefix, end = "IDLE_FRAME ", None
        count = int(args["idle_steps"])
        image_names = [f"zoom.bmp.idle{i}.bmp" for i in range(count)]
        starts = [fields(l) for l in lines if l.startswith("IDLE_BEGIN ")]
        ends = [fields(l) for l in lines if l.startswith("IDLE_END ")]
        if (len(starts)!=1 or len(ends)!=1 or ends[0].get("status")!="pass" or
                starts[0].get("steps")!=str(count) or starts[0].get("warmup")!=str(args.get("idle_warmup",10)) or
                starts[0].get("pose_hz")!="15" or starts[0].get("paced")!="0" or
                starts[0].get("units")!=str(args.get("idle_units",0)) or starts[0].get("tile_width")!=str(args["tile_width"]) or
                (starts[0].get("x"),starts[0].get("y"))!=("75","39")):
            raise ValueError("Missing stationary animation contract")
    elif scenario == "zoom":
        _, all_rows = run("zoom", directory, "zoom")
        prefix, end = "ZOOM cycle=", None
        count = len(all_rows)
        image_names = [f"zoom.bmp.z{r['width']}.bmp" for r in all_rows if r["cycle"] == "0"]
    elif scenario == "animation":
        prefix, end = "ANIMATION temporal frame=", "ANIMATION temporal: pass changed_frames=5"
        count = 6
        image_names = ["zoom.bmp"] + [f"zoom.bmp.animation-{i}.bmp" for i in range(count)]
        if "ANIMATION zoom-return parity: pass" not in lines:
            raise ValueError("Missing animation zoom-return witness")
        for case in ("scroll", "removal"):
            records = [fields(l) for l in lines if l.startswith(f"ANIMATION {case} parity: pass ")]
            if len(records) != 1 or int(records[0]["error"]) != 0:
                raise ValueError("Missing exact animation scroll/removal witness")
    else:
        raise ValueError("Select a resident pan, zoom, distant, or animation sweep")
    rows = [fields(line) for line in lines if line.startswith(prefix)]
    if scenario == "animation":
        rows = [dict(r, step=r["frame"], built=r["terrain_built"], upload_bytes=r["terrain_upload"], result="1") for r in rows]
    if count < 1 or len(rows) != count or (end and lines.count(end) != 1) or any(r["result"] != "1" for r in rows):
        raise ValueError("Incomplete or failed measured sweep")
    if scenario != "zoom" and [int(r["step"]) for r in rows] != list(range(count)):
        raise ValueError("Reordered or duplicated samples")
    if scenario == "idle":
        if any(int(r["ticks"])!=1000000+(i+args.get("idle_warmup",10)+1)*1000000//15 or int(r["visible"])<1 or
               int(r["built"])!=0 or int(r["upload_bytes"])!=0 or int(r["recoveries"])!=0 for i,r in enumerate(rows)):
            raise ValueError("Idle clocks or retained geometry contract violated")
        if int(ends[0]["changed_frames"])!=sum(r["changed"]=="1" for r in rows[1:]):
            raise ValueError("Inconsistent idle pose changes")
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
    report = {"endpoint": "standalone animation completed render; no native presentation" if scenario in ("animation","idle") else "standalone capture plus completed render; no native presentation",
              "scenario": scenario, "viewport": [args["width"], args["height"]],
              "waves": args["waves"], "clock": f"unpaced stationary 15 Hz authored pose samples; {args.get('idle_warmup',10)} warmup renders" if scenario == "idle" else "six changing animation clocks" if scenario == "animation" else "fixed replay clock; changing animation is a separate witness",
              "binaries": receipt["binaries"], "images": images,
              "image_receipt_verified": "images" in completion,
              "camera_requests": [{k: r[k] for k in ("cycle", "step", "x", "y", "pixel_y", "width", "ticks") if k in r} for r in rows],
              "timing": {k: distribution([float(r[k]) for r in measured])
                         for k in ("ms", "capture_ms", "geometry_ms", "draw_ms", "readback_ms", "map_ms", "copy_ms", "units_ms")
                         if all(k in r for r in measured)},
              "mesh_builds": sum(int(r["built"]) for r in measured),
              "mesh_counter_scope": "Static terrain counters; wave/animation buffers are separate",
              "upload_bytes": sum(int(r["upload_bytes"]) for r in measured) if all("upload_bytes" in r for r in measured) else None,
              "native_first_response_ms": None, "native_presented_frames": None,
              "map_prepared_before_sweep": False if scenario == "distant" else None}
    if scenario == "idle":
        observed_changes=sum(images[image_names[i]]!=images[image_names[i-1]] for i in range(1,count))
        if observed_changes!=int(ends[0]["changed_frames"]) or (count>1 and observed_changes==0):
            raise ValueError("Idle images do not establish changing poses")
        report["changed_frames"]=observed_changes
        report["visible_animations"]={"min":min(int(r["visible"]) for r in rows),"max":max(int(r["visible"]) for r in rows)}
        report["fixture_objects"]={k:int(starts[0][k]) for k in ("units","cities","roads","farms","mines","camps","resources") if k in starts[0]}
        warmup=[fields(l) for l in lines if l.startswith("IDLE_WARMUP ")]
        if "idle_warmup" in args:
            if len(warmup)!=1 or warmup[0].get("frames")!=str(args["idle_warmup"]):
                raise ValueError("Missing declared idle warmup")
            report["warmup_completed_render_ms"]=distribution([float(warmup[0]["ms"])])["median_ms"]
        if args.get("dense_scene") and (starts[0].get("dense")!="1" or
                any(report["fixture_objects"].get(k,0)<1 for k in ("cities","roads","farms","mines","camps","resources"))):
            raise ValueError("Dense fixture is missing a required object category")
        if args.get("idle_units") and any(int(r.get("units",0))!=args["idle_units"] or "units_ms" not in r for r in rows):
            raise ValueError("Missing measured unit plane")
        report["unit_actions"]=starts[0].get("unit_actions","idle")
        if report["unit_actions"]!=args.get("unit_actions","idle"):
            raise ValueError("Unexpected unit action workload")
        if report["unit_actions"]=="mixed":
            report["unit_action_draws"]={k:sum(int(r.get(k,0)) for r in rows) for k in ("moving","attacking","fortifying","idling")}
            if any(v==0 for v in report["unit_action_draws"].values()):
                raise ValueError("Mixed unit witness omits an action class")
    if scenario == "zoom":
        report["warm_zoom"] = distribution([float(r["ms"]) for r in rows if int(r["cycle"]) > 0])
        report["first_zoom_changes"] = distribution([float(r["ms"]) for r in rows[1:] if r["cycle"] == "0"])
    if args.get("camera_view"):
        camera = [fields(l) for l in lines if l.startswith("CAMERA ticket=")][-count:]
        if len(camera) != count or any(r.get("identical_coalesced") != "1" or r.get("stale_rejected") != "1" or r["result"] != "1" for r in camera):
            raise ValueError("Missing verified camera completion/coalescing records")
        report["standalone_queue"] = {k: distribution([float(r[k]) for r in camera])
                                      for k in ("accepted_ms", "final_ms", "poll_max_ms", "repeat_max_ms")}
        report["standalone_queue"]["poll_measurement"] = "Distribution of per-request maximum call times, including publication identity validation; no native UI callback"
    memory = [fields(l) for l in lines if l.startswith("CAMERA memory ")]
    if memory:
        report["sampled_address_space_not_peak"] = {k: min(int(r[k]) for r in memory)
                                                   for k in ("available_virtual", "largest_free_region")}
    trace = (directory / "renderer.log").read_text().splitlines()
    measured_sequences=None
    if scenario=="idle":
        # The bounded trace may end early. Match actual measured clocks instead
        # of silently substituting warm-up records from its last N entries.
        clocks={str(int(r["ticks"])//(1000000//15)) for r in rows}
        measured_sequences={r["sequence"] for l in trace if "stage=animation-frame " in l
                            for r in [fields(l)] if r.get("clock") in clocks and "sequence" in r}
    def stage_rows(stage):
        selected=[fields(l) for l in trace if f"stage={stage} " in l]
        return ([r for r in selected if r.get("sequence") in measured_sequences]
                if measured_sequences is not None else selected[-count:])
    animation_phases=stage_rows("animation-phases")
    if scenario!="animation" and animation_phases and (scenario=="idle" or len(animation_phases)==count):
        report["animation_phases"]={k:distribution([float(r[k]) for r in animation_phases])
                                    for k in ("pose_prepare_ms","backdrop_submit_ms","animated_submit_ms",
                                              "readback_submit_ms","readback_wait_ms","cpu_copy_ms")}
    animation = stage_rows("animation-frame")
    if scenario != "animation" and animation and (scenario=="idle" or len(animation)==count):
        report["animation"] = {"timing": distribution([float(r["ms"]) for r in animation]),
                               "totals": {k: sum(int(r[k]) for r in animation) for k in
                                          ("wave_upload_bytes", "wave_cells_built", "wave_cells_reused", "backdrop_hits", "backdrop_misses")}}
    if scenario=="idle":
        report["idle_trace_coverage"]={"measured_frames":count,"animation_frames":len(animation),
                                      "phase_frames":len(animation_phases),"animation_trace_complete":len(animation)==count}
        unit_rows=stage_rows("unit-body")
        unit_count=args.get("idle_units",0)
        if unit_count:
            grouped={sequence:[r for r in unit_rows if r.get("sequence")==sequence] for sequence in measured_sequences}
            complete=[r for group in grouped.values() if len(group)==unit_count for r in group]
            if complete:
                report["unit_pose_cache"]={"complete_traced_frames":len(complete)//unit_count,
                    "hits":sum(r.get("cache_hit")=="1" for r in complete),
                    "misses":sum(r.get("cache_hit")=="0" for r in complete),
                    "maximum_sampled_pixel_capacity_bytes":max(int(r["cache_bytes"]) for r in complete)}
    center = stage_rows("center-shore-cache")
    if scenario != "animation" and center and (scenario=="idle" or len(center)==count):
        report["center_shore"] = {"timing": distribution([float(r["ms"]) for r in center]),
                                  "maximum_bytes": max(int(r["bytes"]) for r in center)}
    phases = stage_rows("navigation-phases")
    if scenario != "animation" and phases and (scenario=="idle" or len(phases)==count):
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
    controls = {"dependency_control", "index_control", "center_shore_control", "world_regions_control", "three_zoom_memory", "camera_view",
                "backdrop_control", "wave_control", "composition_casters_control", "backdrop_dependencies", "unit_pose_memory", "unit_pose_memory_mib", "composition_receiver_index", "mountain_samples", "material_samples", "production_defaults"}
    ignored = controls | {"out", "binaries"}
    # Receipts made before these opt-in witnesses existed represent their
    # disabled defaults. Nondefault scene/unit settings still must match.
    defaults={"idle_steps":100,"idle_units":0,"idle_warmup":10,"dense_scene":False,"unit_actions":"idle"}
    if {k: v for k, v in (defaults|before["args"]).items() if k not in ignored} != {k: v for k, v in (defaults|after["args"]).items() if k not in ignored}:
        raise ValueError("Paired cameras or quality settings differ")
    env_controls = {"C3X_RENDERER_REGION_DEPENDENCY_CONTROL", "C3X_RENDERER_REGION_INDEX_CONTROL",
                    "C3X_RENDERER_CENTER_SHORE_CONTROL", "C3X_RENDERER_WORLD_REGIONS_CONTROL",
                    "C3X_RENDERER_BACKDROP_REUSE_CONTROL", "C3X_RENDERER_WAVE_REUSE_CONTROL",
                    "C3X_RENDERER_BACKDROP_DEPENDENCIES",
                    "C3X_RENDERER_WORLD_BACKDROPS", "C3X_RENDERER_WORLD_WAVES",
                    "C3X_RENDERER_COMPOSITION_RECEIVER_INDEX",
                    "C3X_RENDERER_MOUNTAIN_SAMPLES",
                    "C3X_RENDERER_MATERIAL_SAMPLES",
                    "C3X_RENDERER_UNIT_POSE_MEMORY",
                    "C3X_RENDERER_COMPOSITION_CASTERS_CONTROL", "C3X_RENDERER_THREE_ZOOM_MEMORY", "C3X_RENDERER_TRACE_FILE",
                    "C3X_RENDERER_PREVIEW_CAMERA_QUEUE", "C3X_RENDERER_PREVIEW_CAMERA_VIEW"}
    env_defaults={"C3X_RENDERER_PREVIEW_IDLE_STEPS":"","C3X_RENDERER_PREVIEW_IDLE_UNITS":"0","C3X_RENDERER_PREVIEW_IDLE_WARMUP":"10","C3X_RENDERER_PREVIEW_DENSE_SCENE":"","C3X_RENDERER_PREVIEW_UNIT_ACTIONS":"idle"}
    if ({k: v for k, v in (env_defaults|before.get("environment", {})).items() if k not in env_controls} !=
            {k: v for k, v in (env_defaults|after.get("environment", {})).items() if k not in env_controls} or
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

"""Summarize verified standalone workloads without claiming native presentation.

Optional paired comparisons require identical binaries, runtime inputs, cameras,
and quality settings. Only explicit cache-control switches may differ.
"""
import argparse
import hashlib
import gzip
import json
import math
from pathlib import Path
import re
import statistics

from Renderer.native.compare_zoom_benchmark import pixels, run
from Renderer.lab.platform import ROOT


def fields(line):
    return dict(re.findall(r"(\w+)=([^ ]+)", line))


def digest(path):
    result = hashlib.sha256()
    with (path.open("rb") if path.exists() or path.suffix!=".bmp" else gzip.open(path.with_suffix(".bmp.gz"),"rb")) as stream:
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


def endpoint_accounting(lines, trace_lines=()):
    """Account disjoint host spans; never add overlapping CPU/GPU intervals.

    Endpoints certify API success/ownership, not independent full-redraw parity.
    Missing trace/query coverage stays explicit, including the final delayed GPU
    sample. A cold first render includes lazy loading and is preparation, not playback.
    """
    setup = [fields(l) for l in lines if l.startswith("TIMING_SETUP ")]
    if len(setup) != 1:
        return {"status":"unmeasured", "reason":"missing unique versioned endpoint record"}
    setup=setup[0]
    if setup.get("schema")!="1" or int(setup["frequency"])<=0:
        raise ValueError("Invalid timing schema/frequency")
    scale=1000/int(setup["frequency"])
    def ordered(row, names):
        values=[int(row[n]) for n in names]
        if values!=sorted(values) or values[0]<0:
            raise ValueError("Invalid timing endpoint order")
        return values
    marks=ordered(setup,("process_enter","source_done","dll_done","definitions_done","initial_begin","initial_done"))
    trace=[fields(l) for l in trace_lines if "qpc=" in l and "stage=" in l]
    requests=[]
    for line in lines:
        if not line.startswith("TIMING_REQUEST "):continue
        row=fields(line)
        if int(row["id"])!=len(requests):raise ValueError("Missing or duplicate timing request")
        a,b,c,d,e=ordered(row,("capture_begin","capture_end","caller_enter","caller_return","correct_done"))
        phases={n: int(row[n+"_ticks"])*scale for n in ("geometry","draw","readback")}
        if any(v<0 for v in phases.values()):raise ValueError("Negative renderer phase")
        call=(d-c)*scale
        known=sum(phases.values())
        matching=[r for r in trace if c<=int(r["qpc"])<=d]
        render=[r for r in matching if r["stage"]=="render-begin"]
        sequence=render[0].get("sequence") if len(render)==1 else None
        gpu=[r for r in trace if r["stage"]=="gpu-timing" and r.get("sample_sequence")==sequence] if sequence else []
        submission=[r for r in matching if r["stage"]=="submission-phases"]
        animation=[r for r in matching if r["stage"]=="animation-frame"]
        if len(animation)==1:
            phases["animation_composition"]=float(animation[0]["ms"])
            known=sum(phases.values())
        # Report nested diagnostics separately: they are contained by caller and
        # readback, and cannot be added to their parent intervals.
        nested={"blocking_map_wait_ms":None,"cpu_bitmap_copy_ms":None,
                "gpu_execution_ms":None,"gpu_copy_ms":None}
        if len(submission)==1:
            nested.update(blocking_map_wait_ms=float(submission[0]["map_wait_ms"]),
                          cpu_bitmap_copy_ms=float(submission[0]["cpu_copy_ms"]))
        if len(gpu)==1 and gpu[0].get("valid")=="1":
            nested.update(gpu_execution_ms=float(gpu[0]["gpu_draw_ms"]),gpu_copy_ms=float(gpu[0]["gpu_copy_ms"]))
        request={"id":int(row["id"]),"role":"initial_preparation" if not requests else "playback",
                 "status":"success_ownership_checked" if row["result"]=="1" else "failed",
                 "captured_tiles":int(row["tiles"]), "renderer_sequence":sequence,
                 "capture_ms":(b-a)*scale,"capture_to_caller_ms":(c-b)*scale,
                 "caller_ms":call,"ownership_check_ms":(e-d)*scale,
                 "request_to_result_ms":(d-a)*scale,"request_to_checked_result_ms":(e-a)*scale,
                 "renderer_cpu_spans_ms":phases,"nested_diagnostics":nested,
                 "unexplained_caller_ms":max(0,call-known),
                 "cpu_phase_accounting_valid":known<=call+0.01,
                 "independent_pixel_parity":"unmeasured"}
        calls=[r for r in matching if r["stage"]=="call-endpoints"]
        request["caller_cpu_spans_ms"]=None
        if len(calls)==1:
            callrow=calls[0]
            names=("entered","locked","submitted","worker_begin","worker_rendered","worker_published","returned") if callrow["queued"]=="1" else ("entered","locked","returned")
            values=ordered(callrow,names)
            if values[0]<c or values[-1]>d:raise ValueError("Caller trace outside request")
            labels=("lock_wait","snapshot_and_camera_drain","queue_wait","worker_render",
                    "worker_publication_and_preparation","completion_wakeup") if callrow["queued"]=="1" else ("lock_wait","retained_result")
            spans=dict(zip(labels,[(y-x)*scale for x,y in zip(values,values[1:])]))
            spans["api_entry_and_return"]=(values[0]-c+d-values[-1])*scale
            request["caller_cpu_spans_ms"]=spans
            request["unexplained_caller_ms"]=max(0,call-sum(spans.values()))
            request["unexplained_worker_render_ms"]=max(0,spans.get("worker_render",0)-known) if callrow["queued"]=="1" else None
        request["missing_measurements"]=[k for k,v in nested.items() if v is None]
        if not calls:request["missing_measurements"] += ["caller_lock_wait", "worker_queue_wait", "publication_cpu"]
        request["accounting_complete"]=not request["missing_measurements"] and request["cpu_phase_accounting_valid"] and max(
            request["unexplained_caller_ms"], request.get("unexplained_worker_render_ms") or 0)<=max(1,call*.05)
        requests.append(request)
    dropped=int(setup["dropped"])
    setup_trace=[r for r in trace if marks[0]<=int(r["qpc"])<=marks[-1]]
    return {"status":"measured_with_gaps", "schema":1,"requests":requests,"dropped_requests":dropped,
            "host_span_through_last_check_ms":(e-marks[0])*scale if requests else None,
            "setup_ms":dict(zip(("source_fixture_and_host_startup","dll_load","definition_configuration",
                                  "initial_capture_and_host_setup","initial_render_preparation"),
                                 [(b-a)*scale for a,b in zip(marks,marks[1:])])),
            "load_diagnostics":[r for r in setup_trace if r["stage"].startswith("load-")],
            "setup_nested_diagnostics":[r for r in setup_trace if r["stage"] in ("setup-device","setup-shader")],
            "missing_setup_measurements":["process_spawn_before_main", "shader_setup_outside_primary_material_entries"]+
                ([] if any(r["stage"]=="setup-device" for r in trace) else ["device_creation_not_observed_in_this_case"]),
            "trace_dropped":next((int(fields(l)["dropped"]) for l in trace_lines if l.startswith("TRACE_BUFFER ")),None),
            "playback_samples":max(0,len(requests)-1),
            "performance_claim":"endpoint accounting only; no optimization or native presentation pass"}


def session_accounting(lines, trace_lines=()):
    """Separate untimed bootstrap/reset/warmup from each bounded case."""
    cases=[];current=None;playback=None;warmup_start=None
    ends=[fields(l) for l in lines if l.startswith("CASE_SESSION_END ")]
    for i,line in enumerate(lines):
        if line.startswith("CASE_BEGIN "):
            if current is not None:raise ValueError("Overlapping persistent cases")
            current={"identity":fields(line),"resets":[],"warmup_ran":False};playback=None
        elif current is not None and line.startswith("CASE_RESET "):
            reset=fields(line)
            if reset["result"]!="1" or reset["budgets"]!="unchanged" or int(reset["geometry_evictions"]) or int(reset["pose_evictions"]):
                raise ValueError("Persistent reset changed budgets or failed")
            if reset["mode"]=="1" and any(int(reset[k]) for k in ("retained_geometry","retained_natural","retained_ground","geometry_entries")):
                raise ValueError("Scene content survived assets-loaded reset")
            current["resets"].append(reset)
        elif current is not None and line=="CASE_WARMUP_BEGIN":warmup_start=i+1
        elif current is not None and line.startswith("CASE_WARMUP_END "):
            if fields(line)["result"]!="0" or warmup_start is None:raise ValueError("Failed persistent warmup")
            current["warmup_ran"]=True
            current["warmup_endpoints"]=endpoint_accounting(lines[warmup_start:i],trace_lines)
            warmup_start=None
        elif current is not None and line=="CASE_PLAYBACK_BEGIN":playback=i+1
        elif line.startswith("CASE_END "):
            end=fields(line)
            if current is None or playback is None or end["id"]!=current["identity"]["id"] or end["result"]!="0":
                raise ValueError("Failed/missing persistent case")
            body=lines[playback:i]
            current["endpoints"]=endpoint_accounting(body,trace_lines)
            if current["endpoints"]["status"]=="unmeasured" or not current["endpoints"]["playback_samples"] or any(r["status"]!="success_ownership_checked" for r in current["endpoints"]["requests"]):
                raise ValueError("Missing persistent timing/ownership")
            offsets=[fields(l) for l in body if l.startswith("SCROLL_SEQUENCE ")]
            current["request_offsets"]=[int(r["offset_columns"]) for r in offsets] if offsets else [int(fields(l)["delta_columns"]) for l in body if l.startswith("SCROLL_ABLATION ")]
            if offsets and (any(r["exact"]!="1" for r in offsets) or not any(l.startswith("SCROLL_SEQUENCE_END status=pass") for l in body)):
                raise ValueError("Persistent reversal parity failed")
            modes=[r["mode"] for r in current["resets"]]
            policy=current["identity"]["reset"]
            if modes!=({"process_cold":[],"assets_loaded":["1"],"prepared_resident":["1","2"]}.get(policy)) or current["warmup_ran"]!=(policy=="prepared_resident"):
                raise ValueError("Persistent reset/warmup coverage mismatch")
            cases.append(current);current=None
    if current is not None or len(ends)!=1 or ends[0]["result"]!="0" or ends[0]["time_limit_exceeded"]!="0" or not cases or int(ends[0]["completed"])!=len(cases) or int(ends[0]["requested"])!=len(cases):
        raise ValueError("Incomplete/timed-out persistent session")
    checks=[fields(l) for l in lines if l.startswith("CASE_INPUT_CHECK ")]
    if len(checks)!=len(cases)+1 or any(r.get("unchanged")!="1" for r in checks):
        raise ValueError("Persistent session inputs changed or were not checked between cases")
    return {"schema":1,"status":"measured_with_gaps","cases":cases,"session":ends[0],
            "input_checks":checks,
            "bootstrap":[fields(l) for l in lines if l.startswith("CASE_BOOTSTRAP_END ")],
            "performance_claim":"setup amortization only; no gameplay performance pass"}


def inspect(directory):
    receipt = json.loads((directory / "inputs.json").read_text())
    completion = json.loads((directory / "evidence.json").read_text())
    if ((directory / "completion.txt").read_text().strip() != "0" or
            completion.get("returncode") != 0 or completion.get("invocation") != receipt.get("invocation") or
            completion.get("inputs_unchanged") is not True or completion.get("binaries_unchanged") is not True or
            completion.get("sources_unchanged",True) is not True or completion.get("provisional",False)):
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
    if receipt.get("case_manifest"):
        report=session_accounting(lines,(directory/"renderer.log").read_text().splitlines() if (directory/"renderer.log").exists() else [])
        manifest=receipt["case_manifest"]
        if len(report["cases"])!=manifest["repeats"]:raise ValueError("Changed case count")
        for i,case in enumerate(report["cases"]):
            if case["request_offsets"]!=manifest["request"]["offsets"] or case["endpoints"]["playback_samples"]!=len(case["request_offsets"]):
                raise ValueError("Changed persistent request sequence")
            if case["identity"]!={"id":manifest["case_id"]+f"-{i}","config":manifest["config_id"],
                                  "request_digest":manifest["request_digest"],"reset":manifest["reset"],"warmup":manifest["warmup"]}:
                raise ValueError("Changed persistent case identity")
        names=[f"zoom.bmp.case{i}.bmp{suffix}" for i in range(manifest["repeats"]) for suffix in ("",".result.bmp")]
        if any(completion.get("images",{}).get(n)!=digest(directory/n) for n in names):
            raise ValueError("Changed persistent case image")
        report["images"]={n:completion["images"][n] for n in names}
        report["repeated_images_exact"]=all(report["images"][n]==report["images"]["zoom.bmp.case0.bmp"+(".result.bmp" if n.endswith(".result.bmp") else "")] for n in names)
        report["wrapper_timing_ms"]=completion.get("wrapper_timing_ms")
        report["timing"]={"ms":distribution([r["request_to_checked_result_ms"] for c in report["cases"] for r in c["endpoints"]["requests"][1:]])}
        return receipt,report
    if not lines or not lines[-1].startswith("BIQ ") or "0 fallback" not in lines[-1]:
        raise ValueError("Missing completed zero-fallback witness")
    scenario = args["scenario"]
    if scenario == "navigation" and args.get("resident"):
        run("resident warmup", directory, "navigation")
        prefix, end = "RESIDENT_NAV ", "RESIDENT_END status=pass"
        count = int(args["resident_steps"])
        image_names = [f"zoom.bmp.resident{i}.bmp" for i in range(count)]
    elif scenario == "replay":
        prefix, end = "REPLAY_FRAME ", None
        starts = [fields(l) for l in lines if l.startswith("REPLAY_BEGIN ")]
        prepare_starts = [fields(l) for l in lines if l.startswith("REPLAY_PREPARE_BEGIN ")]
        prepare_ends = [fields(l) for l in lines if l.startswith("REPLAY_PREPARE_END ")]
        timed_ends = [fields(l) for l in lines if l.startswith("REPLAY_TIMED_END ")]
        ends = [fields(l) for l in lines if l.startswith("REPLAY_END ")]
        parity = [fields(l) for l in lines if l.startswith("REPLAY_PARITY ")]
        count = int(args.get("replay_samples_per_phase",25))*8
        mode=args.get("preparation_mode","baseline")
        if (len(starts)!=1 or len(prepare_starts)!=1 or len(prepare_ends)!=1 or
                len(timed_ends)!=1 or len(ends)!=1 or
                any(r.get("status")!="pass" for r in (prepare_ends[0],timed_ends[0],ends[0])) or
                any(r.get("mode")!=mode for r in (starts[0],prepare_starts[0],prepare_ends[0],timed_ends[0],ends[0])) or
                starts[0].get("clock")!="logical" or starts[0].get("final_map_cache")!="cleared" or
                starts[0].get("native_presented")!="0" or int(starts[0].get("requests",0))!=count or
                int(ends[0].get("coverage_complete",0))!=1 or not parity or
                any(r.get("status")!="pass" for r in parity) or
                int(ends[0].get("verified",0))!=len(parity) or int(ends[0].get("snapshots",0))!=len(parity)):
            raise ValueError("Missing completed deterministic retained-preparation replay")
        image_names=[f"zoom.bmp.replay-{r['label']}-{r['tile_width']}.bmp" for r in parity]
    elif scenario == "session":
        prefix, end = "SESSION_FRAME ", None
        mode=args.get("preparation_mode","baseline")
        prepare_starts = [fields(l) for l in lines if l.startswith("SESSION_PREPARE_BEGIN ")]
        prepare_ends = [fields(l) for l in lines if l.startswith("SESSION_PREPARE_END ")]
        preparation_contract="preparation_mode" in args
        if not preparation_contract and not prepare_starts and not prepare_ends:
            prepare_starts=[{"mode":"baseline"}]
            prepare_ends=[{"status":"pass","mode":"baseline","requests":"0","unit_requests":"0","ms":"0",
                "builds":"0","upload_bytes":"0","cleared_viewport":"0","cleared_regions":"0",
                "cleared_blocks":"0","cleared_backdrops":"0","cleared_publication":"0",
                "retained_geometry":"0","retained_natural":"0","retained_ground":"0","retained_waves":"0",
                "retained_pose":"0","retained_payload":"0","retained_shadow":"0","retained_other":"0",
                "geometry_entries":"0","pose_entries":"0","wave_entries":"0"}]
        starts = [fields(l) for l in lines if l.startswith("SESSION_BEGIN ")]
        timed_ends = [fields(l) for l in lines if l.startswith("SESSION_TIMED_END ")]
        ends = [fields(l) for l in lines if l.startswith("SESSION_END ")]
        parity = [fields(l) for l in lines if l.startswith("SESSION_PARITY ")]
        if (len(prepare_starts)!=1 or len(prepare_ends)!=1 or len(starts)!=1 or len(timed_ends)!=1 or len(ends)!=1 or
                prepare_ends[0].get("status")!="pass" or prepare_starts[0].get("mode")!=mode or
                prepare_ends[0].get("mode")!=mode or (preparation_contract and starts[0].get("preparation_mode")!=mode) or
                starts[0].get("clock")!="wall" or starts[0].get("unit_warmup")!="0" or
                starts[0].get("native_presented")!="0" or starts[0].get("dense")!="1" or
                starts[0].get("units_per_zone")!=str(args.get("idle_units")) or
                timed_ends[0].get("status")!="pass" or ends[0].get("status")!="pass" or
                not parity or any(r.get("status")!="pass" for r in parity) or
                ends[0].get("verified")!=str(len(parity)) or ends[0].get("snapshots")!=str(len(parity))):
            raise ValueError("Missing completed busy session and independent snapshots")
        if ((mode=="baseline" and int(prepare_ends[0]["requests"])!=0) or
                (mode=="oracle" and (not 0<int(prepare_ends[0]["requests"])<=200 or
                 int(prepare_ends[0].get("requested",0))!=200 or prepare_ends[0].get("memory_safe")!="1" or
                 sum(int(prepare_ends[0][k]) for k in ("cleared_viewport","cleared_regions","cleared_blocks",
                                                        "cleared_backdrops","cleared_publication"))<=0))):
            raise ValueError("Busy-session preparation mode was not honored")
        count=int(ends[0]["frames"])
        if timed_ends[0].get("frames")!=str(count):
            raise ValueError("Busy session frame counts disagree")
        image_names=["zoom.bmp"]+[f"zoom.bmp.session-{r['phase']}-{r['tile_width']}.bmp" for r in parity]
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
    elif scenario == "scroll":
        sequence=args.get("scroll_sequence",False)
        prefix="SCROLL_SEQUENCE " if sequence else "SCROLL_ABLATION "
        end="SCROLL_SEQUENCE_END status=pass unique_cameras=8 exact_revisits=1" if sequence else None
        count=14 if sequence else 1
        image_names=["zoom.bmp"]
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
    if scenario == "scroll":
        rows=[dict(r,step=r.get("step",str(i)),ms=r["total_ms"]) for i,r in enumerate(rows)]
        if any(int(r["fallback"]) or int(r["recoveries"]) or r.get("exact","1")!="1" for r in rows):
            raise ValueError("Failed scroll correctness/ownership witness")
        if args.get("scroll_sequence") and [int(r["offset_columns"]) for r in rows]!=[1,2,4,8,4,2,1,0,-2,-4,-8,-4,-2,0]:
            raise ValueError("Changed scroll request sequence")
    if scenario == "animation":
        rows = [dict(r, step=r["frame"], built=r["terrain_built"], upload_bytes=r["terrain_upload"], result="1") for r in rows]
    elif scenario == "session":
        rows = [dict(r,step=r["frame"]) for r in rows]
    if count < 1 or len(rows) != count or (end and lines.count(end) != 1) or any(r["result"] != "1" for r in rows):
        raise ValueError("Incomplete or failed measured sweep")
    if scenario != "zoom" and [int(r["step"]) for r in rows] != list(range(count)):
        raise ValueError("Reordered or duplicated samples")
    if scenario == "replay":
        expected_samples=int(args.get("replay_samples_per_phase",25));phase_counts=[0]*8
        events=[];previous=-1;zooms=set()
        for index,row in enumerate(rows):
            logical=int(row["logical_us"]);phase=int(row["phase"]);event=int(row["event"])
            if (int(row["step"])!=index or logical<=previous or phase not in range(8) or
                    int(row["tile_width"]) not in (128,160,192) or
                    not 0<int(row["units"])<=int(args["idle_units"]) or int(row["fallback"])!=0 or
                    int(row["recoveries"])!=0 or not row.get("request_hash")):
                raise ValueError("Deterministic replay order, semantics or completion is invalid")
            previous=logical;phase_counts[phase]+=1;zooms.add(int(row["tile_width"]))
            if event>=0:events.append((event,phase,int(row["tile_width"]),logical))
        expected_events=[(0,2,160,20000000),(1,2,192,22000000),(2,2,160,24000000),
                         (3,2,128,26000000),(4,3,128,28000000),(5,5,128,40000000),
                         (6,7,128,50000000)]
        if phase_counts!=[expected_samples]*8 or events!=expected_events or zooms!={128,160,192}:
            raise ValueError("Deterministic replay phase, event or zoom coverage is incomplete")
        if (int(timed_ends[0]["frames"])!=count or int(timed_ends[0]["phase_mask"])!=255 or
                int(timed_ends[0]["event_mask"])!=127 or int(timed_ends[0]["zoom_mask"])!=7 or
                int(timed_ends[0]["recoveries"])!=0):
            raise ValueError("Deterministic replay summary disagrees with frames")
        if mode=="baseline" and (int(prepare_ends[0]["requests"])!=0 or
                                  int(prepare_ends[0]["unit_requests"])!=0):
            raise ValueError("Baseline replay was warmed before timing")
        if mode=="oracle" and (not 0<int(prepare_ends[0]["requests"])<=count or
                int(prepare_ends[0].get("requested",0))!=count or prepare_ends[0].get("memory_safe")!="1" or
                sum(int(prepare_ends[0][k]) for k in ("cleared_viewport","cleared_regions","cleared_blocks",
                                                       "cleared_backdrops","cleared_publication"))<=0 or
                int(rows[0]["raster_cached_pixels"])!=0):
            raise ValueError("Oracle did not prepare all requests or clear completed map images")
    if scenario == "session":
        slot_us=int(starts[0]["input_slot_us"]);duration=int(starts[0]["duration_us"])
        previous_slot=-1;previous_done=0;phase_mask=zoom_mask=0;skipped=0
        queued=starts[0].get("input_model")=="queued_discrete_v1"
        event_times=[20000000,22000000,24000000,26000000,28000000,40000000,50000000]
        next_event=0
        if slot_us!=33333 or duration!=60000000 or starts[0].get("input_model","latest_state_v1") not in ("latest_state_v1","queued_discrete_v1"):
            raise ValueError("Unexpected busy-session input schedule")
        for row in rows:
            dispatch,done=int(row["dispatch_us"]),int(row["done_us"])
            slot=dispatch//slot_us;phase=int(row["phase"]);width=int(row["tile_width"])
            event=int(row.get("input_event",-1));requested=slot*slot_us
            if queued and event>=0:
                if event!=next_event or event>=len(event_times):
                    raise ValueError("Busy-session discrete event order is invalid")
                requested=event_times[event];next_event+=1
                if (phase,width)!=[(2,160),(2,192),(2,160),(2,128),(3,128),(5,128),(7,128)][event]:
                    raise ValueError("Busy-session discrete camera event is invalid")
            if queued and (int(row["dispatch_delay_us"])!=dispatch-requested or dispatch<requested or
                           (event<0 and next_event<len(event_times) and dispatch>=event_times[next_event])):
                raise ValueError("Busy-session queued input delay is invalid")
            if (not 0<=dispatch<(duration*3 if queued and event>=0 else duration) or dispatch<previous_done or done<dispatch or
                    int(row["requested_us"])!=requested or
                    int(row["skipped_slots"])!=slot-previous_slot-1 or slot<=previous_slot or
                    phase not in range(8) or width not in (128,160,192) or int(row["recoveries"])!=0):
                raise ValueError("Busy-session clock, camera, or completion order is invalid")
            previous_slot,previous_done=slot,done;skipped+=int(row["skipped_slots"])
            phase_mask|=1<<phase;zoom_mask|={128:1,160:2,192:4}[width]
        if queued and (next_event!=7 or int(timed_ends[0].get("discrete_events",0))!=7):
            raise ValueError("Busy-session discrete events are incomplete")
        coverage=int(phase_mask==255 and zoom_mask==7)
        if (int(timed_ends[0]["skipped_slots"])!=skipped or int(timed_ends[0]["phase_mask"])!=phase_mask or
                int(timed_ends[0]["zoom_mask"])!=zoom_mask or
                any(r.get("coverage_complete")!=str(coverage) for r in (timed_ends[0],ends[0])) or
                float(timed_ends[0]["wall_ms"])<max(duration,previous_done)/1000):
            raise ValueError("Busy-session schedule coverage is inconsistent")
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
              "endpoint_accounting": endpoint_accounting(lines,(directory/"renderer.log").read_text().splitlines() if (directory/"renderer.log").exists() else []),
              "wrapper_timing_ms":completion.get("wrapper_timing_ms"),
              "scenario": scenario, "viewport": [args["width"], args["height"]],
              "waves": args["waves"], "clock": f"unpaced stationary 15 Hz authored pose samples; {args.get('idle_warmup',10)} warmup renders" if scenario == "idle" else "six changing animation clocks" if scenario == "animation" else "fixed logical-time busy-session replay" if scenario == "replay" else "fixed replay clock; changing animation is a separate witness",
              "binaries": receipt["binaries"], "images": images,
              "image_receipt_verified": "images" in completion,
              "camera_requests": [{k: r[k] for k in ("cycle", "step", "x", "y", "pixel_y", "width", "tile_width", "ticks", "logical_us", "event", "request_hash", "dispatch_us", "phase") if k in r} for r in rows],
              "timing": {k: distribution([float(r[k]) for r in measured])
                         for k in ("ms", "capture_ms", "geometry_ms", "draw_ms", "readback_ms", "map_ms", "copy_ms", "units_ms")
                         if all(k in r for r in measured)},
              "mesh_builds": sum(int(r["built"]) for r in measured),
              "mesh_counter_scope": "Static terrain counters; wave/animation buffers are separate",
              "upload_bytes": sum(int(r["upload_bytes"]) for r in measured) if all("upload_bytes" in r for r in measured) else None,
              "native_first_response_ms": None, "native_presented_frames": None,
              "map_prepared_before_sweep": False if scenario == "distant" else None}
    if scenario == "replay":
        retained_fields=("retained_geometry","retained_natural","retained_ground","retained_waves",
                         "retained_pose","retained_payload","retained_shadow","retained_other")
        cleared_fields=("cleared_viewport","cleared_regions","cleared_blocks","cleared_backdrops","cleared_publication")
        report["endpoint"]="standalone deterministic capture, map rendering and unit/GDI completion; no native presentation"
        report["preparation"]={"mode":mode,"duration_ms":float(prepare_ends[0]["ms"]),
            "requests_examined":int(prepare_ends[0]["requests"]),
            "requests_requested":int(prepare_ends[0].get("requested",prepare_ends[0]["requests"])),
            "capacity_limited":prepare_ends[0].get("capacity_limited","0")=="1",
            "unit_requests":int(prepare_ends[0]["unit_requests"]),
            "unique_views":int(prepare_ends[0]["unique_views"]),
            "unique_poses":int(prepare_ends[0]["unique_poses"]),
            "geometry_admissions":int(prepare_ends[0]["geometry_admissions"]),
            "geometry_evictions":int(prepare_ends[0]["geometry_evictions"]),
            "capacity_geometry_evictions":int(prepare_ends[0].get("capacity_geometry_evictions",0)),
            "capacity_pose_evictions":int(prepare_ends[0].get("capacity_pose_evictions",0)),
            "geometry_upload_bytes":int(prepare_ends[0]["upload_bytes"]),
            "cleared_bytes":{k:int(prepare_ends[0][k]) for k in cleared_fields},
            "retained_bytes":{k:int(prepare_ends[0][k]) for k in retained_fields},
            "retained_entries":{k:int(prepare_ends[0][k]) for k in ("geometry_entries","pose_entries","wave_entries")}}
        report["replay"]={"samples_per_phase":expected_samples,"phase_counts":phase_counts,
            "events":events,"observed_zooms":sorted(zooms),"representative_parity_count":len(parity),
            "structural_builds":int(timed_ends[0]["built"]),
            "structural_evictions":int(timed_ends[0]["evicted"]),
            "structural_upload_bytes":int(timed_ends[0]["upload_bytes"]),
            "structural_preparation_complete":timed_ends[0]["structural_complete"]=="1"}
    if scenario == "session":
        report["clock"]="60-second wall-clock input script; no unit warm-up; post-session snapshot checks excluded"
        report["endpoint"]="standalone capture, map rendering and unit/GDI completion; no native presentation"
        workloads={}
        for phase in sorted({r["phase"] for r in rows},key=int):
            selected=[r for r in rows if r["phase"]==phase]
            workloads[selected[0]["label"]]={
                "timing":{k:distribution([float(r[k]) for r in selected]) for k in ("ms","capture_ms","map_ms","copy_ms","units_ms")},
                "visible_unit_occurrences":{"min":min(int(r["units"]) for r in selected),"max":max(int(r["units"]) for r in selected)},
                "late_camera_completions":sum(r["superseded"]=="1" for r in selected),
                "input_slots_skipped":sum(int(r["skipped_slots"]) for r in selected)}
        scheduled=math.ceil(duration/slot_us)
        report["preparation"]={"mode":mode,"duration_ms":float(prepare_ends[0]["ms"]),
            "requests_examined":int(prepare_ends[0]["requests"]),"unit_requests":int(prepare_ends[0]["unit_requests"]),
            "requests_requested":int(prepare_ends[0].get("requested",prepare_ends[0]["requests"])),
            "capacity_limited":prepare_ends[0].get("capacity_limited","0")=="1",
            "geometry_admissions":int(prepare_ends[0]["builds"]),
            "capacity_geometry_evictions":int(prepare_ends[0].get("capacity_geometry_evictions",0)),
            "capacity_pose_evictions":int(prepare_ends[0].get("capacity_pose_evictions",0)),
            "geometry_upload_bytes":int(prepare_ends[0]["upload_bytes"]),
            "cleared_bytes":{k:int(prepare_ends[0][k]) for k in ("cleared_viewport","cleared_regions","cleared_blocks","cleared_backdrops","cleared_publication")},
            "retained_bytes":{k:int(prepare_ends[0][k]) for k in ("retained_geometry","retained_natural","retained_ground","retained_waves","retained_pose","retained_payload","retained_shadow","retained_other")},
            "retained_entries":{k:int(prepare_ends[0][k]) for k in ("geometry_entries","pose_entries","wave_entries")}}
        report["session"]={"workloads":workloads,"initial_map_render_ms":float(starts[0]["initial_render_ms"]),
            "wall_ms":float(timed_ends[0]["wall_ms"]),"scheduled_input_slots":scheduled,
            "undispatched_input_slots":scheduled-sum(int(r["dispatch_us"])<duration for r in rows),
            "input_model":starts[0].get("input_model","latest_state_v1"),
            "discrete_events":[{"event":int(r["input_event"]),"phase":r["label"],"tile_width":int(r["tile_width"]),
                "dispatch_delay_ms":int(r["dispatch_delay_us"])/1000,
                "requested_to_completion_ms":(int(r["done_us"])-int(r["requested_us"]))/1000}
                for r in rows if int(r.get("input_event",-1))>=0],
            "post_input_settle_ms":max(0,float(timed_ends[0]["wall_ms"])-duration/1000),"schedule_coverage_complete":bool(coverage),
            "missing_phases":[i for i in range(8) if not phase_mask&(1<<i)],
            "observed_zooms":sorted({int(r["tile_width"]) for r in rows}),
            "snapshot_parity_count":len(parity),"snapshot_pixel_bytes":int(timed_ends[0]["snapshot_bytes"]),
            "evidence_overhead_ms":float(timed_ends[0]["evidence_ms"]),
            "completed_updates_per_second":count*1000/float(timed_ends[0]["wall_ms"]),
            "note":"The scripted producer advances while synchronous rendering blocks. Continuous slots coalesce; queued_discrete_v1 retains ordered zoom/minimap actions and reports their delay. This is a simulated input model, not native input handling. Missing phases remain missing; independent snapshot checks cannot turn them into a workload pass. Initial DLL load/configuration precedes the separately timed initial map render."}
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
    if scenario=="session":
        # Initial map render is followed by the timed synchronous calls, then
        # independent snapshot replays. Never use the last N trace records:
        # those may be cold verification work, or an incomplete file prefix.
        session_trace=trace
        if mode=="oracle":
            trims=[i for i,l in enumerate(trace) if "stage=oracle-trim " in l]
            session_trace=trace[trims[-1]+1:] if trims else []
        views=[fields(l) for l in session_trace if "stage=usage-view " in l][(0 if mode=="oracle" else 1):count+(0 if mode=="oracle" else 1)]
        aligned=[]
        for expected,observed in zip(rows,views):
            if observed.get("clock")!=str(1000000+int(expected["dispatch_us"])):
                break
            aligned.append((expected,observed))
        report["session_trace_coverage"]={"timed_frames":count,"aligned_views":len(aligned),
            "complete":len(aligned)==count,"post_session_replays_excluded":True}
        for phase in report["session"]["workloads"]:
            selected=[view for row,view in aligned if row["label"]==phase]
            report["session"]["workloads"][phase]["logged_object_count_samples"]=len(selected)
            if selected:
                report["session"]["workloads"][phase]["maximum_logged_objects"]={
                    k:max(int(v.get(k,0)) for v in selected) for k in ("visible","cities","roads","railroads","farms","mines","camps","resources")}
        report["unit_action_draws"]={k:sum(int(r[k]) for r in rows) for k in ("moving","attacking","fortifying","idling")}
        return receipt, report
    if scenario=="replay":
        reset_indices=[i for i,l in enumerate(trace) if "stage=reset " in l]
        trim_indices=[i for i,l in enumerate(trace) if "stage=oracle-trim " in l]
        timed_trace=[];preparation_trace=[]
        if mode=="oracle" and trim_indices:
            trim=trim_indices[-1]
            before=max((i for i in reset_indices if i<trim),default=-1)
            after=min((i for i in reset_indices if i>trim),default=len(trace))
            preparation_trace=trace[before+1:trim]
            timed_trace=trace[trim+1:after]
        elif mode=="baseline":
            segments=[]
            boundaries=[-1,*reset_indices,len(trace)]
            for first,last in zip(boundaries,boundaries[1:]):
                segment=trace[first+1:last]
                views=sum("stage=usage-view " in line for line in segment)
                segments.append((views,segment))
            timed_trace=max(segments,key=lambda item:item[0])[1] if segments else []
        usage=[fields(l) for l in timed_trace if "stage=usage-view " in l]
        unit_rows=[fields(l) for l in timed_trace if "stage=unit-body " in l]
        expected_units=sum(int(r["units"]) for r in rows)
        pose_hits=sum(r.get("cache_hit")=="1" for r in unit_rows)
        pose_misses=sum(r.get("cache_hit")=="0" for r in unit_rows)
        prepare_units=[fields(l) for l in preparation_trace if "stage=unit-body " in l]
        prepare_misses=sum(r.get("cache_hit")=="0" for r in prepare_units)
        retained_poses=report["preparation"]["retained_entries"]["pose_entries"]
        inferred_evictions=max(0,prepare_misses-retained_poses)
        report["replay_trace_coverage"]={"timed_frames":count,"usage_views":len(usage),
            "expected_unit_requests":expected_units,"unit_records":len(unit_rows),
            "complete":len(usage)==count and len(unit_rows)==expected_units,
            "post_replay_cold_checks_excluded":True}
        report["unit_pose_cache"]={"hits":pose_hits,"misses":pose_misses,
            "miss_rate":pose_misses/max(1,pose_hits+pose_misses),"preparation_admissions":prepare_misses,
            "inferred_minimum_preparation_evictions":inferred_evictions,
            "maximum_sampled_pixel_capacity_bytes":max((int(r["cache_bytes"]) for r in unit_rows),default=0)}
        animation_phases=[fields(l) for l in timed_trace if "stage=animation-phases " in l]
        if len(animation_phases)==count:
            report["animation_phases"]={k:distribution([float(r[k]) for r in animation_phases])
                for k in ("pose_prepare_ms","backdrop_submit_ms","animated_submit_ms",
                          "readback_submit_ms","readback_wait_ms","cpu_copy_ms")}
        animation=[fields(l) for l in timed_trace if "stage=animation-frame " in l]
        if len(animation)==count:
            report["animation"]={"timing":distribution([float(r["ms"]) for r in animation]),
                "totals":{k:sum(int(r[k]) for r in animation) for k in
                    ("wave_upload_bytes","wave_cells_built","wave_cells_reused","backdrop_hits","backdrop_misses")}}
        phases=[fields(l) for l in timed_trace if "stage=navigation-phases " in l]
        if len(phases)==count:
            report["cpu_phases"]={k:distribution([float(r[k]) for r in phases])
                for k in phases[0] if k.endswith("_ms") and all(k in r for r in phases)}
        wave_misses=report.get("animation",{}).get("totals",{}).get("wave_cells_built",0)
        missing=[]
        if report["preparation"]["requests_examined"]!=report["preparation"]["requests_requested"]:
            missing.append("request_coverage")
        if not report["replay"]["structural_preparation_complete"]:missing.append("structural_geometry")
        if pose_misses:missing.append("unit_poses")
        if wave_misses:missing.append("waves")
        if not report["replay_trace_coverage"]["complete"]:missing.append("diagnostic_trace_coverage")
        report["preparation"]["incomplete_owners"]=missing
        report["preparation"]["coverage"]="perfect" if mode=="oracle" and not missing else "partial" if mode=="oracle" else "baseline"
        return receipt,report
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
    controls = {"dependency_control", "index_control", "center_shore_control", "world_regions_control", "three_zoom_memory", "camera_view", "preparation_mode",
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
                    "C3X_RENDERER_PREVIEW_CAMERA_QUEUE", "C3X_RENDERER_PREVIEW_CAMERA_VIEW",
                    "C3X_RENDERER_PREVIEW_PREPARATION_MODE"}
    env_defaults={"C3X_RENDERER_PREVIEW_IDLE_STEPS":"","C3X_RENDERER_PREVIEW_IDLE_UNITS":"0","C3X_RENDERER_PREVIEW_IDLE_WARMUP":"10","C3X_RENDERER_PREVIEW_DENSE_SCENE":"","C3X_RENDERER_PREVIEW_UNIT_ACTIONS":"idle"}
    if ({k: v for k, v in (env_defaults|before.get("environment", {})).items() if k not in env_controls} !=
            {k: v for k, v in (env_defaults|after.get("environment", {})).items() if k not in env_controls} or
            old["camera_requests"] != new["camera_requests"]):
        raise ValueError("Paired environment or camera requests differ")
    exact=old["images"]==new["images"]
    result={"reference":old,"candidate":new,"all_images_exact":exact}
    if old["scenario"]=="replay":
        baseline,oracle=(old,new) if old["preparation"]["mode"]=="baseline" else (new,old)
        streams_identical=old["camera_requests"]==new["camera_requests"]
        valid=(baseline["preparation"]["mode"]=="baseline" and oracle["preparation"]["mode"]=="oracle" and
               streams_identical and exact and oracle["preparation"]["coverage"] in ("perfect","partial"))
        baseline_p95=baseline["timing"]["ms"]["p95_ms"];oracle_p95=oracle["timing"]["ms"]["p95_ms"]
        speedup=baseline_p95/oracle_p95 if oracle_p95 else math.inf
        components={k:oracle["timing"][k]["p95_ms"] for k in ("geometry_ms","draw_ms","readback_ms","units_ms")
                    if k in oracle["timing"]}
        if valid and speedup>=2 and oracle_p95<=125:
            next_experiment="causal preparation that survives camera cancellation"
        elif (valid and oracle["replay"]["structural_preparation_complete"] and
              components.get("draw_ms",0)>10 and components.get("draw_ms",0)==max(components.values(),default=0)):
            next_experiment="world-space regional batching"
        elif valid and (components.get("readback_ms",0)>10 or
                        components.get("readback_ms",0)==max(components.values(),default=-1)):
            next_experiment="staging-ring and GDI-compatible-surface readback"
        elif valid and (oracle["unit_pose_cache"]["miss_rate"]>.01 or components.get("units_ms",0)>10):
            next_experiment="causal unit-pose preparation and capacity"
        else:
            next_experiment="do not build a broad preparation scheduler"
        result["oracle_decision"]={"valid_comparison":valid,"streams_identical":streams_identical,
            "p95_speedup":speedup,"oracle_p95_ms":oracle_p95,"oracle_component_p95_ms":components,
            "next_experiment":next_experiment}
    return result


def compare_session_reference(reference, candidate):
    original,base=inspect(reference);receipt,report=inspect(candidate)
    if not receipt.get("case_manifest") or original.get("case_manifest"):
        raise ValueError("Use a persistent candidate and fresh one-shot reference")
    if receipt["inputs"]!=original["inputs"] or receipt["binaries"]!=original["binaries"]:
        raise ValueError("Session reference inputs/binaries differ")
    ignored={"out","binaries","case_repeats","case_reset","case_time_limit"}
    if {k:v for k,v in receipt["args"].items() if k not in ignored}!={k:v for k,v in original["args"].items() if k not in ignored}:
        raise ValueError("Session reference workload/config differs")
    completion=json.loads((reference/"evidence.json").read_text())
    exact=True
    for suffix in ("",".result.bmp"):
        name="zoom.bmp"+suffix
        expected=digest(reference/name)
        if completion.get("images",{}).get(name)!=expected:raise ValueError("Unverified fresh reference image")
        exact=exact and all(report["images"][f"zoom.bmp.case{i}.bmp"+suffix]==expected for i in range(len(report["cases"])))
    report["fresh_one_shot_images_exact"]=exact
    candidate_completion=json.loads((candidate/"evidence.json").read_text())
    report["setup_comparison"]={"one_shot_wrapper_ms":completion.get("wrapper_total_ms"),
        "session_wrapper_ms":candidate_completion.get("wrapper_total_ms"),"cases":len(report["cases"]),
        "session_wrapper_per_case_ms":candidate_completion["wrapper_total_ms"]/len(report["cases"]),
        "scope":"iteration wait amortization; not renderer or native presentation speedup"}
    report["all_images_exact"]=exact
    return report


def read_dense_diagnostic_case(folder, offsets):
    receipt = json.loads((folder / "inputs.json").read_text())
    evidence = json.loads((folder / "evidence.json").read_text())
    exclusivity = json.loads((folder / "exclusivity.json").read_text(encoding="utf-8-sig"))
    if exclusivity["checks"] < 2 or exclusivity["conflicts"]:
        raise ValueError("Concurrent renderer/compiler observed")
    if (evidence["returncode"] != 0 or evidence["invocation"] != receipt["invocation"] or
            not all(evidence.get(k) is True for k in
                    ("inputs_unchanged", "sources_unchanged", "binaries_unchanged")) or
            evidence.get("provisional") or evidence["endpoints"]["status"] == "invalid"):
        raise ValueError("Invalid diagnostic invocation: " + str(folder.relative_to(ROOT)))
    for name, expected in receipt["binaries"].items():
        if digest(folder / name) != expected:
            raise ValueError("Diagnostic binary changed")
    cases = evidence["endpoints"]["cases"]
    if len(cases) != 1 or cases[0]["request_offsets"] != offsets:
        raise ValueError("Diagnostic camera sequence differs")
    case = cases[0]
    requests = [r for r in case["endpoints"]["requests"] if r["role"] == "playback"]
    if len(requests) != len(offsets) or any(r["status"] != "success_ownership_checked" or
                                            not r["cpu_phase_accounting_valid"] for r in requests):
        raise ValueError("Missing diagnostic endpoint or ownership coverage")
    lines = (folder / "benchmark.log").read_text().splitlines()
    fixture = [fields(l) for l in lines if l.startswith("DENSE_FIXTURE ")]
    if not fixture or any(int(fixture[-1][k]) <= 0 for k in ("cities", "roads", "rails", "improvements", "resources")):
        raise ValueError("Dense fixture does not exercise all requested content")
    memory = [int(fields(l)["largest_free_region"]) for l in lines if l.startswith("CAMERA memory ")]
    if not memory:
        raise ValueError("Missing address-space samples")
    # Warmup and initial preparation stay out of the transition measurements.
    seen = {0}
    samples = []
    for offset, request in zip(offsets, requests):
        samples.append({"offset": offset, "kind": "revisit" if offset in seen else "first_exposure",
                        "captured_tiles": request["captured_tiles"],
                        "total_ms": request["request_to_checked_result_ms"],
                        **request["renderer_cpu_spans_ms"]})
        seen.add(offset)
    images = {name: evidence["images"][name] for name in
              ("zoom.bmp.case0.bmp", "zoom.bmp.case0.bmp.result.bmp")}
    trace = (folder / "renderer.log").read_text().splitlines()
    budgets = [fields(l) for l in trace if "gpu_geometry_cap=" in l]
    if not budgets:
        raise ValueError("Missing cache budget receipt")
    memory.extend(int(b["largest_free_region"]) for b in budgets)
    caps = {k: v for k, v in budgets[-1].items() if k.endswith("_cap")}
    if any({k: v for k, v in b.items() if k.endswith("_cap")} != caps for b in budgets):
        raise ValueError("Budgets changed inside diagnostic case")
    return {"folder": folder.relative_to(ROOT).as_posix(), "samples": samples,
            "mean_transition_ms": statistics.mean(s["total_ms"] for s in samples),
            "whole_trace_ms": sum(s["total_ms"] for s in samples),
            "phase_mean_ms": {k: statistics.mean(s[k] for s in samples)
                              for k in requests[0]["renderer_cpu_spans_ms"]},
            "initial_preparation_ms": case["endpoints"]["setup_ms"]["initial_render_preparation"],
            "warmup_ms": case.get("warmup_endpoints", {}).get("host_span_through_last_check_ms"),
            "resets": case["resets"], "fixture": fixture[-1], "budgets": caps,
            "exclusivity": exclusivity,
            "min_largest_free_bytes": min(memory), "headroom_pass": min(memory) >= 512 * 1024**2,
            "images": images, "wrapper_total_ms": evidence["wrapper_total_ms"],
            "identity": {k: receipt[k] for k in ("binaries", "inputs", "source_at_run")}}


def compare_dense_diagnostic_runs(baseline, candidate):
    """Few matched pairs reject large effects; overlap remains inconclusive."""
    b = [r["mean_transition_ms"] for r in baseline]
    c = [r["mean_transition_ms"] for r in candidate]
    savings = [x-y for x, y in zip(b, c)]
    useful = max(20.0, .1 * statistics.median(b))
    if len(b) < 2:
        decision = "inconclusive_insufficient_repetitions"
    elif min(savings) >= useful and max(c) < min(b):
        decision = "useful_causal_effect"
    elif max(savings) < useful:
        decision = "reject_as_primary_target"
    else:
        decision = "inconclusive_repeat_variation"
    return {"decision": decision, "baseline_ms": b, "candidate_ms": c,
            "paired_savings_ms": savings, "minimum_useful_ms": useful,
            "phase_savings_ms": {k: [x["phase_mean_ms"][k]-y["phase_mean_ms"][k]
                                     for x, y in zip(baseline, candidate)]
                                 for k in baseline[0]["phase_mean_ms"]}}


def summarize_dense_diagnostic(runs, arms=("full", "route_draws_omitted", "route_surfaces_omitted", "prepared_content", "half_geometry_pixels")):
    comparisons = {}
    for workload in ("four_columns", "reversal"):
        selected = {arm: [r for r in runs if r["workload"] == workload and r["arm"] == arm]
                    for arm in arms}
        # Restore repetition order after alternating the actual dispatch order.
        for values in selected.values():
            values.sort(key=lambda r: r["repeat"])
        base = selected["full"]
        if not base:
            continue
        for arm, values in selected.items():
            if len(values) != len(base):
                continue
            if any(r["images"] != values[0]["images"] for r in values):
                raise ValueError("Repeated diagnostic images differ: " + arm)
            if arm == "prepared_content" and any(r["images"] != b["images"] for r, b in zip(values, base)):
                raise ValueError("Prepared content differs from fresh content rendering")
            if arm != "full":
                comparisons[workload + "/" + arm] = {
                    **compare_dense_diagnostic_runs(base, values),
                    "correctness": "fresh-content endpoint pixels exact; intermediate full-redraw parity unmeasured"
                    if arm == "prepared_content" else "intentional pixel ablation; cannot pass production correctness",
                }
        if selected["route_draws_omitted"] and len(selected["route_surfaces_omitted"]) == len(selected["route_draws_omitted"]):
            comparisons[workload + "/route_construction_only"] = compare_dense_diagnostic_runs(
                selected["route_draws_omitted"], selected["route_surfaces_omitted"])
    return comparisons


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--case-reference", type=Path, help="Independent fresh process for persistent case output reproduction")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = compare_session_reference(args.case_reference,args.candidate) if args.case_reference else compare(args.reference, args.candidate) if args.reference else inspect(args.candidate)[1]
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"report": str(args.out), "all_images_exact": result.get("all_images_exact"),
                      "timing": result.get("candidate", result)["timing"]["ms"]}))
    return 1 if result.get("all_images_exact") is False else 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Read-only joins for bounded native routes and immutable presented-source witnesses.

Camera acceptance, native adoption, successful Present and physical scanout have
different meanings. Only explicit source serials join publication/workload to a
successful Present; counter polls never invent a same-frame correctness endpoint.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import statistics


ROOT = Path(__file__).resolve().parents[2]
TOKEN = re.compile(r"([A-Za-z_][A-Za-z_0-9]*)=([^\s]+)")
MAX_BYTES = 128 * 1024 * 1024
MAX_LINES = 1_000_000
STAGES = {
    "publication": "route-publication",
    "presented": "route-presented",
    "workload": "route-workload",
}


def integer(value):
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and re.fullmatch(r"-?\d+", value):
        return int(value)
    return None


def distribution(values):
    if not values:
        return {"n": 0, "status": "unavailable"}
    ordered = sorted(values)
    def quantile(fraction):
        position = (len(ordered)-1)*fraction
        lower = int(position)
        return ordered[lower]+(ordered[min(lower+1,len(ordered)-1)]-ordered[lower])*(position-lower)
    return {"n":len(values),"min":ordered[0],"p50":quantile(.5),
            "p95":quantile(.95),"max":ordered[-1],"mean":statistics.fmean(values)}


def parse_log(text):
    lines = text.splitlines()
    if len(lines) > MAX_LINES:
        raise ValueError("Log exceeds bounded line count")
    records = []
    for number,line in enumerate(lines,1):
        if "[C3X renderer]" not in line:
            continue
        fields = dict(TOKEN.findall(line.split("[C3X renderer]",1)[1]))
        if "stage" in fields:
            records.append({"line":number,**fields})
    return records


def pair(value):
    if isinstance(value,str) and re.fullmatch(r"-?\d+,-?\d+",value):
        return tuple(map(int,value.split(",")))
    return None


def scoped(records,begin,end):
    return [row for row in records if (qpc:=integer(row.get("qpc"))) is not None and begin <= qpc < end]


def present_records(core):
    output=[]
    for row in core:
        if row.get("stage")!=STAGES["presented"] or row.get("result")!="1":continue
        logged=integer(row.get("qpc"))
        exact="present_qpc" in row
        endpoint=integer(row.get("present_qpc")) if exact else logged
        if exact and (endpoint is None or endpoint<0 or logged is None or endpoint>logged):
            raise ValueError("Invalid raw successful-Present clock boundary")
        if endpoint is not None:
            if output:
                previous=output[-1];previous_index=integer(previous.get("present_index"));index=integer(row.get("present_index"))
                if index is not None and previous_index is not None and (index<=previous_index or endpoint<=previous["qpc"]):
                    raise ValueError("Successful-Present clock/index is not strictly ordered")
            output.append({**row,"qpc":endpoint,"log_qpc":logged,
                           "clock_endpoint":"successful Present return" if exact else "legacy diagnostic write"})
    return sorted(output,key=lambda row:row["qpc"])


def frame_budgets(presents,core,frequency):
    """Partition elapsed owner-thread intervals, never rounded durations or GPU time."""
    preparations={};budgets={};conflicts=set()
    phase_edges=("begin","candidate_end","selection_end","assets_end","render_begin","end")
    phase_names=("candidates","selection","assets","target","render")
    for row in core:
        stage=row.get("stage")
        if stage not in {"frame-preparation-ready","route-frame-budget"}:continue
        serial=integer(row.get("source_serial"));generation=integer(row.get("source_generation"))
        if stage=="frame-preparation-ready":
            if "begin" not in row or "end" not in row:continue
            begin,end=integer(row["begin"]),integer(row["end"])
            if begin is None or end is None or begin<0 or end<begin:
                raise ValueError("Negative or reversed raw frame-preparation clock interval")
            if integer(row.get("frequency"))!=frequency:continue
            edges=[integer(row.get(key)) for key in phase_edges]
            if all(key in row for key in phase_edges):
                if any(value is None or value<0 for value in edges) or any(b<a for a,b in zip(edges,edges[1:])):
                    raise ValueError("Negative or reversed raw frame-preparation subphase clock interval")
            else:edges=None
            if serial and serial>0 and generation and generation>0:
                preparations.setdefault((serial,generation),[]).append((begin,end,edges))
        else:
            index=integer(row.get("present_index"));times=[integer(row.get(key)) for key in ("begin","sampled","end")]
            if any(value is None or value<0 for value in times) or any(b<a for a,b in zip(times,times[1:])):
                raise ValueError("Negative or reversed raw frame-budget clock interval")
            if index is None or index<=0:continue
            key=(index,serial,generation)
            if key in budgets and any(budgets[key].get(field)!=row.get(field) for field in ("begin","sampled","end","frequency","operations","assemblies","copies","copied_pixels","assembly_pixels")):
                conflicts.add(key)
            budgets.setdefault(key,row)
    frames=[];refusals=[]
    for present in presents:
        key=(integer(present.get("present_index")),integer(present.get("source_serial")),integer(present.get("source_generation")))
        budget=budgets.get(key)
        if not budget:continue
        if key in conflicts or integer(budget.get("frequency"))!=frequency or integer(present.get("frequency"))!=frequency:
            refusals.append({"present_index":key[0],"reason":"conflicting budget or incompatible QPC frequency"});continue
        begin,sampled,end=(integer(budget[name]) for name in ("begin","sampled","end"))
        if present["clock_endpoint"]!="successful Present return" or end!=present["qpc"]:
            refusals.append({"present_index":key[0],"reason":"budget end differs from explicit successful-Present boundary or that boundary is unavailable"});continue
        spans={}
        for prep_begin,prep_end,edges in preparations.get(key[1:],[]):
            span=(prep_begin,prep_end)
            if span in spans and spans[span]!=edges:
                raise ValueError("Conflicting raw preparation boundaries for one source/generation")
            spans[span]=edges
        ordered=sorted(spans)
        if any(right[0]<left[1] for left,right in zip(ordered,ordered[1:])):
            raise ValueError("Overlapping raw preparation intervals for one source/generation")
        overlap=0;subphases={name:0 for name in phase_names};phase_complete=True
        for prep_begin,prep_end in ordered:
            clipped=max(0,min(sampled,prep_end)-max(begin,prep_begin));overlap+=clipped
            edges=spans[(prep_begin,prep_end)]
            if clipped and edges is None:phase_complete=False
            if edges:
                for name,left,right in zip(phase_names,edges,edges[1:]):
                    subphases[name]+=max(0,min(sampled,right)-max(begin,left))
        if overlap>sampled-begin:raise ValueError("Preparation overlap exceeds enclosing composition interval")
        milliseconds=lambda ticks:ticks*1000/frequency
        frame={"present_index":key[0],"source_serial":key[1],"source_generation":key[2],"begin_qpc":begin,"sampled_qpc":sampled,"end_qpc":end,
               "compose_ms":milliseconds(sampled-begin),"same_generation_preparation_overlap_ms":milliseconds(overlap),
               "residual_compose_ms":milliseconds(sampled-begin-overlap),"present_call_ms":milliseconds(end-sampled),"total_frame_work_ms":milliseconds(end-begin),
               "preparation_span_records":len(spans),"preparation_subphases_ms":{name:milliseconds(value) for name,value in subphases.items()} if phase_complete and spans else None,
               "preparation_coverage":"explicit same-source/generation completed-preparation intervals" if spans else "no matching completed-preparation interval; residual may contain other preparation work"}
        for name in ("operations","assemblies","copies","copied_pixels","assembly_pixels"):
            value=integer(budget.get(name))
            if value is not None and value>=0:frame[name]=value
        frames.append(frame)
    by_index={row["present_index"]:row for row in frames};gaps=[]
    for left,right in zip(presents,presents[1:]):
        first,last=integer(left.get("present_index")),integer(right.get("present_index"))
        frame=by_index.get(last)
        if first is None or last!=first+1 or frame is None:continue
        if left["clock_endpoint"]!="successful Present return":continue
        if frame["begin_qpc"]<left["qpc"]:raise ValueError("Raw frame clock begins before previous successful Present boundary")
        gaps.append({"present_index":last,"end_qpc":frame["end_qpc"],
                     "unclassified_cadence_queue_ownership_gap_ms":(frame["begin_qpc"]-left["qpc"])*1000/frequency,
                     "end_to_end_present_gap_ms":(right["qpc"]-left["qpc"])*1000/frequency})
    return {"frames":frames,"consecutive_gaps":gaps,"refusals":refusals,
            "scope":"raw CPU/API elapsed intervals; compose = same-generation preparation overlap + residual; frame = compose + Present; adjacent Present gap = unclassified cadence/queue/ownership gap + next frame. Rounded parent/subphase fields are not added; GPU duration and physical scanout unavailable"}


def budget_summary(budget,begin,end):
    frames=[row for row in budget["frames"] if begin<=row["end_qpc"]<end]
    gaps=[row for row in budget["consecutive_gaps"] if begin<=row["end_qpc"]<end]
    fields=("compose_ms","same_generation_preparation_overlap_ms","residual_compose_ms","present_call_ms","total_frame_work_ms","operations","assemblies","copies","copied_pixels","assembly_pixels")
    output={"frames":len(frames),"fields":{key:distribution([row[key] for row in frames if key in row]) for key in fields}}
    for key in ("unclassified_cadence_queue_ownership_gap_ms","end_to_end_present_gap_ms"):
        output["fields"][key]=distribution([row[key] for row in gaps])
    output["preparation_subphases_ms"]={name:distribution([row["preparation_subphases_ms"][name] for row in frames if row["preparation_subphases_ms"] is not None]) for name in ("candidates","selection","assets","target","render")}
    output["status"]="observed" if frames else "unavailable"
    return output


def stage_statistics(records):
    # Each row/stage is independently summarized. No subphase totals are added
    # to enclosing frame time, background work, native time or Present time.
    keys = {
        "fresh-scene-phases":("prepare","reflection","static","water","units","reconstruct"),
        "frame":("render_ms","geometry_ms","draw_submit_ms","readback_wait_ms","built","reused","evicted","upload_bytes","gpu_bytes","viewport_bytes"),
        "frame-preparation-ready":("age_ms","render_ms","units","turns","bytes","zoom","candidates_ms","selection_ms","assets_ms","target_ms","total_ms"),
        "map-complete":("total_ms",),
        "direct-visual":("prepare_ms","sample_ms","present_ms","total_ms","bytes"),
    }
    output={}
    for stage,names in keys.items():
        rows=[row for row in records if row.get("stage")==stage]
        if not rows:
            continue
        summary={"events":len(rows),"fields":{}}
        for name in names:
            values=[]
            for row in rows:
                try:value=float(row[name])
                except (KeyError,TypeError,ValueError):continue
                if math.isfinite(value):values.append(value)
            if values:summary["fields"][name]=distribution(values)
        output[stage]=summary
    return output


def counters(samples,begin,end,frequency):
    selected=[row for row in samples if begin <= integer(row.get("qpc")) < end]
    if len(selected)<2:
        return {"status":"unavailable","polls":len(selected)}
    deltas=[b["frames"]-a["frames"] for a,b in zip(selected,selected[1:])]
    if any(delta<0 for delta in deltas):
        return {"status":"invalid","reason":"presentation counter reversed"}
    elapsed=(selected[-1]["qpc"]-selected[0]["qpc"])/frequency
    memory={}
    for key in ("game_private_bytes","game_working_bytes","helper_private_bytes","helper_working_bytes"):
        values=[integer(row.get(key)) for row in selected]
        values=[value for value in values if value is not None]
        if values:memory[key]={"first":values[0],"last":values[-1],"peak":max(values)}
    return {"status":"observed","polls":len(selected),"observed_seconds":elapsed,
            "successful_present_counter_delta":selected[-1]["frames"]-selected[0]["frames"],
            "successful_present_counter_per_second":(selected[-1]["frames"]-selected[0]["frames"])/elapsed,
            "counter_increment_histogram":dict(sorted(Counter(deltas).items())),"memory":memory,
            "scope":"polled successful presentations; poll windows are not exact Present intervals"}


def workload_signature(row,members=None):
    # The producer documents the digest and actual pass-mask semantics. Missing
    # coverage or overflow refuses eligibility; an empty valid scene is allowed.
    if not row or row.get("complete") != "1" or row.get("overflow") != "0":
        return None
    keys=("facts_digest","main_units","reflected_units","shadow_units")
    if any(key not in row for key in keys):
        return None
    count=integer(row.get("count"))
    if count is None or count<0 or count>4096:return None
    segments=(count+31)//32
    records={}
    for segment in members or []:
        index=integer(segment.get("segment"));total=integer(segment.get("total_segments"))
        if index is None or index<0 or index>=segments or total!=segments:return None
        ids=segment.get("unit_ids","");masks=segment.get("pass_masks","")
        if index in records and records[index]!=(ids,masks):return None
        records[index]=(ids,masks)
    if len(records)!=segments:return None
    ids=[];masks=[]
    for index in range(segments):
        raw_ids,raw_masks=records[index]
        if not re.fullmatch(r"\d+(?:,\d+)*",raw_ids) or not re.fullmatch(r"[0-7](?:,[0-7])*",raw_masks):return None
        segment_ids=list(map(int,raw_ids.split(",")));segment_masks=list(map(int,raw_masks.split(",")))
        if len(segment_ids)!=min(32,count-index*32) or len(segment_ids)!=len(segment_masks):return None
        ids.extend(segment_ids);masks.extend(segment_masks)
    # A native unit can have several authoritative wrap occurrences. Preserve
    # their ordered entries/masks; the facts digest includes their exact anchors.
    for key,bit in (("main_units",1),("shadow_units",2),("reflected_units",4)):
        if integer(row[key])!=sum(bool(mask&bit) for mask in masks):return None
    return {**{key:row[key] for key in keys},"unit_ids":ids,"pass_masks":masks,"part_samples":row.get("part_samples")}


def analyze_capture(plan,result,inputs,cadence,native,core):
    frequency=integer(cadence.get("qpc_frequency"))
    if not frequency or frequency<=0:
        raise ValueError("Cadence requires positive raw QPC frequency")
    samples=cadence.get("samples",[])
    if len(samples)>20_000 or any(integer(row.get("qpc")) is None or integer(row.get("frames")) is None for row in samples):
        raise ValueError("Invalid or unbounded cadence samples")
    if any(b["qpc"]<=a["qpc"] for a,b in zip(samples,samples[1:])):
        raise ValueError("Cadence QPC is not strictly ordered")
    records=native+core
    failures=[row for row in records if row.get("stage") in {
        "native-operation-failed","async-publication-failed","visual-failure","worker-error",
        "unit-animation-failed","scripted-route-refused"}]
    accepted={};resolved={};adopted={};duplicate_accepts=[]
    for row in native:
        step=integer(row.get("step"));qpc=integer(row.get("qpc"))
        if not step or qpc is None:continue
        stage=row.get("stage")
        if stage=="scripted-route-accepted":
            if step in accepted:duplicate_accepts.append(step)
            else:accepted[step]=row
        elif stage=="scripted-route-resolved":resolved.setdefault(step,row)
        elif stage=="scripted-route-adopted" and row.get("valid")=="1":adopted.setdefault(step,row)
    native_sources={};native_requests={};local_sources={};local_requests={};ambiguous_cameras=set()
    for row in native:
        stage=row.get("stage")
        if stage not in {"route-native-source","route-native-request"}:continue
        field=("local_image_ticket" if "local_image_ticket" in row else "source_serial") if stage=="route-native-source" else ("local_camera_ticket" if "local_camera_ticket" in row else "camera_ticket")
        key=integer(row.get(field))
        camera=pair(row.get("camera"));qpc=integer(row.get("qpc"))
        if not key or key<0 or camera is None or qpc is None:continue
        if row.get("valid")!="1" or row.get("genuine")!="1" or integer(row.get("frequency"))!=frequency:continue
        mapping={"source_serial":native_sources,"camera_ticket":native_requests,"local_image_ticket":local_sources,"local_camera_ticket":local_requests}[field]
        namespace=field if field.startswith("local_") else stage
        if key in mapping and pair(mapping[key]["camera"])!=camera:ambiguous_cameras.add((namespace,key))
        mapping.setdefault(key,row)
    bindings=[];binding_local={};ambiguous_bindings=set()
    binding_fields=("local_camera_ticket","local_image_ticket","remote_camera_ticket","map_ticket","remote_session")
    for row in records:
        if row.get("stage")!="route-ticket-binding":continue
        values=tuple(integer(row.get(field)) for field in binding_fields)
        if any(value is None or value<=0 for value in values):continue
        if row.get("valid")!="1" or row.get("genuine")!="1" or integer(row.get("frequency"))!=frequency or integer(row.get("qpc")) is None or integer(row["qpc"])<0:continue
        for field,value in zip(binding_fields[:2],values[:2]):
            key=(field,value)
            if key in binding_local and binding_local[key]!=values:ambiguous_bindings.add(key)
            binding_local.setdefault(key,values)
        bindings.append(row)
    publications={};ambiguous_serials=set();ambiguous_workloads=set();workloads={};members={}
    contract_keys=("camera","camera_ticket","map_ticket","remote_camera_ticket","remote_session","phase_x","phase_y","identity","tile_width","tile_height","anchor_basis","anchor_x","anchor_y","width","height","tiles","map_epoch","viewer_epoch","visibility_epoch","scene_epoch")
    for row in core:
        serial=integer(row.get("serial" if row.get("stage")==STAGES["publication"] else "source_serial"))
        if not serial or serial<0:continue
        if row.get("stage")==STAGES["publication"]:
            if serial in publications and any(publications[serial].get(key)!=row.get(key) for key in contract_keys):
                ambiguous_serials.add(serial)
            publications.setdefault(serial,row)
        elif row.get("stage")==STAGES["workload"]:
            generation=integer(row.get("source_generation"))
            if generation is None or generation<=0:continue
            key=(serial,generation)
            # Only identical facts/actual contributor lists can share one source.
            if key in workloads and any(workloads[key].get(field)!=row.get(field) for field in ("facts_digest","count","main_units","reflected_units","shadow_units","camera_x","camera_y","zoom")):
                ambiguous_workloads.add(key)
            workloads.setdefault(key,row)
        elif row.get("stage")=="route-workload-members":
            generation=integer(row.get("source_generation"))
            if generation is not None and generation>0:members.setdefault((serial,generation),[]).append(row)
    def camera_proofs(serial,publication):
        # Older traces keep their original namespaces and contradictions. New
        # traces must copy the local-to-remote relationship at actual adoption;
        # equality of camera reservations and source serials proves nothing.
        ticket=integer(publication.get("camera_ticket"))
        if ("route-native-source",serial) in ambiguous_cameras or ("route-native-request",ticket) in ambiguous_cameras:return None
        proofs=[proof for proof in (native_sources.get(serial),native_requests.get(ticket)) if proof is not None]
        if "map_ticket" not in publication:return proofs
        remote=tuple(integer(publication.get(field)) for field in ("map_ticket","remote_camera_ticket","remote_session"))
        if any(value is None or value<=0 for value in remote):return None
        matches=[binding for binding in bindings if tuple(integer(binding[field]) for field in ("map_ticket","remote_camera_ticket","remote_session"))==remote]
        if not matches:return None
        bound=[]
        for binding in matches:
            local_camera=integer(binding["local_camera_ticket"]);local_image=integer(binding["local_image_ticket"])
            if ("local_camera_ticket",local_camera) in ambiguous_bindings or ("local_image_ticket",local_image) in ambiguous_bindings:return None
            if ("local_camera_ticket",local_camera) in ambiguous_cameras or ("local_image_ticket",local_image) in ambiguous_cameras:return None
            for proof in (local_sources.get(local_image),local_requests.get(local_camera)):
                if proof is not None:
                    bound.append({**proof,**{field:binding[field] for field in binding_fields},
                                  "native_qpc":proof["qpc"],"binding_qpc":binding["qpc"],
                                  "qpc":max(integer(proof["qpc"]),integer(binding["qpc"]))})
        return proofs+bound if bound else None
    presents=present_records(core)
    budget=frame_budgets(presents,core,frequency)
    posted={integer(row.get("step")):row for row in result.get("events",[]) if row.get("kind")=="posted"}
    route=[]
    for item in plan["steps"]:
        step=item["step"];target=(item["x"],item["y"]);width=item["width"]
        request=accepted.get(step);resolution=resolved.get(step);camera=adopted.get(step)
        row={"step":step,"name":item["name"],"requested_camera":list(target),"requested_width":width,"actual_workload":None}
        reasons=[]
        if not request:reasons.append("native acceptance with explicit step/raw QPC unavailable")
        if not resolution or pair(resolution.get("native"))!=target:
            reasons.append("resolved native camera missing or different from planned absolute target")
        if request and pair(request.get("requested"))!=target:reasons.append("accepted target differs from plan")
        if request and integer(request.get("target_width"))!=width:reasons.append("accepted zoom target differs from plan")
        if request and integer(request.get("frequency"))!=frequency:reasons.append("native and cadence QPC frequencies differ or are absent")
        if request and any(integer(request.get(key))!=value for key,value in (("capture_width",128),("map_width",plan["map_width"]),("map_height",plan["map_height"]))):
            reasons.append("accepted native map dimensions or canonical capture basis differ")
        if step in duplicate_accepts:reasons.append("duplicate acceptance for one step")
        begin=integer(request.get("qpc")) if request else None
        later=[integer(value.get("qpc")) for key,value in accepted.items() if key>step]
        end=min(later) if later else (samples[-1]["qpc"]+1 if samples else (begin or 0)+1)
        if begin is not None:
            if step in posted:
                latency=(begin-posted[step]["qpc"])*1000/frequency
                row["posted_to_native_acceptance_ms"]=latency
                if latency<0:reasons.append("native acceptance precedes its explicitly linked posted input")
            unchanged=camera is not None and camera.get("camera_already_adopted")=="1"
            camera_ok=camera is not None and pair(camera.get("requested"))==target and pair(camera.get("displayed"))==target and integer(camera.get("qpc"))>=begin
            row["native_camera_adoption"]={"status":"already correct at acceptance" if unchanged else "observed" if camera_ok else "unavailable"}
            if camera_ok:row["native_camera_adoption"]["native_acceptance_to_adoption_ms"]=(integer(camera["qpc"])-begin)*1000/frequency
            if item.get("advance")!="accepted" and not (camera_ok or unchanged):reasons.append("exact camera-adoption acknowledgement unavailable")
            endpoint=None;contract=None;camera_proof=None;selected_workload=None
            for present in scoped(presents,begin,end):
                serial=integer(present.get("source_serial"));generation=integer(present.get("source_generation"));publication=publications.get(serial)
                if not serial or serial in ambiguous_serials or not publication or present.get("mixed")!="0":continue
                if generation is None or generation<=0 or (serial,generation) in ambiguous_workloads:continue
                proofs=camera_proofs(serial,publication)
                if not proofs or any(pair(proof["camera"])!=target for proof in proofs):continue
                available=[proof for proof in proofs if integer(proof["qpc"])<=integer(present["qpc"])]
                if not available:continue
                metadata=min(available,key=lambda proof:integer(proof["qpc"]))
                if integer(present.get("zoom_q16"))!=width*512:continue
                if integer(present.get("frequency"))!=frequency:continue
                if integer(publication.get("width"))!=plan["client_width"] or integer(publication.get("height"))!=plan["client_height"]:continue
                if integer(publication.get("tile_width"))!=128 or integer(publication.get("tile_height"))!=64:continue
                if integer(publication.get("qpc")) is None or integer(publication["qpc"])>integer(present["qpc"]):continue
                workload=workloads.get((serial,generation))
                signature=workload_signature(workload,members.get((serial,generation)))
                if signature is None:continue
                if integer(workload.get("qpc")) is None or integer(workload["qpc"])>integer(present["qpc"]):continue
                # Producer origins belong to this exact copied frame. They
                # never infer a transform of Civ III's native camera.
                if integer(workload.get("camera_x"))!=integer(publication.get("anchor_x")) or integer(workload.get("camera_y"))!=integer(publication.get("anchor_y")):continue
                try:workload_zoom=float(workload["zoom"])
                except (KeyError,TypeError,ValueError):continue
                if workload_zoom!=width/128:continue
                # A rounded presentation counter can reach its target before
                # the frame's float projection settles. Keep searching until
                # all existing source, workload and exact-projection guards pass.
                endpoint=present;contract=publication;camera_proof=metadata;selected_workload=signature;break
            if endpoint:
                serial=integer(endpoint["source_serial"]);generation=integer(endpoint["source_generation"])
                row["correct_destination_present"]={"status":"observed","source_serial":serial,"source_generation":generation,"qpc":integer(endpoint["qpc"]),
                    "native_acceptance_to_present_ms":(integer(endpoint["qpc"])-begin)*1000/frequency,
                    "posted_to_present_ms":(integer(endpoint["qpc"])-posted[step]["qpc"])*1000/frequency if step in posted else None,
                    "clock_endpoint":endpoint["clock_endpoint"],"diagnostic_log_qpc":endpoint["log_qpc"],
                    "scope":"successful Present of an explicitly identified copied map source at requested projection; legacy logs without present_qpc provide the subsequent diagnostic-write boundary; physical scanout unavailable"}
                row["publication_contract"]={key:contract.get(key) for key in contract_keys if key not in {"map_ticket","remote_camera_ticket","remote_session"} or key in contract}
                row["camera_source_proof"]={key:camera_proof.get(key) for key in ("stage","source_serial","camera_ticket","camera","qpc")}
                row["camera_source_proof"].update({key:camera_proof[key] for key in (*binding_fields,"native_qpc","binding_qpc") if key in camera_proof})
                row["actual_workload"]=selected_workload
            else:
                row["correct_destination_present"]={"status":"unavailable","reason":"no unique source-serial publication and complete actual workload joined to successful Present at exact requested camera/zoom"}
                reasons.append("same-frame correct-destination presentation endpoint unavailable")
            row["request_window"]={"begin_qpc":begin,"end_qpc_exclusive":end}
            row["counter_observations"]=counters(samples,begin,end,frequency)
            row["stage_statistics"]=stage_statistics(scoped(records,begin,end))
            row["frame_budget_summary"]=budget_summary(budget,begin,end)
        row["route_eligible"]=not reasons;row["eligibility_reasons"]=reasons
        if item["name"]=="eviction-candidate-return":
            row["evicted_jump_qualification"]={"status":"unavailable","reason":"requires explicit destination eviction/restoration evidence; route distance and generic eviction counters do not identify it"}
        route.append(row)
    idle_begin=integer(result.get("idle_begin_qpc"));idle_end=integer(result.get("idle_end_qpc"))
    idle_workloads=[];idle_present_count=0;idle_joined_count=0
    if idle_begin is not None and idle_end is not None:
        for present in scoped(presents,idle_begin,idle_end):
            idle_present_count+=1
            serial=integer(present.get("source_serial"));generation=integer(present.get("source_generation"))
            publication=publications.get(serial)
            if not publication or present.get("mixed")!="0" or serial in ambiguous_serials or (serial,generation) in ambiguous_workloads:continue
            proofs=camera_proofs(serial,publication)
            if not proofs or any(pair(proof["camera"])!=tuple(plan.get("initial_camera",())) for proof in proofs):continue
            if not any(integer(proof["qpc"])<=integer(present["qpc"]) for proof in proofs):continue
            if integer(present.get("zoom_q16"))!=plan.get("initial_width",128)*512:continue
            workload=workloads.get((serial,generation))
            if not workload or integer(workload.get("qpc")) is None or integer(workload["qpc"])>integer(present["qpc"]):continue
            signature=workload_signature(workload,members.get((serial,generation)))
            if signature is None:continue
            idle_joined_count+=1
            if signature not in idle_workloads:idle_workloads.append(signature)
    ready=integer(result.get("ready_map_qpc"))
    intervals=[]
    after_ready=[row for row in presents if ready is not None and integer(row["qpc"])>=ready]
    gaps=[]
    for left,right in zip(after_ready,after_ready[1:]):
        elapsed=(integer(right["qpc"])-integer(left["qpc"]))*1000/frequency
        first,last=integer(left.get("present_index")),integer(right.get("present_index"))
        if first is not None and last==first+1:intervals.append(elapsed)
        else:gaps.append({"begin_qpc":integer(left["qpc"]),"end_qpc":integer(right["qpc"]),"observed_span_ms":elapsed,"present_index_delta":last-first if first is not None and last is not None else None})
    integrity=all(result.get(key) is True for key in ("original_save_unchanged","disposable_save_unchanged","ini_restored"))
    return {"capture_passed":result.get("passed") is True,"capture_failure":result.get("failure"),"integrity_passed":integrity,
        "native_failures":len(failures),"raw_qpc_frequency":frequency,"loading_ready_seconds":result.get("ready_map_seconds"),
        "idle":counters(samples,idle_begin,idle_end,frequency) if idle_begin is not None and idle_end is not None else {"status":"unavailable"},
        "idle_actual_workloads":idle_workloads,"idle_workload_coverage":{"successful_present_witnesses":idle_present_count,"exact_source_workload_joins":idle_joined_count},
        "idle_workload_eligible":idle_present_count>0 and idle_present_count==idle_joined_count,
        "route":route,"all_route_endpoints_eligible":bool(route) and all(row["route_eligible"] for row in route),
        "source_serial_refusals":sorted(ambiguous_serials),"source_generation_refusals":[list(key) for key in sorted(ambiguous_workloads)],"native_camera_refusals":[list(key) for key in sorted(ambiguous_cameras)],"ticket_binding_refusals":[list(key) for key in sorted(ambiguous_bindings)],"ready_successful_present_intervals_ms":distribution(intervals),
        "unqualified_present_spans":gaps,
        "ready_frame_budget":budget_summary(budget,ready,samples[-1]["qpc"]+1) if ready is not None and samples else {"status":"unavailable"},
        "idle_frame_budget":budget_summary(budget,idle_begin,idle_end) if idle_begin is not None and idle_end is not None else {"status":"unavailable"},
        "frame_budget":budget,
        "long_present_intervals":{str(limit):sum(value>limit for value in intervals) for limit in (1000/60,1000/30,50,100)},
        "present_interval_scope":"consecutive present_index witnesses use exact successful Present returns where present_qpc exists; older logs use explicitly labeled diagnostic-write boundaries; missing indices remain aggregated spans; physical scanout unavailable",
        "identity":{key:inputs.get(key) for key in ("game_sha256","save_sha256","plan_sha256","binaries","common_injected_diagnostic_delta","baseline_full_system_pristine")}}


def compare(baseline,candidate):
    identity_keys=("game_sha256","save_sha256","plan_sha256")
    fixture_match=all(baseline["identity"].get(key) and baseline["identity"].get(key)==candidate["identity"].get(key) for key in identity_keys)
    capture_integrity=all(arm["capture_passed"] and arm["integrity_passed"] and arm["native_failures"]==0 for arm in (baseline,candidate))
    paired=[]
    for left,right in zip(baseline["route"],candidate["route"]):
        target_match=all(left.get(key)==right.get(key) for key in ("step","name","requested_camera","requested_width"))
        workload_match=left.get("actual_workload") is not None and left.get("actual_workload")==right.get("actual_workload")
        eligible=capture_integrity and fixture_match and target_match and workload_match and left["route_eligible"] and right["route_eligible"]
        paired.append({"step":left["step"],"name":left["name"],"same_actual_route":target_match,
                       "same_actual_workload":workload_match,"performance_comparison_eligible":eligible,
                       "baseline_present_latency":left.get("correct_destination_present"),"candidate_present_latency":right.get("correct_destination_present")})
    complete=len(baseline["route"])==len(candidate["route"])==len(paired)
    idle_match=baseline["idle_workload_eligible"] and candidate["idle_workload_eligible"] and baseline["idle_actual_workloads"]==candidate["idle_actual_workloads"]
    return {"same_save_plan_common_injected_executable":fixture_match,"capture_integrity_passed":capture_integrity,"paired_steps":paired,
        "idle_performance_comparison_eligible":capture_integrity and fixture_match and idle_match,"same_actual_idle_workload":idle_match,
        "all_step_comparisons_eligible":complete and bool(paired) and all(row["performance_comparison_eligible"] for row in paired),
        "baseline_full_system_pristine":False,"scope":"common explicitly instrumented injected executable; renderer-side comparison only; no causal speedup inferred from unequal or missing workload"}


def read_capture(directory,receipts):
    directory=directory.resolve()
    if not directory.is_relative_to(ROOT):raise ValueError("Capture must be inside the repository")
    def read(leaf,required=True):
        path=(directory/leaf).resolve()
        if not path.is_relative_to(ROOT):raise ValueError("Capture leaf resolves outside repository")
        if not path.exists() and not required:return b""
        if path.stat().st_size>MAX_BYTES:raise ValueError("Input exceeds bounded bytes")
        data=path.read_bytes();receipts[path.relative_to(ROOT).as_posix()]={"sha256":hashlib.sha256(data).hexdigest(),"bytes":len(data)}
        return data
    plan=json.loads(read("route-plan.json"));result=json.loads(read("result.json"));inputs=json.loads(read("inputs.json"));cadence=json.loads(read("cadence.json"))
    native=parse_log(read("renderer.log").decode("utf-8-sig",errors="replace"))
    core=parse_log(read("renderer-core.log.x64",False).decode("utf-8-sig",errors="replace"))
    return analyze_capture(plan,result,inputs,cadence,native,core)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline",type=Path,required=True);parser.add_argument("--candidate",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True);args=parser.parse_args()
    receipts={};baseline=read_capture(args.baseline,receipts);candidate=read_capture(args.candidate,receipts)
    for relative,receipt in receipts.items():
        if hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()!=receipt["sha256"]:raise ValueError("Input changed during analysis")
    report={"schema":"c3x.matched_native_route_analysis.v1","baseline":baseline,"candidate":candidate,
            "comparison":compare(baseline,candidate),"inputs":receipts,
            "timing_policy":"raw QPC endpoints; event stages separately summarized; no summed overlapping CPU/GPU/background spans",
            "gpu_duration":"unavailable unless separate supported nonblocking GPU timestamp witness is supplied"}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+"\n")
    print("Matched route analysis written; eligibility is reported separately from capture completion.")


if __name__=="__main__":main()

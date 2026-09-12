"""Run an isolated current-code navigation/zoom witness with matched input receipts.

Uses the existing Windows dispatcher. Completion is render-plus-capture, never
native game presentation. Asset hashing and BMP writing are outside that interval.
"""
import argparse
import hashlib
import json
import os
import platform
import re
from pathlib import Path
import shutil
import time
import uuid

from Renderer.lab.platform import ROOT, native_command_result, windows_root


def storage_preflight(root, width, height, samples):
    if width <= 0 or height <= 0 or samples < 0:
        raise ValueError("Invalid evidence dimensions or sample count")
    # Reserve startup/navigation images and logs in addition to the saved sweep.
    # Keep eight GiB free after the estimate; never rely on filesystem compression.
    estimated = (width * height * 4 + 54) * (samples + 64) + 128 * 1024**2
    reserve = 8 * 1024**3
    free = shutil.disk_usage(root).free
    if free < estimated + reserve:
        raise RuntimeError(f"Insufficient evidence disk space: {free / 1024**3:.1f} GiB free; "
                           f"need {(estimated + reserve) / 1024**3:.1f} GiB including reserve. "
                           "Clean obsolete generated images before another run.")
    return {"free_bytes": free, "estimated_output_bytes": estimated, "reserve_bytes": reserve}


def digest(path):
    with path.open("rb") as stream:
        result=hashlib.sha256()
        for chunk in iter(lambda:stream.read(1024*1024),b""):
            result.update(chunk)
        return result.hexdigest()


def preparation_mode(scenario, requested):
    if requested is not None and scenario not in ("replay", "session"):
        raise ValueError("--preparation-mode is valid only for replay or session scenarios")
    return requested or "baseline"


def inputs():
    paths = set()
    # Offline source studies are never loaded by the DLL. Hash the definition
    # packs and all currently hard-coded production companion packs instead.
    packs = {"TerrainProfileR1", "ResourceAnimationRuntime", "TileSitesRuntime",
             "NaturalFidelityRuntime", "CityCompositionRuntime", "CoastalWavesRuntime",
             "UnitAnimationRuntime", "UnitNormalFidelity"}
    for definitions in (ROOT / "Renderer").glob("*.custom_rendering.txt"):
        packs.update(re.findall(r"Renderer[\\/]packs[\\/]([\w-]+)", definitions.read_text()))
    for directory in [*("Renderer/packs/" + p for p in sorted(packs)), "Renderer/native/render_core",
                      "Renderer/native/source_fidelity", "Renderer/native/environment_refresh",
                      "Renderer/native/city_fidelity", "Renderer/lab/shared"]:
        paths.update(p for p in (ROOT / directory).rglob("*") if p.is_file()
                     and not any(part in ("__pycache__", ".cache", "out", "build") for part in p.parts))
    paths = {p for p in paths if p.suffix.lower() not in (".py", ".pyc", ".cpp", ".h", ".md")}
    paths.update((ROOT / "Renderer").glob("*.custom_rendering.txt"))
    paths.add(ROOT / "Renderer/lab/.local/verification/world.csv")
    return {p.relative_to(ROOT).as_posix(): {"bytes": p.stat().st_size, "sha256": digest(p)}
            for p in sorted(paths)}


def metadata_snapshot(paths):
    result={}
    for relative in paths:
        try:
            info=(ROOT/relative).stat()
            result[relative]={"bytes":info.st_size,"mtime_ns":info.st_mtime_ns}
        except FileNotFoundError:result[relative]=None
    return result


def main(argv=None):
    wrapper_started=time.perf_counter()
    timing={}
    phase_started=wrapper_started
    def phase(name):
        nonlocal phase_started
        now=time.perf_counter();timing[name]=(now-phase_started)*1000;phase_started=now
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binaries", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--scenario", choices=("navigation", "zoom", "animation", "ambient", "idle", "distant", "replay", "session", "scroll"), default="navigation")
    parser.add_argument("--preparation-mode", choices=("baseline", "oracle"), default=None,
                        help="Cold baseline or complete untimed retained preparation for replay/session")
    parser.add_argument("--replay-samples-per-phase", type=int, choices=range(1,101), default=25)
    parser.add_argument("--idle-steps", type=int, choices=range(1,1001), default=100)
    parser.add_argument("--idle-warmup", type=int, choices=range(10,151), default=10)
    parser.add_argument("--unit-pose-memory", action="store_true", help="Opt-in 256 MiB / 4096-entry exact unit-pose retention")
    parser.add_argument("--unit-pose-memory-mib", type=int, choices=(256,512), default=256, help="Bounded pose-pixel budget when --unit-pose-memory is enabled")
    parser.add_argument("--idle-units", type=int, choices=(0,8,24,64), default=0,
                        help="Draw this many separate native-directed idle unit bodies in a synthetic GDI canvas")
    parser.add_argument("--unit-actions", choices=("idle","realistic","mixed"), default="idle",
                        help="Use frozen idle units, a realistic active subset, or worst-case mixed native timelines")
    parser.add_argument("--dense-scene", action="store_true", help="Stable synthetic cities, infrastructure, camps and supported resources on the captured world")
    parser.add_argument("--boundary-fixture", action="store_true", help="Small guarded-region correctness witness with independent full redraws")
    parser.add_argument("--prepared-resource-pass", action="store_true", help="Use the explicit animated body/shadow binding contract")
    parser.add_argument("--width", type=int, default=2240)
    parser.add_argument("--height", type=int, default=1192)
    parser.add_argument("--distant-steps", type=int, choices=range(1,1001), default=100)
    parser.add_argument("--cycles", type=int, choices=range(2,41), default=2)
    parser.add_argument("--resident", action="store_true")
    parser.add_argument("--cold", action="store_true")
    parser.add_argument("--waves", choices=("0", "1"), default="1")
    parser.add_argument("--tier", choices=("normal", "384", "768"), required=True)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--instrumentation", choices=("diagnostic", "timing"), default="diagnostic",
                        help="Buffered detailed traces or low-overhead endpoints with unmeasured internal phases")
    parser.add_argument("--case-repeats", type=int, choices=range(1,17), help="Bounded same-config persistent scroll cases")
    parser.add_argument("--case-reset", choices=("process_cold","assets_loaded","prepared_resident"), default="assets_loaded")
    parser.add_argument("--case-time-limit", type=int, choices=range(1,601), default=60)
    parser.add_argument("--exclusive-gpu", action="store_true",
                        help="Reject a case if another Lab renderer/compiler is observed by the child watchdog")
    parser.add_argument("--verification", choices=("acceptance","quick"), default="acceptance",
                        help="Quick checks are provisional; acceptance hashes inputs before and after")
    parser.add_argument("--scroll-sequence", action="store_true", help="Use the existing 14-offset reversal instead of one four-column move")
    parser.add_argument("--block-clip", choices=("0", "1"), default="0")
    parser.add_argument("--resident-steps", type=int, default=14)
    parser.add_argument("--tile-width", type=int, choices=(64,96,128,160,192), default=128)
    parser.add_argument("--caster-control", action="store_true")
    parser.add_argument("--raster-control", action="store_true")
    parser.add_argument("--region-size", type=int, choices=(128,256,512,2240), default=128)
    parser.add_argument("--bounded-post", action="store_true", help="Limit experimental strip reconstruction to guarded output regions")
    parser.add_argument("--reflection-ablation", action="store_true", help="Diagnostic only: omit reflections to estimate their cost; images are not current quality")
    parser.add_argument("--diagnostic-routes", choices=("full", "draw", "all"), default="full",
                        help="Benchmark only: omit route surface draws or also construction; bridge objects remain")
    parser.add_argument("--diagnostic-half-pixels", action="store_true",
                        help="Benchmark only: halve geometry scissor coverage with unchanged scene and draw candidates")
    parser.add_argument("--world-grid", action="store_true")
    parser.add_argument("--world-regions", action="store_true", help="Reuse completed static world-grid render regions")
    parser.add_argument("--region-metadata-mib", type=int, choices=(32,96), default=96)
    parser.add_argument("--center-shore-control", action="store_true", help="Independently recompute tile-center shoreline samples")
    parser.add_argument("--index-control", action="store_true", help="Scan every chunk independently for each region")
    parser.add_argument("--dependency-control", action="store_true", help="Recompute shadow dependency proofs each request")
    parser.add_argument("--region-diagnostics", action="store_true", help="Trace component fingerprints; diagnostic timings are not performance evidence")
    parser.add_argument("--region-receiver-shadows", action="store_true", help="Limit completed-region shadow dependencies to receiver reach")
    parser.add_argument("--tight-natural-bounds", action="store_true", help="Use actual projected natural mesh extrema for culling")
    parser.add_argument("--region-input-ring", type=int, choices=(2,4), default=2, help="Use this many tiles of current captured appearance support")
    parser.add_argument("--world-regions-control", action="store_true", help="Draw identical full regions independently without caching")
    parser.add_argument("--camera-view", action="store_true", help="Exercise the versioned camera queue and exact publication identity")
    parser.add_argument("--three-zoom-memory", action="store_true", help="Opt-in bounded 64 MiB viewport / 832 MiB MSAA backdrop retention experiment")
    parser.add_argument("--water-coverage", action="store_true", help="Omit provably empty water/bed passes and unused reflections")
    parser.add_argument("--world-backdrops", action="store_true", help="Retain world-anchored animation backdrops across camera translations")
    parser.add_argument("--backdrop-dependencies", action="store_true", help="Reuse exact static region dependencies for retained linear color/depth backgrounds")
    parser.add_argument("--composition-receiver-index", action="store_true", help="Use the retained static spatial index for exact animation shadow-receiver selection")
    parser.add_argument("--production-defaults", action="store_true", help="Exercise new DLL defaults inherited from the existing world-region cache switch")
    parser.add_argument("--backdrop-control", action="store_true", help="Independently redraw every static animation backdrop")
    parser.add_argument("--world-waves", action="store_true", help="Retain immutable coast-cell wave buffers across cameras")
    parser.add_argument("--wave-control", action="store_true", help="Rebuild coast-cell wave buffers for independent comparisons")
    parser.add_argument("--composition-casters-control", action="store_true", help="Rebuild caster preparation independently for each animation region")
    parser.add_argument("--animation-readback-atlas", action="store_true", help="Pack exact animated blocks into a compact staging atlas before CPU readback")
    args = parser.parse_args(argv)
    try:
        args.preparation_mode=preparation_mode(args.scenario,args.preparation_mode)
    except ValueError as error:
        parser.error(str(error))
    if args.idle_units and args.scenario not in ("ambient", "idle", "replay", "session"):
        parser.error("--idle-units requires --scenario ambient, idle, replay or session")
    if args.unit_actions=="mixed" and (args.scenario not in ("idle","replay","session") or not args.idle_units):
        parser.error("--unit-actions mixed requires --scenario idle, replay or session and --idle-units")
    if args.unit_actions=="realistic" and (args.scenario not in ("ambient","idle") or not args.idle_units):
        parser.error("--unit-actions realistic requires --scenario ambient or idle and --idle-units")
    if args.scenario in ("replay","session") and (not args.idle_units or not args.dense_scene or args.waves!="1" or args.reflection_ablation or args.camera_view or args.tile_width!=128 or args.unit_actions!="mixed"):
        parser.error("A busy replay/session starts at width 128 and requires units, mixed actions, --dense-scene, waves/reflections on and the synchronous native-compatible render API")
    if args.case_repeats and (args.scenario!="scroll" or args.idle_units or args.camera_view):
        parser.error("Persistent cases currently cover synchronous scroll with one fixed constructor configuration")
    if args.case_repeats and args.case_reset=="process_cold" and args.case_repeats!=1:
        parser.error("Process-cold cases require a fresh process per case")
    if args.exclusive_gpu and not args.case_repeats:
        parser.error("--exclusive-gpu requires the bounded session watchdog")
    if args.boundary_fixture and (args.case_repeats or args.dense_scene or args.scenario!="scroll" or
                                 (args.width,args.height,args.tile_width)!=(384,256,128)):
        parser.error("Boundary fixture requires one-shot scroll, 384x256, tile width 128, without dense population")
    out = args.out.resolve()
    relative = out.relative_to(ROOT)
    samples = (args.idle_steps if args.scenario == "idle" else args.distant_steps if args.scenario == "distant" else
               15 if args.scenario == "replay" else args.resident_steps if args.resident else 0)
    storage = storage_preflight(ROOT, args.width, args.height, samples)
    out.mkdir(parents=True, exist_ok=False)
    for name in ("C3XRenderer.dll", "biq_preview.exe"):
        shutil.copy2(args.binaries / name, out / name)
    build_record=args.binaries / "build-evidence.json"
    if build_record.is_file():
        shutil.copy2(build_record,out / "build-evidence.json")
    phase("arguments_storage_binary_copy_ms")
    print("Hashing current runtime inputs before execution", flush=True)
    before = inputs()
    before_metadata=metadata_snapshot(before)
    phase("input_verification_before_ms")
    env = {key: "" for key in os.environ if key.startswith(("C3X_RENDERER_", "C3X_LAB_"))}
    env.update({"C3X_RENDERER_VISUAL_PROFILE": "", "C3X_RENDERER_TRACE": "2" if args.instrumentation=="diagnostic" else "0",
           "C3X_RENDERER_TRACE_BUFFERED": "1" if args.instrumentation=="diagnostic" else "0",
           "C3X_RENDERER_PREVIEW_TIMING": "1",
           "C3X_RENDERER_PREVIEW_SCROLL_ABLATION": ("sequence" if args.scroll_sequence else "full") if args.scenario=="scroll" else "",
           "C3X_RENDERER_PROFILE": "1" if args.profile else "0",
           "C3X_RENDERER_BLOCK_CLIP": args.block_clip,
           "C3X_RENDERER_CASTER_BOUNDS_CONTROL": "1" if args.caster_control else "0",
           "C3X_RENDERER_RASTER_REUSE_CONTROL": "1" if args.raster_control else "0",
           "C3X_RENDERER_REGION_SIZE": str(args.region_size),
           "C3X_RENDERER_BOUNDED_POST": "1" if args.bounded_post else "0",
           "C3X_RENDERER_REFLECTION_CONTROL": "1" if args.reflection_ablation else "0",
           "C3X_RENDERER_DIAGNOSTIC_ROUTES": args.diagnostic_routes,
           "C3X_RENDERER_DIAGNOSTIC_HALF_PIXELS": "1" if args.diagnostic_half_pixels else "0",
           "C3X_RENDERER_PREVIEW_RETAINED_BOUNDARY": "1" if args.boundary_fixture else "",
           "C3X_RENDERER_PREPARED_RESOURCE_PASS": "1" if args.prepared_resource_pass else "0",
           "C3X_RENDERER_WORLD_RASTER_GRID": "1" if args.world_grid else "0",
           "C3X_RENDERER_WORLD_REGIONS": "1" if args.world_regions else "0",
           "C3X_RENDERER_THREE_ZOOM_MEMORY": "1" if args.three_zoom_memory else "0",
           "C3X_RENDERER_CENTER_SHORE_CONTROL": "1" if args.center_shore_control else "0",
           "C3X_RENDERER_REGION_INDEX_CONTROL": "1" if args.index_control else "0",
           "C3X_RENDERER_REGION_DEPENDENCY_CONTROL": "1" if args.dependency_control else "0",
           "C3X_RENDERER_REGION_DIAGNOSTICS": "1" if args.region_diagnostics else "0",
           "C3X_RENDERER_REGION_RECEIVER_SHADOWS": "1" if args.region_receiver_shadows else "0",
           "C3X_RENDERER_TIGHT_NATURAL_BOUNDS": "1" if args.tight_natural_bounds else "0",
           "C3X_RENDERER_REGION_INPUT_RING": str(args.region_input_ring),
           "C3X_RENDERER_REGION_METADATA_MIB": str(args.region_metadata_mib),
           "C3X_RENDERER_WORLD_REGIONS_CONTROL": "1" if args.world_regions_control else "0",
           "C3X_RENDERER_WATER_COVERAGE": "1" if args.water_coverage else "0",
           "C3X_RENDERER_WORLD_BACKDROPS": "1" if args.world_backdrops else "0",
           "C3X_RENDERER_BACKDROP_DEPENDENCIES": "1" if args.backdrop_dependencies else "0",
           "C3X_RENDERER_COMPOSITION_RECEIVER_INDEX": "1" if args.composition_receiver_index else "0",
           "C3X_RENDERER_BACKDROP_REUSE_CONTROL": "1" if args.backdrop_control else "0",
           "C3X_RENDERER_WORLD_WAVES": "1" if args.world_waves else "0",
           "C3X_RENDERER_WAVE_REUSE_CONTROL": "1" if args.wave_control else "0",
           "C3X_RENDERER_COMPOSITION_CASTERS_CONTROL": "1" if args.composition_casters_control else "0",
           "C3X_RENDERER_ANIMATION_READBACK_ATLAS": "1" if args.animation_readback_atlas else "0",
           "C3X_RENDERER_PREVIEW_RESIDENT_STEPS": str(args.resident_steps),
           "C3X_RENDERER_CAMERA_PREVIEW": "0", "C3X_RENDERER_PREVIEW_CAMERA_QUEUE": "1" if args.camera_view else "",
           "C3X_RENDERER_PREVIEW_CAMERA_VIEW": "1" if args.camera_view else "",
           "C3X_RENDERER_PREVIEW_AMBIENT_ASYNC": "1" if args.scenario == "ambient" else "",
           "C3X_RENDERER_PREVIEW_SUPPORTED_ZOOMS": "1" if args.scenario == "zoom" else "",
           "C3X_RENDERER_PREVIEW_CYCLES": str(args.cycles),
           "C3X_RENDERER_PREVIEW_DISTANT_STEPS": str(args.distant_steps) if args.scenario == "distant" else "",
           "C3X_RENDERER_PREVIEW_IDLE_STEPS": str(args.idle_steps) if args.scenario == "idle" else "",
           "C3X_RENDERER_PREVIEW_IDLE_UNITS": str(args.idle_units),
           "C3X_RENDERER_PREVIEW_UNIT_ACTIONS": args.unit_actions,
           "C3X_RENDERER_PREVIEW_IDLE_WARMUP": str(args.idle_warmup),
           "C3X_RENDERER_UNIT_POSE_MEMORY": ("512" if args.unit_pose_memory_mib==512 else "1") if args.unit_pose_memory else "0",
           "C3X_RENDERER_PREVIEW_DENSE_SCENE": "1" if args.dense_scene else "",
           "C3X_RENDERER_PREVIEW_SEASON": "0",
           "C3X_RENDERER_PREVIEW_ANIMATION": "1", "C3X_RENDERER_WAVES": args.waves,
           "C3X_RENDERER_PREVIEW_NAVIGATION": "1" if args.scenario == "navigation" else "",
           "C3X_RENDERER_PREVIEW_ZOOM": "1" if args.scenario == "zoom" else "",
           "C3X_RENDERER_PREVIEW_RETAINED_REPLAY": "1" if args.scenario == "replay" else "",
           "C3X_RENDERER_PREVIEW_BUSY_SESSION": "1" if args.scenario == "session" else "",
           "C3X_RENDERER_PREVIEW_PREPARATION_MODE": args.preparation_mode,
           "C3X_RENDERER_PREVIEW_REPLAY_SAMPLES": str(args.replay_samples_per_phase),
           "C3X_RENDERER_PREVIEW_RESIDENT_SWEEP": "1" if args.resident else "",
           "C3X_RENDERER_PREVIEW_RESIDENT_COLD": "1" if args.cold else ""})
    win_root = Path(ROOT) if os.name == "nt" else windows_root()
    win_out = win_root / str(relative)
    env["C3X_RENDERER_TRACE_FILE"] = str(win_out / "renderer.log")
    env["C3X_RENDERER_PREVIEW_CUSTOM_DEFINITIONS"] = str(win_root / "Renderer/custom.custom_rendering.txt")
    if args.production_defaults:
        for name in ("THREE_ZOOM_MEMORY", "WORLD_BACKDROPS", "WORLD_WAVES",
                     "BACKDROP_DEPENDENCIES", "COMPOSITION_RECEIVER_INDEX", "UNIT_POSE_MEMORY"):
            env["C3X_RENDERER_" + name] = ""
    case_manifest=None
    if args.case_repeats:
        request={"schema":1,"offsets":[1,2,4,8,4,2,1,0,-2,-4,-8,-4,-2,0] if args.scroll_sequence else [4],
                 "width":args.width,"height":args.height,"center":[75,39],"tile_width":args.tile_width,"hour":12,"clock":1000000}
        request_digest=hashlib.sha256(json.dumps(request,sort_keys=True).encode()).hexdigest()
        config_digest=hashlib.sha256(json.dumps({k:v for k,v in env.items() if k!="C3X_RENDERER_TRACE_FILE"},sort_keys=True).encode()).hexdigest()
        warmup="sequence" if args.case_reset=="prepared_resident" else "initial"
        case_manifest={"schema":1,"case_id":"scroll","config_id":config_digest,"request":request,
                       "request_digest":request_digest,"reset":args.case_reset,"warmup":warmup,
                       "repeats":args.case_repeats,"time_limit_ms":args.case_time_limit*1000,
                       "environment_policy":"one immutable constructor configuration per process"}
        (out/"cases.json").write_text(json.dumps(case_manifest,indent=2))
        (out/"cases.txt").write_text(f"C3X_PREVIEW_CASES_V1 scroll {config_digest} {request_digest} {args.case_reset} {warmup} {args.case_repeats} {args.case_time_limit*1000}\n")
        env["C3X_RENDERER_PREVIEW_SESSION"]=str(win_out/"cases.txt")
        watched=set(before)
        if build_record.is_file():watched.update(json.loads(build_record.read_text()).get("sources",{}))
        native_paths=[str(win_root/name) for name in sorted(watched)]
        native_paths += [str(win_out/name) for name in ("C3XRenderer.dll","biq_preview.exe","cases.txt")]
        (out/"session-inputs.txt").write_text("\n".join(native_paths)+"\n")
        env["C3X_RENDERER_SESSION_INPUTS"]=str(win_out/"session-inputs.txt")
    source_paths={p for p in (ROOT/"Renderer/native").glob("*.cpp")}
    if build_record.is_file():
        build=json.loads(build_record.read_text())
        for closure in build.get("unit_inputs",{}).values():
            for name,expected in closure.items():
                path=ROOT/name
                if digest(path)!=expected:raise ValueError("Build inputs changed; end session and rebuild before running")
                source_paths.add(path)
    source_before={p.relative_to(ROOT).as_posix():digest(p) for p in sorted(source_paths)}
    receipt = {"case_manifest":case_manifest,"verification":args.verification,
               "input_metadata":before_metadata,"invocation": uuid.uuid4().hex, "endpoint": "standalone capture plus completed render; no native presentation",
               "storage_preflight": storage,
               "quality_mode": "diagnostic_pixel_ablation" if args.diagnostic_routes!="full" or args.diagnostic_half_pixels else "diagnostic_reflections_disabled" if args.reflection_ablation else "current",
               "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
               "environment": env, "inputs": before,
               "host": {"os": platform.platform(), "architecture": platform.machine(), "logical_processors": os.cpu_count()},
               "source_at_run":source_before,
               "binaries": {n: digest(out / n) for n in ("C3XRenderer.dll", "biq_preview.exe")},
               "started_unix": time.time()}
    phase("source_binary_verification_and_configuration_ms")
    receipt["instrumentation_mode"]=args.instrumentation
    (out / "inputs.json").write_text(json.dumps(receipt, indent=2))
    # Every path is quoted for cmd, and generated/configured values may not inject commands.
    values = [str(win_out / "biq_preview.exe"), str(win_out / "C3XRenderer.dll"), str(win_root),
              str(win_root / "Renderer/default.custom_rendering.txt"),
              str(win_root / "Renderer/lab/.local/verification/world.csv"), str(win_out / "zoom.bmp")]
    if any(any(c in v for c in '"%\r\n') for v in values + list(env.values())):
        raise ValueError("Unsupported cmd characters in configured paths")
    command = " ".join(f'"{v}"' for v in values)
    command += f' {args.width} {args.height} 75 39 {args.tile_width} 12 >"{win_out / "benchmark.log"}" 2>&1'
    # Match the category dispatcher's short transport command. Passing the
    # entire environment through prlctl intermittently fails before execution.
    batch = "@echo off\n" + "\n".join(f'set "{k}={v}"' for k, v in env.items())
    if args.case_repeats:
        # Bound the owned process itself, including driver/DLL teardown and WER,
        # rather than timing out a transport that can leave an orphaned render.
        quote=lambda value:"'"+str(value).replace("'","''")+"'"
        child_args=" ".join(f'"{v}"' for v in values[1:])+f" {args.width} {args.height} 75 39 {args.tile_width} 12"
        script=f"""$ErrorActionPreference='Stop'
$exclusive=${str(args.exclusive_gpu).lower()}
$checks=0
$conflicts=@()
function Other-Workloads($owned) {{
 @(Get-Process biq_preview,native_preview,cl,link -ErrorAction SilentlyContinue | Where-Object {{$_.Id -ne $owned}} | Select-Object ProcessName,Id)
}}
if($exclusive) {{
 $checks++;$conflicts=@(Other-Workloads 0)
 if($conflicts.Count) {{
  @{{checks=$checks;conflicts=$conflicts;interval_ms=1000}} | ConvertTo-Json -Depth 4 | Set-Content {quote(win_out/'exclusivity.json')}
  [IO.File]::WriteAllText({quote(win_out/'child-completion.txt')},'{receipt['invocation']} 126')
  exit 126
 }}
}}
$child=Start-Process -FilePath {quote(values[0])} -ArgumentList {quote(child_args)} -RedirectStandardOutput {quote(win_out/'benchmark.log')} -RedirectStandardError {quote(win_out/'error.log')} -PassThru
$childHandle=$child.Handle
[IO.File]::WriteAllText({quote(win_out/'process.txt')},'{receipt['invocation']} '+$child.Id)
$deadline=[DateTime]::UtcNow.AddMilliseconds({args.case_time_limit*1000+10000})
do {{
 $exited=$child.WaitForExit(1000)
 if($exclusive) {{$checks++;$conflicts+=@(Other-Workloads $child.Id)}}
}} while(-not $exited -and -not $conflicts.Count -and [DateTime]::UtcNow -lt $deadline)
if($exited) {{$child.WaitForExit();$childCode=$child.ExitCode;if($null -eq $childCode){{$childCode=125}}}}
else {{
 $null=& taskkill.exe /PID $child.Id /T /F 2>&1
 $childCode=124
}}
if($exclusive) {{
 @{{checks=$checks;conflicts=$conflicts;interval_ms=1000}} | ConvertTo-Json -Depth 4 | Set-Content {quote(win_out/'exclusivity.json')}
 if($conflicts.Count) {{$childCode=126}}
}}
[IO.File]::WriteAllText({quote(win_out/'child-completion.txt')},'{receipt['invocation']} '+$childCode)
exit $childCode
"""
        (out/'watch.ps1').write_text(script)
        command=f'powershell -NoProfile -ExecutionPolicy Bypass -File "{win_out / "watch.ps1"}"'
    batch += "\n" + command + "\nexit /b %errorlevel%\n"
    (out / "run.bat").write_bytes(batch.replace("\n", "\r\n").encode("utf-8"))
    phase("dispatch_preparation_and_receipt_ms")
    result = native_command_result("Renderer/native", f'call "{win_out / "run.bat"}"')
    if args.case_repeats:
        child=(out/'child-completion.txt').read_text().split() if (out/'child-completion.txt').exists() else []
        if len(child)==2 and child[0]==receipt['invocation']:
            result['returncode']=int(child[1])
        else:
            result['returncode']=None
            result['status']='unconfirmed_child; inspect published PID before retry'
    phase("dispatch_process_playback_ms")
    (out / "completion.txt").write_text(str(result["returncode"]) + "\n")
    print("Hashing inputs after execution", flush=True)
    after_metadata=metadata_snapshot(before)
    after=inputs() if args.verification=="acceptance" else None
    phase("input_verification_after_ms")
    changed = sorted(k for k in before.keys() | after.keys() if before.get(k) != after.get(k)) if after is not None else [k for k in before_metadata if before_metadata[k]!=after_metadata[k]]
    completion = {"invocation": receipt["invocation"], "finished_unix": time.time(),
                  "returncode": result["returncode"], "changed_inputs": changed,
                  "inputs_unchanged":not changed if after is not None else None,
                  "metadata_unchanged":before_metadata==after_metadata,
                  "verification":args.verification,"provisional":args.verification=="quick"}
    completion["sources_unchanged"]=all((ROOT/name).is_file() and digest(ROOT/name)==value for name,value in source_before.items())
    completion["binaries_unchanged"] = all(digest(out / name)==value for name,value in receipt["binaries"].items())
    completion["images"] = {p.name: digest(p) for p in sorted(out.glob("*.bmp")) if p.is_file()}
    if args.boundary_fixture:
        log=(out/"benchmark.log").read_text(errors="replace")
        completion["boundary_correctness_pass"]=bool(re.search(r"^RETAINED_BOUNDARY_END status=pass checks=6 independent_full_redraw=1$",log,re.M))
        if not completion["boundary_correctness_pass"]:completion["returncode"]=1;result["returncode"]=1
    phase("binary_image_verification_ms")
    from Renderer.native.analyze_navigation_run import endpoint_accounting, session_accounting
    try:
        completion["endpoints"]=(session_accounting if args.case_repeats else endpoint_accounting)(
            (out/"benchmark.log").read_text(errors="replace").splitlines(),
            (out/"renderer.log").read_text(errors="replace").splitlines() if (out/"renderer.log").exists() else [])
    except (OSError,ValueError) as error:
        completion["endpoints"]={"status":"invalid", "reason":str(error)}
    phase("endpoint_analysis_ms")
    completion["wrapper_timing_ms"]=timing
    completion["compile_link"]={"status":"not_run", "reason":"uses supplied isolated binaries",
                                "build_timing_ms":json.loads(build_record.read_text()).get("timing_ms") if build_record.is_file() else None}
    (out / "evidence.json").write_text(json.dumps(completion, indent=2))
    phase("evidence_output_ms")
    completion["wrapper_total_ms"]=(time.perf_counter()-wrapper_started)*1000
    completion["wrapper_endpoint"]="before final receipt rewrite and console output"
    (out / "evidence.json").write_text(json.dumps(completion, indent=2))
    print(json.dumps({"out":str(out),"returncode":completion["returncode"],
        "verification":completion["verification"],"inputs_unchanged":completion["inputs_unchanged"],
        "sources_unchanged":completion["sources_unchanged"],"binaries_unchanged":completion["binaries_unchanged"],
        "endpoints":completion["endpoints"]["status"],"wrapper_total_ms":completion["wrapper_total_ms"]}),flush=True)
    raise SystemExit(0 if result["returncode"] == 0 and not changed and completion["binaries_unchanged"] and completion["sources_unchanged"] and completion["endpoints"]["status"]!="invalid" else 1)


if __name__ == "__main__":
    main()

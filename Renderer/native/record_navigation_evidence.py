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
        return hashlib.file_digest(stream, "sha256").hexdigest()


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binaries", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--scenario", choices=("navigation", "zoom", "animation", "idle", "distant"), default="navigation")
    parser.add_argument("--idle-steps", type=int, choices=range(1,1001), default=100)
    parser.add_argument("--idle-warmup", type=int, choices=range(10,151), default=10)
    parser.add_argument("--unit-pose-memory", action="store_true", help="Opt-in 256 MiB / 4096-entry exact unit-pose retention")
    parser.add_argument("--idle-units", type=int, choices=(0,8,24,64), default=0,
                        help="Draw this many separate native-directed idle unit bodies in a synthetic GDI canvas")
    parser.add_argument("--unit-actions", choices=("idle","mixed"), default="idle",
                        help="Script separate per-unit native move, attack, return, fortify and idle timelines")
    parser.add_argument("--dense-scene", action="store_true", help="Stable synthetic cities, infrastructure, camps and supported resources on the captured world")
    parser.add_argument("--width", type=int, default=2240)
    parser.add_argument("--height", type=int, default=1192)
    parser.add_argument("--distant-steps", type=int, choices=range(1,1001), default=100)
    parser.add_argument("--cycles", type=int, choices=range(2,41), default=2)
    parser.add_argument("--resident", action="store_true")
    parser.add_argument("--cold", action="store_true")
    parser.add_argument("--waves", choices=("0", "1"), default="1")
    parser.add_argument("--tier", choices=("normal", "384", "768"), required=True)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--block-clip", choices=("0", "1"), default="0")
    parser.add_argument("--resident-steps", type=int, default=14)
    parser.add_argument("--tile-width", type=int, choices=(64,96,128,160,192), default=128)
    parser.add_argument("--caster-control", action="store_true")
    parser.add_argument("--raster-control", action="store_true")
    parser.add_argument("--region-size", type=int, choices=(128,256,512,2240), default=128)
    parser.add_argument("--bounded-post", action="store_true", help="Limit experimental strip reconstruction to guarded output regions")
    parser.add_argument("--reflection-ablation", action="store_true", help="Diagnostic only: omit reflections to estimate their cost; images are not current quality")
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
    parser.add_argument("--backdrop-control", action="store_true", help="Independently redraw every static animation backdrop")
    parser.add_argument("--world-waves", action="store_true", help="Retain immutable coast-cell wave buffers across cameras")
    parser.add_argument("--wave-control", action="store_true", help="Rebuild coast-cell wave buffers for independent comparisons")
    parser.add_argument("--composition-casters-control", action="store_true", help="Rebuild caster preparation independently for each animation region")
    args = parser.parse_args()
    if args.idle_units and args.scenario != "idle":
        parser.error("--idle-units requires --scenario idle")
    if args.unit_actions=="mixed" and (args.scenario!="idle" or not args.idle_units):
        parser.error("--unit-actions mixed requires --scenario idle and --idle-units")
    out = args.out.resolve()
    relative = out.relative_to(ROOT)
    storage = storage_preflight(ROOT, args.width, args.height, args.idle_steps if args.scenario == "idle" else args.distant_steps if args.scenario == "distant" else args.resident_steps if args.resident else 0)
    out.mkdir(parents=True, exist_ok=False)
    for name in ("C3XRenderer.dll", "biq_preview.exe"):
        shutil.copy2(args.binaries / name, out / name)
    build_record=args.binaries / "build-evidence.json"
    if build_record.is_file():
        shutil.copy2(build_record,out / "build-evidence.json")
    print("Hashing current runtime inputs before execution", flush=True)
    before = inputs()
    env = {key: "" for key in os.environ if key.startswith(("C3X_RENDERER_", "C3X_LAB_"))}
    env.update({"C3X_RENDERER_VISUAL_PROFILE": "", "C3X_RENDERER_TRACE": "2",
           "C3X_RENDERER_PROFILE": "1" if args.profile else "0",
           "C3X_RENDERER_BLOCK_CLIP": args.block_clip,
           "C3X_RENDERER_CASTER_BOUNDS_CONTROL": "1" if args.caster_control else "0",
           "C3X_RENDERER_RASTER_REUSE_CONTROL": "1" if args.raster_control else "0",
           "C3X_RENDERER_REGION_SIZE": str(args.region_size),
           "C3X_RENDERER_BOUNDED_POST": "1" if args.bounded_post else "0",
           "C3X_RENDERER_REFLECTION_CONTROL": "1" if args.reflection_ablation else "0",
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
           "C3X_RENDERER_BACKDROP_REUSE_CONTROL": "1" if args.backdrop_control else "0",
           "C3X_RENDERER_WORLD_WAVES": "1" if args.world_waves else "0",
           "C3X_RENDERER_WAVE_REUSE_CONTROL": "1" if args.wave_control else "0",
           "C3X_RENDERER_COMPOSITION_CASTERS_CONTROL": "1" if args.composition_casters_control else "0",
           "C3X_RENDERER_PREVIEW_RESIDENT_STEPS": str(args.resident_steps),
           "C3X_RENDERER_CAMERA_PREVIEW": "0", "C3X_RENDERER_PREVIEW_CAMERA_QUEUE": "1" if args.camera_view else "",
           "C3X_RENDERER_PREVIEW_CAMERA_VIEW": "1" if args.camera_view else "",
           "C3X_RENDERER_PREVIEW_SUPPORTED_ZOOMS": "1" if args.scenario == "zoom" else "",
           "C3X_RENDERER_PREVIEW_CYCLES": str(args.cycles),
           "C3X_RENDERER_PREVIEW_DISTANT_STEPS": str(args.distant_steps) if args.scenario == "distant" else "",
           "C3X_RENDERER_PREVIEW_IDLE_STEPS": str(args.idle_steps) if args.scenario == "idle" else "",
           "C3X_RENDERER_PREVIEW_IDLE_UNITS": str(args.idle_units),
           "C3X_RENDERER_PREVIEW_UNIT_ACTIONS": args.unit_actions,
           "C3X_RENDERER_PREVIEW_IDLE_WARMUP": str(args.idle_warmup),
           "C3X_RENDERER_UNIT_POSE_MEMORY": "1" if args.unit_pose_memory else "0",
           "C3X_RENDERER_PREVIEW_DENSE_SCENE": "1" if args.dense_scene else "",
           "C3X_RENDERER_PREVIEW_SEASON": "0",
           "C3X_RENDERER_PREVIEW_ANIMATION": "1", "C3X_RENDERER_WAVES": args.waves,
           "C3X_RENDERER_PREVIEW_NAVIGATION": "1" if args.scenario == "navigation" else "",
           "C3X_RENDERER_PREVIEW_ZOOM": "1" if args.scenario == "zoom" else "",
           "C3X_RENDERER_PREVIEW_RESIDENT_SWEEP": "1" if args.resident else "",
           "C3X_RENDERER_PREVIEW_RESIDENT_COLD": "1" if args.cold else ""})
    win_root = Path(ROOT) if os.name == "nt" else windows_root()
    win_out = win_root / str(relative)
    env["C3X_RENDERER_TRACE_FILE"] = str(win_out / "renderer.log")
    env["C3X_RENDERER_PREVIEW_CUSTOM_DEFINITIONS"] = str(win_root / "Renderer/custom.custom_rendering.txt")
    receipt = {"invocation": uuid.uuid4().hex, "endpoint": "standalone capture plus completed render; no native presentation",
               "storage_preflight": storage,
               "quality_mode": "diagnostic_reflections_disabled" if args.reflection_ablation else "current",
               "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
               "environment": env, "inputs": before,
               "host": {"os": platform.platform(), "architecture": platform.machine(), "logical_processors": os.cpu_count()},
               "source_at_run": {p.relative_to(ROOT).as_posix(): digest(p)
                                 for p in (ROOT / "Renderer/native").glob("*.cpp")},
               "binaries": {n: digest(out / n) for n in ("C3XRenderer.dll", "biq_preview.exe")},
               "started_unix": time.time()}
    (out / "inputs.json").write_text(json.dumps(receipt, indent=2))
    # Every path is quoted for cmd, and generated/configured values may not inject commands.
    values = [str(win_out / "biq_preview.exe"), str(win_out / "C3XRenderer.dll"), str(win_root),
              str(win_root / "Renderer/default.custom_rendering.txt"),
              str(win_root / "Renderer/lab/.local/verification/world.csv"), str(win_out / "zoom.bmp")]
    if any(any(c in v for c in '"%\r\n') for v in values + list(env.values())):
        raise ValueError("Unsupported cmd characters in configured paths")
    command = " && ".join(f'set "{k}={v}"' for k, v in env.items()) + " && "
    command += " ".join(f'"{v}"' for v in values)
    command += f' {args.width} {args.height} 75 39 {args.tile_width} 12 >"{win_out / "benchmark.log"}" 2>&1'
    result = native_command_result("Renderer/native", command)
    (out / "completion.txt").write_text(str(result["returncode"]) + "\n")
    print("Hashing inputs after execution", flush=True)
    after = inputs()
    changed = sorted(k for k in before.keys() | after.keys() if before.get(k) != after.get(k))
    completion = {"invocation": receipt["invocation"], "finished_unix": time.time(),
                  "returncode": result["returncode"], "changed_inputs": changed,
                  "inputs_unchanged": not changed}
    completion["binaries_unchanged"] = all(digest(out / name)==value for name,value in receipt["binaries"].items())
    completion["images"] = {p.name: digest(p) for p in sorted(out.glob("*.bmp")) if p.is_file()}
    (out / "evidence.json").write_text(json.dumps(completion, indent=2))
    print(json.dumps(completion), flush=True)
    raise SystemExit(0 if result["returncode"] == 0 and not changed and completion["binaries_unchanged"] else 1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Category-based workbench and integration checks for the current C3X code."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import uuid

ROOT = Path(__file__).resolve().parent.parent
LAB = ROOT / "Renderer/lab"
sys.path.insert(0, str(ROOT))


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", prefix=".renderer-", suffix=".tmp",
                                     dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(json.dumps(value, indent=2) + "\n")
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def checksum(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def relative(path):
    return path.resolve().relative_to(ROOT).as_posix()


def local(path):
    target = (ROOT / path).resolve()
    if not target.is_relative_to(ROOT):
        raise ValueError("Path escapes the repository")
    return target


def catalog():
    return read(LAB / "catalog.json")["categories"]


def standard_path(category):
    entries = catalog()
    if category not in entries:
        raise ValueError("Unknown category: " + category)
    return LAB / "categories" / entries[category] / "standard.json"


def standard(category):
    return read(standard_path(category))


def declared_affected(category):
    """Transitive consumers select a bounded, deduplicated preview set."""
    selected = {category}
    changed = True
    entries = {key: standard(key) for key in catalog()}
    while changed:
        changed = False
        for key, value in entries.items():
            if key not in selected and selected.intersection(value["depends_on"]):
                selected.add(key)
                changed = True
    return sorted(selected)


def asset_receipts():
    return {path.stem: read(path) for path in (LAB / ".cache/assets").glob("*.json")}


def category_signatures():
    from Renderer.lab.dependencies import signatures
    shaders = LAB / ".cache/shader-preparation.json"
    generated = read(shaders).get("outputs", {}) if shaders.exists() else {}
    return signatures(implementation_inputs(), {key: standard(key) for key in catalog()},
                      assets=asset_receipts(), generated_shaders=generated)


def affected(category):
    """Visual dependents of the category the user actually requested."""
    return declared_affected(category)


def reexec_with_workspace_python(packages):
    """Select an optional-dependency runtime before doing expensive work."""
    missing = [name for name in packages if importlib.util.find_spec(name) is None]
    if not missing:
        return
    configured = os.environ.get("C3X_RENDERER_PYTHON")
    candidates = ([Path(configured).expanduser()] if configured else []) + [
        Path.home() / ".cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3"
    ]
    for candidate in candidates:
        if not candidate.is_file() or candidate.resolve() == Path(sys.executable).resolve():
            continue
        probe = subprocess.run([str(candidate), "-c", "import " + ",".join(packages)],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if probe.returncode == 0:
            print("Using optional-dependency Python: " + str(candidate), flush=True)
            os.execv(str(candidate), [str(candidate), str(Path(__file__).resolve()), *sys.argv[1:]])
    raise ValueError("Missing Python packages: " + ", ".join(missing) +
                     ". Set C3X_RENDERER_PYTHON to a Python with Pillow and NumPy installed.")


def asset_jobs_for(categories=None):
    """Asset builders used by the requested category set."""
    if categories is None:
        return None
    from Renderer.lab.dependencies import consumers
    selected = set(categories)
    return {name for name, owners in consumers({key: standard(key) for key in catalog()}).items()
            if selected.intersection(owners)}


def prepare_sources(categories=None):
    from Renderer.lab.preparation import prepare
    changed = prepare(ROOT)
    print(f"Prepared production shader bindings: {len(changed)} changed files", flush=True)
    from Renderer.lab.asset_preparation import prepare as prepare_assets
    assets = prepare_assets(selected=asset_jobs_for(categories))
    print(f"Prepared current category assets: {len(assets)} refreshed files", flush=True)
    return changed + assets


def require_prepared(categories=None):
    from Renderer.lab.preparation import require_current
    require_current(ROOT)
    from Renderer.lab.asset_preparation import prepare as prepare_assets
    prepare_assets(selected=asset_jobs_for(categories), check_only=True)


def implementation_inputs():
    """Bounded current source/runtime read closure used by preview freshness."""
    paths = set()
    for category in catalog():
        paths.update(local(p) for p in standard(category)["implementation"])
    paths.update(local(path) for path in native_inputs())
    paths.update([ROOT / "C3X.h", ROOT / "injected_code.c"])
    paths.update([ROOT / "Renderer/default.custom_rendering.txt", ROOT / "Renderer/custom.custom_rendering.txt"])
    paths.update([ROOT / "Renderer/renderer.py", LAB / "native_preview.cpp", LAB / "dependencies.py", LAB / "platform.py"])
    from Renderer.lab.preparation import input_paths
    paths.update(ROOT / path for path in input_paths(ROOT))
    from Renderer.lab.asset_preparation import jobs as asset_jobs
    paths.add(LAB / "asset_preparation.py")
    paths.update(ROOT / job[3] for job in asset_jobs())
    receipts = asset_receipts()
    for record in receipts.values():
        paths.update(local(path) for path in record.get("inputs", {})
                     if not path.startswith("Renderer/packs/"))
    records = {relative(path): checksum(path) for path in sorted(paths) if path.is_file()}
    # Preparation receipts contain the exact source/runtime bytes read by each
    # builder. Commands verify the relevant receipt before using this identity,
    # so historical pack paths never enter the hot loop.
    for record in receipts.values():
        records.update(record.get("inputs", {}))
        records.update(record.get("outputs", {}))
    return records


def implementation_identity():
    """Internal stale-preview guard; category pages remain free of hash ledgers."""
    return hashlib.sha256(json.dumps(implementation_inputs(), sort_keys=True).encode()).hexdigest()


def native_inputs():
    """Actual C++ translation units and their local include closure."""
    native = ROOT / "Renderer/native"
    pending = [native / name for name in ("c3x_renderer.cpp", "terrain_scene_runtime.cpp",
        "environment_runtime.cpp", "terrain_definition_runtime.cpp", "scene_export.cpp",
        "frame_scheduler.cpp", "c3x_renderer.def", "BUILD.bat")]
    return source_inputs(pending)


def source_inputs(pending):
    """Follow local quoted includes for either production or preview builds."""
    pending = list(pending)
    visited = set()
    while pending:
        path = pending.pop().resolve()
        if path in visited:
            continue
        visited.add(path)
        for include in re.findall(r'^\s*#include\s*"([^"]+)"', path.read_text(), re.MULTILINE):
            child = (path.parent / include).resolve()
            if child.is_file():
                child.relative_to(ROOT)
                pending.append(child)
    return {relative(path): checksum(path) for path in sorted(visited)}


class CandidateBuildRequired(ValueError):
    """Known missing/stale build, distinct from broken or unreadable inputs."""


def require_current_candidate():
    dll = ROOT / "Renderer/native/build/candidate/C3XRenderer.dll"
    if not dll.exists():
        raise CandidateBuildRequired("Candidate DLL is missing")
    digest = checksum(dll)
    inputs = native_inputs()
    receipt = LAB / ".cache/native-build.json"
    if receipt.exists():
        built = read(receipt)
        if built.get("dll_sha256") == digest and built.get("inputs") == inputs:
            return
    raise CandidateBuildRequired("Candidate DLL does not match the current C++ inputs; run Renderer/renderer.py build")


def ensure_candidate():
    """Reuse a verified candidate; build only when compiled inputs require it."""
    try:
        require_current_candidate()
    except CandidateBuildRequired:
        print("Rebuilding candidate for current compiled inputs; production is not staged", flush=True)
        build_candidate()
        # A successful process alone is not evidence of a matching build.
        require_current_candidate()


def ensure_preview_tool(*, force=False):
    from Renderer.lab.platform import native_command_result
    sources = (LAB / "native_preview.cpp", LAB / "build_native_preview.bat",
               ROOT / "Renderer/native/biq_preview.cpp", ROOT / "Renderer/native/c3x_renderer_api.h")
    inputs = source_inputs(sources)
    binary = LAB / ".cache/native_preview.exe"
    receipt = LAB / ".cache/preview-build.json"
    if not force and binary.exists() and receipt.exists():
        built = read(receipt)
        if built.get("inputs") == inputs and built.get("binary") == checksum(binary):
            return
    result = native_command_result("Renderer/lab", "call build_native_preview.bat")
    if result["status"] != "pass" or not binary.exists():
        raise ValueError("Standalone category witness build failed")
    if source_inputs(sources) != inputs:
        raise ValueError("Preview source changed during compilation; rerun")
    write(receipt, {"inputs": inputs, "binary": checksum(binary)})


def scene(category, case, destination, *, world_size=32):
    """Small complete synthetic world with no implied live-game provenance."""
    recipe = standard(category)["recipe"]
    watershed_river = {}
    coastal_mountains = {
        # A range arriving perpendicular to the upper coast.
        (15, -4), (16, -4), (17, -4), (18, -4), (19, -4), (20, -4),
        # A shorter range following the center shoreline.
        (18, -1), (19, -1), (19, 0), (19, 1), (19, 2),
        # An isolated headland peak against the lower coast.
        (18, 4),
    }
    if category == "rivers":
        # A complete, deterministic watershed is more useful than a bare
        # diagonal line: one inland headwater crosses a mountain pass, meanders
        # through forested lowlands and reaches a real coast/sea/ocean shelf.
        # Civ III records each shared edge on both incident raw-coordinate tiles.
        river_nodes = (
            (10, -4), (11, -4), (11, -3), (12, -3), (13, -3),
            (13, -2), (13, -1), (13, 0), (14, 0), (15, 0),
            (16, 0), (17, 0), (17, 1), (18, 1), (19, 1),
            (20, 1), (20, 2), (21, 2), (22, 2),
        )

        def add_river_bit(c, r, bit):
            raw = (c + r, c - r)
            if 0 <= raw[0] < world_size and 0 <= raw[1] < world_size:
                watershed_river[raw] = watershed_river.get(raw, 0) | bit

        river_incident_tiles = set()
        for a, b in zip(river_nodes, river_nodes[1:]):
            c, r = min(a, b)
            if a[1] == b[1]:
                add_river_bit(c, r, 32)
                add_river_bit(c, r - 1, 2)
                river_incident_tiles.update(((c, r), (c, r - 1)))
            else:
                add_river_bit(c, r, 128)
                add_river_bit(c - 1, r, 8)
                river_incident_tiles.update(((c, r), (c - 1, r)))

        mountain_chain = {
            # The near ridge follows one bank. Every adjacent river edge keeps
            # its opposite incident tile open, giving the relief-aware spline
            # somewhere to bend instead of trapping it beneath two peaks.
            (11, -4), (12, -4), (13, -3), (13, -2), (13, -1),
            (14, -1), (15, -1), (16, -1),
            # The remainder extends that ridge and supplies a second, wider
            # valley wall without pinching the navigable channel.
            (13, -5), (14, -5), (14, -4), (15, -4), (15, -3),
            (16, -3), (16, -2), (17, -2), (18, -2),
            (12, 2), (13, 2), (14, 2), (15, 2), (16, 2), (17, 2),
        }
        foothills = {
            (11, -5), (12, -5), (13, -4), (14, -3), (15, -2),
            (17, -3), (18, -3), (18, -1),
            (11, -2), (12, -2), (12, -1), (12, 0), (13, 1),
            (14, 1), (15, 1), (16, 1), (17, 1), (18, 2), (19, 2),
        }
        forests = {
            # Upper-course forest lies on both incident sides of the source
            # edges; production clearance must leave the water body exposed.
            (8, -6), (9, -6), (10, -6), (8, -5), (9, -5), (10, -5),
            (8, -4), (9, -4), (10, -4), (8, -3), (9, -3), (10, -3),
            (9, -2), (10, -2), (11, -2), (10, -1), (11, -1), (12, -1),
            # A smaller riparian belt bridges the mountain and jungle zones.
            (17, 2), (18, 2), (19, 2), (17, 3), (18, 3), (19, 3),
        }
        jungles = {
            # Lower-course jungle occupies both banks without using a river
            # edge tile, keeping the legacy jungle bodies out of the channel.
            (18, -1), (19, -1), (20, -1), (21, -1),
            (18, 3), (19, 3), (20, 3), (21, 3),
            (17, 4), (18, 4), (19, 4), (20, 4), (21, 4),
            (18, 5), (19, 5), (20, 5),
        }
    rows = []
    for y in range(world_size):
        for x in range(y % 2, world_size, 2):
            base, real = recipe["terrain"], recipe["feature"]
            river = 0
            if recipe["feature"] in (5, 6, 7, 8):
                base, real = 2, 2
                # Mountain review must exercise true edge adjacency.  The raw
                # Civ III lattice advances an edge neighbor on both axes; the
                # older three points were separated by a tile and could only
                # demonstrate isolated peaks.
                feature_points = ((15, 15), (16, 16), (17, 17), (18, 18),
                                  (17, 15)) if recipe["feature"] == 6 else (
                                      (16, 16), (18, 16), (16, 18))
                if (x, y) in feature_points:
                    real = recipe["feature"]
            if category == "mountains":
                c, r = (x + y) // 2, (x - y) // 2
                if case == "coastal":
                    # Three coastal arrangements in one stable review scene:
                    # a range ending at shore, one running alongside it, and
                    # an isolated coastal peak. The stepped coast prevents a
                    # single orientation from hiding transition artifacts.
                    shore = 21 if r <= -3 else 20 if r <= 2 else 19
                    if c >= shore:
                        base = real = 11 if c == shore else 12 if c <= shore + 2 else 13
                    else:
                        base = real = 0 if r <= -3 else 2 if r <= 2 else 3
                        if (c, r) in coastal_mountains:
                            real = recipe["feature"]
                else:
                    # One compact witness exercises every requested perimeter:
                    # plains, grass, tundra and desert under the same connected
                    # range, with its eastern shoulder descending into a coast.
                    if c >= 19:
                        base = real = 11 if c == 19 else 12 if c <= 21 else 13
                    else:
                        base = 1 if c <= 15 else 3 if r >= 1 else 0 if c >= 17 else 2
                        real = recipe["feature"] if (x, y) in feature_points else base
            if "gameplay" in case.split("-"):
                if x < 12:
                    base = real = 1
                if (x, y) == (18, 14):
                    base, real = 2, 5
                if (x, y) == (14, 12):
                    base, real = 2, 7
                if (x, y) == (20, 10):
                    base, real = 2, 6
            if category == "transitions":
                base = real = (2 if x < 16 else 1) if y < 16 else (0 if x < 16 else 3)
            if category in ("shorelines", "seas-oceans"):
                if category == "shorelines":
                    # Exercise the feature this category actually owns: cliffs
                    # occur only where a hill tile meets the shore.  The detail
                    # case keeps a long rocky run in frame; gameplay mixes it
                    # with ordinary lowland beach so both joins remain visible.
                    shore = 15 if 7 <= y < 14 else 11 if 23 <= y < 28 else 13
                    if x < shore:
                        rocky_run = case != "lowland" and (case == "detail" or 8 <= y < 24)
                        base, real = 2, 5 if rocky_run and x >= shore - 4 else 2
                    else:
                        depth = x - shore
                        base = real = 11 if depth < 4 else 12 if depth < 10 else 13
                else:
                    base = real = 2 if x < 13 else 11 if x < 17 else 12 if x < 23 else 13
                    if case == "detail":
                        base = real = 12 if x < 16 else 13
            if category == "rivers":
                c, r = (x + y) // 2, (x - y) // 2
                shore = 22 + (1 if r <= -2 else 0) + (1 if r >= 4 else 0)
                if c >= shore:
                    base = real = 11 if c == shore else 12 if c <= shore + 2 else 13
                else:
                    base = real = 1 if r <= -3 else 2
                    if r >= 5 and c < 17:
                        base = real = 0
                    if (c, r) in foothills:
                        real = 5
                    if (c, r) in mountain_chain:
                        real = 6
                    if (c, r) in forests:
                        base, real = 2, 7
                    if (c, r) in jungles:
                        base, real = 2, 8
                river = watershed_river.get((x, y), 0)
            if category == "floodplains":
                # Opposite edge bits agree on the diagonal raw-coordinate chain.
                river = 2 if x == y else 32 if x == y + 2 else 0
            if category == "resources" and case.startswith("water-"):
                base = real = 12 if x < 16 else 13
                if case == "water-gameplay" and x < 12:
                    base = real = 2 if x < 10 else 11
            if category == "shadows":
                # Stable adjacent systems, all under one captured environment.
                # Keep the lower center flat for the native unit sprite API.
                base = real = 2
                if (x, y) in ((13, 13), (14, 14)):
                    real = 6
                if (x, y) == (18, 14):
                    real = 7
                if case == "gameplay" and x > 21:
                    base = real = 11 if x < 24 else 12
            rows.append(f"{x},{y},{base},{real},0,0,{river}")
    destination.write_text(f"C3X_BIQ_TERRAIN_V3,{world_size},{world_size},{len(rows)}\n" + "\n".join(rows) + "\n")


def native_render(category, case, hour, zoom, output, *, behavior=None, center=(16, 16), diagnostics=False, candidate=None, preview=None):
    from Renderer.lab.platform import run_native_fixture
    if behavior not in (None, "replay", "edits", "animation", "units"):
        raise ValueError("Unknown native behavior check")
    if category == "rivers" and center == (16, 16):
        # The paired views inspect opposite ends of the same watershed instead
        # of wasting both captures on its middle reach.
        center = (8, 14) if case == "detail" else (21, 19)
    if category == "shorelines" and case == "lowland" and center == (16, 16):
        center = (10, 18)
    if category == "mountains" and case == "coastal" and center == (16, 16):
        center = (18, 18)
    output.mkdir(parents=True, exist_ok=True)
    csv = output / "scene.csv"
    scene(category, case, csv, world_size=100 if behavior else 32)
    name = f"{case}-h{hour:02}-z{zoom}"
    image = output / (name + ".bmp")
    dll = candidate or ROOT / "Renderer/native/build/candidate/C3XRenderer.dll"
    if not dll.is_file():
        raise ValueError("Build the candidate DLL before native rendering")
    # Clear all diagnostic overrides that could otherwise change the scene.
    env = {
        "C3X_RENDERER_VISUAL_PROFILE": "", "C3X_RENDERER_TRACE": "0",
        "C3X_RENDERER_TRACE_FILE": "", "C3X_RENDERER_PREVIEW_COLOR": "",
        "C3X_RENDERER_PREVIEW_CUSTOM_DEFINITIONS": r"..\..\Renderer\custom.custom_rendering.txt",
        "C3X_RENDERER_PREVIEW_OBJECTS": "1" if standard(category)["recipe"]["objects"] and category != "shadows" else "",
        "C3X_RENDERER_PREVIEW_CITY": "0,3,1,1",
        "C3X_RENDERER_PREVIEW_REPLAY": "", "C3X_RENDERER_PREVIEW_MINIMAP": "",
        "C3X_RENDERER_PREVIEW_EDITS": "", "C3X_RENDERER_PREVIEW_ANIMATION": "",
        "C3X_RENDERER_PREVIEW_UNITS": "", "C3X_RENDERER_PREVIEW_SEASON": "0",
        "C3X_RENDERER_UNIT_CASES": "", "C3X_RENDERER_UNIT_PACK": "",
        "C3X_RENDERER_PREVIEW_ACTIVE_VOLCANO": "",
        "C3X_RENDERER_FIDELITY_SHADOW_CONTROL": "", "C3X_RENDERER_REFLECTION_CONTROL": "",
        "C3X_RENDERER_CITY_LIGHT_CONTROL": "", "C3X_RENDERER_CITY_GLOW_CONTROL": "",
        "C3X_LAB_UNIT_STUDY": "shadows" if category == "shadows" else "1" if category in ("units", "animation") else "",
        "C3X_LAB_ACTION_CURSOR": "0" if category == "animation" and case.endswith("start") else "7",
        "C3X_LAB_OBJECT_STUDY": category if category in ("resources", "infrastructure", "shadows", "huts-camps") else "",
        "C3X_LAB_WATER_STUDY": "1" if case.startswith("water-") else "",
    }
    unit_sizing = category == "units" and case in ("sizing", "sizing-gameplay", "sizing-move")
    if unit_sizing:
        from Renderer.lab.studies.units.prepare import build as build_unit_study
        build_unit_study()
        env["C3X_RENDERER_UNIT_PACK"] = "UnitQualityStudy"
        env["C3X_LAB_UNIT_STUDY"] = case
    if behavior:
        env["C3X_RENDERER_PREVIEW_" + behavior.upper()] = "1"
        # The resource playback/removal witness supplies its own resources.
        # It cannot share the static-object assertion requiring those resources
        # to remain present after their intentional removal.
        if behavior == "animation":
            env["C3X_RENDERER_PREVIEW_OBJECTS"] = ""
    def windows(path):
        return "..\\..\\" + relative(path).replace("/", "\\")
    run_id = uuid.uuid4().hex
    env["C3X_LAB_PID_FILE"] = windows(output / "process.txt")
    env["C3X_LAB_RUN_ID"] = run_id
    if diagnostics:
        env["C3X_RENDERER_TRACE"] = "2"
        env["C3X_RENDERER_TRACE_FILE"] = windows(output / "renderer.log")
    command = " && ".join(f'set "{key}={value}"' for key, value in env.items())
    if preview is None:
        ensure_preview_tool()
        executable = "..\\lab\\.cache\\native_preview.exe"
    else:
        if not preview.is_file():
            raise ValueError("Build the isolated preview executable before native rendering")
        executable = '"' + windows(preview) + '"'
    width, height = (960, 640) if behavior else (640, 480)
    if unit_sizing:
        width, height = 1200, 880
    command += (f' && {executable} "{windows(dll)}" ..\\.. '
                f'..\\default.custom_rendering.txt "{windows(csv)}" "{windows(image)}" '
                f'{width} {height} {center[0]} {center[1]} {zoom} {hour}')
    print(f"Rendering {category}: {name} through the production renderer", flush=True)
    # Keep the VM transport argument short; Parallels rejects some long command
    # strings before launching cmd. This generated script is disposable output.
    batch = output / "render.bat"
    # Fixed disposable paths stay bounded; the invocation ID prevents an older
    # successful frame from certifying a command that never started/completed.
    log = windows(output / "native.log")
    receipt = windows(output / "completion.txt")
    batch.write_text("@echo off\nsetlocal\n" + command.replace(" && ", "\n") +
        f' > "{log}" 2>&1\nset "C3X_LAB_EXIT=%errorlevel%"\n' +
        f'> "{receipt}" echo {run_id} %C3X_LAB_EXIT%\n' +
        f'type "{log}"\nexit /b %C3X_LAB_EXIT%\n')
    result = run_native_fixture(output, f'call "{windows(batch)}"', run_id)
    if behavior:
        # Disposable diagnostic output belongs to this current check, not a
        # growing historical campaign directory.
        (output / "witness.txt").write_text(result.get("output_tail", ""))
    if result["status"] != "pass" or "0 fallback, output=" not in result.get("output_tail", "") or not image.is_file():
        raise ValueError("Production renderer failed: " + name)
    if category in ("units", "animation", "shadows") and "PASS category unit study" not in result.get("output_tail", ""):
        raise ValueError("Unit body witness did not complete")
    if category in ("resources", "infrastructure", "shadows", "huts-camps") and "PASS category object study" not in result.get("output_tail", ""):
        raise ValueError("Category object ownership witness did not complete")
    if behavior:
        verify_behavior_output(behavior, result.get("output_tail", ""))
    return {"image": relative(image), "sha256": checksum(image), "dll_sha256": checksum(dll),
            "case": case, "hour": hour, "zoom": zoom, "backend": "d3d11", "fixture": "synthetic"}


def verify_behavior_output(behavior, output):
    """Require executed witnesses, not merely a successful process exit."""
    if "FAIL" in output:
        raise ValueError("Native behavior witness failed")
    if behavior == "units":
        required = ["UNIT body matrix drawn=288 status=pass", "UNIT cached anchor translation: pass",
                    "UNIT repeated native cursor: pass", "UNIT retained terrain unchanged: pass",
                    "UNIT post-draw terrain parity: pass", "UNIT config-off preserves canvas: pass",
                    "UNIT action interruption and held endpoint: pass draws=564"]
        for zoom in (0, 1):
            required.append(f"UNIT magenta underlay parity zoom={zoom}")
            for mode in ("RGB555", "RGB565"):
                required.extend((f"UNIT {mode} clipped zoom={zoom}",
                                 f"UNIT {mode} magenta clipped parity zoom={zoom} status=pass"))
        if not all(marker in output for marker in required):
            raise ValueError("Incomplete native unit action/compositing witness")
        return
    if behavior == "animation":
        required = ("ANIMATION temporal: pass changed_frames=5",
                    "ANIMATION scroll parity: pass", "ANIMATION removal parity: pass")
        if not all(marker in output for marker in required):
            raise ValueError("Incomplete animation lifecycle witness")
        return
    if behavior == "replay":
        if "PICKUP selection p95_ms=" not in output or output.count("PICKUP jump=") != 5:
            raise ValueError("Incomplete scrolling/reuse witness")
    elif "PICKUP authoritative edit: pass" not in output:
        raise ValueError("Incomplete authoritative-edit witness")
    match = re.search(r"PICKUP (?:edit )?pixel parity: changed=(\d+) error=(\d+) bytes=(\d+)", output)
    if not match:
        raise ValueError("Missing warm/cold pixel comparison")
    changed, error, size = map(int, match.groups())
    # Preserve the existing production warm/cold thresholds. This is distinct
    # from exact approved-image comparisons and is not a Metal parity tolerance.
    if not size or changed > size // 4000 or error > size // 100:
        raise ValueError("Warm/cold pixel comparison exceeded the production budget")


def integration_replay_cases(category, *, full=False):
    """Select focused witnesses; keep the exhaustive sweep explicit."""
    cases = []
    selected = {category}
    if category == "shadows":
        selected.update(("resources", "units"))
    if full:
        cases.extend((("scroll", "replay", 128, (50, 50), 12),
                      ("reduced-scroll", "replay", 64, (50, 50), 12),
                      ("world-wrap", "replay", 128, (0, 50), 12)))
        selected = set(affected(category))
    terrain = {"grassland", "plains", "desert", "tundra", "floodplains", "transitions",
               "hills", "mountains", "forests", "jungles", "shorelines", "seas-oceans",
               "rivers", "day-night", "shadows"}
    if selected.intersection(terrain):
        cases.append(("terrain-edit", "edits", 128, (50, 50), 12))
    if selected.intersection(("resources", "animation")):
        cases.append(("resource-playback", "animation", 128, (50, 50), 12))
    if selected.intersection(("units", "animation")):
        cases.extend((("unit-actions-day", "units", 128, (50, 50), 12),
                      ("unit-actions-night", "units", 128, (50, 50), 0)))
    return cases


def integration_replays(category, *, full=False):
    """Run the behavior witnesses selected for the current category."""
    cases = integration_replay_cases(category, full=full)
    results = []
    # Independent category tasks must not truncate each other's native logs,
    # process receipts, scenes or output bitmaps during a shared full sweep.
    replay_root = LAB / "out/integration/replays" / category
    from Renderer.lab.platform import NativeFixturePending
    for name, behavior, zoom, center, hour in cases:
        print("Checking production behavior: " + name, flush=True)
        try:
            scene_category, scene_case = "grassland", "gameplay"
            if behavior == "edits":
                # Creating the first coast in an all-land world legitimately
                # invalidates every empty nearest-coast certificate. Exercise
                # local reuse beside an existing coast; keep the edit and
                # warm/cold pixel assertions unchanged.
                scene_category, scene_case, center = "shorelines", "lowland", (10, 18)
            native_render(scene_category, scene_case, hour, zoom,
                          replay_root / name, behavior=behavior, center=center)
            results.append({"name": name, "status": "pass"})
        except NativeFixturePending as error:
            results.append({"name": name, "status": "fail", "reason": str(error)})
            # Completion may still be unconfirmed. Never overlap this case
            # with another process or let its cleanup stop somebody else's run.
            break
        except ValueError as error:
            results.append({"name": name, "status": "fail", "reason": str(error)})
    write(replay_root / "results.json", results)
    failed = [r["name"] for r in results if r["status"] != "pass"]
    if failed:
        raise ValueError("Production behavior checks failed: " + ", ".join(failed))
    return results


def render(category, *, selected_case=None):
    value = standard(category)
    if selected_case is not None and selected_case not in value["recipe"]["cases"]:
        raise ValueError("Unknown fixture case for " + category)
    require_prepared([category])
    ensure_candidate()
    out = LAB / "out" / category
    outputs = []
    signature = category_signatures()[category]
    recipe = value["recipe"]
    for case in ([selected_case] if selected_case else recipe["cases"]):
        for hour in recipe["hours"]:
            for zoom in recipe["zooms"]:
                outputs.append(native_render(category, case, hour, zoom, out / case))
    if category_signatures()[category] != signature:
        raise ValueError("Renderer inputs changed during rendering; rerun the preview")
    result = {"category": category, "outputs": outputs,
              "implementation_identity": signature, "input_signature": signature, "recipe": recipe}
    write(out / "render.json", result)
    print(relative(out / "render.json"))
    return result


def compare(category):
    require_prepared([category])
    from PIL import Image, ImageChops, ImageDraw
    value = standard(category)
    result = read(LAB / "out" / category / "render.json")
    signature = category_signatures()[category]
    if result.get("input_signature") != signature or result.get("recipe") != value["recipe"]:
        raise ValueError("Candidate preview is stale; rerender " + category)
    refs = value["references"].get("d3d11", [])
    if not refs:
        raise ValueError("No approved native references for " + category)
    panels = []
    summaries = []
    for entry in result["outputs"]:
        if checksum(local(entry["image"])) != entry["sha256"]:
            raise ValueError("Candidate image changed after rendering")
        matches = [r for r in refs if all(r[k] == entry[k] for k in ("case", "hour", "zoom", "backend"))]
        if not matches:
            b = Image.open(local(entry["image"])).convert("RGB")
            panel = Image.new("RGB", (b.width * 2, b.height + 28), "#222222")
            panel.paste(b, (b.width, 28))
            draw = ImageDraw.Draw(panel)
            draw.text((8, 8), "No fixed reference for this case/time/zoom", fill="white")
            draw.text((b.width + 8, 8), "Current code", fill="white")
            panels.append(panel)
            summaries.append({"case": entry["case"], "hour": entry["hour"], "zoom": entry["zoom"],
                              "identical_pixels": None, "reference": "unavailable"})
            continue
        if len(matches) != 1:
            raise ValueError("Candidate has ambiguous fixed references")
        ref = matches[0]
        if checksum(local(ref["image"])) != ref["sha256"]:
            raise ValueError("Approved image was modified")
        a = Image.open(local(ref["image"])).convert("RGB")
        b = Image.open(local(entry["image"])).convert("RGB")
        if a.size != b.size:
            raise ValueError("Candidate dimensions differ from reference")
        difference = ImageChops.difference(a, b)
        panel = Image.new("RGB", (a.width * 2, a.height + 28), "#222222")
        panel.paste(a, (0, 28)); panel.paste(b, (a.width, 28))
        draw = ImageDraw.Draw(panel)
        draw.text((8, 8), "Reference", fill="white")
        draw.text((a.width + 8, 8), "Current code", fill="white")
        panels.append(panel)
        summaries.append({"case": entry["case"], "hour": entry["hour"], "zoom": entry["zoom"],
                          "identical_pixels": difference.getbbox() is None})
    sheet = Image.new("RGB", (max(p.width for p in panels), sum(p.height for p in panels)))
    offset = 0
    for panel in panels:
        sheet.paste(panel, (0, offset)); offset += panel.height
    target = LAB / "out" / category / "compare.png"
    sheet.save(target)
    write(target.with_suffix(".json"), summaries)
    print(json.dumps(summaries, indent=2))
    print(relative(target))
    return summaries


def gallery(category=None):
    from PIL import Image, ImageDraw
    cards = []
    for key in ([category] if category else catalog()):
        value = standard(key)
        refs = value["references"].get("d3d11", [])
        if not refs:
            continue
        if category:
            for ref in refs:
                if checksum(local(ref["image"])) != ref["sha256"]:
                    raise ValueError("Modified reference: " + key)
                card = Image.new("RGB", (640, 504), "#222222")
                label = f'{value["title"]} / {ref["case"]} / hour {ref["hour"]} / tile {ref["zoom"]}'
                ImageDraw.Draw(card).text((8, 7), label, fill="white")
                im = Image.open(local(ref["image"])).convert("RGB")
                im.thumbnail((640, 480))
                card.paste(im, (0, 24))
                cards.append(card)
            continue
        selected = [next((r for r in refs if r["case"].startswith(case) and r["hour"] == 12), None)
                    for case in ("detail", "gameplay")]
        card = Image.new("RGB", (640, 264), "#222222")
        ImageDraw.Draw(card).text((8, 7), value["title"] + " / current production", fill="white")
        for index, ref in enumerate(selected):
            if ref:
                if checksum(local(ref["image"])) != ref["sha256"]:
                    raise ValueError("Modified reference: " + key)
                im = Image.open(local(ref["image"])).convert("RGB")
                im.thumbnail((320, 240))
                card.paste(im, (index * 320, 24))
        cards.append(card)
    if not cards:
        raise ValueError("No captured category references")
    height = 504 if category else 264
    sheet = Image.new("RGB", (1280, ((len(cards) + 1) // 2) * height), "#222222")
    for index, card in enumerate(cards):
        sheet.paste(card, ((index % 2) * 640, (index // 2) * height))
    destination = LAB / "out" / category / "reference.png" if category else LAB / "out/gallery.png"
    destination.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(destination)
    print(relative(destination))


def describe(category):
    value = standard(category)
    print((standard_path(category).parent / "README.md").read_text().strip())
    print("Shared dependencies: " + (", ".join(value["depends_on"]) or "none"))
    print("Current implementation:")
    for path in value["implementation"]:
        print("  " + path)
    print("Previews: " + ", ".join(value["recipe"]["cases"]))
    print("Reference gallery: python3 Renderer/renderer.py gallery " + category)
    candidate = LAB / "out" / category / "render.json"
    if candidate.is_file():
        print("Current-code preview: available; compare verifies freshness")
    for limitation in value.get("limitations", []):
        print("Limit: " + limitation)


def validate():
    entries = catalog()
    errors = []
    for category in entries:
        value = standard(category)
        if value["id"] != category:
            errors.append(category + ": mismatched identity")
        for dependency in value["depends_on"]:
            if dependency not in entries:
                errors.append(category + ": missing dependency " + dependency)
        for path in value["implementation"]:
            if not local(path).is_file():
                errors.append(category + ": missing implementation " + path)
    visiting, visited = set(), set()
    def visit(category):
        if category in visiting:
            raise ValueError("Category dependency cycle: " + category)
        if category in visited:
            return
        visiting.add(category)
        for dependency in standard(category)["depends_on"]:
            if dependency in entries:
                visit(dependency)
        visiting.remove(category)
        visited.add(category)
    for category in entries:
        visit(category)
    if errors:
        raise ValueError("\n".join(errors))
    return len(entries)


def test_modules(category=None):
    selected = [category] if category else list(catalog())
    modules = {"Renderer.lab.test_workflow", "Renderer.lab.test_dependencies", "Renderer.lab.test_platform",
               "Renderer.lab.test_backend_bindings", "Renderer.lab.test_preparation"}
    for key in selected:
        modules.update(standard(key)["tests"])
    return sorted(modules)


def run_tests(category=None, *, integration=False, full=False):
    modules = set(test_modules(category))
    if integration:
        modules.update(("Renderer.definitions.test_definition_parser",
                        "Renderer.definitions.test_rule_resolver", "Renderer.scenes.test_scene_contract",
                        "Renderer.native.test_native_bridge_contract",
                        "Renderer.native.test_zoom_mesh_cache",
                        "Renderer.native.test_frame_publication",
                        "Renderer.native.test_asset_content_hash"))
        if full:
            modules.update("Renderer.native." + name for name in (
                "test_scroll_damage", "test_unit_bridge", "test_unit_input_guard",
                "test_unit_shadow", "test_unit_animation_runtime", "test_animation_runtime"))
        else:
            terrain = {"grassland", "plains", "desert", "tundra", "floodplains", "transitions",
                       "hills", "mountains", "forests", "jungles", "shorelines", "seas-oceans",
                       "rivers", "day-night", "shadows"}
            if category in terrain:
                modules.add("Renderer.native.test_scroll_damage")
            if category in ("resources", "animation", "shadows"):
                modules.add("Renderer.native.test_animation_runtime")
            if category in ("units", "animation", "shadows"):
                modules.update("Renderer.native." + name for name in (
                    "test_unit_bridge", "test_unit_input_guard", "test_unit_shadow",
                    "test_unit_animation_runtime", "test_animation_runtime"))
    result = subprocess.run([sys.executable, "-m", "unittest", *sorted(modules)], cwd=ROOT)
    if result.returncode:
        raise ValueError("Current rendering regression checks failed")
    return sorted(modules)


def run_affected_tests(category):
    modules = set(test_modules(category))
    for key in affected(category):
        modules.update(standard(key)["tests"])
    result = subprocess.run([sys.executable, "-m", "unittest", *sorted(modules)], cwd=ROOT)
    if result.returncode:
        raise ValueError("Current rendering regression checks failed")
    return sorted(modules)


def build_candidate():
    from Renderer.lab.platform import native_command_result
    prepare_sources()
    before = checksum(ROOT / "Renderer/bin/C3XRenderer.dll")
    inputs = native_inputs()
    result = native_command_result("Renderer/native", "call BUILD.bat candidate-compile")
    if result["status"] != "pass":
        raise ValueError("Candidate build failed")
    ensure_preview_tool()
    if checksum(ROOT / "Renderer/bin/C3XRenderer.dll") != before:
        raise ValueError("Candidate build unexpectedly changed the staged DLL")
    if native_inputs() != inputs:
        raise ValueError("Native source changed during compilation; rebuild")
    write(LAB / ".cache/native-build.json", {"inputs": inputs,
        "dll_sha256": checksum(ROOT / "Renderer/native/build/candidate/C3XRenderer.dll")})


def verify_integration(category, *, build=False, full=False, renderer_only=False):
    """Automated delivery evidence, never an assertion of a live-game test."""
    receipt_path = LAB / "out/integration" / (category + ".json")
    scope = "full" if full else "focused"
    # Invalidate an older successful receipt before attempting new verification.
    write(receipt_path, {"status": "running", "category": category, "scope": scope})
    try:
        return verify_integration_checks(category, build=build, full=full, renderer_only=renderer_only)
    except Exception as error:
        write(receipt_path, {"status": "fail", "category": category, "scope": scope,
                             "reason": str(error),
                             "live_game": "not_tested"})
        raise


def verify_integration_checks(category, *, build=False, full=False, renderer_only=False):
    receipt_path = LAB / "out/integration" / (category + ".json")
    selected = affected(category) if full else [category]
    prepare_sources(selected)
    if build:
        build_candidate()
    else:
        ensure_candidate()
    identity = implementation_identity()
    modules = run_tests(category, integration=True, full=full)
    from Renderer.lab.platform import changed_injected_sources, injected_compile_result
    if not renderer_only and changed_injected_sources() and injected_compile_result()["status"] != "pass":
        raise ValueError("Approved injected compile/injection smoke test failed")
    replays = integration_replays(category, full=full)
    if identity != implementation_identity():
        raise ValueError("Inputs changed during integration verification")
    signatures = category_signatures()
    result = {"status": "pass", "category": category,
              "scope": "full" if full else "focused", "categories": selected,
              "input_signatures": {key: signatures[key] for key in selected},
              "implementation_identity": identity,
              "dll_sha256": checksum(ROOT / "Renderer/native/build/candidate/C3XRenderer.dll"),
              "tests": modules, "replays": replays, "live_game": "not_tested",
              "injected_compile": "not_requested_renderer_only" if renderer_only else "checked_if_changed"}
    write(receipt_path, result)
    print("PASS current-code integration checks. Reference comparison is opt-in; Civ III was not launched.")
    return result


def approve(category, note):
    """Replace the category's optional comparison snapshot after user acceptance."""
    if not note.strip():
        raise ValueError("Record the user's explicit approval of the affected appearance")
    require_prepared([category])
    selected = [category]
    signatures = category_signatures()
    identity = signatures[category]
    updates = []
    for key in selected:
        value = standard(key)
        path = LAB / "out" / key / "render.json"
        if not path.is_file():
            raise ValueError("Render the affected category before approval: " + key)
        result = read(path)
        if result.get("recipe") != value["recipe"]:
            raise ValueError("Preview is stale: " + key)
        if result.get("input_signature") != signatures[key]:
            raise ValueError("Preview input selection is stale: " + key)
        expected = {(case, hour, zoom) for case in value["recipe"]["cases"]
                    for hour in value["recipe"]["hours"] for zoom in value["recipe"]["zooms"]}
        actual = {(r["case"], r["hour"], r["zoom"]) for r in result["outputs"]}
        if actual != expected or len(actual) != len(result["outputs"]):
            raise ValueError("Approval requires the complete category preview: " + key)
        folder = LAB / "references" / key / "approved"
        for entry in result["outputs"]:
            if checksum(local(entry["image"])) != entry["sha256"]:
                raise ValueError("Preview image changed: " + key)
        updates.append((key, value, result, folder))
    for key, value, result, folder in updates:
        staged = folder.parent / (".approved-" + uuid.uuid4().hex)
        staged.mkdir(parents=True)
        references = []
        for entry in result["outputs"]:
            target = staged / entry["case"] / Path(entry["image"]).name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(local(entry["image"]), target)
            references.append(dict(entry, image=relative(folder / entry["case"] / target.name)))
        backup = folder.parent / (".previous-" + uuid.uuid4().hex)
        try:
            if folder.exists():
                folder.replace(backup)
            staged.replace(folder)
        except Exception:
            if backup.exists() and not folder.exists():
                backup.replace(folder)
            raise
        else:
            if backup.exists():
                shutil.rmtree(backup)
        value.update(approval={"user_statement": note, "implementation_identity": identity},
                     references={"d3d11": references}, reference_inputs=signatures[key])
        value.pop("candidate", None)
        value.pop("reviewed_inputs", None)
        write(standard_path(key), value)
    print("Updated reference: " + ", ".join(key for key, _, _, _ in updates))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest="command", required=True)
    commands.add_parser("list")
    commands.add_parser("prepare", help="Refresh shader bindings and changed category assets without staging")
    gallery_parser = commands.add_parser("gallery")
    gallery_parser.add_argument("category", nargs="?", choices=list(catalog()))
    tests = commands.add_parser("test")
    tests.add_argument("category", nargs="?", choices=list(catalog()))
    tests.add_argument("--affected", action="store_true", help="Include visual dependents; category tests are the default")
    tests.add_argument("--backends", choices=("metal","both"), help="Check GPU bindings and compile production Metal shaders; not full scene parity")
    commands.add_parser("build", help="Build a candidate and preview tools without staging")
    commands.add_parser("check", help="Validate the category catalog without rendering or hashing reference images")
    integration = commands.add_parser("integration", help="Verify the current code without staging or launching Civ III")
    integration.add_argument("category", choices=list(catalog()))
    integration.add_argument("--build", action="store_true")
    integration.add_argument("--full", action="store_true", help="Run the exhaustive cross-category regression sweep")
    integration.add_argument("--renderer-only", action="store_true", help="Verify standalone renderer changes without compiling unrelated injected C edits")
    approval = commands.add_parser("approve")
    approval.add_argument("category", choices=list(catalog()))
    approval.add_argument("--user-approval", required=True, help="Quote the user's actual approval of the complete category preview")
    for name in ("show", "affected", "lab", "compare"):
        parser = commands.add_parser(name)
        parser.add_argument("category", choices=list(catalog()))
        if name in ("lab", "compare"):
            parser.add_argument("--affected", action="store_true", help="Include visual dependents; the requested category is the default")
        if name == "lab":
            parser.add_argument("--case", help="One fixture case listed by show CATEGORY")
    args = p.parse_args()
    try:
        if args.command == "list":
            for key in catalog():
                value = standard(key)
                print(f'{key:18} {value["title"]}')
        elif args.command == "show":
            describe(args.category)
        elif args.command == "gallery":
            reexec_with_workspace_python(("PIL",))
            gallery(args.category)
        elif args.command == "test":
            reexec_with_workspace_python(("PIL", "numpy"))
            selected = affected(args.category) if args.category and args.affected else (
                [args.category] if args.category else None)
            prepare_sources(selected)
            if args.category and args.affected:
                run_affected_tests(args.category)
            else:
                run_tests(args.category)
            if args.backends:
                from Renderer.lab.test_backend_bindings import gpu_check
                gpu_check(windows=args.backends=="both")
        elif args.command == "build":
            build_candidate()
        elif args.command == "prepare":
            prepare_sources()
        elif args.command == "affected":
            print("\n".join(affected(args.category)))
        elif args.command == "lab":
            selected = affected(args.category) if getattr(args, "affected", False) else [args.category]
            prepare_sources(selected)
            for key in selected:
                selected_case = getattr(args, "case", None) if key == args.category else None
                render(key, selected_case=selected_case)
        elif args.command == "compare":
            reexec_with_workspace_python(("PIL",))
            for key in affected(args.category) if args.affected else [args.category]:
                compare(key)
        elif args.command == "check":
            print(f"PASS {validate()} category definitions")
        elif args.command == "integration":
            reexec_with_workspace_python(("PIL", "numpy"))
            verify_integration(args.category, build=args.build, full=args.full, renderer_only=args.renderer_only)
        elif args.command == "approve":
            if not args.user_approval.strip():
                raise ValueError("An actual user approval statement is required")
            approve(args.category, args.user_approval)
        return 0
    except (ValueError, OSError) as exc:
        print("ERROR: " + str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

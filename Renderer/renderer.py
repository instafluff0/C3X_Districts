#!/usr/bin/env python3
"""Category-based C3X visual workbench. Production appearance is the baseline."""
from __future__ import annotations

import argparse
import hashlib
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


def dirty_categories():
    from Renderer.lab.dependencies import dirty
    return dirty({key: standard(key) for key in catalog()}, category_signatures())


def affected(category):
    # Source-selected categories already represent actual fixture consumers.
    # Do not expand a local city-light change into a global sun/moon change just
    # because the day/night fixture is one of its consumers.
    return sorted(set(declared_affected(category)) | set(dirty_categories()))


def prepare_sources():
    from Renderer.lab.preparation import prepare
    changed = prepare(ROOT)
    print(f"Prepared production shader bindings: {len(changed)} changed files", flush=True)
    from Renderer.lab.asset_preparation import prepare as prepare_assets
    assets = prepare_assets()
    print(f"Prepared current category assets: {len(assets)} refreshed files", flush=True)
    return changed + assets


def require_prepared():
    from Renderer.lab.preparation import require_current
    require_current(ROOT)
    from Renderer.lab.asset_preparation import prepare as prepare_assets
    prepare_assets(check_only=True)


def pack_files():
    """Inventory the same pack tree with one cached directory-entry stat.

    Directory symlinks are not followed (matching Path.rglob); file symlinks
    still resolve to their repository-local source and cannot escape the root.
    """
    root = (ROOT / "Renderer/packs").resolve()
    root.relative_to(ROOT)
    pending = [root] if root.is_dir() else []
    while pending:
        with os.scandir(pending.pop()) as entries:
            for entry in entries:
                if entry.name == "__pycache__":
                    continue
                if entry.is_dir(follow_symlinks=False):
                    pending.append(entry.path)
                elif entry.is_file():
                    path = Path(entry.path)
                    key = relative(path) if entry.is_symlink() else path.relative_to(ROOT).as_posix()
                    yield key, path, entry.stat()


def implementation_inputs():
    """Current bytes, shared by stale-preview guards and category selection."""
    paths = set()
    for category in catalog():
        paths.update(local(p) for p in standard(category)["implementation"])
    paths.update((ROOT / "Renderer/native").rglob("*.hlsl"))
    paths.update((ROOT / "Renderer/native").rglob("*.cpp"))
    paths.update((ROOT / "Renderer/native").rglob("*.h"))
    paths.update((LAB / "shared").rglob("*.py"))
    for extension in ("*.hlsl", "*.h", "*.csv", "*.json"):
        paths.update((LAB / "shared").rglob(extension))
    for extension in ("*.py", "*.cpp", "*.mm", "*.h", "*.hlsl"):
        paths.update((LAB / "backends").rglob(extension))
        paths.update((LAB / "scene").rglob(extension))
    paths.update((LAB / "contracts").rglob("*.h"))
    paths.update([ROOT / "C3X.h", ROOT / "injected_code.c"])
    paths.update([ROOT / "Renderer/default.custom_rendering.txt", ROOT / "Renderer/custom.custom_rendering.txt"])
    paths.update([ROOT / "Renderer/renderer.py", LAB / "native_preview.cpp", LAB / "dependencies.py", LAB / "platform.py"])
    from Renderer.lab.preparation import input_paths
    paths.update(ROOT / path for path in input_paths(ROOT))
    from Renderer.lab.asset_preparation import jobs as asset_jobs
    paths.add(LAB / "asset_preparation.py")
    paths.update(ROOT / job[3] for job in asset_jobs())
    for record in asset_receipts().values():
        paths.update(local(path) for path in record.get("inputs", {})
                     if not path.startswith("Renderer/packs/"))
    records = {relative(path): checksum(path) for path in sorted(paths) if path.is_file()}
    # Texture/material edits must invalidate previews even when the binary index
    # is unchanged. Cache content hashes by size/mtime/ctime for fast repeated
    # local iteration; this is disposable internal cache, not a handoff ledger.
    cache_path = LAB / ".cache/input-hashes.json"
    cache = read(cache_path) if cache_path.exists() else {}
    current = {}
    for key, path, stat in pack_files():
        stamp = [stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns]
        previous = cache.get(key, {})
        digest = previous.get("digest") if previous.get("stamp") == stamp else None
        digest = digest or checksum(path)
        current[key] = {"stamp": stamp, "digest": digest}
        records[key] = digest
    if current != cache:
        write(cache_path, current)
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
    baseline = read(LAB / "baseline.json")
    if digest == baseline["dll_sha256"]:
        check = subprocess.run(["git", "diff", "--quiet", baseline["source_commit"], "--", *inputs], cwd=ROOT)
        tracked = subprocess.run(["git", "ls-files", "--error-unmatch", "--", *inputs], cwd=ROOT,
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if check.returncode == 0 and tracked.returncode == 0:
            # The unchanged production binary is an existing verified build.
            write(receipt, {"dll_sha256": digest, "inputs": inputs})
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
    rows = []
    for y in range(world_size):
        for x in range(y % 2, world_size, 2):
            base, real = recipe["terrain"], recipe["feature"]
            river = 0
            if recipe["feature"] in (5, 6, 7, 8):
                base, real = 2, 2
                if (x, y) in ((16, 16), (18, 16), (16, 18)):
                    real = recipe["feature"]
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
                base = real = 2 if x < 13 else 11 if x < 17 else 12 if x < 23 else 13
                if category == "seas-oceans" and case == "detail":
                    base = real = 12 if x < 16 else 13
            if category in ("rivers", "floodplains"):
                # Opposite edge bits agree on the diagonal raw-coordinate chain.
                river = 2 if x == y else 32 if x == y + 2 else 0
            if category == "resources" and case.startswith("water-"):
                base = real = 12 if x < 16 else 13
                if case == "water-gameplay" and x < 12:
                    base = real = 2 if x < 10 else 11
            rows.append(f"{x},{y},{base},{real},0,0,{river}")
    destination.write_text(f"C3X_BIQ_TERRAIN_V3,{world_size},{world_size},{len(rows)}\n" + "\n".join(rows) + "\n")


def native_render(category, case, hour, zoom, output, *, baseline=False, behavior=None, center=(16, 16), diagnostics=False):
    from Renderer.lab.platform import run_native_fixture
    if behavior not in (None, "replay", "edits", "animation", "units"):
        raise ValueError("Unknown native behavior check")
    output.mkdir(parents=True, exist_ok=True)
    csv = output / "scene.csv"
    scene(category, case, csv, world_size=100 if behavior else 32)
    name = f"{case}-h{hour:02}-z{zoom}"
    image = output / (name + ".bmp")
    dll = ROOT / "Renderer/bin/C3XRenderer.dll" if baseline else ROOT / "Renderer/native/build/candidate/C3XRenderer.dll"
    if baseline and checksum(dll) != read(LAB / "baseline.json")["dll_sha256"]:
        raise ValueError("Staged DLL differs from the user-approved baseline")
    if not dll.is_file():
        raise ValueError("Build the candidate DLL before native rendering")
    # Clear all diagnostic overrides that could otherwise change the scene.
    env = {
        "C3X_RENDERER_VISUAL_PROFILE": "", "C3X_RENDERER_TRACE": "0",
        "C3X_RENDERER_TRACE_FILE": "", "C3X_RENDERER_PREVIEW_COLOR": "",
        "C3X_RENDERER_PREVIEW_CUSTOM_DEFINITIONS": r"..\..\Renderer\custom.custom_rendering.txt",
        "C3X_RENDERER_PREVIEW_OBJECTS": "1" if standard(category)["recipe"]["objects"] else "",
        "C3X_RENDERER_PREVIEW_CITY": "0,3,1,1",
        "C3X_RENDERER_PREVIEW_REPLAY": "", "C3X_RENDERER_PREVIEW_MINIMAP": "",
        "C3X_RENDERER_PREVIEW_EDITS": "", "C3X_RENDERER_PREVIEW_ANIMATION": "",
        "C3X_RENDERER_PREVIEW_UNITS": "", "C3X_RENDERER_PREVIEW_SEASON": "0",
        "C3X_RENDERER_UNIT_CASES": "", "C3X_RENDERER_UNIT_PACK": "",
        "C3X_RENDERER_PREVIEW_ACTIVE_VOLCANO": "",
        "C3X_RENDERER_FIDELITY_SHADOW_CONTROL": "", "C3X_RENDERER_REFLECTION_CONTROL": "",
        "C3X_RENDERER_CITY_LIGHT_CONTROL": "", "C3X_RENDERER_CITY_GLOW_CONTROL": "",
        "C3X_LAB_UNIT_STUDY": "1" if category in ("units", "animation") else "",
        "C3X_LAB_ACTION_CURSOR": "0" if category == "animation" and case.endswith("start") else "7",
        "C3X_LAB_OBJECT_STUDY": category if category in ("resources", "infrastructure") else "",
        "C3X_LAB_WATER_STUDY": "1" if case.startswith("water-") else "",
    }
    if behavior:
        env["C3X_RENDERER_PREVIEW_" + behavior.upper()] = "1"
        # The resource playback/removal witness supplies its own resources.
        # It cannot share the static-object assertion requiring those resources
        # to remain present after their intentional removal.
        if behavior == "animation":
            env["C3X_RENDERER_PREVIEW_OBJECTS"] = ""
    def windows(path):
        return "..\\..\\" + relative(path).replace("/", "\\")
    if diagnostics:
        env["C3X_RENDERER_TRACE"] = "2"
        env["C3X_RENDERER_TRACE_FILE"] = windows(output / "renderer.log")
    command = " && ".join(f'set "{key}={value}"' for key, value in env.items())
    ensure_preview_tool()
    executable = "..\\lab\\.cache\\native_preview.exe"
    width, height = (960, 640) if behavior else (640, 480)
    command += (f' && {executable} "{windows(dll)}" ..\\.. '
                f'..\\default.custom_rendering.txt "{windows(csv)}" "{windows(image)}" '
                f'{width} {height} {center[0]} {center[1]} {zoom} {hour}')
    print(f"Rendering {category}: {name} through the production renderer", flush=True)
    # Keep the VM transport argument short; Parallels rejects some long command
    # strings before launching cmd. This generated script is disposable output.
    batch = output / "render.bat"
    run_id = uuid.uuid4().hex
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
    if category in ("units", "animation") and "PASS category unit study" not in result.get("output_tail", ""):
        raise ValueError("Unit body witness did not complete")
    if category in ("resources", "infrastructure") and "PASS category object study" not in result.get("output_tail", ""):
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


def integration_replays():
    cases = (("scroll", "replay", 128, (50, 50), 12),
             ("reduced-scroll", "replay", 64, (50, 50), 12),
             ("world-wrap", "replay", 128, (0, 50), 12),
             ("terrain-edit", "edits", 128, (50, 50), 12),
             ("resource-playback", "animation", 128, (50, 50), 12),
             ("unit-actions-day", "units", 128, (50, 50), 12),
             ("unit-actions-night", "units", 128, (50, 50), 0))
    results = []
    for name, behavior, zoom, center, hour in cases:
        print("Checking production behavior: " + name, flush=True)
        try:
            native_render("grassland", "gameplay", hour, zoom,
                          LAB / "out/integration/replays" / name, behavior=behavior, center=center)
            results.append({"name": name, "status": "pass"})
        except ValueError as error:
            results.append({"name": name, "status": "fail", "reason": str(error)})
    write(LAB / "out/integration/replays/results.json", results)
    failed = [r["name"] for r in results if r["status"] != "pass"]
    if failed:
        raise ValueError("Production behavior checks failed: " + ", ".join(failed))
    return results


def render(category, *, baseline=False, selected_case=None):
    value = standard(category)
    if selected_case is not None and selected_case not in value["recipe"]["cases"]:
        raise ValueError("Unknown fixture case for " + category)
    if not baseline:
        require_prepared()
        ensure_candidate()
    out = LAB / ("references" if baseline else "out") / category
    if baseline:
        out /= "r1"
        if value["approved_revision"] != 1 or value["references"]:
            raise ValueError("Baseline references already captured or superseded")
    outputs = []
    identity = implementation_identity()
    signature = category_signatures()[category]
    recipe = value["recipe"]
    for case in ([selected_case] if selected_case else recipe["cases"]):
        for hour in recipe["hours"]:
            for zoom in recipe["zooms"]:
                outputs.append(native_render(category, case, hour, zoom, out / case, baseline=baseline))
    if implementation_identity() != identity:
        raise ValueError("Renderer inputs changed during rendering; rerun the preview")
    result = {"category": category, "revision": value["revision"], "outputs": outputs,
              "implementation_identity": identity, "input_signature": signature, "recipe": recipe,
              "dependency_revisions": {key: standard(key)["approved_revision"] for key in value["depends_on"]}}
    write(out / "render.json", result)
    if baseline:
        value["references"] = {"d3d11": outputs}
        write(standard_path(category), value)
    else:
        value["candidate"] = {"render": relative(out / "render.json"),
                              "implementation_identity": identity}
        write(standard_path(category), value)
    print(relative(out / "render.json"))
    return result


def compare(category):
    require_prepared()
    from PIL import Image, ImageChops, ImageDraw
    value = standard(category)
    result = read(LAB / "out" / category / "render.json")
    if result.get("implementation_identity") != implementation_identity() or result.get("recipe") != value["recipe"]:
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
        if len(matches) != 1:
            raise ValueError("Candidate has no corresponding approved reference")
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
        draw.text((8, 8), "Approved baseline", fill="white")
        draw.text((a.width + 8, 8), "Candidate", fill="white")
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
    # Exact pixels establish that these inputs still reproduce the existing
    # approval. This is not a new revision, permission for different pixels,
    # or evidence of a game test. Partial comparisons never clear review scope.
    expected = {(case, hour, zoom) for case in value["recipe"]["cases"]
                for hour in value["recipe"]["hours"] for zoom in value["recipe"]["zooms"]}
    actual = {(row["case"], row["hour"], row["zoom"]) for row in summaries}
    if actual == expected and len(summaries) == len(expected) and all(row["identical_pixels"] for row in summaries):
        signatures = category_signatures()
        if result.get("input_signature") == signatures[category] and result["implementation_identity"] == implementation_identity():
            value["reviewed_inputs"] = {"revision": value["approved_revision"],
                "signature": signatures[category], "basis": "exact_approved_reference_comparison"}
            write(standard_path(category), value)
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
                label = f'{value["title"]} r{value["approved_revision"]} / {ref["case"]} / hour {ref["hour"]} / tile {ref["zoom"]}'
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
    integrated = read(ROOT / "Renderer/integration/status.json")["categories"].get(category, {})
    print((standard_path(category).parent / "README.md").read_text().strip())
    print(f'\nApproved r{value["approved_revision"]}; integrated r{integrated.get("integrated_revision", "none")}.')
    print("Shared dependencies: " + (", ".join(value["depends_on"]) or "none"))
    print("Current implementation:")
    for path in value["implementation"]:
        print("  " + path)
    print("Previews: " + ", ".join(value["recipe"]["cases"]))
    print("Reference gallery: python3 Renderer/renderer.py gallery " + category)
    candidate = value.get("candidate")
    if candidate:
        fresh = candidate.get("implementation_identity") == implementation_identity()
        print("Unapproved candidate: " + ("available" if fresh else "stale; rerender before review"))
    for limitation in value.get("limitations", []):
        print("Limit: " + limitation)


def validate(*, complete=False):
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
        for refs in value["references"].values():
            for ref in refs:
                if not local(ref["image"]).is_file() or checksum(local(ref["image"])) != ref["sha256"]:
                    errors.append(category + ": missing or modified reference " + ref["image"])
        if complete:
            expected = {(case, hour, zoom) for case in value["recipe"]["cases"]
                        for hour in value["recipe"]["hours"] for zoom in value["recipe"]["zooms"]}
            for backend in ("d3d11", "metal"):
                references = value["references"].get(backend, [])
                actual = {(r["case"], r["hour"], r["zoom"]) for r in references}
                if actual != expected or len(actual) != len(references):
                    errors.append(category + ": " + backend + " reference coverage is incomplete")
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


def pending():
    integrated = read(ROOT / "Renderer/integration/status.json")["categories"]
    changes = []
    for category in catalog():
        value = standard(category)
        before = integrated.get(category, {}).get("integrated_revision")
        if value["approved_revision"] != before:
            changes.append({"category": category, "approved": value["approved_revision"], "integrated": before})
    return changes


def test_modules(category=None):
    selected = affected(category) if category else list(catalog())
    modules = {"Renderer.lab.test_workflow", "Renderer.lab.test_dependencies", "Renderer.lab.test_platform", "Renderer.lab.test_backend_bindings",
               "Renderer.lab.test_preparation", "Renderer.lab.test_natural_scene"}
    for key in selected:
        modules.update(standard(key)["tests"])
    return sorted(modules)


def run_tests(category=None, *, integration=False):
    modules = set(test_modules(category))
    if integration:
        modules.update(("Renderer.definitions.test_definition_parser",
                        "Renderer.definitions.test_rule_resolver", "Renderer.scenes.test_scene_contract"))
        modules.update("Renderer.native." + name for name in (
            "test_native_bridge_contract", "test_scroll_damage", "test_unit_bridge",
            "test_unit_input_guard", "test_unit_shadow", "test_unit_animation_runtime", "test_asset_content_hash",
            "test_animation_runtime"))
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
    ensure_preview_tool(force=True)
    if checksum(ROOT / "Renderer/bin/C3XRenderer.dll") != before:
        raise ValueError("Candidate build unexpectedly changed the staged DLL")
    if native_inputs() != inputs:
        raise ValueError("Native source changed during compilation; rebuild")
    write(LAB / ".cache/native-build.json", {"inputs": inputs,
        "dll_sha256": checksum(ROOT / "Renderer/native/build/candidate/C3XRenderer.dll")})


def verify_integration(category, *, build=False):
    """Automated delivery evidence, never an assertion of a live-game test."""
    receipt_path = LAB / "out/integration" / (category + ".json")
    # Invalidate an older successful receipt before attempting new verification.
    write(receipt_path, {"status": "running", "category": category})
    try:
        return verify_integration_checks(category, build=build)
    except Exception as error:
        write(receipt_path, {"status": "fail", "category": category, "reason": str(error),
                             "live_game": "not_tested"})
        raise


def verify_integration_checks(category, *, build=False):
    receipt_path = LAB / "out/integration" / (category + ".json")
    prepare_sources()
    if build:
        build_candidate()
    else:
        ensure_candidate()
    identity = implementation_identity()
    modules = run_tests(category, integration=True)
    from Renderer.lab.platform import changed_injected_sources, injected_compile_result
    if changed_injected_sources() and injected_compile_result()["status"] != "pass":
        raise ValueError("Approved injected compile/injection smoke test failed")
    selected = affected(category)
    comparisons = {}
    for key in selected:
        render(key)
        comparisons[key] = compare(key)
        if not all(row["identical_pixels"] for row in comparisons[key]):
            raise ValueError("Delivery differs from the approved D3D11 appearance: " + key)
    replays = integration_replays()
    if identity != implementation_identity():
        raise ValueError("Inputs changed during integration verification")
    result = {"status": "pass", "category": category, "categories": selected,
              "approved_revisions": {key: standard(key)["approved_revision"] for key in selected},
              "implementation_identity": identity,
              "dll_sha256": checksum(ROOT / "Renderer/native/build/candidate/C3XRenderer.dll"),
              "tests": modules, "comparisons": comparisons, "replays": replays, "live_game": "not_tested"}
    write(receipt_path, result)
    print("PASS automated delivery checks; Civ III has not been launched or certified.")
    return result


def integration_receipt(category):
    require_prepared()
    result = read(LAB / "out/integration" / (category + ".json"))
    if result.get("status") != "pass" or result.get("implementation_identity") != implementation_identity():
        raise ValueError("Run fresh integration verification first")
    if result["dll_sha256"] != checksum(ROOT / "Renderer/native/build/candidate/C3XRenderer.dll"):
        raise ValueError("Candidate DLL changed after verification")
    for key in result["categories"]:
        if result["approved_revisions"][key] != standard(key)["approved_revision"]:
            raise ValueError("Approval changed after verification: " + key)
    return result


def record_integration(category, note):
    if not note.strip():
        raise ValueError("Describe the actual Civ III check; automated replay alone is insufficient")
    result = integration_receipt(category)
    if checksum(ROOT / "Renderer/bin/C3XRenderer.dll") != result["dll_sha256"]:
        raise ValueError("The staged DLL is not the verified candidate")
    path = ROOT / "Renderer/integration/status.json"
    status = read(path)
    for key in result["categories"]:
        status["categories"][key] = {"integrated_revision": result["approved_revisions"][key],
            "verification": "automated_replay_and_recorded_game_check", "game_check": note,
            "dll_sha256": result["dll_sha256"], "implementation_identity": result["implementation_identity"]}
    write(path, status)
    print("Recorded integration: " + ", ".join(result["categories"]))


def approve(category, note):
    """Record an actual user decision; a technical pass is never approval."""
    if not note.strip():
        raise ValueError("Record the user's explicit approval of the affected appearance")
    require_prepared()
    selected = affected(category)
    identity = implementation_identity()
    signatures = category_signatures()
    updates = []
    for key in selected:
        value = standard(key)
        path = LAB / "out" / key / "render.json"
        if not path.is_file():
            raise ValueError("Render the affected category before approval: " + key)
        result = read(path)
        if result.get("implementation_identity") != identity or result.get("recipe") != value["recipe"]:
            raise ValueError("Preview is stale: " + key)
        if result.get("input_signature") != signatures[key]:
            raise ValueError("Preview input selection is stale: " + key)
        expected = {(case, hour, zoom) for case in value["recipe"]["cases"]
                    for hour in value["recipe"]["hours"] for zoom in value["recipe"]["zooms"]}
        actual = {(r["case"], r["hour"], r["zoom"]) for r in result["outputs"]}
        if actual != expected or len(actual) != len(result["outputs"]):
            raise ValueError("Approval requires the complete category preview: " + key)
        revision = value["approved_revision"] + 1
        folder = LAB / "references" / key / f"r{revision}"
        if folder.exists():
            raise ValueError("Reference destination already exists: " + relative(folder))
        for entry in result["outputs"]:
            if checksum(local(entry["image"])) != entry["sha256"]:
                raise ValueError("Preview image changed: " + key)
        updates.append((key, value, result, revision, folder))
    # Validate the entire affected set before writing any approved reference.
    # Prior references remain intact if an I/O error interrupts the new set.
    for key, value, result, revision, folder in updates:
        folder.mkdir(parents=True)
        references = []
        for entry in result["outputs"]:
            target = folder / Path(entry["image"]).name
            shutil.copyfile(local(entry["image"]), target)
            references.append(dict(entry, image=relative(target)))
        value.update(revision=revision, approved_revision=revision,
                     approval={"user_statement": note, "affected_categories": selected,
                               "implementation_identity": identity},
                     references={"d3d11": references}, candidate=None)
        value["reviewed_inputs"] = {"revision": revision, "signature": signatures[key],
                                    "basis": "explicit_user_approval"}
        write(standard_path(key), value)
    print("Approved " + ", ".join(f"{key} r{revision}" for key, _, _, revision, _ in updates))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest="command", required=True)
    commands.add_parser("list")
    commands.add_parser("prepare", help="Refresh shader bindings and changed category assets without staging")
    gallery_parser = commands.add_parser("gallery")
    gallery_parser.add_argument("category", nargs="?", choices=list(catalog()))
    tests = commands.add_parser("test")
    tests.add_argument("category", nargs="?", choices=list(catalog()))
    tests.add_argument("--backends", choices=("metal","both"), help="Check GPU bindings and compile production Metal shaders; not full scene parity")
    commands.add_parser("build", help="Build a candidate and preview tools without staging")
    check = commands.add_parser("check")
    check.add_argument("--complete", action="store_true")
    integration = commands.add_parser("integration")
    integration.add_argument("action", choices=["pending", "verify", "record"])
    integration.add_argument("category", nargs="?", choices=list(catalog()))
    integration.add_argument("--build", action="store_true")
    integration.add_argument("--game-check", help="Describe the actual live Civ III test and result")
    approval = commands.add_parser("approve")
    approval.add_argument("category", choices=list(catalog()))
    approval.add_argument("--user-approval", required=True, help="Quote the user's actual approval of the complete affected preview set")
    for name in ("show", "affected", "lab", "compare", "baseline"):
        parser = commands.add_parser(name)
        parser.add_argument("category", choices=list(catalog()))
        if name in ("lab", "compare"):
            parser.add_argument("--backend", choices=("d3d11", "metal"), default="d3d11",
                                help="Metal currently supports the grassland detail pilot only")
        if name == "lab":
            parser.add_argument("--case", help="One fixture case listed by show CATEGORY")
            parser.add_argument("--affected", action="store_true", help="Include dependents even with a focused --case run; complete runs do this automatically")
    args = p.parse_args()
    try:
        if args.command == "list":
            for key in catalog():
                value = standard(key)
                print(f'{key:18} approved r{value["approved_revision"]} | {value["title"]}')
        elif args.command == "show":
            describe(args.category)
        elif args.command == "gallery":
            gallery(args.category)
        elif args.command == "test":
            prepare_sources()
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
        elif args.command in ("lab", "baseline"):
            if getattr(args, "backend", None) == "metal":
                from Renderer.lab.backends import natural_scene
                natural_scene.request(args.category, args.case, args.affected)
                prepare_sources()
                natural_scene.render(args.category, args.case)
                return 0
            if args.command == "lab":
                prepare_sources()
            complete_lab = args.command == "lab" and not args.case
            selected = affected(args.category) if complete_lab or getattr(args, "affected", False) else [args.category]
            for key in selected:
                selected_case = getattr(args, "case", None) if key == args.category else None
                render(key, baseline=args.command == "baseline", selected_case=selected_case)
        elif args.command == "compare":
            if args.backend == "metal":
                from Renderer.lab.backends import natural_scene
                natural_scene.compare(args.category)
                return 0
            for key in affected(args.category):
                compare(key)
        elif args.command == "check":
            print(f"PASS {validate(complete=args.complete)} category definitions and current references")
        elif args.command == "integration":
            if args.action == "pending":
                changes = pending()
                print(json.dumps(changes, indent=2) if changes else "No approved revisions await integration.")
            elif not args.category:
                raise ValueError("Select the category to verify or record")
            elif args.action == "verify":
                verify_integration(args.category, build=args.build)
            elif not args.game_check:
                raise ValueError("--game-check must describe the actual Civ III check")
            else:
                record_integration(args.category, args.game_check)
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

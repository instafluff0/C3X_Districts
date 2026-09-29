"""Prepare one Civ VI SettlerBuilder as an isolated Lab unit."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from Renderer.lab.platform import ROOT, native_command_result
from Renderer.tools.asset_compiler.build_unit_animation_runtime import build as build_runtime
from Renderer.tools.asset_compiler.unit_family_action_validator import validate_family_actions
from Renderer.tools.asset_compiler.unit_family_asset_importer import compile_unit_families
from Renderer.tools.asset_compiler.unit_family_pose_cache_builder import build_family_pose_caches
from Renderer.tools.asset_compiler.unit_member_resolver import ASSETS_ROOT


HERE = Path(__file__).resolve().parent
OUT = ROOT / "Renderer/lab/out/units/settler-carrier"
PACK = ROOT / "Renderer/packs/UnitSettlerCarrierLab"
RUNTIME = ROOT / "Renderer/packs/UnitSettlerCarrierRuntime"
STRATEGY = HERE / "settler_carrier_strategy.json"
ACTIONS = ("idle", "move", "fidget", "fortify", "capture", "build", "death")


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def prepare() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    report = compile_unit_families(ASSETS_ROOT, STRATEGY, PACK, OUT / "import-report.json")
    if report["outputs"]["units"] != 1 or report["outputs"]["components"] != 4:
        raise ValueError("Expected one carrier with backpack, armor, body and head")
    if any(not (PACK / "animations/unit/settler" / (action + ".c3anim")).is_file()
           for action in ACTIONS):
        result = native_command_result("Renderer/lab/studies/units",
                                       "call CONVERT_SETTLER_CARRIER_ANIMATIONS.bat",
                                       timeout_seconds=120)
        if result["status"] != "pass":
            raise ValueError("Carrier clip conversion failed")
        report = compile_unit_families(ASSETS_ROOT, STRATEGY, PACK, OUT / "import-report.json")
    if report["outputs"]["converted_actions"] != len(ACTIONS):
        raise ValueError("The carrier's seven clips have not all been converted")
    recipe = json.loads((PACK / "units/settler_recipe.json").read_text(encoding="utf-8"))
    if recipe["member"]["count"] != 1 or recipe["civ3_ids"] != ["PRTO_Lab_SettlerCarrier"]:
        raise ValueError("Lab recipe must contain exactly one private carrier")
    action_report = validate_family_actions(PACK)
    _write(OUT / "action-validation.json", action_report)
    pose_report = build_family_pose_caches(PACK, OUT / "pose-validation.json")
    runtime = build_runtime([PACK], RUNTIME)
    if set(runtime["units"]) != {"unit/settler"}:
        raise ValueError("Lab runtime contains a non-carrier unit")
    receipt = {
        "status": "lab_only",
        "unit_key": "PRTO_Lab_SettlerCarrier",
        "member_index": 2,
        "source_member_count": 2,
        "actions": list(ACTIONS),
        "driver_matching_tracks": {action: record["driver_matched_tracks"]
            for action, record in action_report["units"]["unit/settler"]["actions"].items()},
        "pose_caches": pose_report["unique_component_pose_caches"],
        "runtime_pack": str(RUNTIME.relative_to(ROOT)),
        "backpack_socket": "Pelvis (inferred; preview visually checked)",
    }
    _write(OUT / "receipt.json", receipt)
    return receipt


def preview_source() -> None:
    source = (ROOT / "Renderer/lab/native_preview.cpp").read_text(encoding="utf-8")
    for name in ("c3x_renderer_api.h", "biq_preview.cpp"):
        source = source.replace("../native/" + name,
                                os.path.relpath(ROOT / "Renderer/native" / name, OUT))

    def replace(old: str, new: str) -> None:
        nonlocal source
        if source.count(old) != 1:
            raise ValueError("Native unit preview changed near " + old[:60])
        source = source.replace(old, new)

    replace("int main(int argc, char** argv) {",
            'int main(int argc, char** argv) {\n'
            '    SetEnvironmentVariableA("C3X_RENDERER_UNIT_PACK", "UnitSettlerCarrierRuntime");')
    replace('sprintf_s(unit.unit_key, "PRTO_%s", keys[index%6]);',
            'strcpy_s(unit.unit_key, "PRTO_Lab_SettlerCarrier");')
    replace("unit.action = 2;",
            "unit.action = index == 0 ? 1 : index == 1 ? 2 : index == 2 ? 8 : "
            "index == 3 ? 10 : index == 4 ? 12 : 6;")
    replace("unit.action_cursor = cursor[0] ? std::atoi(cursor) : 7;",
            "unit.action_cursor = index == 0 ? 0 : (cursor[0] ? std::atoi(cursor) : 7);")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "native_preview.cpp").write_text(source, encoding="utf-8")
    setup = (ROOT / "Renderer/lab/build_native_preview.bat").read_text(encoding="utf-8").split("cl /nologo")[0]
    (OUT / "build_preview.bat").write_text(
        setup + "cl /nologo /std:c++17 /EHsc /O2 /W4 /WX native_preview.cpp "
        "/Fo:native_preview.obj /Fe:native_preview.exe /link /LARGEADDRESSAWARE "
        "gdi32.lib user32.lib\nexit /b %errorlevel%\n", encoding="utf-8")


def render() -> dict:
    from Renderer.lab.categories.terrain.floodplains.source_preview import read_bmp
    from Renderer.preview.render_textured_patch import write_png
    from Renderer.renderer import native_inputs, native_render

    prepare()
    preview_source()
    result = native_command_result("Renderer/lab/out/units/settler-carrier",
                                   "call build_preview.bat", timeout_seconds=120)
    if result["status"] != "pass":
        raise ValueError("Carrier Lab preview build failed")
    staged = ROOT / "Renderer/bin/C3XRenderer.dll"
    staged_hash = hashlib.sha256(staged.read_bytes()).hexdigest() if staged.is_file() else None
    inputs = native_inputs()
    result = native_command_result("Renderer/native", "call BUILD.bat candidate-compile",
                                   timeout_seconds=180)
    if result["status"] != "pass" or native_inputs() != inputs:
        raise ValueError("Current native Lab candidate build failed or its inputs changed")
    if (hashlib.sha256(staged.read_bytes()).hexdigest() if staged.is_file() else None) != staged_hash:
        raise ValueError("Lab build changed the staged renderer")
    outputs = [native_render("units", case, 12, zoom, OUT / case,
                             candidate=ROOT / "Renderer/native/build/candidate/C3XRenderer.dll",
                             preview=OUT / "native_preview.exe")
               for case, zoom in (("detail", 128), ("gameplay", 128), ("detail", 192))]
    for output in outputs:
        path = ROOT / output["image"]
        write_png(read_bmp(path), path.with_suffix(".png"))
    receipt = {"status": "lab_only", "outputs": outputs}
    _write(OUT / "renders.json", receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render", action="store_true", help="build and render the D3D11 Lab fixture")
    args = parser.parse_args()
    result = render() if args.render else prepare()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

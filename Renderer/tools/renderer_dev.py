#!/usr/bin/env python3
"""One-command development workflows for the C3X renderer."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath
from typing import Any


RENDERER_ROOT = Path(__file__).resolve().parents[1]
C3X_ROOT = RENDERER_ROOT.parent
DEFAULT_VM = "Windows 11"
DEFAULT_WINDOWS_ROOT = PureWindowsPath(
    r"Y:\fun\Civilization III Complete\Conquests\C3X_Districts"
)
DEFAULT_WINDOWS_LIVE_TARGET = PureWindowsPath(
    r"\\Mac\Home\fun\Civilization III Complete\Conquests\C3X_Districts"
)
DEFAULT_CIV3_CONQUESTS = PureWindowsPath(
    r"C:\Program Files (x86)\GOG Galaxy\Games\Civilization III Complete\Conquests"
)
INJECTED_VERIFY_LINK = "C3X_Shared_Verify"
LIVE_MOD_DIR = "C3X_Districts"
DEFAULT_REPORT = RENDERER_ROOT / "verification" / "latest_iteration.json"

sys.path.insert(0, str(RENDERER_ROOT / "tools"))
import check_project_state


def command_result(command: list[str], cwd: Path = C3X_ROOT) -> dict[str, Any]:
    started = datetime.now(timezone.utc)
    result = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    output = result.stdout or ""
    if output:
        print(output, end="" if output.endswith("\n") else "\n")
    return {
        "command": command,
        "cwd": str(cwd),
        "status": "pass" if result.returncode == 0 else "fail",
        "returncode": result.returncode,
        "started_utc": started.isoformat(),
        "output_tail": output[-4000:],
    }


def changed_injected_sources() -> bool:
    result = subprocess.run(
        ["git", "status", "--porcelain", "--", "C3X.h", "injected_code.c"],
        cwd=C3X_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    # If Git is unavailable, fail safe and compile rather than silently skipping.
    return result.returncode != 0 or bool(result.stdout.strip())


def windows_root() -> PureWindowsPath:
    configured = os.environ.get("C3X_RENDERER_WINDOWS_ROOT")
    return PureWindowsPath(configured) if configured else DEFAULT_WINDOWS_ROOT


def windows_live_target() -> PureWindowsPath:
    configured = os.environ.get("C3X_RENDERER_WINDOWS_LIVE_TARGET")
    return PureWindowsPath(configured) if configured else DEFAULT_WINDOWS_LIVE_TARGET


def windows_command_result(relative_cwd: str, command: str) -> dict[str, Any]:
    vm = os.environ.get("C3X_RENDERER_VM", DEFAULT_VM)
    cwd = windows_root() / PureWindowsPath(relative_cwd)
    remote = f'cd /d "{cwd}" && {command}'
    started = datetime.now(timezone.utc)
    result = subprocess.run(
        ["prlctl", "exec", vm, "cmd", "/d", "/s", "/c", remote],
        cwd=C3X_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    output = result.stdout or ""
    if output:
        print(output, end="" if output.endswith("\n") else "\n")
    return {
        "command": command,
        "cwd": str(cwd),
        "host": vm,
        "status": "pass" if result.returncode == 0 else "fail",
        "returncode": result.returncode,
        "started_utc": started.isoformat(),
        "output_tail": output[-4000:],
    }


def native_command_result(relative_cwd: str, command: str) -> dict[str, Any]:
    if os.name == "nt":
        return command_result(
            ["cmd", "/d", "/s", "/c", command], C3X_ROOT / relative_cwd
        )
    return windows_command_result(relative_cwd, command)


def integrated_terrain_preview_results() -> list[dict[str, Any]]:
    """Render the approved river scene at both game zooms and lighting phases."""
    cases = (
        ("integrated_terrain_preview_near", "native_i13a_noon.bmp", 900, 520, 42, 32, 128, 12),
        ("integrated_terrain_preview_far", "native_i13a_sunset_zoom2.bmp", 1200, 700, 42, 32, 64, 18),
    )
    results: list[dict[str, Any]] = []
    for name, output, width, height, center_x, center_y, tile_width, hour in cases:
        command = (
            r'build\biq_preview.exe build\candidate\C3XRenderer.dll ..\.. '
            r'..\default.custom_rendering.txt '
            r'..\preview\out\terrain_lab\test_biq_l13_rivers_192.csv '
            f'..\\preview\\out\\{output} '
            f'{width} {height} {center_x} {center_y} {tile_width} {hour}'
        )
        result = native_command_result("Renderer/native", command)
        result["name"] = name
        result["output"] = str(RENDERER_ROOT / "preview" / "out" / output)
        results.append(result)
        if result["status"] != "pass":
            break
    return results


def approved_terrain_smoke_result() -> dict[str, Any]:
    """Replay the complete approved L9-L19 production payload in its own VM command.

    Parallels can stop relaying output from one long remote command after the
    compile/synthetic-smoke phase. Keeping this licensed-payload replay in a
    second command makes its pass/fail result explicit in every Integration and
    Full report while preserving portable builds when ignored local packs are
    unavailable.
    """
    required = (
        "TerrainNormalized/manifest.json",
        "VegetationNormalized/vegetation_runtime.bin",
        "DecalsNormalized/manifest.json",
        "TerrainElementsNormalized/manifest.json",
        "ShoreNormalized/shore_runtime.bin",
        "RouteStylesNormalized/manifest.json",
        "RouteDoodadsNormalized/bridge_runtime.bin",
        "ResourceNormalized/resource_runtime.bin",
        "CityComponentsNormalized/city_runtime.bin",
        "CityAdjunctsNormalized/wall_runtime.bin",
        "ImprovementsNormalized/mine_runtime.bin",
        "ImprovementsNormalized/farm_runtime.bin",
    )
    missing = [entry for entry in required if not (RENDERER_ROOT / "packs" / entry).is_file()]
    if missing:
        detail = "ignored normalized L9-L19 payloads are unavailable: " + ", ".join(missing)
        print(f"SKIP approved_terrain_integration: {detail}")
        return {"name": "approved_terrain_integration", "status": "skip", "reason": detail}
    command = (
        r'build\native_smoke.exe build\candidate\C3XRenderer.dll '
        r'--definitions ..\.. ..\..\Renderer\default.custom_rendering.txt'
    )
    result = native_command_result("Renderer/native", command)
    if result["status"] == "fail":
        # Parallels occasionally returns 255 without starting or relaying this
        # otherwise deterministic D3D process immediately after a compiler VM
        # command. Retry once in a fresh guest command; a real renderer failure
        # remains a failure on the second run and is never hidden.
        first_returncode = result.get("returncode")
        print(f"RETRY approved_terrain_integration after VM return code {first_returncode}.")
        result = native_command_result("Renderer/native", command)
        result["retry_after_returncode"] = first_returncode
    result["name"] = "approved_terrain_integration"
    if result["status"] == "pass" and \
       "PASS approved_terrain_integration" not in result.get("output_tail", ""):
        result["status"] = "fail"
        result["detail"] = "approved production smoke did not report its completion marker"
    return result


def injected_compile_result() -> dict[str, Any]:
    if os.name == "nt" and (C3X_ROOT.parent / "Civ3Conquests.exe").is_file():
        return command_result(
            ["cmd", "/d", "/s", "/c", "call TEST_INJECTED_CODE_COMPILE.bat"]
        )
    conquests = PureWindowsPath(
        os.environ.get("C3X_RENDERER_CIV3_CONQUESTS", str(DEFAULT_CIV3_CONQUESTS))
    )
    link = conquests / INJECTED_VERIFY_LINK
    target = windows_root()
    command = (
        f'mklink /D "{link}" "{target}" >nul 2>nul & '
        f'cd /d "{link}" && call TEST_INJECTED_CODE_COMPILE.bat'
    )
    if os.name == "nt":
        return command_result(["cmd", "/d", "/s", "/c", command])
    return windows_command_result("", command)


def live_checkout_result() -> dict[str, Any]:
    """Verify that the interactive C3X root is the shared Git checkout."""
    conquests = PureWindowsPath(
        os.environ.get("C3X_RENDERER_CIV3_CONQUESTS", str(DEFAULT_CIV3_CONQUESTS))
    )
    live = conquests / LIVE_MOD_DIR
    target = windows_root()
    link_target = windows_live_target()
    expected_root = link_target.as_posix()
    command = (
        f'git -C "{live}" rev-parse --show-toplevel 2>nul | '
        f'find /i "{expected_root}" >nul & '
        f'if errorlevel 1 (echo Live C3X root does not resolve to the shared Git checkout: "{live}" 1>&2 & exit /b 1) '
        "else (echo PASS live_checkout_link: the installed C3X_Districts path is the shared Git checkout. Launch INSTALL.bat through cmd.exe using its local C-drive path before testing; direct Git Bash execution resolves the link to UNC.)"
    )
    if os.name == "nt":
        result = command_result(["cmd", "/d", "/s", "/c", command])
    else:
        result = windows_command_result("", command)
    result["name"] = "live_checkout_link"
    result["live_mod_directory"] = str(live)
    result["source_checkout"] = str(target)
    return result


def quote_windows_arg(value: str) -> str:
    if value and not any(character in value for character in ' \t"'):
        return value
    return '"' + value.replace('"', '\\"') + '"'


def python_executable() -> str:
    configured = os.environ.get("C3X_RENDERER_PYTHON")
    candidates = [
        configured,
        str(
            Path.home()
            / ".cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3"
        ),
        shutil.which("python3.12"),
        shutil.which("python3.11"),
        shutil.which("python3.10"),
        sys.executable if sys.version_info >= (3, 10) else None,
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return str(candidate)
    raise RuntimeError(
        "Renderer workflows require Python 3.10+; set C3X_RENDERER_PYTHON to a compatible interpreter."
    )


def validate_state() -> dict[str, Any]:
    errors = check_project_state.validate_project_state()
    if errors:
        for error in errors:
            print(f"error: {error}", file=sys.stderr)
        return {"name": "project_state", "status": "fail", "errors": errors}
    status = json.loads(check_project_state.DEFAULT_STATUS.read_text(encoding="utf-8"))
    step = status["next_step"]
    print(f"PASS project_state: next step {step['id']} - {step['title']}")
    return {"name": "project_state", "status": "pass", "next_step": step["id"]}


def run_workflow(name: str, with_injected: bool, report_path: Path) -> int:
    results: list[dict[str, Any]] = [validate_state()]
    if results[-1]["status"] == "fail":
        return write_report(name, results, report_path)

    if name == "state":
        return write_report(name, results, report_path)

    try:
        python = python_executable()
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        results.append({"name": "python_runtime", "status": "fail", "detail": str(exc)})
        return write_report(name, results, report_path)

    if name == "full":
        command = [
            python,
            str(RENDERER_ROOT / "tools" / "verify_project.py"),
            "--portable-only",
            "--source-only",
        ]
        results.append(command_result(command))
        if results[-1]["status"] == "pass":
            results.append(native_command_result("Renderer/native", "call BUILD.bat portable"))
        if results[-1]["status"] == "pass":
            results.append(approved_terrain_smoke_result())
        if results[-1]["status"] == "pass":
            results.extend(integrated_terrain_preview_results())
        if results[-1]["status"] == "pass":
            results.append(injected_compile_result())
        if results[-1]["status"] == "pass":
            results.append(live_checkout_result())
        return write_report(name, results, report_path)

    if name == "lab":
        results.append(command_result([
            python, "-m", "unittest",
            "Renderer.tools.test_project_state",
            "Renderer.tools.asset_compiler.test_generic_decal_compiler",
            "Renderer.tools.asset_compiler.test_clutter_blp_extractor",
            "Renderer.tools.asset_compiler.test_route_doodad_importer",
            "Renderer.tools.asset_compiler.test_build_route_bridge_runtime",
            "Renderer.tools.asset_compiler.test_city_asset_importer",
            "Renderer.tools.asset_compiler.test_improvement_asset_importer",
            "Renderer.tools.asset_compiler.test_unit_family_asset_importer",
            "Renderer.tools.asset_compiler.test_worker_builder_action_compiler",
            "Renderer.tools.asset_compiler.test_effect_graph_compiler",
            "Renderer.tools.asset_compiler.test_pack_loader_abi",
            "Renderer.tools.asset_compiler.test_tile_fit_calibrator",
            "Renderer.tools.test_state_provenance_compiler",
            "Renderer.tools.test_visual_qa",
            "Renderer.preview.test_render_route_doodad_sheet",
            "Renderer.preview.test_render_city_day_night_sheet",
            "Renderer.preview.test_render_improvement_sheet",
            "Renderer.preview.test_render_unit_family_sheet",
            "Renderer.preview.test_render_compound_tile_fit_sheet",
            "Renderer.terrain_lab.test_continuous_surface_contract",
            "Renderer.terrain_lab.test_canonical_reference_contract",
            "Renderer.terrain_lab.test_build_l14_road_scenario",
            "Renderer.terrain_lab.test_l14_road_contract",
            "Renderer.terrain_lab.test_build_l15_railroad_scenario",
            "Renderer.terrain_lab.test_l15_railroad_contract",
            "Renderer.terrain_lab.test_build_l16_resource_scenario",
            "Renderer.terrain_lab.test_l16_resource_contract",
            "Renderer.terrain_lab.test_build_l17_city_scenario",
            "Renderer.terrain_lab.test_l17_city_contract",
            "Renderer.terrain_lab.test_build_l18_mine_scenario",
            "Renderer.terrain_lab.test_l18_mine_contract",
            "Renderer.terrain_lab.test_build_l19_farm_scenario",
            "Renderer.terrain_lab.test_l19_farm_contract",
            "Renderer.terrain_lab.test_build_l19a_tile_object_scenario",
            "Renderer.terrain_lab.test_l19a_tile_object_contract",
            "Renderer.terrain_lab.test_build_l19b_infrastructure_scenario",
            "Renderer.terrain_lab.test_l19b_infrastructure_contract",
            "Renderer.terrain_lab.test_build_l20_unit_scenario",
            "Renderer.terrain_lab.test_l20_unit_contract",
            "Renderer.terrain_lab.test_l21_complete_scene_contract",
        ]))
        if results[-1]["status"] == "pass":
            results.append(command_result(
                ["node", str(RENDERER_ROOT / "tools" / "test_export_biq_terrain_scene.js")]
            ))
        if results[-1]["status"] == "pass" and results[0]["next_step"] in ("L20", "L21"):
            results.append(command_result([
                python,
                str(RENDERER_ROOT / "tools" / "asset_compiler" /
                    "build_l20_unit_runtime.py"),
                "--pack", str(RENDERER_ROOT / "packs" / "UnitFamilyLab"),
            ]))
        if results[-1]["status"] == "pass" and results[0]["next_step"] in ("L20", "L21"):
            results.append(command_result([
                python,
                str(RENDERER_ROOT / "tools" / "asset_compiler" /
                    "build_l20_compound_unit_runtime.py"),
                "--pack", str(RENDERER_ROOT / "packs" / "CompoundUnitLab"),
            ]))
        if results[-1]["status"] == "pass" and results[0]["next_step"] in ("L20", "L21"):
            results.append(command_result([
                python,
                str(RENDERER_ROOT / "terrain_lab" / "build_l20_unit_scenario.py"),
                str(RENDERER_ROOT / "preview" / "out" / "terrain_lab" /
                    "test_biq_l13_rivers_192.csv"),
                str(RENDERER_ROOT / "terrain_lab" / "fixtures" / "l17_cities_192.csv"),
                str(RENDERER_ROOT / "terrain_lab" / "fixtures" /
                    "l19a_tile_objects_192.csv"),
                str(RENDERER_ROOT / "terrain_lab" / "fixtures" /
                    "l19b_infrastructure_192.csv"),
                str(RENDERER_ROOT / "terrain_lab" / "fixtures" / "l20_units_192.csv"),
            ]))
        if results[-1]["status"] == "pass":
            lab_script_name = f"RUN_{results[0]['next_step']}.bat"
            lab_script = RENDERER_ROOT / "terrain_lab" / lab_script_name
            if lab_script.is_file():
                results.append(native_command_result(
                    "Renderer/terrain_lab", f"call {lab_script_name}"
                ))
            else:
                detail = f"missing Renderer/terrain_lab/{lab_script_name}"
                print(f"error: {detail}", file=sys.stderr)
                results.append({
                    "name": "lab_gate_script",
                    "status": "fail",
                    "detail": detail,
                })
        return write_report(name, results, report_path)

    results.append(command_result([
        python, "-m", "unittest",
        "Renderer.tools.test_project_state",
        "Renderer.tools.test_verification",
        "Renderer.tools.test_state_provenance_compiler",
        "Renderer.tools.asset_compiler.test_pack_loader_abi",
        "Renderer.native.test_native_bridge_contract",
    ]))
    if results[-1]["status"] == "pass":
        results.append(native_command_result("Renderer/native", "call BUILD.bat portable"))
    if results[-1]["status"] == "pass":
        results.append(approved_terrain_smoke_result())
    if results[-1]["status"] == "pass":
        results.extend(integrated_terrain_preview_results())
    should_compile_injected = with_injected or changed_injected_sources()
    if results[-1]["status"] == "pass" and should_compile_injected:
        results.append(injected_compile_result())
    elif results[-1]["status"] == "pass":
        print("SKIP injected_compile: C3X.h and injected_code.c are unchanged (use --with-injected to force).")
        results.append({
            "name": "injected_compile",
            "status": "skip",
            "reason": "C3X.h and injected_code.c are unchanged",
        })
    if results[-1]["status"] in {"pass", "skip"}:
        results.append(live_checkout_result())
    return write_report(name, results, report_path)


def write_report(name: str, results: list[dict[str, Any]], path: Path) -> int:
    passed = all(item["status"] in {"pass", "skip"} for item in results)
    report = {
        "schema": "c3x.renderer_iteration_report.v0",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "workflow": name,
        "results": results,
        "passed": passed,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Report: {path}")
    return 0 if passed else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one C3X renderer development workflow")
    parser.add_argument("workflow", choices=("state", "lab", "integration", "full"))
    parser.add_argument("--with-injected", action="store_true")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    return run_workflow(args.workflow, args.with_injected, args.report)


if __name__ == "__main__":
    raise SystemExit(main())

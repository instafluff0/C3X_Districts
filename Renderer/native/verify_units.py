"""Verify DLL unit rendering and native capture offscreen; never launch Civ III."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from Renderer.native.test_unit_bridge import prepare
from Renderer.tools.renderer_dev import windows_command_result, command_result, python_executable


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-build", action="store_true")
    args = parser.parse_args()
    output = ROOT / "Renderer/lab/out/verification/animation"
    output.mkdir(parents=True, exist_ok=True)
    checks=command_result([python_executable(), "-m", "unittest",
        "Renderer.native.test_unit_shadow", "Renderer.native.test_unit_input_guard",
        "Renderer.native.test_unit_bridge", "-q"])
    checks.pop("cwd", None)
    (output/"unit-presentation-contracts.json").write_text(json.dumps(checks, indent=2)+"\n")
    if checks["status"] != "pass":return 1
    prepare()
    if not args.skip_build:
        built = windows_command_result("Renderer/native", "call BUILD.bat candidate-compile")
        built.pop("cwd", None)
        (output/"unit-body-build.json").write_text(json.dumps(built, indent=2)+"\n")
        if built["status"] != "pass":
            return 1
    results = []
    for hour in (12, 0):
        name = f"units-h{hour}"
        settings = {
            "C3X_RENDERER_PREVIEW_CUSTOM_DEFINITIONS": "..\\..\\Renderer\\custom.custom_rendering.txt",
            "C3X_RENDERER_VISUAL_PROFILE": "pickup-r1",
            "C3X_RENDERER_PREVIEW_ANIMATION": "1",
            "C3X_RENDERER_PREVIEW_UNITS": "1",
            "C3X_RENDERER_TRACE": "2",
            "C3X_RENDERER_TRACE_FILE": f"..\\lab\\out\\verification\\animation\\{name}.log",
        }
        command = " && ".join(f'set "{key}={value}"' for key, value in settings.items())
        command += (" && build\\biq_preview.exe build\\candidate\\C3XRenderer.dll ..\\.. "
                    "..\\..\\Renderer\\default.custom_rendering.txt ..\\lab\\.local\\verification\\world.csv "
                    f"..\\lab\\out\\verification\\animation\\{name}.bmp 960 640 75 39 128 {hour}")
        result = windows_command_result("Renderer/native", command)
        result.pop("cwd", None)
        text = result.get("output_tail", "")
        required = ("UNIT body matrix drawn=288 status=pass", "UNIT cached anchor translation: pass",
            "UNIT repeated native cursor: pass", "UNIT magenta underlay parity zoom=0", "UNIT magenta underlay parity zoom=1", "UNIT retained terrain unchanged: pass",
            "UNIT post-draw terrain parity: pass", "ANIMATION temporal: pass",
            "ANIMATION scroll parity: pass", "ANIMATION removal parity: pass",
            "UNIT config-off preserves canvas: pass", "UNIT RGB555 clipped zoom=0",
            "UNIT action interruption and held endpoint: pass draws=582",
            "UNIT independent ambient phases and exact repeat: pass",
            "UNIT RGB555 magenta clipped parity zoom=0 status=pass", "UNIT RGB555 magenta clipped parity zoom=1 status=pass",
            "UNIT RGB565 magenta clipped parity zoom=0 status=pass", "UNIT RGB565 magenta clipped parity zoom=1 status=pass",
            "UNIT RGB555 clipped zoom=1", "UNIT RGB565 clipped zoom=0", "UNIT RGB565 clipped zoom=1")
        if any(marker not in text for marker in required) or "FAIL" in text:
            result["status"] = "fail"
        result["case"] = name
        results.append(result)
        (output/"units.json").write_text(json.dumps(results, indent=2)+"\n")
        if result["status"] != "pass":
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

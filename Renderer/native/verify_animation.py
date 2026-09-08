"""Headless Windows resource animation witnesses; never launches Civ III."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from Renderer.tools.renderer_dev import windows_command_result


def main():
    output = ROOT / "Renderer/lab/out/verification/animation"
    output.mkdir(parents=True, exist_ok=True)
    results = []
    for zoom, hour in ((128, 12), (64, 12), (128, 0)):
        name = f"resources-z{zoom}-h{hour}"
        settings = {
            "C3X_RENDERER_PREVIEW_CUSTOM_DEFINITIONS": "..\\..\\Renderer\\custom.custom_rendering.txt",
            "C3X_RENDERER_VISUAL_PROFILE": "pickup-r1",
            "C3X_RENDERER_PREVIEW_ANIMATION": "1",
            "C3X_RENDERER_TRACE": "2",
            "C3X_RENDERER_TRACE_FILE": f"..\\lab\\out\\verification\\animation\\{name}.log",
        }
        command = " && ".join(f'set "{key}={value}"' for key, value in settings.items())
        command += (" && build\\biq_preview.exe build\\candidate\\C3XRenderer.dll ..\\.. "
                    "..\\..\\Renderer\\default.custom_rendering.txt ..\\lab\\.local\\verification\\world.csv "
                    f"..\\lab\\out\\verification\\animation\\{name}.bmp 960 640 75 39 {zoom} {hour}")
        result = windows_command_result("Renderer/native", command)
        result.pop("cwd", None)
        text = result.get("output_tail", "")
        if ("ANIMATION temporal: pass changed_frames=5" not in text or
                "ANIMATION scroll parity: pass" not in text or "ANIMATION removal parity: pass" not in text or "FAIL" in text):
            result["status"] = "fail"
        result["case"] = name
        results.append(result)
        (output / "temporal.json").write_text(json.dumps(results, indent=2)+"\n")
        if result["status"] != "pass":
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Headless Windows pickup witnesses; never installs or launches Civ III.

Requires the full authoritative CSV exported to verification/pickup/world.csv.
The volcano witness is explicitly synthetic and leaves the source BIQ untouched.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from Renderer.tools.renderer_dev import windows_command_result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("replay", "matrix", "gpu", "edits"))
    args = parser.parse_args()
    destination = ROOT / "Renderer/verification/pickup"
    destination.mkdir(parents=True, exist_ok=True)
    results = []
    if args.mode == "gpu":
        cases = [("source-casters-linear-msaa", "Renderer/native/profile_v2", "call VERIFY.bat")]
    else:
        source = destination / "world.csv"
        if not source.is_file():
            raise SystemExit("Export the complete authoritative world.csv before verification")
        synthetic = []
        for line in source.read_text().splitlines():
            fields = line.split(",")
            if fields[:2] == ["70", "50"]:
                fields[2:4] = ["2", "10"]
            synthetic.append(",".join(fields))
        (destination / "synthetic-volcano.csv").write_text("\n".join(synthetic) + "\n")
        scenes = [("prepared-scroll", 75, 39, 128, 12, False, False)] if args.mode == "replay" else [
            (f"coastal-z{zoom}-h{hour}", 85, 38, zoom, hour, False, False)
            for zoom in (128, 64) for hour in (12, 18, 0, 6)
        ] + [
            (f"objects-z{zoom}-h{hour}", 69, 50, zoom, hour, True, False)
            for zoom in (128, 64) for hour in (12, 0)
        ] + [
            (f"synthetic-volcano-z{zoom}-h{hour}", 70, 50, zoom, hour, False, True)
            for zoom in (128, 64) for hour in (12, 0)
        ] + [(f"wrap-z{zoom}", 0, 50, zoom, 12, False, False) for zoom in (128, 64)]
        if args.mode == "edits":
            scenes = [("terrain-edit", 69, 50, 128, 12, False, False),
                      ("volcano-activity-edit", 70, 50, 128, 0, False, True)]
        cases = []
        for name, x, y, zoom, hour, objects, volcano in scenes:
            settings = {
                "C3X_RENDERER_VISUAL_PROFILE": "pickup-r1",
                "C3X_RENDERER_TRACE": "2",
                "C3X_RENDERER_TRACE_FILE": f"..\\verification\\pickup\\{name}.log",
                "C3X_RENDERER_PREVIEW_REPLAY": "1" if args.mode == "replay" else "",
                "C3X_RENDERER_PREVIEW_OBJECTS": "1" if objects else "",
                "C3X_RENDERER_PREVIEW_EDITS": "1" if args.mode == "edits" else "",
                "C3X_RENDERER_PREVIEW_ACTIVE_VOLCANO": "1" if volcano else "",
            }
            command = " && ".join(f'set "{key}={value}"' for key, value in settings.items())
            csv = "synthetic-volcano.csv" if volcano else "world.csv"
            command += (f" && build\\biq_preview.exe build\\candidate\\C3XRenderer.dll ..\\.. "
                        f"..\\..\\Renderer\\default.custom_rendering.txt ..\\verification\\pickup\\{csv} "
                        f"..\\verification\\pickup\\{name}.bmp 960 640 {x} {y} {zoom} {hour}")
            cases.append((name, "Renderer/native", command))
    for name, directory, command in cases:
        print("Checking " + name, flush=True)
        result = windows_command_result(directory, command)
        # Keep reports portable; dispatcher cwd includes a machine-local path.
        result.pop("cwd", None)
        output = result.get("output_tail", "")
        marker = "pickup MSAA4 linear/premultiplied/transfer: pass" if args.mode == "gpu" else "0 fallback, output="
        if marker not in output or "FAIL" in output:
            result["status"] = "fail"
            result["detail"] = "Required completion marker missing or native check reported failure"
        if args.mode in ("replay", "edits"):
            parity = re.search(r"PICKUP (?:edit )?pixel parity: changed=(\d+) error=(\d+) bytes=(\d+)", output)
            if parity:
                changed, error, size = map(int, parity.groups())
                result["pixel_parity"] = {"changed": changed, "error": error, "bytes": size}
            if not parity or changed > size // 4000 or error > size // 100:
                result["status"] = "fail"
                result["detail"] = "Cached/cold parity did not pass the unchanged native thresholds"
        result["name"] = name
        image = destination / (name + ".bmp")
        if image.is_file():
            result["image_sha256"] = hashlib.sha256(image.read_bytes()).hexdigest()
        results.append(result)
        (destination / (args.mode + ".json")).write_text(json.dumps(results, indent=2) + "\n")
        if result["status"] != "pass":
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Build and run the sandbox client against the unchanged BIQ fixture."""
from pathlib import Path
import subprocess
import sys

SANDBOX = Path(__file__).resolve().parent
RENDERER = SANDBOX.parent
sys.path.insert(0, str(RENDERER))
from lab import platform


def main():
    output = SANDBOX / "out" / "test-biq.csv"
    result = subprocess.run([
        "node", str(SANDBOX / "export_biq.js"),
        str(RENDERER / "packs" / "RendererSourceStudies" / "maps" / "test.biq"),
        str(output),
    ], cwd=SANDBOX, check=False)
    if result.returncode:
        return result.returncode
    if "--build" in sys.argv:
        build = platform.native_command_result("Renderer/sandbox", "call build_reference_x64.bat",
                                               timeout_seconds=180)
        if build["status"] != "pass":
            return 1
    run = platform.native_command_result("Renderer/sandbox", "call run_client.bat",
                                         timeout_seconds=180)
    return 0 if run["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())

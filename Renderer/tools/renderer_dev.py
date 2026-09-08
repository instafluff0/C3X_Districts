"""Temporary imports for source probes; category commands live in Renderer/renderer.py.

No milestone state, campaign execution, automatic staging or installation.
Remove this shim after remaining retained source probes use lab.platform directly.
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from Renderer.lab.platform import (native_command_result, command_result,
    windows_root, python_executable, changed_injected_sources, injected_compile_result)

windows_command_result = native_command_result
windows_live_target = windows_root

if __name__ == "__main__":
    raise SystemExit("Retired workflow. Use python3 Renderer/renderer.py --help")

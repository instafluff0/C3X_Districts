"""Compatibility entry point for the shared tool installer."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from Renderer.lab.backends.bootstrap_tools import main
if __name__ == "__main__":
    raise SystemExit(main())

"""Migration import; current city source tooling lives in Renderer/lab/shared."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from Renderer.lab.shared.cities.prepare_materials import *

if __name__ == "__main__":
    main()

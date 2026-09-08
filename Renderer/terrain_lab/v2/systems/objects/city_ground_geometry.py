"""Migration import; current reusable city code lives in Renderer/lab/shared."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[5]))
from Renderer.lab.shared.cities.clipping import *

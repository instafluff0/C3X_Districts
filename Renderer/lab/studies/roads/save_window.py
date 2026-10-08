"""Render a window of a Civ III save through the production renderer.

Reproduces in-game route, bridge and tunnel layouts in the Lab without a game
run. save_window.js cuts the 32x32 scene (terrain, rivers, overlay bits and
cities); this script renders it in the infrastructure category with the
scene's own overlays (C3X_LAB_TILE_OVERLAYS) and cities.

    node Renderer/lab/studies/roads/save_window.js <save.SAV> scene.csv cities.json 113 75
    python3 Renderer/lab/studies/roads/save_window.py scene.csv cities.json Renderer/lab/out/infrastructure/save-window 16,16 128,256

The view center is in Lab coordinates (the save tile given to the cutter is
16,16). Every view needs a railroad and a mine in sight, as the category's
ownership witness requires.
"""
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
import Renderer.renderer as renderer  # noqa: E402


def main(argv):
    scene_csv, cities_json, out = Path(argv[0]), Path(argv[1]), Path(argv[2])
    cx, cy = (int(value) for value in argv[3].split(","))
    zooms = [int(value) for value in argv[4].split(",")]
    cities = json.loads(cities_json.read_text())

    def scene(category, case, destination, *, world_size=32):
        shutil.copyfile(scene_csv, destination)

    renderer.scene = scene
    # "dx,dy,culture,era,size,capital,walled" from the view center.
    spec = ";".join(f"{c['x'] - cx},{c['y'] - cy},0,2,{0 if (c.get('pop') or 0) < 7 else 1},0,0" for c in cities)
    renderer.prepare_sources(["infrastructure"])
    renderer.ensure_candidate(["infrastructure"])
    renderer.ensure_preview_tool()
    for zoom in zooms:
        print(renderer.native_render("infrastructure", "save-window", 12, zoom, out / f"z{zoom}_{cx}_{cy}",
                                     center=(cx, cy), extra_env={"C3X_LAB_TILE_OVERLAYS": "1",
                                                                 "C3X_LAB_TILE_CITIES": spec}))


if __name__ == "__main__":
    main(sys.argv[1:])

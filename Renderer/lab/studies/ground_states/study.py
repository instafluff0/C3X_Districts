"""Ground states and site sizes on the 1498 AD save, through the production renderer.

Cuts the save around the big volcano (save tile 96,90), whose land neighbours
take the eruption pollution and whose surroundings hold Osaka's real pollution,
adds craters, city ruins, a goody hut and a barbarian camp on chosen tiles
(their routes removed, as bombardment and eruptions remove them), and renders
the same views with the production site pack ("before") and a candidate pack
("after", C3X_RENDERER_SITE_PACK).

    python3 Renderer/lab/studies/ground_states/study.py [--pack TileSitesLab] [--build]
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
import Renderer.renderer as renderer  # noqa: E402

SAVE = ROOT / "Renderer/.cache/composition-integration-step/input-1498AD.SAV"
OUT = ROOT / "Renderer/lab/out/ground_states"
CENTER = (97, 91)  # save tile drawn at Lab (16, 16)
POLLUTION = [(95, 91), (97, 91), (96, 92)]        # eruption damage beside the volcano
CRATERS = [(94, 94), (93, 95), (95, 95)]
RUINS = [(93, 93)]
HUT, CAMP = (98, 92), (100, 94)


def lab(tile):
    return tile[0] - CENTER[0] + 16, tile[1] - CENTER[1] + 16


def cut(folder: Path):
    folder.mkdir(parents=True, exist_ok=True)
    scene, cities = folder / "scene.csv", folder / "cities.json"
    subprocess.run(["node", "Renderer/lab/studies/roads/save_window.js", str(SAVE), str(scene), str(cities),
                    str(CENTER[0]), str(CENTER[1])], cwd=ROOT, check=True, capture_output=True)
    rows = scene.read_text().splitlines()
    edits = {lab(t): 0x40 for t in POLLUTION}
    edits.update({lab(t): 0x100 for t in CRATERS})
    edits[lab(HUT)], edits[lab(CAMP)] = 0x20, 0x80
    out = [rows[0]]
    for row in rows[1:]:
        x, y, base, real, bonus, overlays, river = (int(v) for v in row.split(","))
        if (x, y) in edits:
            overlays = (overlays & ~0x1f) | edits[(x, y)]   # routes and improvements removed
        out.append(f"{x},{y},{base},{real},{bonus},{overlays},{river}")
    scene.write_text("\n".join(out) + "\n")
    return scene, json.loads(cities.read_text())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", default="TileSitesLab")
    parser.add_argument("--build", action="store_true", help="rebuild the candidate pack first")
    parser.add_argument("--labels", default="before,after", help="which of before/after to render")
    parser.add_argument("--views", default="128:16:16,256:14:18,256:18:16",
                        help="zoom:lab_x:lab_y views (Lab 16,16 is save 97,91)")
    args = parser.parse_args(argv)
    if args.build:
        from Renderer.tools.asset_compiler.build_site_runtime import build
        build(ROOT / "Renderer/packs" / args.pack)
    scene_csv, cities = cut(OUT / "scene")
    renderer.scene = lambda category, case, destination, *, world_size=32: shutil.copyfile(scene_csv, destination)
    renderer.prepare_sources(["infrastructure"])
    renderer.ensure_candidate(["infrastructure"])
    renderer.ensure_preview_tool()
    for view in args.views.split(","):
        zoom, cx, cy = (int(v) for v in view.split(":"))
        # The tile lists are relative to the view centre.
        city_spec = ";".join(f"{c['x'] - cx},{c['y'] - cy},0,2,{0 if (c.get('pop') or 0) < 7 else 1},0,0" for c in cities)
        ruin_spec = ";".join(f"{lab(t)[0] - cx},{lab(t)[1] - cy}" for t in RUINS)
        for label, pack in (("before", ""), ("after", args.pack)):
            if label not in args.labels.split(","):
                continue
            env = {"C3X_LAB_TILE_OVERLAYS": "1", "C3X_LAB_TILE_CITIES": city_spec, "C3X_LAB_TILE_RUINS": ruin_spec,
                   "C3X_RENDERER_SITE_PACK": pack}
            print(renderer.native_render("infrastructure", "save-window", 12, zoom, OUT / f"z{zoom}_{cx}_{cy}_{label}",
                                         center=(cx, cy), extra_env=env), flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])

#!/usr/bin/env python3
"""Build the Winter review page from the study's renders (self-contained HTML).

    python3 Renderer/lab/studies/winter/review.py LOOK [--alt LOOK_LADEN] [--out PATH]

Images are embedded as JPEG data URIs (ffmpeg); the page lands under
Renderer/lab/out/winter/review/ unless --out says otherwise.
"""
from __future__ import annotations

import argparse
import base64
import html
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
import numpy as np

from Renderer.lab.studies.winter import water_classes
from Renderer.lab.studies.winter.study import OUT, TARGET, VIEWS, load

TEMPLATE = Path(__file__).with_name("review_template.html")


def jpeg(path: Path, width: int, crop=None, quality=4) -> str:
    """Base64 JPEG of an image (optionally cropped by fractions l t r b)."""
    filters = []
    if crop:
        l, t, r, b = crop
        filters.append(f"crop=iw*{r - l}:ih*{b - t}:iw*{l}:ih*{t}")
    filters.append(f"scale={width}:-2:flags=lanczos")
    data = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(path), "-vf", ",".join(filters),
                           "-q:v", str(quality), "-f", "mjpeg", "-"], capture_output=True, check=True).stdout
    return "data:image/jpeg;base64," + base64.b64encode(data).decode()


def render(label: str, view: str) -> Path:
    path = OUT / label / view / "preview.png"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def target(view: str) -> Path:
    """The concept, resized to the view (it frames the concept view only)."""
    out = OUT / "review" / "target.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    w, h = VIEWS[view][3]
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(TARGET), "-vf", f"scale={w}:{h}:flags=lanczos",
                    str(out)], check=True)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("look")
    parser.add_argument("--alt", help="look with snow-laden broadleaf crowns")
    parser.add_argument("--light", help="look for the evening and night frames (identical to LOOK by day)")
    parser.add_argument("--uniform", help="earlier look with one snow for every land type (for the land table)")
    parser.add_argument("--out", type=Path, default=OUT / "review" / "index.html")
    parser.add_argument("--flurry", default="flurry", help="label of the Renderer64 flurry clips")
    parser.add_argument("--crop-day", nargs=4, type=float, default=(0.32, 0.10, 0.52, 0.50))
    parser.add_argument("--crop-night", nargs=4, type=float, default=(0.32, 0.10, 0.52, 0.50))
    parser.add_argument("--flurry-note", default="")
    parser.add_argument("--roofs-before", default="final", help="label with the roof snow before the user's note")
    parser.add_argument("--banks-before", default="final23", help="label with the river banks before the user's note")
    args = parser.parse_args()
    look, alt = args.look, args.alt
    args.out.parent.mkdir(parents=True, exist_ok=True)
    full, half = 1600, 800
    images = {
        # The hero uses the concept framing through the tile overlays, so the
        # map's own pine forests draw as in game.
        "concept_target": jpeg(target("pines"), full),
        "concept_summer": jpeg(render("summer", "pines"), full),
        "concept_base": jpeg(render("base", "pines"), full),
        "concept_winter": jpeg(render(look, "pines"), full),
        "village_summer": jpeg(render("summer", "village-close"), half),
        "village_winter": jpeg(render(look, "village-close"), half),
        "salamanca_summer": jpeg(render("summer", "salamanca-close"), half),
        "salamanca_winter": jpeg(render(look, "salamanca-close"), half),
        "evening_summer": jpeg(render("summer", "pines-evening" if (OUT / "summer/pines-evening/preview.png").is_file()
                                      else "evening"), half),
        "evening_winter": jpeg(render(args.light or look, "pines-evening"), half),
        "night_summer": jpeg(render("summer", "village-night"), half),
        "night_winter": jpeg(render(args.light or look, "village-night"), half),
        "crop_roofs": jpeg(render(look, "village-close"), 700, (0.40, 0.24, 0.66, 0.44)),
        "roofs_before": jpeg(render(args.roofs_before, "salamanca-close"), 800, (0.50, 0.30, 0.68, 0.48)),
        "roofs_after": jpeg(render(look, "salamanca-close"), 800, (0.50, 0.30, 0.68, 0.48)),
        "banks_before": jpeg(render(args.banks_before, "village-close"), 800, (0.30, 0.64, 0.48, 0.84)),
        "banks_after": jpeg(render(look, "village-close"), 800, (0.30, 0.64, 0.48, 0.84)),
        "crop_farms": jpeg(render(look, "village-close"), 700, (0.60, 0.76, 0.95, 0.98)),
        "crop_rocks": jpeg(render(look, "concept"), 700, (0.70, 0.52, 0.92, 0.98)),
        "crop_pines_sp": jpeg(render(look, "village-close"), 700, (0.0, 0.33, 0.44, 0.80)),
        "crop_laden": jpeg(render(look, "concept"), 700, (0.26, 0.33, 0.70, 0.78)),
        "crop_river": jpeg(render(look, "village-close"), 800, (0.20, 0.42, 0.48, 0.98)),
        "crop_river_summer": jpeg(render("summer", "village-close"), 800, (0.20, 0.42, 0.48, 0.98)),
    }
    images["seas_summer"] = jpeg(render("summer", "seas"), half)
    images["seas_winter"] = jpeg(render(look, "seas"), half)
    rows = []
    for name, label in (("Summer", "summer"), ("Winter today", "base"), ("Winter, class step only", "v10"),
                        ("Winter", look)):
        if not (OUT / label / "seas" / "preview.png").is_file():
            continue
        centre = {k: v.mean(0) for k, v in water_classes.class_means(load(OUT / label / "seas" / "preview.png"), "seas").items()}
        cells = [f"{np.linalg.norm(centre[a] - centre[b]) * 100:.1f}" if a in centre and b in centre else "–"
                 for a, b in ((11, 12), (12, 13), (11, 13))]
        rows.append(f"<tr><td>{html.escape(name)}</td>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
    images["water_rows"] = "".join(rows)
    # Land types (concept view): separation between type means, every pair.
    pairs = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
    rows = []
    for name, label in (("Summer", "summer"), ("Winter, one snow for all", args.uniform), ("Winter", look)):
        if not label or not (OUT / label / "lands" / "preview.png").is_file():
            continue
        centre = {k: v.mean(0) for k, v in water_classes.class_means(load(OUT / label / "lands" / "preview.png"), "lands",
                                                                    water_classes.LAND).items()}
        cells = [f"{np.linalg.norm(centre[a] - centre[b]) * 100:.1f}" if a in centre and b in centre else "–" for a, b in pairs]
        rows.append(f"<tr><td>{html.escape(name)}</td>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
    images["land_rows"] = "".join(rows)
    images["land_summer"] = jpeg(render("summer", "lands"), half, (0.12, 0.0, 0.62, 0.45))
    images["land_winter"] = jpeg(render(look, "lands"), half, (0.12, 0.0, 0.62, 0.45))
    images["tundra_summer"] = jpeg(render("summer", "lands"), half, (0.08, 0.55, 0.58, 1.0))
    images["tundra_winter"] = jpeg(render(look, "lands"), half, (0.08, 0.55, 0.58, 1.0))
    images["seas_step"] = jpeg(render("v10", "seas"), half, (0.0, 0.0, 0.5, 0.5)) if (OUT / "v10/seas/preview.png").is_file() else ""
    images["flurry_note"] = html.escape(args.flurry_note)
    # Renderer64 effect clips (flurry_tree.py), cropped to the mountains.
    for clip, view, crop in (("flurry-day.mp4", "pines", args.crop_day), ("flurry-night.mp4", "pines-night", args.crop_night)):
        source = OUT / args.flurry / view / "effect.mp4"
        if source.is_file():
            l, t_, r, b = crop
            subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(source), "-vf",
                            f"crop=trunc(iw*{r - l}/2)*2:trunc(ih*{b - t_}/2)*2:iw*{l}:ih*{t_},scale=720:-2:flags=lanczos",
                            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-movflags", "+faststart",
                            str(args.out.parent / clip)], check=True)
    page = TEMPLATE.read_text()
    for key, value in images.items():
        page = page.replace("{{" + key + "}}", value)
    record = OUT / look / "render.json"
    shown = json.loads(record.read_text()).get("look", look) if record.is_file() else look
    page = page.replace("{{look}}", html.escape(shown + (f" · {args.light} at dusk and night" if args.light and args.light != look else "")))
    args.out.write_text(page)
    print(args.out, f"{len(page) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()

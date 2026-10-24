#!/usr/bin/env python3
"""Build the tile-buildings gallery: one self-contained HTML page.

Reads the crops (study.py crops) and Civ III reference sprites (study.py refs)
and writes Renderer/lab/out/tile_buildings/gallery.html with every image
embedded as JPEG.

    python3 Renderer/lab/studies/tile_buildings/page.py
"""
from __future__ import annotations

import base64
import html
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "Renderer/lab/out/tile_buildings"

# Per family: title, what Civ III does, its reference sprites, the catalog,
# the recommended cells (by name) and a note on the recommendation.
FAMILIES = [
    {"key": "tb-fortress", "title": "Fortress",
     "civ3": "Built by workers anywhere. One sprite per era: a stone keep on a mound, a tower keep, a star fort, "
             "a concrete bunker. Units stand in it.",
     "refs": [("fortress_0", "Ancient"), ("fortress_1", "Medieval"), ("fortress_2", "Industrial"),
              ("fortress_3", "Modern")],
     "pick": {"FortKeep An", "FortKeep Md", "FortKeep In", "FortKeep Mo"},
     "why": "The Fort's keep alone: as compact as Civ III's sprite, a castle ring for the first two eras, a brick "
            "star fort in the Industrial era (Civ III's own industrial fortress is a star fort) and a concrete fort "
            "in the Modern era. Each keeps an open courtyard where the unit stands. Civ VI has no ancient fort, so "
            "the Ancient row borrows the medieval castle. The full Fort adds rocks and props; the Alcázar reads as "
            "a palace with gardens rather than a military post."},
    {"key": "tb-barricade", "title": "Barricade",
     "civ3": "The Conquests upgrade of a fortress. Civ III draws the barricade instead of the fortress: the keep "
             "with stakes, a moated castle, a moated star fort, a bunker in barbed wire.",
     "refs": [("barricade_0", "Ancient"), ("barricade_1", "Medieval"), ("barricade_2", "Industrial"),
              ("barricade_3", "Modern")],
     "pick": {"BarFortW An", "BarFortW Md", "BarFortW In", "BarFortW Mo"},
     "why": "The same keep as the fortress inside a ring of the era's walls (city walls first, the Encampment's "
            "walls and watchtowers later): it reads at once as the fortress upgraded. Neighbouring barricades meet "
            "wall to wall, so a line of them reads as a fortified line. The alternative below joins a line with "
            "Civ VI's Great Wall instead."},
    {"key": "tb-barricade-line", "title": "Barricade line (alternative)",
     "civ3": "Barricades are often built in lines, like the Japanese line in the 1498 save.",
     "refs": [], "pick": set(),
     "why": "Civ VI's Great Wall pieces joined edge to edge, a tower on each tile: a striking line, but stone in "
            "every era and with no fort for the unit to stand in. Shown for comparison."},
    {"key": "tb-airfield", "title": "Airfield",
     "civ3": "Two sprites: an early grass field with hangars, then a modern airport with a tower and tarmac.",
     "refs": [("airfield_0", "Early"), ("airfield_1", "Modern")],
     "pick": {"Air In", "AeroAir Mo"},
     "why": "Civ VI's Airstrip (Quonset hangar, tents, watchtower) for the early sprite, and the Aerodrome with "
            "hangar and airport (tarmac and control tower) for the modern one, following Civ III's own change."},
    {"key": "tb-outpost", "title": "Outpost",
     "civ3": "A lookout tower that extends the owner's sight. Three sprites: a timber lookout, a stone tower, "
             "an iron watchtower with a telegraph pole.",
     "refs": [("outpost_0", "Ancient–Medieval"), ("outpost_1", "Industrial"), ("outpost_2", "Modern")],
     "pick": {"Watch An", "Watch Md", "Watch In", "Watch Mo"},
     "why": "The Encampment's own watchtowers, which Civ VI gives a classical, an industrial and a modern "
            "version: the closest match to Civ III's three lookouts."},
    {"key": "tb-radar", "title": "Radar tower",
     "civ3": "One sprite: a radar dish on a concrete pedestal.",
     "refs": [("radar", "Modern")], "pick": {"RadarDome"},
     "why": "Amundsen-Scott Station's radar dish is the one Civ VI piece that is a radar, and it matches Civ III's "
            "dish on a pedestal."},
    {"key": "tb-colony", "title": "Colony",
     "civ3": "A worker-built outpost on a resource outside the borders. One sprite per era; the resource stays "
             "visible in front of it.",
     "refs": [("colony_0", "Ancient"), ("colony_1", "Medieval"), ("colony_2", "Industrial"), ("colony_3", "Modern")],
     "pick": {"Col An B", "Col Md B", "Col In B", "Col Mo A"},
     "why": "The 7 October draft's concepts, now through the production renderer; the marked cells are that "
            "draft's recommendations (no change). The middle of the tile is kept clear for the resource, which "
            "these renders leave out. Owner pennants are not drawn yet."},
    {"key": "tb-victory", "title": "Victory point",
     "civ3": "Scenario victory locations only: a small white obelisk.",
     "refs": [("victory", "All eras")], "pick": {"VicAntique"},
     "why": "The antiquity obelisk: white, slender and plain, as Civ III's marker is. The Sphinx's obelisk has "
            "a gilded cap and darker stone."},
]


def jpeg_b64(png: Path, quality: int = 80) -> str:
    with tempfile.TemporaryDirectory() as folder:
        target = Path(folder) / "x.jpg"
        subprocess.run(["sips", "-s", "format", "jpeg", "-s", "formatOptions", str(quality), str(png),
                        "--out", str(target)], check=True, capture_output=True)
        return base64.b64encode(target.read_bytes()).decode()


def png_b64(png: Path) -> str:
    return base64.b64encode(png.read_bytes()).decode()


def build() -> Path:
    from Renderer.lab.studies.tile_buildings import study  # noqa: F401  (points kits at this study)
    from Renderer.lab.studies.tile_buildings import plan
    catalogs = plan.catalogs()
    index = json.loads((OUT / "crops/index.json").read_text())
    body = []
    for family in FAMILIES:
        spec = catalogs[family["key"]]
        body.append(f"<section><h2>{html.escape(family['title'])}</h2>"
                    f"<p class=note><b>Civ III:</b> {html.escape(family['civ3'])}</p>")
        if family["refs"]:
            body.append("<div class=refs>" + "".join(
                f"<figure><img src='data:image/png;base64,{png_b64(OUT / 'civ3' / (name + '.png'))}' alt='Civ III "
                f"{html.escape(label)}'><figcaption>Civ III · {html.escape(label)}</figcaption></figure>"
                for name, label in family["refs"]) + "</div>")
        body.append(f"<p class=why><span class=pick>Recommended</span>{html.escape(family['why'])}</p>")
        crops = index.get(family["key"], {})
        game, close = crops.get("-z128-h12"), crops.get("-z256-h12")
        if isinstance(game, str):
            # A whole scene (the barricade line).
            body.append(f"<figure class=scene><img src='data:image/jpeg;base64,{jpeg_b64(OUT / 'crops' / close)}' "
                        f"alt='{html.escape(family['title'])}'><figcaption>Close zoom (2x)</figcaption></figure>"
                        f"<figure class=scene><img src='data:image/jpeg;base64,{jpeg_b64(OUT / 'crops' / game)}' "
                        f"alt='{html.escape(family['title'])}'><figcaption>Game zoom</figcaption></figure></section>")
            continue
        rows = sorted({cell["row"] for cell in spec["cells"]})
        for row in rows:
            cells = sorted((cell for cell in spec["cells"] if cell["row"] == row), key=lambda cell: cell["col"])
            era = cells[0].get("era")
            body.append(f"<h3>{html.escape(era or '')}</h3><div class=grid>")
            for cell in cells:
                key = f"{cell['row']},{cell['col']}"
                picked = cell["name"] in family["pick"]
                label = cell["label"].split(" · ", 1)[-1]
                images = "".join(
                    f"<img src='data:image/jpeg;base64,{jpeg_b64(OUT / 'crops' / shots[key])}' alt='{html.escape(label)}' "
                    f"class={kind}>" for kind, shots in (("close", close), ("game", game)) if shots and key in shots)
                body.append(f"<figure class='card{' picked' if picked else ''}'>{images}<figcaption>"
                            f"{'<span class=pick>Recommended</span>' if picked else ''}{html.escape(label)}"
                            f"</figcaption></figure>")
            body.append("</div>")
        body.append("</section>")
    page = PAGE_HEAD + "".join(body) + PAGE_TAIL
    target = OUT / "gallery.html"
    target.write_text(page)
    return target


PAGE_HEAD = """<!doctype html><html lang=en><head><meta charset=utf-8>
<meta name=viewport content='width=device-width,initial-scale=1'><title>Tile Buildings Gallery</title><style>
:root{--bg:#f6f5f2;--fg:#1d1d1b;--muted:#5c5a55;--card:#fff;--line:#dedbd4;--ref:#efe9dc;--pick:#1f6f43}
@media (prefers-color-scheme: dark){:root:not([data-theme=light]){--bg:#1b1a18;--fg:#ecebe7;--muted:#a8a59e;--card:#252421;--line:#3a3833;--ref:#2f2a22;--pick:#6fcf97}}
:root[data-theme=dark]{--bg:#1b1a18;--fg:#ecebe7;--muted:#a8a59e;--card:#252421;--line:#3a3833;--ref:#2f2a22;--pick:#6fcf97}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 -apple-system,Segoe UI,Helvetica,Arial,sans-serif}
main{max-width:1240px;margin:0 auto;padding:24px 16px 64px}h1{font-size:26px;margin:0 0 6px}h2{font-size:21px;margin:44px 0 6px}
h3{font-size:15px;margin:18px 0 4px;color:var(--muted);text-transform:uppercase;letter-spacing:.05em}
.lede,.note,.why{color:var(--muted);max-width:940px}.note,.why{font-size:14px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(230px,1fr));gap:12px;margin-top:6px}
.card{margin:0;background:var(--card);border:1px solid var(--line);border-radius:10px;overflow:hidden}
.card.picked{border:2px solid var(--pick)}
.card img{display:block;width:100%;height:auto}.card img.game{width:50%;margin:0 auto;border-top:1px solid var(--line)}
.card figcaption{font-size:13px;padding:6px 10px 8px}
.pick{display:inline-block;font-size:11px;font-weight:600;color:#fff;background:var(--pick);border-radius:4px;padding:1px 6px;margin-right:6px;vertical-align:1px}
.refs{display:flex;flex-wrap:wrap;gap:10px;margin:8px 0}.refs figure{margin:0;background:var(--ref);border:1px solid var(--line);border-radius:10px;padding:6px;flex:0 1 270px}
.refs img{width:100%;image-rendering:pixelated;display:block}.refs figcaption,.scene figcaption{font-size:12px;color:var(--muted)}
.scene{margin:10px 0}.scene img{max-width:100%;display:block;border-radius:8px}
.box{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 16px;margin-top:18px}
</style></head><body><main><h1>Tile buildings: Civ VI stand-ins (Lab draft)</h1>
<p class=lede>With the custom renderer on, Civ III's own tile plane is switched off, and nothing yet replaces its
fortresses, barricades, airfields, outposts, radar towers, colonies or victory points: they are invisible on the map.
Each section shows Civ III's sprites, then Civ VI candidates rendered through the production renderer (Lab, noon).
In each card the large image is close zoom (2x) and the small one game zoom. Nothing here is in the game yet.</p>
<div class=box><b>How to choose:</b> pick one option per family (per era row where they differ). Recommended cards
have a green border. Integration follows your picks: Civ III's state is read per tile (fortress, barricade, airfield,
radar, outpost, colony, victory point), and the era follows Civ III's own rule for each family.</div>
"""
PAGE_TAIL = "</main></body></html>\n"


if __name__ == "__main__":
    print(build())

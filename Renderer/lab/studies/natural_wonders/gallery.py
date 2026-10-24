#!/usr/bin/env python3
"""Page comparing each C3X natural-wonder sprite with its Renderer64 Lab kit render (Lab only).

Reads the crops written by kits.py and writes a self-contained page folder under
Renderer/lab/out/natural_wonders/site/. Optional in-game source reference shots
may be placed in that folder's img/ directory as ref_<key>.jpg.

    python3 Renderer/lab/studies/natural_wonders/gallery.py
"""
from __future__ import annotations

import html
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from Renderer.lab.studies.natural_wonders import kits

SITE = kits.OUT / "site"
CROPS = kits.OUT / "crops"
ORDER = ("angel_falls", "yosemite", "mount_fuji", "yellowstone", "mount_everest", "zhangjiajie_mountains",
         "mount_kilimanjaro", "great_barrier_reef", "matterhorn", "moraine_lake", "tropical_rainforest", "wadi_rum",
         "eyjafjallajokull", "ha_long_bay", "lofoten_skerries", "geirangerfjord", "delicate_arch", "savanna")
SOURCES = {
    "angel_falls": ("Workshop 2984762028 · installed", "Height field and plant children, its falls turned to face the river (C3X: southeast); animated waterfall and spray"),
    "yosemite": ("Civ VI base game", "The east (Half Dome) side of the height field and granite at true proportions (its normal map's cracks and the skin's fine rock grain baked into the colour), a third of its trees; its sheer face turned to the viewer's left as C3X draws it"),
    "mount_fuji": ("Authored for C3X", "A broad concave stratovolcano: summit crater, gullies, green foot, snow cap"),
    "yellowstone": ("Workshop 1461463960 · installed", "Height field, travertine and pines; animated geyser and steaming vents"),
    "mount_everest": ("Civ VI base game", "Height field on a broad massif: about 1.8x a standard mountain's height, 1.4x the usual footprint"),
    "zhangjiajie_mountains": ("Terra Mirabilis' Wulingyuan reuses the Indonesia & Khmer pack's Hạ Long Bay asset",
                              "Its karst pillars stretched slender and 14 more packed across the tile, in jungle"),
    "mount_kilimanjaro": ("Civ VI base game", "Height field only"),
    "great_barrier_reef": ("Civ VI base game", "Reef meshes on its submerged height field"),
    "matterhorn": ("Rise and Fall", "Height field on a slightly broader massif, same height"),
    "moraine_lake": ("Rise and Fall (Matterhorn element) + authored basin", "A moraine-rimmed lake in Renderer64's production water, with the Matterhorn's peak behind and skin pines"),
    "tropical_rainforest": ("Renderer64 Civ V skin vegetation", "A canopy dome: one emergent palm ringed by big dense jungle trees, then palms and understory"),
    "wadi_rum": ("Workshop 2949095377 · installed", "The mod's buttes laid out like Monument Valley (the two Mittens side by side, Merrick Butte in front, a mesa behind), each on its own red talus slope above flat sand, no brush"),
    "eyjafjallajokull": ("Vikings Scenario pack", "Height field only; the ash plume is a missing effect"),
    "ha_long_bay": ("Indonesia & Khmer pack", "Karst pillar children standing in coastal water, each in a lacy foam collar as in Civ VI"),
    "lofoten_skerries": ("Rise and Fall (Matterhorn element)", "A tight, irregular cluster of the Matterhorn's peak at nine sizes and turns as rock islets in open coastal water, with lacy foam collars at their waterlines; the skin rock's relief as geometry and shading (one tundra neighbour, as the rule allows)"),
    "geirangerfjord": ("Vikings Scenario pack", "Height field with its fjord channel drawn as Renderer64 river water, fanning out across the beach into the sea; no trees"),
    "delicate_arch": ("Rise and Fall", "Arch mesh only, turned to face southeast as in Civ VI"),
    "savanna": ("Workshop 3263126211 · installed", "Authored: JNR's acacias, shrubs and desert rocks (grain baked from the skin's rock), under a short, even cover of golden grass made from Renderer64's wheat tufts"),
}


def rule(key: str) -> str:
    item = kits.definitions()[key]
    text = item["terrain_type"].replace("-", " ").capitalize()
    if item.get("adjacent_to"):
        near = item["adjacent_to"].replace("-", " ")
        text += f" · next to {near}"
        if item.get("adjacency_dir"):
            text += f" ({item['adjacency_dir']})"
    return text


def jpeg(source: Path, target: Path, quality: int = 82) -> None:
    subprocess.run(["sips", "-s", "format", "jpeg", "-s", "formatOptions", str(quality), str(source),
                    "--out", str(target)], check=True, capture_output=True)


SEASONS = ("autumn", "winter", "spring")


def build() -> Path:
    (SITE / "img").mkdir(parents=True, exist_ok=True)
    report = json.loads((kits.OUT / "render_report.json").read_text()) if (kits.OUT / "render_report.json").is_file() else {}
    stamp_report = json.loads((kits.OUT / "render_report_stamp.json").read_text()) \
        if (kits.OUT / "render_report_stamp.json").is_file() else {}
    rows, rendered, seasonal = [], 0, 0
    for key in ORDER:
        name = kits.definitions()[key]["name"]
        sprite = f"img/c3x_{key}.png"
        if not (SITE / sprite).is_file():
            raise FileNotFoundError(f"Missing C3X sprite {sprite}; run the sprite export first")
        cell, context = CROPS / f"{key}-cell-stamp-z256.png", CROPS / f"{key}-context-stamp-z256.png"
        status = stamp_report.get(key, {})
        method = "Relief in Renderer64's terrain" if kits.stamped(key) else "Mesh kit on the tile"
        if not (cell.is_file() and status.get("z256") == "drawn"):
            cell, context = CROPS / f"{key}-cell-z256.png", CROPS / f"{key}-context-z256.png"
            status = report.get(key, {})
            method = "Mesh kit on the tile"
        has_render = cell.is_file() and status.get("z256") == "drawn"
        if has_render:
            rendered += 1
            shutil.copy2(cell, SITE / f"img/r64_{key}_cell.png")
            jpeg(context, SITE / f"img/r64_{key}_context.jpg")
        pick = kits.PICKS[key]
        source, parts = SOURCES[key]
        reference = SITE / f"img/ref_{key}.jpg"
        ref_html = (f'<figure class="ref"><img src="img/ref_{key}.jpg" alt="{html.escape(pick["title"])} in Civ VI" '
                    f'loading="lazy"><figcaption>In Civ VI (reference shot)</figcaption></figure>'
                    if reference.is_file() else "")
        fold = status.get("fold")
        scale_note = f"Scaled ×{fold:.2f} to fill the tile" if isinstance(fold, (int, float)) else ""
        clip = CROPS / f"{key}-clip-z256.mp4"
        if has_render and clip.is_file():
            # Animated effects: a Renderer64 motion clip in place of the context still.
            shutil.copy2(clip, SITE / f"img/r64_{key}_clip.mp4")
            view_html = (f'<figure class="context"><video src="img/r64_{key}_clip.mp4" poster="img/r64_{key}_context.jpg" '
                         f'autoplay loop muted playsinline aria-label="Renderer64 {html.escape(name)} in motion"></video>'
                         f'<figcaption>In motion: 3.2 s of Renderer64 frames (15 fps), on its C3X terrain</figcaption></figure>')
        else:
            view_html = (f'<figure class="context"><img src="img/r64_{key}_context.jpg" alt="Renderer64 {html.escape(name)} '
                         f'with neighbouring tiles" loading="lazy"><figcaption>On its C3X terrain, with neighbours'
                         f'</figcaption></figure>')
        # Seasonal variants (kits.py --season): the same framing in Autumn, Winter and Spring.
        cells, contexts = [f'<img class="s-summer" src="img/r64_{key}_cell.png" width="256" height="176" '
                           f'alt="Renderer64 {html.escape(name)} in summer">'], [f'<div class="s-summer">{view_html}</div>']
        for season in SEASONS:
            s_cell = CROPS / f"{key}-cell-stamp-{season}-z256.png"
            s_context = CROPS / f"{key}-context-stamp-{season}-z256.png"
            if not (has_render and s_cell.is_file()):
                cells.append(f'<img class="s-{season}" src="img/r64_{key}_cell.png" width="256" height="176" '
                             f'alt="Renderer64 {html.escape(name)} (no {season} render yet)">')
                contexts.append(f'<div class="s-{season}">{view_html}</div>')
                continue
            seasonal += 1
            shutil.copy2(s_cell, SITE / f"img/r64_{key}_cell_{season}.png")
            jpeg(s_context, SITE / f"img/r64_{key}_context_{season}.jpg")
            cells.append(f'<img class="s-{season}" src="img/r64_{key}_cell_{season}.png" width="256" height="176" '
                         f'alt="Renderer64 {html.escape(name)} in {season}" loading="lazy">')
            contexts.append(f'<div class="s-{season}"><figure class="context"><img src="img/r64_{key}_context_{season}.jpg" '
                            f'alt="Renderer64 {html.escape(name)} in {season}, with neighbouring tiles" loading="lazy">'
                            f'<figcaption>{season.capitalize()}, on its C3X terrain with neighbours</figcaption></figure></div>')
        if has_render:
            render_html = (f'<figure class="cell">{"".join(cells)}<figcaption>Renderer64 kit, same framing</figcaption></figure>'
                           f'<div class="seasonal">{"".join(contexts)}</div>')
        else:
            reason = status.get("error") or status.get("z256") or "not rendered yet"
            render_html = f'<div class="missing">Not rendered: {html.escape(str(reason))}</div>'
        rows.append(f'''<article class="nw" id="{key}">
<header class="head"><h2>{html.escape(name)}</h2><p class="rule">{html.escape(rule(key))}</p><span class="method">{html.escape(method)}</span></header>
<div class="strip">
  <figure class="cell sprite"><div class="tile"><img src="{sprite}" width="256" height="176" alt="C3X {html.escape(name)} sprite"></div><figcaption>C3X sprite (2×)</figcaption></figure>
  {render_html}
</div>
<div class="meta">
  <div class="src"><h3>{html.escape(pick["title"])}</h3><p class="mono">{html.escape(source)}</p><p>{html.escape(parts)}</p><p class="muted">{html.escape(scale_note)}</p></div>
  {ref_html}
</div>
</article>''')
    page = TEMPLATE.replace("{rows}", "\n".join(rows)).replace("{rendered}", str(rendered)).replace("{total}", str(len(ORDER)))
    page = page.replace("{seasonal}", str(seasonal))
    (SITE / "index.html").write_text(page, encoding="utf-8")
    return SITE / "index.html"


TEMPLATE = """<title>C3X Natural Wonder Sources</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wdth,wght@62..100,500..800&family=Source+Sans+3:ital,wght@0,400;0,600;1,400&family=JetBrains+Mono:wght@400;500&display=swap">
<style>
/* Layout: a survey ledger. Each wonder is one plate: the C3X sprite and the Renderer64
   kit in identical 2x framing, the kit in context, then its source and a Civ VI reference. */
:root {
  --paper: #eef1ee; --plate: #f8faf8; --ink: #1d2829; --muted: #586869; --rule: #cbd4d1;
  --tile: #c7d0c2; --accent: #1b6c75; --warn: #b0452c;
  --display: "Archivo", "Helvetica Neue", Arial, sans-serif;
  --body: "Source Sans 3", "Segoe UI", system-ui, sans-serif;
  --mono: "JetBrains Mono", ui-monospace, Menlo, monospace;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --paper: #131a1b; --plate: #1a2324; --ink: #e2e9e7; --muted: #92a2a1; --rule: #2c3839;
    --tile: #273334; --accent: #62b8bf; --warn: #e5826a; color-scheme: dark;
  }
}
:root[data-theme="dark"] {
  --paper: #131a1b; --plate: #1a2324; --ink: #e2e9e7; --muted: #92a2a1; --rule: #2c3839;
  --tile: #273334; --accent: #62b8bf; --warn: #e5826a; color-scheme: dark;
}
* { box-sizing: border-box; }
body {
  background: var(--paper); color: var(--ink); font-family: var(--body); font-size: 15px; line-height: 1.5;
  padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px));
  padding-block: 28px 56px;
}
.wrap { max-width: 1180px; margin: 0 auto; display: grid; gap: 24px; }
h1, h2, h3 { font-family: var(--display); text-wrap: balance; margin: 0; }
.kicker { font: 500 12px/1 var(--mono); letter-spacing: .08em; text-transform: uppercase; color: var(--accent); margin: 0; }
h1 { font-size: clamp(30px, 5vw, 46px); font-weight: 800; font-stretch: 75%; line-height: 1.02; }
.lede { max-width: 72ch; margin: 0; color: var(--muted); font-size: 16px; }
.lede strong { color: var(--ink); font-weight: 600; }
.intro { display: grid; gap: 12px; }
.rows { display: grid; gap: 16px; }
.nw { background: var(--plate); border: 1px solid var(--rule); border-radius: 6px; padding: 16px; display: grid; gap: 12px; }
.head { display: flex; flex-wrap: wrap; align-items: baseline; gap: 4px 14px; }
.head h2 { font-size: 22px; font-weight: 750; font-stretch: 85%; }
.rule { margin: 0; font: 400 12.5px/1.4 var(--mono); color: var(--accent); }
.method { font: 500 11.5px/1 var(--body); color: var(--muted); border: 1px solid var(--rule); border-radius: 999px; padding: 4px 9px; }
.strip { display: flex; flex-wrap: wrap; gap: 12px; align-items: flex-start; }
figure { margin: 0; display: grid; gap: 4px; }
figcaption { font-size: 11.5px; color: var(--muted); }
.cell img, .tile { width: 256px; height: 176px; display: block; border-radius: 4px; }
.tile { background: var(--tile); }
.tile img { image-rendering: pixelated; }
.context { flex: 1 1 320px; min-width: 0; max-width: 480px; }
.context img, .context video { width: 100%; height: auto; display: block; border-radius: 4px; }
.missing { width: 256px; height: 176px; display: grid; place-items: center; text-align: center; padding: 12px;
  border: 1px dashed var(--warn); color: var(--warn); border-radius: 4px; font-size: 13px; }
.meta { display: flex; flex-wrap: wrap; gap: 12px 24px; align-items: flex-start; border-top: 1px solid var(--rule); padding-top: 10px; }
.src { flex: 1 1 320px; min-width: 0; display: grid; gap: 3px; }
.src h3 { font-size: 16px; font-weight: 700; font-stretch: 85%; }
.src p { margin: 0; max-width: 64ch; }
.mono { font: 400 12px/1.4 var(--mono); color: var(--muted); }
.muted { color: var(--muted); font-size: 13px; }
.ref img { width: 200px; aspect-ratio: 16 / 10; object-fit: cover; border-radius: 3px; display: block; background: var(--tile); }
.notes { display: grid; gap: 10px; border-top: 1px solid var(--rule); padding-top: 18px; }
.notes h2 { font-size: 22px; font-weight: 750; font-stretch: 85%; }
.notes ul { margin: 0; padding-left: 20px; display: grid; gap: 7px; max-width: 80ch; }
code { font: 400 12.5px var(--mono); overflow-wrap: anywhere; }
@media (max-width: 600px) { .context { max-width: 100%; } }
.seasons { position: sticky; top: 0; z-index: 2; display: flex; flex-wrap: wrap; align-items: center; gap: 8px 14px;
  background: var(--paper); padding: 10px 0; border-bottom: 1px solid var(--rule); }
.seasons .label { font: 500 12px/1 var(--mono); letter-spacing: .08em; text-transform: uppercase; color: var(--muted); }
.switch { display: inline-flex; border: 1px solid var(--rule); border-radius: 999px; overflow: hidden; background: var(--plate); }
.switch button { font: 600 13.5px/1 var(--body); color: var(--muted); background: none; border: 0; padding: 9px 14px; cursor: pointer; }
.switch button[aria-pressed="true"] { background: var(--accent); color: var(--plate); }
.switch button:focus-visible { outline: 2px solid var(--accent); outline-offset: -2px; }
.seasonal { flex: 1 1 320px; min-width: 0; max-width: 480px; }
.seasonal .context { max-width: none; }
.cell .s-autumn, .cell .s-winter, .cell .s-spring, .seasonal .s-autumn, .seasonal .s-winter, .seasonal .s-spring { display: none; }
body[data-season="autumn"] .s-summer, body[data-season="winter"] .s-summer, body[data-season="spring"] .s-summer { display: none; }
body[data-season="autumn"] .s-autumn, body[data-season="winter"] .s-winter, body[data-season="spring"] .s-spring { display: block; }
</style>

<div class="wrap">
<section class="intro">
  <p class="kicker">Renderer64 Lab · natural wonders · {rendered} of {total} rendered</p>
  <h1>C3X Natural Wonder Sources</h1>
  <p class="lede">Each default C3X natural wonder beside its proposed Renderer64 kit. Most kits are built from the source wonder's own Civ VI or Workshop files (height field, terrain meshes and attached pieces); Fuji, Moraine Lake, Lofoten and the rainforest are authored or composed (see the notes). All use Renderer64's Civ V skin materials and trees, and drawn by the <strong>production Renderer64 renderer</strong> on the wonder's C3X terrain rule. The middle image uses exactly the C3X sprite's 2× framing.</p>
</section>
<nav class="seasons" aria-label="Season">
  <span class="label">Season</span>
  <div class="switch" role="group">
    <button type="button" data-season="summer" aria-pressed="true">Summer</button>
    <button type="button" data-season="autumn" aria-pressed="false">Autumn</button>
    <button type="button" data-season="winter" aria-pressed="false">Winter</button>
    <button type="button" data-season="spring" aria-pressed="false">Spring</button>
  </div>
  <span class="muted">Renderer64's integrated season looks: Autumn v18, Winter v24-laden, Spring v20 · {seasonal} seasonal renders</span>
</nav>
<section class="rows">
{rows}
</section>
<section class="notes">
  <h2>How these Lab kits are built</h2>
  <ul>
    <li><strong>Land wonders put their relief into Renderer64's terrain.</strong> A Lab-only stamp sits beside the production volcano stamp, built in a private copy of the native tree, so the checkout and production packs are untouched. Each source height field (levelled, fitted to the tile, with a smooth foot) becomes part of the terrain mesh. The production mountain shader then does the ground handoff, rock, crevices and snow, and every mesh and tree is seated on that terrain. The wonder tile is hosted as a mountain over its C3X base terrain, so it joins neighbouring mountains (Matterhorn, Everest, Crater Lake).</li>
    <li><strong>Water wonders keep a relief mesh.</strong> Lofoten's skerries and the Great Barrier Reef rise from or sit under water, where terrain relief tapers to nothing, so their relief stays a separate mesh baked with the Civ V skin materials and Renderer64's mountain recipe.</li>
    <li><strong>Meshes and pieces</strong> come from each source's own terrain asset: Hạ Long Bay and Wulingyuan's karst pillars, Monument Valley's buttes, Delicate Arch, Angel Falls' cliff, Crater Lake's rim, and Yosemite's granite and trees.</li>
    <li><strong>Clutter</strong> (Old Faithful's pines, Moraine Lake's pines) uses Renderer64's own Civ V skin forest trees at forest-tile scale. That is what Civ VI shows with the skin installed. The tropical rainforest is built entirely from the skin's jungle trees: one emergent palm ringed by big dense jungle trees above a stepped canopy.</li>
    <li><strong>Authored and composed relief:</strong> Fuji is an authored stratovolcano (green foot, volcanic brown flanks, a snow cap reaching down its gullies), since the mod's peak read as a narrow rock spike. Moraine Lake is an authored lake basin with the Matterhorn's peak behind it, and Lofoten's skerries are the Matterhorn's peak at seven sizes in the sea. Yosemite keeps only its Half Dome side.</li>
    <li><strong>Region colours</strong> come from each source's region IDs through a Lab shader channel on the stamp: Old Faithful's travertine terraces, Matterhorn, Yosemite and Fuji's green lower slopes, Monument Valley's banded sandstone, Angel Falls' warm tepui rock, and authored snow caps on Fuji, Everest, Kilimanjaro and Eyjafjallajökull.</li>
    <li><strong>Effects</strong> are animated Lab stand-ins for Civ VI's particle effects, drawn by Renderer64's city effect layer (the one that draws volcano plumes and chimney smoke). The Angel Falls kit is turned so its falls face its river (the C3X rule's southeast edge); the water drapes from the tepui's rim down that face, frays and mists its plunge pool. Old Faithful's central crater erupts on an 11-second cycle (a light water jet, a spray crown and slow, soft steam) while the cones around it steam gently. The clips are real Renderer64 frames.</li>
    <li><strong>Scale:</strong> each kit is fitted to about 1.2 tiles so it fills the tile as C3X's sprites do. A few iconic wonders get extra height or size as Lab tuning: Fuji, Yosemite, Angel Falls, Eyjafjallajökull, Everest and Old Faithful.</li>
  </ul>
  <h2>Needed in game (M9) so these look right</h2>
  <ul>
    <li><strong>Region materials on terrain-integrated relief.</strong> The Lab carries per-texel region weights through the relief owner channel, with one shader variant per special material (snow, travertine, sandstone). Production needs this as a real stamp channel and a material table, so several special materials can share one frame. Hollows (the Ik-Kil cenote, Lysefjord's channel) also need to cut into the terrain, which the Lab stamp cannot.</li>
    <li><strong>The real region-to-material table.</strong> The mapping here is inferred from height and slope (ID 13 matches the BLANK, host-ground material). Recover the exact table before production.</li>
    <li><strong>Seasons.</strong> The Autumn, Winter and Spring views run Renderer64's own integrated season shading (the approved Autumn v18, Winter v24-laden and Spring v20 looks, from <code>Renderer/lab/shared/seasons</code>) on the same wonder kits, so each wonder changes with the land around it: autumn grass and water grades, winter snow on relief, rock and composition tops with icy shores and rivers, spring meadows on neighbouring grassland. The wonders' own trees and plants take the same autumn and spring crown palettes as natural forests (baked into the kits, broadleaf only; conifers stay evergreen), and winter swaps their pines for the skin's authored snowy pines. Following the season sessions, water stays open in winter, foam collars and effects are unchanged, and wonder slopes get no flowers. Three wonder touches on top: Old Faithful's geyser basin stays bare and steaming in winter, the reef's coral keeps its colour under winter, and wonder foliage sheds most of its snow as natural jungle does.</li>
    <li><strong>Foam collars.</strong> As Civ VI rings Hạ Long Bay's pillars, a lacy foam collar hugs each water wonder's rock at the waterline: Renderer64's own coastal-wave foam texture in ribbons traced round each islet, densest at the rock, its reach uneven and slowly breathing (Lofoten and Hạ Long Bay; the clips show it moving). In the Lab the ribbons join the coastal-wave layer on the wonder's tile; production needs them in the wonder's own data.</li>
    <li><strong>Water by wonder type.</strong> Lofoten, the reef and Hạ Long Bay stand in Renderer64's own sea. Moraine Lake's lake and Geirangerfjord's fjord are drawn by Renderer64's production water layers (depth-tinted bed and water surface: Fresnel, sky light, mirrored objects, glints, shadows), with a rocky shore wherever the terrain falls below the water's surface, so the shoreline follows the fjord walls and the lake basin. The Lab builds them only on a fresh tile compile; production must carry them in the prepared ground and world caches.</li>
    <li><strong>Effects in place of C3X's animations.</strong> The Lab waterfall and geyser ride the city effect layer as two new effect kinds read from the wonder stamp. Production needs them in the wonder's own data, plus Eyjafjallajökull's ash plume and Kilimanjaro's clouds, and the same rules as the volcano plume: hidden on unexplored tiles and kept by the world backing.</li>
    <li><strong>Seasons:</strong> wonder clutter needs the skin's snow pine variants, and the ground decals need snow cover, in winter, just like neighbouring forest tiles.</li>
    <li><strong>The wonder tile's own vegetation.</strong> Jungle and forest wonder tiles (Zhangjiajie, Tropical Rainforest) still grow Renderer64's trees around and through the kit. Production should thin or remove them under the kit footprint.</li>
    <li><strong>Orientation:</strong> point directional sources along the C3X adjacency rule (Angel Falls' river to the southeast, Geirangerfjord's coast to the south). Multi-hex sources must fold onto one tile without losing their blend ramp.</li>
    <li><strong>Budgets and dependencies.</strong> Hạ Long Bay and Wulingyuan carry 180 placed pieces, near the 256-instance composition limit. Wulingyuan needs the Indonesia &amp; Khmer pack, and four picks need their Workshop items installed locally. Converted kits stay local and are never redistributed.</li>
  </ul>
  <p class="muted">Lab-only output under <code>Renderer/lab/out/natural_wonders/</code>; built by <code>Renderer/lab/studies/natural_wonders/kits.py</code>. Reference shots: Civilization Wiki and Steam Workshop pages.</p>
</section>
</div>
<script>
(() => {
  const buttons = document.querySelectorAll(".switch button");
  const pick = season => {
    document.body.dataset.season = season;
    buttons.forEach(b => b.setAttribute("aria-pressed", String(b.dataset.season === season)));
    try { localStorage.setItem("nw-season", season); } catch (e) {}
  };
  buttons.forEach(b => b.addEventListener("click", () => pick(b.dataset.season)));
  let saved = "summer";
  try { saved = localStorage.getItem("nw-season") || "summer"; } catch (e) {}
  pick(saved);
})();
</script>
"""

if __name__ == "__main__":
    print(build())

#!/usr/bin/env python3
"""Build the wonder survey page from the Lab renders (kits.py render).

Writes Renderer/lab/out/wonders/page/index.html, one image sheet per catalog
and zoom, the C3X sprite cells and the context scenes under page/img/. Cells
reference their sheet by percentage background position, as the district
survey page does, so the page ships a few dozen images.

    python3 Renderer/lab/studies/wonders/page.py
"""
from __future__ import annotations

import html
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from Renderer.lab.studies.wonders import kits, plan
from Renderer.lab.studies.wonders.content import FINDINGS, PRODUCTION, STATUS, STEAM, WONDERS
from Renderer.lab.studies.districts import kits as dk
from Renderer.lab.studies.districts import page as district_page
from Renderer.lab.studies.resources import audit

PAGE = kits.OUT / "page"
IMG = PAGE / "img"
SPRITES = ROOT / "Art/Districts/Summer/1200"
GROUND = bytes((205, 209, 200))


def installed(item: dict) -> bool:
    return (kits.WORKSHOP / item["id"]).is_dir()


def esc(text: str) -> str:
    return html.escape(text, quote=True)


def jpeg(png: Path, target: Path, quality: int = 84) -> str:
    subprocess.run(["sips", "-s", "format", "jpeg", "-s", "formatOptions", str(quality), str(png), "--out", str(target)],
                   check=True, capture_output=True)
    png.unlink()
    return target.name


def frame_image(key: str, shot: dict):
    folder = dk.frame(key, shot)
    bmp = folder / f"native-district-{key}-h{shot['hour']:02}-z{shot['zoom']}.bmp"
    return audit.read_bmp(bmp) if (folder / "cells.json").is_file() and bmp.is_file() else None


def sheet(key: str, spec: dict, zoom: int, hour: int = 12, tag: str = "", only_row: int | None = None) -> dict | None:
    """Every cell of a catalog frame (one shot: hour and season tag) cropped onto the
    catalog's row x column grid; `only_row` keeps one row (the complete wonders)."""
    image = frame_image(key, {"tag": tag, "zoom": zoom, "hour": hour})
    if image is None:
        return None
    width, height = zoom * 11 // 8, zoom * 5 // 4
    rows = max(cell["row"] for cell in spec["cells"]) + 1
    cols = max(cell["col"] for cell in spec["cells"]) + 1
    pieces = {}
    for dx, dy, cell in dk.layout(spec):
        if only_row is not None and cell["row"] != only_row:
            continue
        cx, cy = image[0] // 2 + dx * zoom // 2, image[1] // 2 + dy * zoom // 4
        w, h, lines = audit.crop(image, cx - zoom * 11 // 16, cy - zoom * 15 // 16, width, height)
        pieces[(0 if only_row is not None else cell["row"], cell["col"])] = \
            [line + GROUND * (width - w) for line in lines] + [GROUND * width] * (height - h)
    if only_row is not None:
        rows = 1
    blank = [GROUND * width] * height
    out = []
    for row in range(rows):
        strips = [pieces.get((row, col), blank) for col in range(cols)]
        out += [b"".join(strip[y] for strip in strips) for y in range(height)]
    name = f"{key}{tag}-z{zoom}" + ("" if hour == 12 else f"-h{hour}") + ("" if only_row is None else f"-r{only_row}")
    png = IMG / f"{name}.png"
    audit.write_png(png, width * cols, height * rows, out)
    return {"path": jpeg(png, IMG / f"{name}.jpg", 86 if zoom == 256 else 82), "rows": rows, "cols": cols}


def scene(key: str, spec: dict) -> dict:
    """Context scene shots cropped around the city."""
    result = {}
    for shot in dk.shots(spec):
        image = frame_image(key, {"tag": "", **shot})
        if image is None:
            continue
        name = f"{key}{shot.get('tag', '')}-z{shot['zoom']}-h{shot['hour']:02}"
        png = IMG / f"{name}.png"
        audit.write_png(png, *audit.crop(image, *dk.scene_box(spec, image, shot["zoom"])))
        result[(shot["zoom"], shot["hour"], shot.get("tag", ""))] = jpeg(png, IMG / f"{name}.jpg")
    return result


def sprite(path: str, row: int, column: int, name: str) -> str:
    """One C3X wonder sprite cell (128 x 64), its key colours made transparent."""
    source = next(item for item in SPRITES.iterdir() if item.name.lower() == path.lower())
    target = IMG / f"c3x-{name}.png"
    if not target.is_file():
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(source), "-vf",
                        f"crop=128:64:{column * 128}:{row * 64},format=rgba,"
                        "colorkey=0xFF00FF:0.01:0,colorkey=0x00FF00:0.01:0", str(target)], check=True)
    return target.name


def cell_style(spec_sheets: dict, cell: dict) -> str:
    big, small = spec_sheets.get(256), spec_sheets.get(128)
    sheet_ = big or small
    x = 0 if sheet_["cols"] == 1 else cell["col"] / (sheet_["cols"] - 1) * 100
    y = 0 if sheet_["rows"] == 1 else cell["row"] / (sheet_["rows"] - 1) * 100
    return (f"--s256:url(img/{(big or small)['path']});--s128:url(img/{(small or big)['path']});"
            f"background-size:{sheet_['cols'] * 100}% {sheet_['rows'] * 100}%;background-position:{x:.4f}% {y:.4f}%")


def shot(spec_sheets: dict, cell: dict, caption: str, sub: str = "") -> str:
    if not spec_sheets:
        return f'<figure><p class="missing">Render pending</p><figcaption>{esc(caption)}</figcaption></figure>'
    return (f'<figure><div class="shot" style="{cell_style(spec_sheets, cell)}" role="img" aria-label="{esc(caption)}">'
            f'</div><figcaption>{esc(caption)}' + (f'<span class="src">{esc(sub)}</span>' if sub else "")
            + '</figcaption></figure>')


def sources(parts: list) -> str:
    names = []
    for part in parts:
        label, origin = kits.label(part["source"])
        names.append(f"{label} ({origin})")
    return " + ".join(dict.fromkeys(names))


def build() -> Path:
    IMG.mkdir(parents=True, exist_ok=True)
    for stale in IMG.glob("*.jpg"):
        stale.unlink()
    catalogs = plan.everything()
    ledger = plan.ledger()
    sheets = {key: {zoom: made for zoom in (128, 256) if (made := sheet(key, spec, zoom))}
              for key, spec in catalogs.items() if not spec.get("cities")}
    # The complete wonders at night and in the other seasons (close-up), and the
    # alternatives at night.
    VARIANTS = (("night", 22, ""), ("autumn", 12, "-autumn"), ("winter", 12, "-winter"), ("spring", 12, "-spring"))
    variant_sheets = {}
    for key, spec in catalogs.items():
        if spec.get("cities"):
            continue
        last = max(cell["row"] for cell in spec["cells"])
        for name, hour, tag in VARIANTS:
            row = last if key.startswith("wonder-stages") else None
            made = sheet(key, spec, 256, hour, tag, row)
            if made:
                variant_sheets[(key, name)] = {256: made}
    scenes = {key: scene(key, spec) for key, spec in catalogs.items() if spec.get("cities")}

    head = district_page.HEAD.replace("<title>C3X District Kit Survey</title>", "<title>C3X Wonder Survey</title>")
    parts = [head, EXTRA_STYLE]
    parts.append('<header class="masthead"><div class="wrap">'
                 '<p class="eyebrow">Renderer64 Lab · Civ VI wonders on Civ III terrain · not promoted</p>'
                 '<h1>C3X Wonder Survey</h1>'
                 '<p class="lede">Which Civ VI or Workshop wonder each of the 23 default C3X wonders should use on the '
                 'Renderer64 map, and how it looks while it is being built. Every picture of 3D art is a '
                 'production-renderer frame, not a mock-up; construction stages come from Civ VI\'s own build movies.</p>'
                 '<div class="toolbar"><span>Tile size</span>'
                 '<button type="button" id="zoom-256" class="on" aria-pressed="true">Close-up · 256 px</button>'
                 '<button type="button" id="zoom-128" aria-pressed="false">Gameplay · 128 px</button></div>'
                 '<nav class="toc" aria-label="Sections"><a href="#findings">Findings</a><a href="#glance">At a glance</a>'
                 '<a href="#context">In context</a><a href="#fit">Tile fit</a><a href="#wonders">Wonder by wonder</a><a href="#district">Wonder District</a>'
                 '<a href="#workshop">Workshop candidates</a><a href="#production">For production</a></nav>'
                 '</div></header><main class="wrap">')
    parts.append('<section id="findings"><h2>Findings</h2><div class="findings">'
                 + "".join(f'<article><h3>{esc(title)}</h3><p>{esc(body)}</p></article>' for title, body in FINDINGS)
                 + '</div></section>')

    # At a glance.
    rows = []
    for _group, _title, wonders in plan.GROUPS:
        for wonder in wonders:
            item, info = WONDERS[wonder], ledger[wonder]
            pick = plan.PICKS[wonder][0]
            others = ", ".join(choice["label"] for choice in plan.PICKS[wonder][1:]) or "–"
            mods = ", ".join(f'<a href="{STEAM}{w["id"]}">{esc(w["title"].split(": ")[-1])}</a>'
                             + ("" if installed(w) else " (not installed)")
                             for w in item.get("workshop", ())) or "–"
            construct, complete = info["native_art"]["construct"], info["native_art"]["complete"]
            c3x = (f'<img class="mini" src="img/{sprite(info["native_art"]["img_path"], *construct, wonder + "-construct")}" '
                   f'alt="C3X construction sprite"><img class="mini" '
                   f'src="img/{sprite(info["native_art"]["img_path"], *complete, wonder + "-complete")}" alt="C3X sprite">')
            rows.append(f'<tr><th scope="row"><a href="#w-{wonder}">{esc(info["c3x_name"])}</a>'
                        f'<span class="class">{info["wonder_class"].capitalize()} wonder</span></th><td class="sprites">{c3x}</td>'
                        f'<td><span class="badge {item["status"]}">{esc(STATUS[item["status"]])}</span></td>'
                        f'<td>{esc(pick["label"])}<span class="src">{esc(sources(pick["parts"]))}</span></td>'
                        f'<td>{esc(others)}</td><td>{mods}</td></tr>')
    parts.append('<section id="glance"><h2>At a glance</h2><p class="intro">The default C3X roster in C3X order. '
                 'C3X sprites are construction, then complete. Workshop items not installed are linked, not rendered.'
                 '</p><div class="table-wrap"><table><thead><tr><th>C3X wonder</th><th>C3X today</th><th>Match</th>'
                 '<th>Recommended</th><th>Also rendered</th><th>Workshop items</th></tr></thead><tbody>'
                 + "".join(rows) + '</tbody></table></div></section>')

    # Context scenes.
    context_html = []
    for key, spec in catalogs.items():
        if not spec.get("cities"):
            continue
        shots_ = scenes.get(key, {})
        legend = "".join(f'<li>{esc(cell["label"])} <span class="src">({esc(ledger.get(cell["wonder"], {"c3x_name": "C3X Wonder District"})["c3x_name"])})</span></li>'
                         for cell in spec["cells"])

        def img(zoom, hour, alt, cls="", tag=""):
            name = shots_.get((zoom, hour, tag))
            return f'<img class="{cls}" src="img/{name}" alt="{esc(alt)}" loading="lazy">' if name else \
                '<p class="missing">Render pending</p>'
        context_html.append(
            f'<article class="scene"><h3>{esc(spec["title"])}</h3><ul class="legend">{legend}</ul>'
            f'<div class="zoomed z256">{img(256, 12, spec["title"] + " at noon, close-up", "wide")}</div>'
            f'<div class="zoomed z128">{img(128, 12, spec["title"] + " at noon, gameplay zoom", "wide")}</div>'
            f'<div class="pairs"><figure class="night">{img(256, 22, spec["title"] + " at night")}'
            f'<figcaption>Night (close-up)</figcaption></figure>'
            f'<figure class="night">{img(256, 12, spec["title"] + " in winter", tag="-winter")}'
            f'<figcaption>Winter (close-up)</figcaption></figure></div></article>')
    parts.append('<section id="context"><h2>Beside production cities</h2><p class="intro">Wonders on the tiles '
                 'around a production city, to judge size, sun, shadows and night light against accepted city art. '
                 'One wonder in each scene is half built.</p>' + "".join(context_html) + '</section>')

    # Fit on the Civ III tile.
    fit_json = kits.OUT / "footprints.json"
    if fit_json.is_file() and (kits.OUT / "footprints.png").is_file():
        import shutil
        shutil.copy2(kits.OUT / "footprints.png", IMG / "footprints.png")
        fits = json.loads(fit_json.read_text())
        fit_rows = "".join(
            f'<tr><th scope="row">{esc(ledger[w]["c3x_name"])}</th><td>{esc(f["pick"])}</td>'
            f'<td class="num">{f["raised_outside"] * 100:.1f}%</td><td class="num">{f["flat_outside"] * 100:.1f}%</td>'
            f'<td class="num">{f["max_overhang_tiles"]:.2f}</td></tr>' for w, f in fits.items())
        parts.append(
            '<section id="fit"><h2>Fit on the Civ III tile</h2><p class="intro">Civ VI authors wonders in a hex, '
            'with buildings at 30 and 60 degree angles and props and plazas filling the hex ring. Each wonder is '
            'turned so its buildings run along the edges of the square Civ III tile (fronts still south-west or '
            'south-east), scaled so its raised buildings span the tile, and keeps only pieces standing on the tile; '
            'ground-level paving and paths reaching past it are pressed to its edge. Top-down footprints below, '
            'one per C3X wonder in C3X order; the red square is the tile.</p>'
            '<figure class="footprints"><img src="img/footprints.png" alt="Top-down footprints of the 23 wonders on '
            'their Civ III tiles" loading="lazy"></figure>'
            '<div class="table-wrap"><table><thead><tr><th>C3X wonder</th><th>Source</th>'
            '<th>Raised geometry off the tile</th><th>Ground geometry off the tile</th><th>Largest overhang (tiles)</th>'
            '</tr></thead><tbody>' + fit_rows + '</tbody></table></div></section>')

    # Wonder by wonder.
    blocks = []
    for group, title, wonders in plan.GROUPS:
        stage_key, pick_key = f"wonder-stages-{group}", f"wonder-picks-{group}"
        blocks.append(f'<h3 class="group">{esc(title)}</h3>')
        for col, wonder in enumerate(wonders):
            item, info = WONDERS[wonder], ledger[wonder]
            art = info["native_art"]
            c3x = [(art["construct"], "construct", "C3X · under construction"), (art["complete"], "complete", "C3X · complete")]
            if art.get("alternate"):
                c3x += [(art["alternate"]["construct"], "alt-construct", "C3X · other direction, construction"),
                        (art["alternate"]["complete"], "alt-complete", "C3X · other direction")]
            sprites = "".join(f'<figure class="c3x"><img src="img/{sprite(art["img_path"], *cell, f"{wonder}-{tag}")}" '
                              f'alt="{esc(caption)}"><figcaption>{esc(caption)}</figcaption></figure>'
                              for cell, tag, caption in c3x)
            stage_cells = sorted((cell for cell in catalogs[stage_key]["cells"] if cell["col"] == col),
                                 key=lambda cell: cell["row"])
            stage_html = "".join(shot(sheets.get(stage_key, {}), cell, cell["label"].split(" · ")[-1])
                                 for cell in stage_cells)
            complete = stage_cells[-1]
            variants = "".join(shot(variant_sheets.get((stage_key, name), {}), {**complete, "row": 0}, caption)
                               for name, caption in (("night", "Night"), ("autumn", "Autumn"), ("winter", "Winter"),
                                                     ("spring", "Spring")))
            variants = (f'<h5>Complete, at night and through the seasons</h5><div class="strip">{variants}</div>')
            alternatives = ""
            if pick_key in catalogs and len(plan.PICKS[wonder]) > 1:
                alt_cells = sorted((cell for cell in catalogs[pick_key]["cells"] if cell["col"] == col and cell["row"] > 0),
                                   key=lambda cell: cell["row"])
                alternatives = ('<h5>Also rendered (complete)</h5><div class="strip">'
                                + "".join(shot(sheets.get(pick_key, {}), cell, cell["label"], sources(cell["parts"]))
                                          for cell in alt_cells) + '</div>')
            mods = "".join(f'<li><a href="{STEAM}{w["id"]}">{esc(w["title"])}</a> by {esc(w["author"])}: '
                           f'{esc(w["note"])}' + ("" if installed(w) else " (not installed)")
                           + '</li>' for w in item.get("workshop", ()))
            rule = ", ".join(info["buildable_on"]) or "any land"
            if info.get("buildable_on_rivers"):
                rule += ", on a river"
            pick = plan.PICKS[wonder][0]
            blocks.append(
                f'<section class="wonder" id="w-{wonder}"><div class="wonder-head"><h4>{esc(info["c3x_name"])}</h4>'
                f'<span class="badge {item["status"]}">{esc(STATUS[item["status"]])}</span>'
                f'<span class="eyebrow">{esc(info["wonder_class"])} wonder · built on {esc(rule)}</span></div>'
                f'<p>{esc(item["text"])}</p><p class="src">Recommended: {esc(sources(pick["parts"]))}</p>'
                f'<div class="pair"><div class="c3x-col"><p class="eyebrow">C3X today</p><div class="c3x-row">{sprites}'
                f'</div></div><div class="r64-col"><p class="eyebrow">Renderer64 · {esc(pick["label"])}</p>'
                f'<div class="strip">{stage_html}</div></div></div>{variants}{alternatives}'
                + (f'<h5>Workshop items</h5><ul class="mods">{mods}</ul>' if mods else "")
                + '</section>')
    district = catalogs.get("wonder-district")
    if district:
        cells = sorted(district["cells"], key=lambda cell: cell["col"])
        noon = "".join(shot(sheets.get("wonder-district", {}), cell, cell["label"].split(" · ")[-1]) for cell in cells)
        night = "".join(shot(variant_sheets.get(("wonder-district", "night"), {}), cell,
                             cell["label"].split(" · ")[-1] + " at night") for cell in cells)
        winter = "".join(shot(variant_sheets.get(("wonder-district", "winter"), {}), cell,
                              cell["label"].split(" · ")[-1] + " in winter") for cell in cells)
        wd_sprites = "".join(f'<figure class="c3x"><img src="img/{sprite("WonderDistrict.PCX", row, 0, f"wonder-district-{row}")}" '
                             f'alt="C3X Wonder District, {era}"><figcaption>C3X · {era}</figcaption></figure>'
                             for row, era in enumerate(plan.WD_ERAS))
        district_html = (
            '<section id="district"><h2>Wonder District before a wonder starts</h2><p class="intro">C3X draws its '
            'Wonder District sprite (a fenced, scaffolded plot by era, a crane in the last) on a district no city has '
            'started a wonder on. Renderer64\'s version: the era\'s paving with Civ VI\'s own construction props: '
            'a scaffold fence, stone, wood and brick piles, and from the Middle Ages the era\'s crane (wooden, '
            'industrial, modern). When a city starts a wonder, its construction stages take over.</p>'
            f'<div class="pair"><div class="c3x-col"><p class="eyebrow">C3X today</p><div class="c3x-row">{wd_sprites}</div></div>'
            f'<div class="r64-col"><p class="eyebrow">Renderer64 · noon</p><div class="strip">{noon}</div>'
            f'<p class="eyebrow">Night</p><div class="strip">{night}</div>'
            f'<p class="eyebrow">Winter</p><div class="strip">{winter}</div></div></div></section>')
    else:
        district_html = ""
    parts.append('<section id="wonders"><h2>Wonder by wonder</h2><p class="intro">C3X\'s two sprites beside '
                 'Renderer64 frames at 25%, half and 75% of the build and complete. Civ VI wonders follow their own '
                 'build movie; district buildings without one rise from the ground. Fronts face south-west or '
                 'south-east as C3X draws them.</p>' + "".join(blocks) + '</section>')
    parts.append(district_html)

    # Workshop table.
    seen, mod_rows = set(), []
    for wonder, item in WONDERS.items():
        for w in item.get("workshop", ()):
            if (w["id"], wonder) in seen:
                continue
            seen.add((w["id"], wonder))
            mod_rows.append(f'<tr><th scope="row">{esc(ledger[wonder]["c3x_name"])}</th>'
                            f'<td><a href="{STEAM}{w["id"]}">{esc(w["title"])}</a><span class="src">Workshop '
                            f'{w["id"]} · {esc(w["author"])}</span></td>'
                            f'<td>{"Installed, rendered" if installed(w) else "Not installed"}</td><td>{esc(w["note"])}</td></tr>')
    parts.append('<section id="workshop"><h2>Workshop candidates</h2><p class="intro">Wonder mods with their own 3D '
                 'model that fill a gap or improve a match. Installed ones are imported, staged and rendered above; '
                 'subscribing to the rest lets the Lab render them the same way. '
                 'Several credit third-party source models, so like Civ VI art they are for local prototyping only.</p>'
                 '<div class="table-wrap"><table><thead><tr><th>C3X wonder</th><th>Workshop item</th><th>Status</th><th>Notes</th>'
                 '</tr></thead><tbody>' + "".join(mod_rows) + '</tbody></table></div></section>')

    parts.append('<section id="production"><h2>For production</h2><ul class="notes">'
                 + "".join(f'<li>{esc(text)}</li>' for text in PRODUCTION) + '</ul></section>')
    parts.append('</main>' + district_page.SCRIPT.replace("district-zoom", "wonder-zoom"))
    target = PAGE / "index.html"
    target.write_text("".join(parts))
    (PAGE / "files.json").write_text(json.dumps(sorted(p.name for p in IMG.iterdir()), indent=1) + "\n")
    return target


EXTRA_STYLE = """<style>
.class{display:block;font:500 12px var(--mono);color:var(--muted);text-transform:uppercase;letter-spacing:.06em}
td.sprites{white-space:nowrap}
img.mini{width:96px;height:48px;image-rendering:pixelated;vertical-align:middle}
.group{margin-top:40px;font-size:28px}
.wonder{margin-top:28px;padding-top:10px;border-top:1px solid var(--line);display:grid;gap:8px}
.wonder-head{display:flex;gap:12px;align-items:baseline;flex-wrap:wrap}
.wonder-head h4{margin:0;color:var(--ink);font-size:23px}
.wonder p{margin:0;max-width:78ch}
.pair{display:flex;flex-wrap:wrap;gap:18px;align-items:flex-start}
.c3x-col,.r64-col{display:grid;gap:6px;min-width:0}
.r64-col{flex:1 1 560px}
.c3x-row{display:flex;flex-wrap:wrap;gap:8px}
.c3x{margin:0}.c3x img{display:block;width:192px;height:96px;image-rendering:pixelated;background:var(--sheet);
  border:1px solid var(--line);border-radius:6px;max-width:100%}
.c3x figcaption{font-size:12.5px;color:var(--muted);margin-top:3px;max-width:192px}
.strip{display:grid;grid-template-columns:repeat(auto-fill,minmax(min(100%,170px),1fr));gap:8px}
.strip figure{margin:0;min-width:0}
.strip figcaption{font-size:13px;line-height:1.35;margin-top:3px}
.mods{margin:0;padding-left:20px;max-width:80ch;font-size:15px}
.notes{max-width:80ch;padding-left:20px}.notes li{margin:6px 0}
.night{margin:0;max-width:640px}.night img{width:100%}
.badge.gap{color:var(--authored);background:var(--authored-bg)}
.footprints{margin:12px 0}.footprints img{display:block;max-width:100%;height:auto;border:1px solid var(--line);border-radius:6px}
td.num{font-variant-numeric:tabular-nums;text-align:right}
</style>
"""


if __name__ == "__main__":
    print(build())

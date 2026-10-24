#!/usr/bin/env python3
"""Build the district recommendation page from the Lab renders (kits.py).

Writes Renderer/lab/out/districts/page/index.html plus one sprite sheet per
catalog and zoom under page/img/. Cells reference their sheet by percentage
background position, so the page ships a few dozen images, not hundreds.

    python3 Renderer/lab/studies/districts/page.py
"""
from __future__ import annotations

import html
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from Renderer.lab.studies.districts import kits, plan
from Renderer.lab.studies.districts.content import DISTRICTS, FINDINGS, CULTURES, MODS, QUIRKS, ERA_MAP, SHEETS
from Renderer.lab.studies.resources import audit

PAGE = kits.OUT / "page"
IMG = PAGE / "img"
SPRITES = ROOT / "Art/Districts/Summer/1200"
GRASS = bytes((118, 140, 62))


def jpeg(png: Path, target: Path, quality: int = 82) -> Path:
    subprocess.run(["sips", "-s", "format", "jpeg", "-s", "formatOptions", str(quality), str(png), "--out", str(target)],
                   check=True, capture_output=True)
    png.unlink()
    return target


def current(folder: Path) -> bool:
    """A frame rendered by the current layout, facing and ground cover (kits.current)."""
    marker = folder / "cells.json"
    return marker.is_file() and kits.current(json.loads(marker.read_text())) and any(folder.glob("*.bmp"))


def cell_box(zoom: int) -> tuple[int, int]:
    return zoom * 11 // 8, zoom * 5 // 4


def sheet(key: str, spec: dict, zoom: int) -> dict | None:
    """One image of every cell of a catalog frame, laid out on its own grid."""
    shot = {"tag": "", "zoom": zoom, "hour": 12}
    bmp = kits.frame(key, shot) / f"native-district-{key}-h12-z{zoom}.bmp"
    if not current(kits.frame(key, shot)):
        return None
    image = audit.read_bmp(bmp)
    width, height = cell_box(zoom)
    rows = max(cell["row"] for cell in spec["cells"]) + 1
    cols = max(cell["col"] for cell in spec["cells"]) + 1
    pieces = {}
    for dx, dy, cell in kits.layout(spec):
        cx, cy = image[0] // 2 + dx * zoom // 2, image[1] // 2 + dy * zoom // 4
        w, h, lines = audit.crop(image, cx - zoom * 11 // 16, cy - zoom * 15 // 16, width, height)
        lines = [line + GRASS * (width - w) for line in lines] + [GRASS * width] * (height - h)
        pieces[(cell["row"], cell["col"])] = lines
    blank = [GRASS * width] * height
    out = []
    for row in range(rows):
        strips = [pieces.get((row, col), blank) for col in range(cols)]
        out += [b"".join(strip[y] for strip in strips) for y in range(height)]
    png = IMG / f"{key}-z{zoom}.png"
    audit.write_png(png, width * cols, height * rows, out)
    path = jpeg(png, IMG / f"{key}-z{zoom}.jpg", 84 if zoom == 256 else 80)
    return {"path": path.name, "rows": rows, "cols": cols}


def scene_images(key: str, spec: dict) -> list[dict]:
    """Context scenes: every shot cropped around the city."""
    result = []
    for shot in kits.shots(spec):
        zoom, hour = shot["zoom"], shot["hour"]
        bmp = kits.frame(key, shot) / f"native-district-{key}-h{hour:02}-z{zoom}.bmp"
        if not current(kits.frame(key, shot)):
            continue
        image = audit.read_bmp(bmp)
        crop = audit.crop(image, *kits.scene_box(spec, image, zoom))
        name = f"{key}{shot['tag']}-z{zoom}-h{hour:02}"
        png = IMG / f"{name}.png"
        audit.write_png(png, *crop)
        result.append({**shot, "path": jpeg(png, IMG / f"{name}.jpg", 84).name})
    return result


def sprite(name: str) -> str | None:
    """A C3X 2D district sheet, magenta/green keyed out, for slot reference."""
    source = next((path for path in SPRITES.iterdir() if path.stem.lower() == name.lower()), None)
    if source is None:
        return None
    target = IMG / f"c3x-{name}.png"
    if not target.is_file():
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(source), "-vf",
                        "format=rgba,colorkey=0xFF00FF:0.02:0,colorkey=0x00FF00:0.02:0", str(target)], check=True)
    return target.name


def esc(text: str) -> str:
    return html.escape(text, quote=True)


def gallery(key: str, spec: dict, sheets: dict, rows=None, note: str = "") -> str:
    big, small = sheets.get((key, 256)), sheets.get((key, 128))
    if not big:
        return f'<p class="missing">Render missing for {esc(key)}.</p>'
    shown = [cell for cell in spec["cells"] if rows is None or cell["row"] in rows]
    used_rows = sorted({cell["row"] for cell in shown})
    used_cols = sorted({cell["col"] for cell in shown})
    figures = []
    for cell in shown:
        x = 0 if big["cols"] == 1 else cell["col"] / (big["cols"] - 1) * 100
        y = 0 if big["rows"] == 1 else cell["row"] / (big["rows"] - 1) * 100
        style = (f"grid-row:{used_rows.index(cell['row']) + 1};grid-column:{used_cols.index(cell['col']) + 1};"
                 f"--s256:url(img/{big['path']});--s128:url(img/{small['path'] if small else big['path']});"
                 f"background-size:{big['cols'] * 100}% {big['rows'] * 100}%;background-position:{x:.4f}% {y:.4f}%")
        sources = " + ".join(part["record"]["entry"] for part in cell["parts"])
        figures.append(f'<figure><div class="shot" style="{style}" role="img" aria-label="{esc(cell["label"])}"></div>'
                       f'<figcaption>{esc(cell["label"])}<span class="src">{esc(sources)}</span></figcaption></figure>')
    return (f'<div class="gallery-wrap"><div class="gallery" style="--cols:{len(used_cols)}">{"".join(figures)}</div></div>'
            + (f'<p class="note">{note}</p>' if note else ""))


def build() -> Path:
    IMG.mkdir(parents=True, exist_ok=True)
    for stale in IMG.glob("*.jpg"):   # sheets this builder wrote earlier; rebuilt below
        stale.unlink()
    catalogs = plan.catalogs()
    sheets = {}
    scenes = {}
    for key, spec in catalogs.items():
        if spec.get("cities") or spec.get("tiles"):
            scenes[key] = scene_images(key, spec)
            continue
        for zoom in (128, 256):
            made = sheet(key, spec, zoom)
            if made:
                sheets[(key, zoom)] = made

    parts = [HEAD]
    parts.append('<header class="masthead"><div class="wrap">'
                 '<p class="eyebrow">Renderer64 Lab · Civ VI district kits on Civ III terrain · not promoted</p>'
                 '<h1>C3X District Kit Survey</h1>'
                 '<p class="lede">Which Civ VI district, building and improvement art each default C3X district should '
                 'use on the Renderer64 map, how its buildings appear, and how it should vary by era and culture. Every '
                 'picture below is a production-renderer frame (the composition format Renderer64 draws in game), not a '
                 'mock-up.</p>'
                 '<div class="toolbar"><span>Tile size</span>'
                 '<button type="button" id="zoom-256" class="on" aria-pressed="true">Close-up · 256 px</button>'
                 '<button type="button" id="zoom-128" aria-pressed="false">Gameplay · 128 px</button></div>'
                 '<nav class="toc" aria-label="Sections"><a href="#findings">Findings</a><a href="#sheets">Sprite sheets</a><a href="#matrix">Recommendations</a>'
                 '<a href="#context">In context</a><a href="#districts">District by district</a>'
                 '<a href="#buildings">Buildings</a><a href="#culture">Culture plan</a><a href="#mods">Mods and gaps</a>'
                 '<a href="#notes">Method</a></nav></div></header><main class="wrap">')

    parts.append('<section id="findings"><h2>Findings</h2><div class="findings">'
                 + "".join(f'<article><h3>{esc(title)}</h3><p>{body}</p></article>' for title, body in FINDINGS)
                 + '</div></section>')

    frames = {}
    sheet_html = []
    for entry in SHEETS:
        made = {zoom: result for zoom in (128, 256) if (result := pcx_sheet(entry, catalogs, zoom, frames))}
        sheet_html.append(sheet_block(entry, made))
    frames.clear()
    parts.append('<section id="sheets"><h2>Sprite sheets: C3X today beside Renderer64</h2>'
                 '<p class="intro">Every C3X district sprite file next to the Renderer64 version laid out on the same '
                 'grid: rows are Civ III eras, columns the buildings C3X counts. Where C3X has one file per culture, each '
                 'culture has its own sheet with its proposed regional art. A dash marks a cell C3X never draws.</p>'
                 + "".join(sheet_html) + '</section>')

    rows = []
    for district in DISTRICTS:
        rows.append(f'<tr><th scope="row"><a href="#{district["id"]}">{esc(district["name"])}</a></th>'
                    f'<td><span class="badge {district["status"]}">{esc(STATUS[district["status"]])}</span></td>'
                    f'<td>{district["source"]}</td><td>{district["buildings"]}</td><td>{district["culture"]}</td></tr>')
    parts.append('<section id="matrix"><h2>Recommendations at a glance</h2>'
                 '<p class="intro">One row per default C3X district. "Buildings" maps each C3X building that C3X draws '
                 'inside the district to the Civ VI model that would appear when it is built.</p>'
                 '<div class="table-wrap"><table><thead><tr><th>C3X district</th><th>Match</th><th>Civ VI source</th>'
                 '<th>Buildings shown when built</th><th>Era and culture</th></tr></thead><tbody>'
                 + "".join(rows) + '</tbody></table></div></section>')

    context_html = []
    for key in ("context-ancient", "context-medieval", "context-industrial", "context-modern"):
        shots = {f"{s['tag']}-z{s['zoom']}-h{s['hour']:02}": s["path"] for s in scenes.get(key, [])}
        spec = catalogs[key]
        legend = "".join(f'<li>{esc(cell["label"])}</li>' for cell in spec["cells"])

        def img(tag, alt, cls=""):
            return (f'<img class="{cls}" src="img/{shots[tag]}" alt="{esc(alt)}" loading="lazy">' if tag in shots
                    else '<p class="missing">missing</p>')
        context_html.append(
            f'<article class="scene"><h3>{esc(spec["title"])}</h3>'
            f'<p class="intro">Production city in the centre, districts on its neighbouring tiles; the tile directly '
            f'below it is a Neighborhood of the city pack\'s houses:</p><ul class="legend">{legend}</ul>'
            f'<div class="zoomed z256">{img("-z256-h12", spec["title"] + " at noon, close-up", "wide")}</div>'
            f'<div class="zoomed z128">{img("-z128-h12", spec["title"] + " at noon, gameplay zoom", "wide")}</div>'
            f'<div class="pairs"><figure>{img("-z256-h22", "night")}<figcaption>Night (close-up)</figcaption></figure>'
            f'<figure>{img("-bare-z256-h12", "without ground cover")}<figcaption>Without the city ground cover (close-up)</figcaption></figure>'
            f'</div></article>')
    parts.append('<section id="context"><h2>Districts beside production cities</h2>'
                 '<p class="intro">Size, sun, shadows and night response judged against the accepted city art. Same '
                 'renderer, same lighting, same frame. District fronts face south-west or south-east, alternating by tile, '
                 'like the city buildings. Every drafted district stands on the city pack\'s era paving; the last '
                 'close-up of each scene leaves it out for comparison.</p>' + "".join(context_html) + '</section>')

    district_html = []
    for district in DISTRICTS:
        refs = "".join(f'<figure class="c3x"><img src="img/{name}" alt="C3X 2D sheet {esc(stem)}" loading="lazy">'
                       f'<figcaption>{esc(stem)}.pcx</figcaption></figure>'
                       for stem in district.get("sprites", ()) if (name := sprite(stem)))
        blocks = []
        for block in district.get("galleries", ()):
            key = block["catalog"]
            blocks.append(f'<h4>{esc(block["title"])}</h4>' + gallery(key, catalogs[key], sheets, block.get("rows"),
                                                                        block.get("note", "")))
        for key in district.get("scenes", ()):
            blocks.append(scene_block(catalogs[key], scenes.get(key, [])))
        district_html.append(
            f'<section class="district" id="{district["id"]}"><div class="district-head"><h3>{esc(district["name"])}</h3>'
            f'<span class="badge {district["status"]}">{esc(STATUS[district["status"]])}</span></div>'
            f'<div class="district-body"><div class="text">{district["text"]}</div>'
            + (f'<div class="refs"><p class="eyebrow">Current C3X 2D sheet (slot reference)</p>{refs}</div>' if refs else "")
            + f'</div>{"".join(blocks)}</section>')
    parts.append('<section id="districts"><h2>District by district</h2><p class="intro">Rows are Civ III eras (Ancient, '
                 'Middle Ages, Industrial, Modern) unless a label says otherwise; columns follow the C3X building count, '
                 'so each grid reads like the C3X sheet it would replace. Each caption lists the Civ VI pieces composed '
                 'in that cell.</p>' + "".join(district_html) + '</section>')

    parts.append('<section id="buildings"><h2>Buildings shown when built</h2><p class="intro">Each Civ VI hero '
                 'building alone at its in-district size. These are the models that appear in a district when a city in '
                 'range completes the C3X building.</p>'
                 + gallery("buildings-a", catalogs["buildings-a"], sheets)
                 + gallery("buildings-b", catalogs["buildings-b"], sheets) + '</section>')

    culture_rows = "".join(f'<tr><th scope="row">{esc(row[0])}</th>' + "".join(f"<td>{cell}</td>" for cell in row[1:]) + "</tr>"
                           for row in CULTURES["rows"])
    parts.append('<section id="culture"><h2>Culture plan</h2>' + CULTURES["intro"]
                 + '<div class="table-wrap"><table><thead><tr>' + "".join(f"<th>{esc(h)}</th>" for h in CULTURES["head"])
                 + f'</tr></thead><tbody>{culture_rows}</tbody></table></div>' + CULTURES["outro"]
 + '</section>')

    parts.append('<section id="mods"><h2>Mods and gaps</h2>' + MODS + '</section>')
    parts.append('<section id="notes"><h2>Method and C3X quirks</h2>' + ERA_MAP + QUIRKS + '</section>')
    parts.append('</main>' + SCRIPT)
    target = PAGE / "index.html"
    target.write_text("".join(parts))
    (PAGE / "files.json").write_text(json.dumps(sorted(p.name for p in IMG.iterdir()), indent=1) + "\n")
    return target


BLANK = bytes((205, 209, 200))


def pcx_sheet(sheet: dict, catalogs: dict, zoom: int, frames: dict) -> dict | None:
    """The Renderer64 cells that fill one C3X sprite file, on its own grid."""
    width, height = zoom * 9 // 8, zoom * 9 // 10
    rows, cols = len(sheet["rows"]), len(sheet["cols"])
    out, filled = [], 0
    for r in range(rows):
        strips = []
        for c in range(cols):
            target, lines = sheet["map"](r, c), None
            if target:
                key, row, col = target
                if (key, zoom) not in frames:
                    shot = {"tag": "", "zoom": zoom, "hour": 12}
                    folder = kits.frame(key, shot)
                    frames[(key, zoom)] = audit.read_bmp(folder / f"native-district-{key}-h12-z{zoom}.bmp") \
                        if current(folder) else None
                image = frames[(key, zoom)]
                place = next(((dx, dy) for dx, dy, cell in kits.layout(catalogs[key])
                              if cell["row"] == row and cell["col"] == col), None)
                if image and place:
                    cx, cy = image[0] // 2 + place[0] * zoom // 2, image[1] // 2 + place[1] * zoom // 4
                    w, h, lines = audit.crop(image, cx - width // 2, cy - zoom * 5 // 8, width, height)
                    lines = [line + BLANK * (width - w) for line in lines] + [BLANK * width] * (height - h)
                    filled += 1
            strips.append(lines or [BLANK * width] * height)
        out += [b"".join(strip[y] for strip in strips) for y in range(height)]
    if not filled:
        return None
    png = IMG / f"sheet-{sheet['file']}-z{zoom}.png"
    audit.write_png(png, width * cols, height * rows, out)
    return {"path": jpeg(png, IMG / f"sheet-{sheet['file']}-z{zoom}.jpg", 86).name, "rows": rows, "cols": cols}


def sheet_block(sheet: dict, made: dict) -> str:
    big, small = made.get(256), made.get(128)
    stem = sheet["file"]
    source = sprite(stem)
    pcx_cols = len(sheet["cols"])
    if source:
        from struct import unpack
        data = (IMG / source).read_bytes()
        pcx_cols = max(1, round(unpack(">I", data[16:20])[0] / (176 if stem in ("Bridge", "Canal") else 128)))
    head = "".join(f'<div class="col-head" style="grid-column:{c + 2}">{esc(name)}</div>'
                   for c, name in enumerate(sheet["cols"]))
    cells = []
    for r, label in enumerate(sheet["rows"]):
        cells.append(f'<div class="row-head" style="grid-row:{r + 2}">{esc(label)}</div>')
        for c in range(len(sheet["cols"])):
            spot = f"grid-row:{r + 2};grid-column:{c + 2}"
            if big and sheet["map"](r, c):
                x = 0 if big["cols"] == 1 else c / (big["cols"] - 1) * 100
                y = 0 if big["rows"] == 1 else r / (big["rows"] - 1) * 100
                cells.append(f'<div class="shot sheet-cell" style="{spot};--s256:url(img/{big["path"]});'
                             f'--s128:url(img/{(small or big)["path"]});background-size:{big["cols"] * 100}% '
                             f'{big["rows"] * 100}%;background-position:{x:.4f}% {y:.4f}%" role="img" '
                             f'aria-label="{esc(sheet["district"])} {esc(label)} {esc(sheet["cols"][c])}"></div>')
            else:
                cells.append(f'<div class="sheet-cell empty" style="{spot}">—</div>')
    culture = f' · {sheet["culture"]}' if sheet.get("culture") else ""
    pcx = (f'<figure class="pcx"><img src="img/{source}" alt="C3X 2D sheet {esc(stem)}" '
           f'style="width:calc(var(--cell) * {pcx_cols})" loading="lazy"><figcaption>C3X 2D sheet today '
           f'({esc(stem)}.pcx)</figcaption></figure>') if source else ""
    grid = (f'<figure class="r64"><div class="sheet-grid" style="--cols:{len(sheet["cols"])}">'
            f'<div class="corner"></div>{head}{"".join(cells)}</div>'
            f'<figcaption>Renderer64, same layout</figcaption></figure>') if big else \
        '<p class="missing">Renderer64 sheet not rendered yet.</p>'
    note = f'<p class="note">{esc(sheet["note"])}</p>' if sheet.get("note") else ""
    return (f'<article class="sheet" id="sheet-{stem}"><div class="district-head"><h3>{esc(sheet["district"])}{culture}</h3>'
            f'<span class="eyebrow">{esc(stem)}.pcx</span></div><div class="sheet-pair">{pcx}{grid}</div>{note}</article>')


def scene_block(spec: dict, shots: list) -> str:
    """A whole-scene catalog (Great Wall runs): noon at both zooms and night."""
    found = {f"{s['tag']}-z{s['zoom']}-h{s['hour']:02}": s["path"] for s in shots}
    def img(tag, alt, cls=""):
        return (f'<img class="{cls}" src="img/{found[tag]}" alt="{esc(alt)}" loading="lazy">' if tag in found
                else '<p class="missing">Render missing.</p>')
    return (f'<h4>{esc(spec["title"])}</h4><div class="scene">'
            f'<div class="zoomed z256">{img("-z256-h12", spec["title"] + ", close-up", "wide")}</div>'
            f'<div class="zoomed z128">{img("-z128-h12", spec["title"] + ", gameplay zoom", "wide")}</div>'
            f'<div class="pairs"><figure>{img("-z256-h22", "night")}<figcaption>Night (close-up)</figcaption></figure></div>'
            f'</div>')


STATUS = {"exact": "Civ VI match", "close": "Civ VI close match", "standin": "Civ VI stand-in",
          "authored": "Needs mod or authored art", "delegated": "Other milestone"}

HEAD = """<title>C3X District Kit Survey</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@500;600;700&family=Source+Sans+3:ital,wght@0,400;0,600;1,400&family=JetBrains+Mono:wght@400;500&display=swap">
<style>
/* Layout: an atlas — wide plates of renders with short captions; prose kept to a reading column. */
:root{
  --paper:#eef1ea; --sheet:#ffffff; --ink:#18201b; --muted:#59645b; --line:#d2d9cf; --accent:#1d6b74;
  --exact:#2d6a3e; --exact-bg:#dcefe0; --close:#2f5d8a; --close-bg:#dde8f4; --standin:#8a5a12; --standin-bg:#f3e6cf;
  --authored:#7a3a7a; --authored-bg:#efdff0; --delegated:#5a5f5b; --delegated-bg:#e5e8e4;
  --display:"Barlow Condensed","Arial Narrow",sans-serif; --body:"Source Sans 3","Segoe UI",system-ui,sans-serif;
  --mono:"JetBrains Mono",ui-monospace,Menlo,monospace;
}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){
  --paper:#141915; --sheet:#1c231e; --ink:#e4e9e2; --muted:#9aa69c; --line:#2f3832; --accent:#6fc2cb;
  --exact:#8fd3a2; --exact-bg:#1e3826; --close:#9cc2ea; --close-bg:#1c2c3d; --standin:#e7bd78; --standin-bg:#3a2d17;
  --authored:#e0a6e0; --authored-bg:#3a2240; --delegated:#b8bfba; --delegated-bg:#2a302c; color-scheme:dark}}
:root[data-theme="dark"]{
  --paper:#141915; --sheet:#1c231e; --ink:#e4e9e2; --muted:#9aa69c; --line:#2f3832; --accent:#6fc2cb;
  --exact:#8fd3a2; --exact-bg:#1e3826; --close:#9cc2ea; --close-bg:#1c2c3d; --standin:#e7bd78; --standin-bg:#3a2d17;
  --authored:#e0a6e0; --authored-bg:#3a2240; --delegated:#b8bfba; --delegated-bg:#2a302c; color-scheme:dark}
body{background:var(--paper);color:var(--ink);font:16px/1.55 var(--body)}
.wrap{max-width:1360px;margin:0 auto;padding-left:max(16px,env(safe-area-inset-left,0px));padding-right:max(16px,env(safe-area-inset-right,0px))}
.masthead{background:var(--sheet);border-bottom:1px solid var(--line)}
.masthead .wrap{padding-block:28px 18px;display:grid;gap:12px}
h1,h2,h3,h4,h5{font-family:var(--display);font-weight:600;letter-spacing:.01em;text-wrap:balance;margin:0}
h1{font-size:clamp(34px,5vw,52px);line-height:1}
h2{font-size:32px;margin-top:56px;padding-bottom:6px;border-bottom:2px solid var(--ink)}
h3{font-size:25px} h4{font-size:20px;margin-top:22px;color:var(--accent)} h5{font-size:17px;margin-top:14px;color:var(--muted)}
.eyebrow{font:500 12px/1.3 var(--mono);text-transform:uppercase;letter-spacing:.08em;color:var(--muted);margin:0}
.lede,.intro,.district .text p,.findings p,section>p{max-width:72ch}
.lede{font-size:18px;margin:0}
.toolbar{display:flex;flex-wrap:wrap;gap:8px;align-items:center;font:500 13px var(--mono);color:var(--muted)}
.toolbar button{font:600 14px var(--body);color:var(--ink);background:var(--paper);border:1px solid var(--line);border-radius:6px;padding:6px 12px;cursor:pointer}
.toolbar button.on{background:var(--ink);color:var(--sheet);border-color:var(--ink)}
.toolbar button:focus-visible,.toc a:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
.toc{display:flex;flex-wrap:wrap;gap:6px 16px;font-size:15px}.toc a{color:var(--accent)}
main{padding-bottom:80px}
.findings{display:grid;grid-template-columns:repeat(auto-fill,minmax(min(100%,400px),1fr));gap:14px;margin-top:16px}
.findings article{background:var(--sheet);border:1px solid var(--line);border-radius:8px;padding:14px 16px;min-width:0}
.findings h3{font-size:20px}.findings p{margin:6px 0 0}
.table-wrap{overflow-x:auto;margin-top:12px;border:1px solid var(--line);border-radius:8px;background:var(--sheet)}
table{border-collapse:collapse;width:100%;min-width:900px;font-size:15px}
th,td{text-align:left;vertical-align:top;padding:9px 12px;border-bottom:1px solid var(--line)}
thead th{font:600 13px var(--mono);text-transform:uppercase;letter-spacing:.06em;color:var(--muted);background:var(--paper)}
tbody th{font-weight:600;white-space:nowrap} tbody tr:last-child>*{border-bottom:0}
td code,p code,li code,.src{font:12.5px var(--mono)}
a{color:var(--accent)}
.badge{display:inline-block;font:600 12px/1.2 var(--body);padding:3px 8px;border-radius:999px;white-space:nowrap}
.badge.exact{color:var(--exact);background:var(--exact-bg)}.badge.close{color:var(--close);background:var(--close-bg)}
.badge.standin{color:var(--standin);background:var(--standin-bg)}.badge.authored{color:var(--authored);background:var(--authored-bg)}
.badge.delegated{color:var(--delegated);background:var(--delegated-bg)}
.scene{margin-top:26px;display:grid;gap:10px}
.scene img{display:block;border-radius:8px;border:1px solid var(--line);width:100%;height:auto}
.legend{display:flex;flex-wrap:wrap;gap:4px 18px;margin:0;padding-left:18px;font-size:15px}
.pairs{display:grid;grid-template-columns:repeat(auto-fill,minmax(min(100%,300px),1fr));gap:10px}
.pairs figure,.c3x{margin:0}.pairs figcaption,.c3x figcaption{font-size:13px;color:var(--muted);margin-top:4px}
.district{margin-top:40px;padding-top:6px;border-top:1px solid var(--line)}
.district-head{display:flex;gap:12px;align-items:center;flex-wrap:wrap}
.district-body{display:grid;grid-template-columns:minmax(0,1fr) auto;gap:20px;align-items:start}
.district .text p{margin:8px 0}.district .text ul{max-width:72ch;margin:6px 0;padding-left:20px}
.refs{display:grid;gap:6px;max-width:300px}.refs img{display:block;width:100%;height:auto;image-rendering:pixelated;background:var(--sheet);border:1px solid var(--line);border-radius:6px}
@media (max-width:760px){.district-body{grid-template-columns:minmax(0,1fr)}}
.gallery-wrap{overflow-x:auto;margin-top:8px;padding-bottom:4px}
.gallery{display:grid;grid-template-columns:repeat(var(--cols),minmax(150px,1fr));gap:8px;min-width:calc(var(--cols)*150px)}
.gallery figure{margin:0;min-width:0}
.shot{aspect-ratio:352/320;max-width:100%;border-radius:6px;background-image:var(--s256);background-repeat:no-repeat;border:1px solid var(--line)}
body.gameplay .shot{background-image:var(--s128)}
.gallery figcaption{font-size:13.5px;line-height:1.35;margin-top:4px}
.src{display:block;color:var(--muted);font-size:11px;word-break:break-word;margin-top:2px}
.zoomed.z128{display:none} body.gameplay .zoomed.z128{display:block} body.gameplay .zoomed.z256{display:none}
.note{font-size:14px;color:var(--muted);max-width:80ch}
.sheet{margin-top:34px;padding-top:6px;border-top:1px solid var(--line);--cell:150px}
body:not(.gameplay) .sheet{--cell:200px}
.sheet-pair{display:flex;flex-wrap:wrap;gap:18px;align-items:flex-start;margin-top:10px}
.sheet-pair figure{margin:0;min-width:0;max-width:100%;overflow-x:auto}
.pcx img{display:block;image-rendering:pixelated;max-width:none;height:auto;background:var(--sheet);border:1px solid var(--line);border-radius:6px}
.sheet-grid{display:grid;grid-template-columns:auto repeat(var(--cols),var(--cell));gap:4px;align-items:center}
.col-head{grid-row:1;font:600 12px/1.25 var(--body);color:var(--muted);text-align:center;padding-bottom:2px}
.row-head{grid-column:1;font:500 12px var(--mono);color:var(--muted);writing-mode:vertical-rl;transform:rotate(180deg);justify-self:center}
.sheet-cell{width:var(--cell);aspect-ratio:1.25;max-width:none}
.sheet-cell.empty{display:flex;align-items:center;justify-content:center;color:var(--muted);background:var(--paper);border:1px dashed var(--line);border-radius:6px}
.pair figcaption,.sheet-pair figcaption{font-size:13px;color:var(--muted);margin-top:4px}
.missing{color:var(--standin);font-size:14px}
@media (prefers-reduced-motion: no-preference){.toolbar button{transition:background .15s,color .15s}}
</style>
"""

SCRIPT = """<script>
(function(){
  var b256=document.getElementById('zoom-256'), b128=document.getElementById('zoom-128');
  function set(mode){
    document.body.classList.toggle('gameplay', mode==='128');
    b256.classList.toggle('on', mode!=='128'); b128.classList.toggle('on', mode==='128');
    b256.setAttribute('aria-pressed', String(mode!=='128')); b128.setAttribute('aria-pressed', String(mode==='128'));
    try{localStorage.setItem('district-zoom', mode)}catch(e){}
  }
  b256.addEventListener('click', function(){set('256')}); b128.addEventListener('click', function(){set('128')});
  try{ if(localStorage.getItem('district-zoom')==='128') set('128') }catch(e){}
})();
</script>"""


if __name__ == "__main__":
    print(build())

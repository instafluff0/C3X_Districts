#!/usr/bin/env python3
"""Build a self-contained before/after review page from city study renders.

    $C3X_RENDERER_PYTHON Renderer/lab/studies/city_readability/gallery.py BEFORE AFTER

Images are embedded as JPEG data URIs; the page is disposable Lab output under
Renderer/lab/out/city-study/gallery/.
"""
from __future__ import annotations

import argparse
import base64
import html
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from Renderer.lab.studies.city_readability import cases as city_cases

OUT = ROOT / "Renderer/lab/out/city-study"


def encode(path: Path, width: int) -> str:
    from PIL import Image
    with Image.open(path) as source:
        image = source.convert("RGB")
    if image.width > width:
        image = image.resize((width, round(image.height * width / image.width)), Image.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", quality=84, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()


def find(label: str, name: str) -> Path | None:
    found = sorted((OUT / label / name).glob("*.bmp"))
    return found[0] if found else None


CIV3_SHEETS = {"american": "rAMER.PCX", "european": "rEURO.PCX", "roman": "rROMAN.PCX",
               "middle_eastern": "rMIDEAST.PCX", "asian": "rASIAN.PCX"}


def luminance_stats(pixels):
    import numpy as np
    values = pixels @ np.array([.2126, .7152, .0722])
    return float(values.mean()), float(np.percentile(values, 10)), float(np.percentile(values, 90))


def metrics(before: str, after: str) -> list[dict]:
    """Metropolis luminance per era: Civ III sprites, production and candidate."""
    import numpy as np
    from PIL import Image
    rows = []
    civ3_root = ROOT.parents[1] / "Art/Cities"
    for era_index, era in enumerate(city_cases.ERAS):
        row = {"era": era}
        for label, key in ((before, "production"), (after, "candidate")):
            samples = []
            for culture in city_cases.CULTURES:
                path = find(label, f"city-ladder-{culture}-z128")
                if not path:
                    continue
                image = np.asarray(Image.open(path).convert("RGB")).astype(float)
                grass = np.median(image.reshape(-1, 3), 0)
                h, w = image.shape[:2]
                x = w // 2 + city_cases.LADDER_COLUMNS[3] * 64 * w // 1472
                y = h // 2 + city_cases.LADDER_ROWS[era_index] * 32 * h // 832
                crop = image[max(0, y - 90):y + 50, max(0, x - 100):x + 100]
                mask = np.abs(crop - grass).sum(-1) > 45
                samples.append(crop[mask])
                row["grass"] = luminance_stats(grass[None, :])[0]
            if samples:
                row[key] = luminance_stats(np.concatenate(samples))
        civ3 = []
        for name in CIV3_SHEETS.values():
            sheet = civ3_root / name
            if sheet.is_file():
                image = np.asarray(Image.open(sheet).convert("RGB")).astype(float)
                cell = image[era_index * 95:(era_index + 1) * 95, 2 * 167:3 * 167]
                mask = ~((cell[..., 0] > 250) & (cell[..., 1] < 5) & (cell[..., 2] > 250))
                civ3.append(cell[mask])
        if civ3:
            row["civ3"] = luminance_stats(np.concatenate(civ3))
        rows.append(row)
    return rows


def metrics_table(rows: list[dict]) -> str:
    def cell(row, key, index, good=False):
        value = row.get(key)
        return f'<td{" class=good" if good else ""}>{value[index]:.0f}</td>' if value else "<td>–</td>"
    body = []
    for row in rows:
        era = "Middle Ages" if row["era"] == "medieval" else row["era"].title()
        body.append(f"<tr><td>{era}</td>{cell(row, 'civ3', 0)}{cell(row, 'civ3', 2)}"
                    f"{cell(row, 'production', 0)}{cell(row, 'production', 2)}"
                    f"{cell(row, 'candidate', 0, True)}{cell(row, 'candidate', 2, True)}"
                    f"<td>{row.get('grass', 0):.0f}</td></tr>")
    return ('<section><h2>Brightness against Civ III</h2>'
            '<p class="muted">Metropolis pixels, sRGB luminance 0–255, averaged over the five culture groups. '
            'Civ III values come from its own city sprite sheets; the others from the ladder renders on grassland.</p>'
            '<div class="table-scroll"><table><thead><tr><th>Era</th><th>Civ III mean</th><th>Civ III 90th</th>'
            '<th>Production mean</th><th>Production 90th</th><th>Candidate mean</th><th>Candidate 90th</th>'
            '<th>Grass</th></tr></thead><tbody>' + "".join(body) + '</tbody></table></div></section>')


def build(before: str, after: str, notes: dict) -> Path:
    groups = []
    overview = []
    case = "city-gameplay-industrial"
    overview.append((case, "Busy late-game map, noon", f"{case}-z128"))
    overview.append((case, "Same map at night", f"{case}-z128-night"))
    overview.append((case, "Same map at reduced zoom", f"{case}-z64"))
    ladders = [(f"city-ladder-{c}", c.replace("_", " ").title(), f"city-ladder-{c}-z128")
               for c in city_cases.CULTURES]
    closeups = [(case, name.replace("-", " "), f"{case}-z256-{name}") for case, _f, name in city_cases.CLOSE_UPS]
    for title, items, width in (("Gameplay map", overview, 1100), ("Era ladders", ladders, 1100),
                                ("Sites and eras up close", closeups, 768)):
        entries = []
        for case, label, name in items:
            a, b = find(before, name), find(after, name)
            if not a or not b:
                continue
            entries.append({"label": label, "name": name, "before": encode(a, width), "after": encode(b, width),
                            "note": notes.get(name, "")})
        groups.append((title, entries))
    page = TEMPLATE
    sections = []
    for index, (title, entries) in enumerate(groups):
        if not entries:
            continue
        figures = []
        for entry in entries:
            ident = "c-" + entry["name"].replace(".", "-")
            figures.append(f'''
<figure class="compare" id="{ident}">
  <figcaption><span class="tag">{html.escape(entry["label"])}</span>{('<span class="note">' + html.escape(entry["note"]) + '</span>') if entry["note"] else ''}</figcaption>
  <div class="frame" style="--split:50%">
    <img class="before" src="{entry["before"]}" alt="{html.escape(entry["label"])}, production pack">
    <img class="after" src="{entry["after"]}" alt="{html.escape(entry["label"])}, readability candidate">
    <span class="side left">Production</span><span class="side right">Candidate</span>
  </div>
  <label class="slider"><span class="visually-hidden">Reveal</span>
    <input type="range" min="0" max="100" value="50" id="{ident}-range" aria-label="Before and after split for {html.escape(entry["label"])}">
  </label>
</figure>''')
        cls = "grid two" if title.startswith("Sites") else "grid"
        sections.append(f'<section><h2>{html.escape(title)}</h2><div class="{cls}">{"".join(figures)}</div></section>')
    page = page.replace("<!--SECTIONS-->", "\n".join(sections))
    page = page.replace("<!--METRICS-->", metrics_table(metrics(before, after)))
    page = page.replace("<!--DECISIONS-->", notes.get("_decisions", ""))
    target = OUT / "gallery" / "city-readability.html"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(page)
    print(target, target.stat().st_size)
    return target


TEMPLATE = Path(__file__).with_name("gallery_template.html").read_text() if \
    Path(__file__).with_name("gallery_template.html").exists() else "<!--SECTIONS-->"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before")
    parser.add_argument("after")
    parser.add_argument("--notes", type=Path, help="JSON {shot name: caption}")
    args = parser.parse_args()
    build(args.before, args.after, json.loads(args.notes.read_text()) if args.notes else {})

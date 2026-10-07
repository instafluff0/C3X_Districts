#!/usr/bin/env python3
"""Audit every vanilla map resource through the production Lab renderer.

Renders the roster (flat grassland) and relief (hill) cases at gameplay and
close-up zoom, records which resources production actually replaces, and writes
per-resource crops beside Civ III's original map sprite in a local gallery.
Outputs are disposable and stay under Renderer/lab/out/resources/audit/.
"""
from __future__ import annotations

import argparse
import html
import json
import re
import struct
import sys
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from Renderer import renderer
from Renderer.lab.studies.resources import roster

OUT = ROOT / "Renderer/lab/out/resources/audit"
ZOOMS = (128, 256)
# Conquests art beside the C3X mod folder; absent installs simply omit sprites.
CIV3_SPRITES = ROOT.parent / "Art/resources.pcx"
NORMALIZED = ROOT / "Renderer/packs/ResourceNormalized/manifest.json"
ANIMATED = ROOT / "Renderer/packs/ResourceAnimationRuntime/bindings.json"


def read_bmp(path: Path):
    data = path.read_bytes()
    offset = struct.unpack_from("<I", data, 10)[0]
    width, height = struct.unpack_from("<ii", data, 18)
    bpp = struct.unpack_from("<H", data, 28)[0] // 8
    stride = (width * bpp + 3) & ~3
    rows = [data[offset + y * stride: offset + y * stride + width * bpp] for y in range(abs(height))]
    if height > 0:
        rows.reverse()
    return width, abs(height), [_bgr_to_rgb(row, bpp) for row in rows]


def _bgr_to_rgb(row: bytes, bpp: int) -> bytes:
    out = bytearray(len(row) // bpp * 3)
    out[0::3], out[1::3], out[2::3] = row[2::bpp], row[1::bpp], row[0::bpp]
    return bytes(out)


def write_png(path: Path, width: int, height: int, rows: list[bytes]) -> None:
    def chunk(kind, payload):
        return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload) & 0xffffffff)
    raw = b"".join(b"\0" + row for row in rows)
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)) +
                     chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


def crop(image, left: int, top: int, width: int, height: int):
    w, h, rows = image
    left, top = max(0, left), max(0, top)
    width, height = min(width, w - left), min(height, h - top)
    return width, height, [row[left * 3:(left + width) * 3] for row in rows[top:top + height]]


def civ3_sprites() -> dict[int, tuple[int, int, list[bytes]]]:
    """Decode Civ III's 6x6 grid of 50px map resource sprites over a neutral ground."""
    if not CIV3_SPRITES.is_file():
        return {}
    data = CIV3_SPRITES.read_bytes()
    width = struct.unpack_from("<H", data, 8)[0] + 1
    height = struct.unpack_from("<H", data, 10)[0] + 1
    line = struct.unpack_from("<H", data, 66)[0]
    palette = data[-768:]
    pixels, at = bytearray(), 128
    while len(pixels) < line * height:
        value = data[at]; at += 1
        if value >= 0xC0:
            pixels += bytes([data[at]]) * (value & 0x3F); at += 1
        else:
            pixels.append(value)
    ground = bytes((126, 138, 82))
    sprites = {}
    for index in range(36):
        x0, y0 = (index % 6) * 50, (index // 6) * 50
        rows = []
        for y in range(y0 + 1, y0 + 50):
            row = bytearray()
            for x in range(x0 + 1, x0 + 50):
                p = pixels[y * line + x]
                row += ground if p >= 254 else palette[p * 3:p * 3 + 3]
            rows.append(bytes(row))
        sprites[index] = (49, 49, rows)
    return sprites


def production_path(name: str) -> str:
    """Mirror the renderer's current name-substring selection for the census label."""
    lowered = name.lower()
    animated = json.loads(ANIMATED.read_text())["bindings"] if ANIMATED.is_file() else {}
    if any(key in lowered for key in animated):
        return "animated"
    if any(key in lowered for key in ("horses", "iron", "uranium", "gold", "dye", "wheat", "cattle", "fish")):
        return "static"
    return "native sprite"


SUBJECT = re.compile(r"(?!(decal|boulder|snow_boulder|tree_pine|jungle_clump|shrub))")


def source_record(artdef: str):
    manifest = json.loads(NORMALIZED.read_text())
    key = "resource/" + {"RESOURCE_NITER": "saltpeter", "RESOURCE_COCOA": "rubber", "RESOURCE_DEER": "game",
                         "RESOURCE_SPICES": "spice", "RESOURCE_DYES": "dye", "RESOURCE_WINE": "wine",
                         "FEATURE_OASIS": "oasis"}.get(artdef, artdef.split("_", 1)[1].lower())
    return manifest["resources"].get(key)


def source_counts(record) -> dict[str, int]:
    """Authored placement counts by asset id, most frequent first."""
    counts: dict[str, int] = {}
    for placement in record["placements"]:
        asset = placement.get("asset")
        if isinstance(asset, str):
            counts[asset] = counts.get(asset, 0) + int(placement.get("count", 1))
    return dict(sorted(counts.items(), key=lambda item: -item[1]))


def source_summary(artdef: str) -> str:
    record = source_record(artdef)
    if record is None:
        return "no normalized record"
    if record.get("landmark_asset") or record.get("landmark_route"):
        return "single landmark model"
    counts = source_counts(record)
    subject = {k: v for k, v in counts.items() if SUBJECT.match(k.split("/")[-1])}
    extra = sum(v for k, v in counts.items() if k not in subject)
    return (f"{sum(subject.values())} authored subject pieces across {len(subject)} models"
            + (f"; {extra} terrain-conditional clutter/decal pieces" if extra else ""))


def source_preview(artdef: str, target: Path) -> bool:
    """Software preview of the most frequent normalized subject model, two yaws."""
    sys.path.insert(0, str(ROOT / "Renderer"))
    from preview.render_feature_asset import render_feature
    from preview.render_textured_patch import write_png as write_canvas
    record = source_record(artdef)
    for asset in source_counts(record) if record else ():
        if not SUBJECT.match(asset.split("/")[-1]):
            continue
        try:
            write_canvas(render_feature(NORMALIZED, asset, 512, 256), target)
            return True
        except (ValueError, KeyError, OSError):
            continue
    return False


def review_cases(resource_pack: str = "") -> tuple[str, ...]:
    """Catalog cases need the catalog pack; every other case renders with any pack."""
    catalog = resource_pack == "ResourceCatalogLab"
    return tuple(case for case in roster.CASES if case.startswith("native-catalog-") == catalog)


def collect(label: str, zooms=ZOOMS, cases=None):
    results = {}
    for case in cases or review_cases():
        for zoom in zooms:
            folder = OUT / label / f"{case}-z{zoom}"
            census = {}
            for line in (folder / "native.log").read_text(errors="replace").splitlines():
                match = re.match(r"ROSTER (-?\d+),(-?\d+) (.+) replaced=([01])", line.strip())
                if match:
                    census[match.group(3)] = match.group(4) == "1"
            results[(case, zoom)] = (folder / f"{case}-h12-z{zoom}.bmp", census)
    return results


def render(label: str, zooms=ZOOMS, resource_pack: str = "", cases=None):
    renderer.prepare_sources(["resources"])
    from Renderer.tools.asset_compiler import build_resource_compositions as compositions
    if resource_pack == "ResourceCompositionLab":
        compositions.build()
    renderer.ensure_candidate(["resources"])
    cases = cases or review_cases(resource_pack)
    for case in cases:
        if resource_pack == "ResourceCatalogLab":
            compositions.build(compositions.CATALOG, catalog=case.removeprefix("native-catalog-"))
        for zoom in zooms:
            renderer.native_render("resources", case, 12, zoom, OUT / label / f"{case}-z{zoom}",
                                   resource_pack=resource_pack)
    return collect(label, zooms, cases)


def gallery(label: str, results) -> Path:
    folder = OUT / label
    crops = folder / "crops"
    crops.mkdir(parents=True, exist_ok=True)
    sprites = civ3_sprites()
    by_name = {m["civ3_name"]: m for m in roster.mappings()}
    sections, census_rows = [], {}
    for case in dict.fromkeys(case for case, _ in results):
        zooms = [zoom for c, zoom in results if c == case]
        images = {zoom: read_bmp(results[(case, zoom)][0]) for zoom in zooms}
        census = results[(case, zooms[0])][1]
        rows = []
        for dx, dy, name, terrain in roster.placements(case):
            mapping = by_name.get(name.split("~")[0], {"civ3_icon_index": -1, "civ6_artdef": "", "match": "catalog",
                                                         "confidence": ""})
            slug = re.sub(r"[^a-z]+", "-", f"{name} {terrain}".lower()).strip("-")
            sprite = sprites.get(mapping["civ3_icon_index"])
            if sprite:
                write_png(crops / f"{slug}-civ3.png", *sprite)
            if case == "roster" and source_preview(mapping["civ6_artdef"], crops / f"{slug}-civ6.png"):
                pass
            cells = [f'<img src="crops/{slug}-civ3.png" width="64">' if sprite else ""]
            for zoom, image in images.items():
                cx = image[0] // 2 + dx * zoom // 2
                cy = image[1] // 2 + dy * zoom // 4
                piece = crop(image, cx - zoom * 9 // 16, cy - zoom * 3 // 4, zoom * 9 // 8, zoom)
                write_png(crops / f"{slug}-{case}-z{zoom}.png", *piece)
                cells.append(f'<img src="crops/{slug}-{case}-z{zoom}.png" width="{256 if zoom == 256 else 144}">')
            status = "replaced" if census.get(name) else "native Civ III sprite"
            if mapping["civ6_artdef"]:
                census_rows[name] = (mapping, production_path(name), status)
            rows.append(f"<tr><td><b>{html.escape(name)}</b><br><small>{html.escape(terrain)} · {status}</small></td>"
                        + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
        heads = "".join(f"<th>{case} @ {zoom}</th>" for zoom in zooms)
        sections.append(f"<h2>{html.escape(case)}</h2><table><tr><th>Resource</th><th>Civ III</th>{heads}</tr>"
                        + "".join(rows) + "</table>")
    page = folder / "index.html"
    page.write_text("<!doctype html><meta charset=utf-8><title>Resource audit</title>"
                    "<style>body{font:13px system-ui;background:#1d2220;color:#dde}td,th{padding:4px;vertical-align:top;"
                    "border-bottom:1px solid #333}img{image-rendering:pixelated}</style>" + "".join(sections))
    (folder / "census.json").write_text(json.dumps(
        [{"name": n, "production_path": p, "status": s, "civ6_artdef": m["civ6_artdef"], "match": m["match"],
          "confidence": m["confidence"], "source": source_summary(m["civ6_artdef"])}
         for n, (m, p, s) in census_rows.items()], indent=2) + "\n")
    return page


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", default="current", help="Output folder name, e.g. before/after")
    parser.add_argument("--zoom", type=int, action="append", help="Zoom levels (default 128 and 256)")
    parser.add_argument("--gallery-only", action="store_true", help="Rebuild the gallery from existing renders")
    parser.add_argument("--resource-pack", default="",
                        help="Render with this pack instead of production (ResourceCompositionLab and "
                             "ResourceCatalogLab are rebuilt first; the catalog pack renders the catalog cases)")
    parser.add_argument("--case", action="append", help="Render only these cases")
    args = parser.parse_args()
    zooms = tuple(args.zoom or ZOOMS)
    cases = tuple(args.case or review_cases(args.resource_pack))
    page = gallery(args.label, collect(args.label, zooms, cases) if args.gallery_only else
                   render(args.label, zooms, args.resource_pack, cases))
    print(f"Wrote {page.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

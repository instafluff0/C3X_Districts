#!/usr/bin/env python3
"""Make native-pixel review atlases and an index of all medieval families."""

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw

from Renderer.lab.studies.cities.build_layouts import ROOT
from Renderer.lab.studies.cities.sheet import font


def atlas(entries, source, output, crop, columns=4):
    cell_width = crop[2]-crop[0]+20
    cell_height = crop[3]-crop[1]+43
    rows = (len(entries)+columns-1)//columns
    header = 38
    image = Image.new("RGB", (cell_width*columns, header+cell_height*rows), (31, 25, 38))
    draw = ImageDraw.Draw(image)
    draw.text((10, 7), "Civ VI Classical art tier  |  Civ III Middle Ages candidates",
              font=font(20), fill=(249, 236, 249))
    for position, entry in enumerate(entries):
        row, col = divmod(position, columns)
        x, y = col*cell_width, header+row*cell_height
        draw.text((x+10, y+8), entry["family"].removeprefix("CIVILIZATION_"),
                  font=font(20), fill=(249, 236, 249))
        with Image.open(source(entry)) as original:
            image.paste(original.convert("RGB").crop(crop), (x+10, y+34))
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    entries = json.loads(args.index.read_text())
    gallery = ROOT / "Renderer/lab/out/cities/test-biq/gallery"
    # Keep the entire City / Walls + Capital cell, including the near wall.
    sheet_crop = (2434+40, 618+10, 2434+750, 618+505)
    map_crop = (405, 235, 875, 535)
    for start in range(0, len(entries), 8):
        page = start//8+1
        batch = entries[start:start+8]
        atlas(batch,
              lambda e: args.index.parent / e["slug"] / "sheet/european-medieval.png",
              args.output_dir / f"medieval-families-sheet-{page}.png", sheet_crop)
        atlas(batch,
              lambda e: gallery / f"medieval-family-{e['slug']}-city-both.png",
              args.output_dir / f"medieval-families-map-{page}.png", map_crop)
    atlas(entries,
          lambda e: args.index.parent / e["slug"] / "sheet/european-medieval.png",
          args.output_dir / "medieval-families-sheet-all.png", sheet_crop)
    atlas(entries,
          lambda e: gallery / f"medieval-family-{e['slug']}-city-both.png",
          args.output_dir / "medieval-families-map-all.png", map_crop)
    lines = ["# Civ VI Classical art tier → Civ III Middle Ages review", "",
             "All 23 source tags in this gallery use the Civ VI `ARTERA_CLASSICAL`",
             "city-art tier. Civ VI assigns that same art tier to its Classical, Medieval,",
             "and Renaissance gameplay eras. We map it to Civ III Middle Ages as an",
             "offline design choice; the models are not all historically medieval.", "",
             "The 23 source tags contain 22 distinct city-component sets. See the",
             "[full source-era flavor matrix](../../../../studies/cities/source_era_inventory.md)",
             "for AncientWood, ModernGlass, and the other era/culture bindings.", "",
             "The magenta sheets use flat grassland and show Town, City, and Metropolis",
             "with Base, Walls, Capital, and Walls + Capital columns. The map images",
             "are headless D3D replays on captured `test.biq` terrain, not live game screenshots.",
             "These are Lab candidates; no art was promoted.", "",
             "| Source art family | Full variant sheet | Town | City | Capital | Walls + capital | Metropolis |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for entry in entries:
        slug = entry["slug"]
        full = Path(slug) / "sheet/european-medieval.png"
        prefix = f"medieval-family-{slug}"
        links = [f"[{label}](../../test-biq/gallery/{prefix}-{variant}.png)"
                 for label, variant in (("Town", "town-base"), ("City", "city-base"),
                                        ("Capital", "city-capital"),
                                        ("Both", "city-both"),
                                        ("Metro", "metro-both"))]
        lines.append(f"| {entry['family']} | [sheet]({full.as_posix()}) | " +
                     " | ".join(links) + " |")
    (args.output_dir / "README.md").write_text("\n".join(lines) + "\n")
    print(args.output_dir / "README.md")


if __name__ == "__main__":
    main()

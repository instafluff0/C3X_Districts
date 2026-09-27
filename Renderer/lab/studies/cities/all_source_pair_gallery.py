#!/usr/bin/env python3
"""Index every source-art city audition without pre-assigning Civ III eras."""

import json
from pathlib import Path

from PIL import Image, ImageDraw

from Renderer.lab.studies.cities.all_source_pair_review import OUT
from Renderer.lab.studies.cities.sheet import font


ERAS = ("ancient", "classical", "industrial", "modern", "future", "unspecified")
LABELS = {
    "ancient": "ARTERA_ANCIENT",
    "classical": "ARTERA_CLASSICAL",
    "industrial": "ARTERA_INDUSTRIAL",
    "modern": "ARTERA_MODERN",
    "future": "ARTERA_FUTURE",
    "unspecified": "Tag_Era=DEFAULT (unspecified)",
}


def atlas(era, entries, output):
    # Show the full unwalled City cell at native sheet pixels. This makes
    # source architecture comparable without provisional walls or palaces.
    crop = (154, 618, 914, 1098)
    cell = (crop[2] - crop[0], crop[3] - crop[1])
    columns = 3
    rows = (len(entries) + columns - 1) // columns
    margin, caption, header = 15, 40, 65
    image = Image.new("RGB", (columns * (cell[0] + margin),
                              header + rows * (cell[1] + caption + margin)),
                      (31, 25, 38))
    draw = ImageDraw.Draw(image)
    draw.text((15, 12), LABELS[era] + " | source-art City comparison",
              font=font(24), fill=(249, 236, 249))
    draw.text((15, 42), "Same scene scale; full Town/City/Metro variant sheets linked in the index",
              font=font(15), fill=(202, 186, 204))
    for slot, entry in enumerate(entries):
        row, column = divmod(slot, columns)
        x = column * (cell[0] + margin)
        y = header + row * (cell[1] + caption + margin)
        draw.text((x + 10, y + 7), entry["source_culture"],
                  font=font(18), fill=(249, 236, 249))
        source = OUT / entry["sheet"]
        with Image.open(source) as full:
            image.paste(full.convert("RGB").crop(crop), (x, y + caption))
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)


def main():
    root = OUT / "review"
    collected = []
    for era in ERAS:
        entries = json.loads((root / era / "index.json").read_text())
        for entry in entries:
            if not (OUT / entry["sheet"]).exists():
                raise FileNotFoundError(entry["sheet"])
            if entry["source_art_era"] != ("DEFAULT" if era == "unspecified"
                                           else "ARTERA_" + era.upper()):
                raise ValueError(f"Source art-era mismatch: {entry}")
        atlas(era, entries, root / f"{era}-city-overview.png")
        collected.extend(entries)
    if len(collected) != 46:
        raise ValueError(f"Expected 46 populated source pairs; found {len(collected)}")
    (root / "index.json").write_text(json.dumps(collected, indent=2) + "\n")
    lines = ["# Civ VI city source-art auditions", "",
             "These 46 sheets cover every populated `Tag_Culture × Tag_Era` city-building",
             "pair in the installed source inventory. They are **source-art comparisons**,",
             "not final Civ III culture or era assignments. The `Tag_Era=DEFAULT` pool is",
             "era-unspecified. Some tags share identical building sets; they remain separate",
             "here because their source selectors and palaces can differ.", "",
             "Every full magenta sheet shows Town, City, and Metropolis with Base, Walls,",
             "Capital, and Walls + Capital. The wall kit is a provisional comparison aid.",
             "Where the source has no matching palace, the sheet uses a generic comparison",
             "palace; its status is recorded below. These software previews compare layout",
             "and source models, not native D3D material fidelity or in-game acceptance.",
             "The rejected count records pieces the importer could not safely compile;",
             "the Future pool omits seven unresolved effect-bearing blocks per tag.",
             "No city ground decal or elevated masonry is added by these recipes.", "",
             "[Source-art era inventory](../../../../studies/cities/source_era_inventory.md)", ""]
    for era in ERAS:
        entries = [entry for entry in collected
                   if entry["source_art_era"] == ("DEFAULT" if era == "unspecified"
                                                  else "ARTERA_" + era.upper())]
        lines.extend([f"## {LABELS[era]} ({len(entries)})", "",
                      f"![{LABELS[era]} City overview]({era}-city-overview.png)", "",
                      "| Source culture tag | Full 12-state sheet | Palace | Usable pieces | Rejected pieces |",
                      "| --- | --- | --- | ---: | ---: |"])
        for entry in entries:
            lines.append(f"| `{entry['source_culture']}` | [view sheet]({entry['sheet'].removeprefix('review/')}) | "
                         f"{entry.get('palace_status', 'previous Classical candidate')} | "
                         f"{entry['selected_components']} | {entry['rejected_components']} |")
        lines.append("")
    (root / "README.md").write_text("\n".join(lines) + "\n")
    print(root / "README.md")


if __name__ == "__main__":
    main()

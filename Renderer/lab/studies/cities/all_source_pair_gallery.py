#!/usr/bin/env python3
"""Index every source-art city audition without pre-assigning Civ III eras."""

import json
from pathlib import Path

from PIL import Image, ImageDraw

from Renderer.lab.studies.cities.all_source_pair_review import OUT, ROOT
from Renderer.lab.studies.cities.medieval_family_review import slug
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


def facade_comparison(output):
    families = ("vietnam", "vikings", "maori")
    width, height, header, gap = 760, 480, 42, 12
    image = Image.new("RGB", (3*(width+gap)-gap, 2*(height+header+gap)-gap),
                      (31, 25, 38))
    draw = ImageDraw.Draw(image)
    for column, family in enumerate(families):
        with Image.open(OUT / "review" / "unspecified" / family / "sheet.png") as sheet:
            for row, (label, start) in enumerate((("City base", 154),
                                                   ("City capital", 1674))):
                x, y = column*(width+gap), row*(height+header+gap)
                draw.text((x+12, y+9), f"{family.title()} | {label}",
                          font=font(20), fill=(249, 236, 249))
                image.paste(sheet.crop((start, 618, start+width, 1098)),
                            (x, y+header))
    image.save(output)


def correction_review(output):
    examples = (
        ("classical", "america", "America | City base", False),
        ("classical", "america", "America | City capital", True),
        ("classical", "baltic", "Baltic | City base", False),
        ("classical", "scottish", "Scottish | City base", False),
        ("classical", "vietnam", "Vietnam | Classical City base", False),
        ("industrial", "default", "Industrial Default | City base", False),
        ("industrial", "rowhouse", "Industrial RowHouse | City base", False),
        ("unspecified", "vikings", "Vikings | City capital", True),
        ("unspecified", "maori", "Māori | City capital", True),
    )
    width, height, header, gap = 760, 480, 42, 12
    image = Image.new("RGB", (3*(width+gap)-gap,
                              3*(height+header+gap)-gap), (31, 25, 38))
    draw = ImageDraw.Draw(image)
    for slot, (era, family, label, capital) in enumerate(examples):
        row, column = divmod(slot, 3)
        x, y = column*(width+gap), row*(height+header+gap)
        draw.text((x+12, y+9), label, font=font(20), fill=(249, 236, 249))
        with Image.open(OUT / "review" / era / family / "sheet.png") as sheet:
            start = 1674 if capital else 154
            image.paste(sheet.crop((start, 618, start+width, 1098)),
                        (x, y+header))
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
    facade_comparison(root / "se-sw-facade-city-comparison.png")
    correction_review(root / "alignment-and-foundations-review.png")
    (root / "index.json").write_text(json.dumps(collected, indent=2) + "\n")
    lines = ["# Civ VI city source-art auditions", "",
             "These 46 sheets cover every populated `Tag_Culture × Tag_Era` city-building",
             "pair in the installed source inventory. They are **source-art comparisons**,",
             "not final Civ III culture or era assignments. The `Tag_Era=DEFAULT` pool is",
             "era-unspecified. Some tags share identical building sets; they remain separate",
             "here because their source selectors and palaces can differ.", "",
             "Every full magenta sheet shows Town with Base, Walls, Capital, and Walls +",
             "Capital. City and Metropolis show Base and Capital; their wall columns",
             "are marked unavailable because walls apply only to Towns. Their sprawl",
             "is unchanged. Town wall kits follow the Civ III target era: Ancient Walls,",
             "Castle, Tsikhe, and the lower spike-free Modern Tower Defense variant.",
             "Where the source has no matching palace, the sheet uses a generic comparison",
             "palace; its status is recorded below. These software previews compare layout",
             "and source models, not native D3D material fidelity or in-game acceptance.",
             "The rejected count records pieces the importer could not safely compile;",
             "the Future pool omits seven unresolved effect-bearing blocks per tag.",
             "For façade alignment, the era-unspecified Vietnam sheet supplements its",
             "combined blocks with individual Vietnam houses from the Classical source",
             "pool. The era-unspecified Vikings and Māori sheets use individual houses",
             "from their own pools instead of mixed-facing multi-house blocks.",
             "No city ground decal or elevated masonry is added by these recipes.",
             "The offline foundation-free study packs remove buried source geometry.",
             "A few exposed source pedestals use per-entry grade cuts; source packs",
             "remain untouched, and the runtime consumes ordinary normalized meshes.",
             "The five-culture Industrial/Modern compositions keep their mapped",
             "Industrial house base. Industrial City/Metropolis swap two/four inner",
             "plots for ordinary ARTERA_MODERN midrises. Modern places one/two",
             "distinct ModernGlass towers. Ordinary cities replace the center plot",
             "with glass; capitals place glass behind the palace in a ten-to-two",
             "o'clock screen arc. Each accent asset appears once per city variant.",
             "Culture-family houses surround downtown. The raw ARTERA_MODERN",
             "source sheets below remain unaltered source-art auditions. Towns retain",
             "their original houses. Skyline and farm-tree plot order use a stable seed",
             "specific to the source family and chosen variation; collision-checked",
             "placements change between seeds without reshuffling on redraw.", "",
             "### Seeded skyline and vegetation comparisons", "",
             "![Industrial City and Metropolis seed comparison](industrial-seeded-skyline-and-trees.png)", "",
             "![Modern source-family house and tree seed comparison](modern-seeded-source-art-and-trees.png)", "",
             "Three authored candidate seeds now cover every source-art pair and",
             "population/capital combination. Seed 0 remains the main full-sheet",
             "gallery; the other layouts change a few ordinary houses and trees",
             "while preserving civic cores and special downtown buildings.", "",
             "![City recipe variation examples](seeded-variant-examples.png)", "",
             "[Variant candidate manifest](seeded-variant-manifest.json)", "",
             "### Five Civ III culture candidates", "",
             "The current candidate family mapping is shown at seed 0. Modern",
             "inherits the mapped Industrial outer-house family and",
             "adds taller ModernGlass towers downtown. The mapping and placements",
             "remain reviewable Lab metadata.", "",
             "![Five culture Industrial and Modern city candidates](civ3-five-culture-late-era-candidates.png)", "",
             "![Modern capital City and Metropolis downtowns](modern-capital-downtown-comparison.png)", "",
             "| Civ III culture | Industrial full sheet | Modern full sheet |",
             "| --- | --- | --- |"]
    strategy = json.loads((ROOT / "Renderer/tools/asset_compiler/city_render_strategy.json")
                          .read_text(encoding="utf-8"))
    for style in sorted(strategy["styles"], key=lambda item: item["civ3_culture_group"]):
        name = style["id"]
        directory = slug(name)
        lines.append(f"| {name.replace('_', ' ').title()} | "
                     f"[view sheet](civ3-culture-compositions/{directory}/industrial/sheet.png) | "
                     f"[view sheet](civ3-culture-compositions/{directory}/modern/sheet.png) |")
    lines.extend(["",
             "### SE/SW façade comparison", "",
             "![Vietnam, Vikings, and Māori City bases and capitals](se-sw-facade-city-comparison.png)", "",
             "### Alignment and foundation review", "",
             "![Nine corrected City examples](alignment-and-foundations-review.png)", "",
             "[Source-art era inventory](../../../../studies/cities/source_era_inventory.md)", ""])
    for era in ERAS:
        entries = [entry for entry in collected
                   if entry["source_art_era"] == ("DEFAULT" if era == "unspecified"
                                                  else "ARTERA_" + era.upper())]
        lines.extend([f"## {LABELS[era]} ({len(entries)})", "",
                      f"![{LABELS[era]} City overview]({era}-city-overview.png)", "",
                      "| Source culture tag | Full variant sheet | Palace | Façade pieces | Usable pieces | Rejected pieces |",
                      "| --- | --- | --- | --- | ---: | ---: |"])
        for entry in entries:
            lines.append(f"| `{entry['source_culture']}` | [view sheet]({entry['sheet'].removeprefix('review/')}) | "
                         f"{entry.get('palace_status', 'previous Classical candidate')} | "
                         f"{entry.get('facade_source', 'existing Classical composition')} | "
                         f"{entry['selected_components']} | {entry['rejected_components']} |")
        lines.append("")
    (root / "README.md").write_text("\n".join(lines) + "\n")
    print(root / "README.md")


if __name__ == "__main__":
    main()

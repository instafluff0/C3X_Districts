#!/usr/bin/env python3
"""Inventory installed Civ VI named volcanoes and volcano art without extraction."""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import re
import struct
import xml.etree.ElementTree as ET


DEFAULT_ASSETS = (Path.home() / "Library/Application Support/Steam/steamapps/common/"
                  "Sid Meier's Civilization VI/Civ6.app/Contents/Assets")
KEYS = ("VOLCANO", "VESUVIUS", "KILIMANJARO", "EYJAFJALLAJOKULL",
        "FUJI", "ASAMA", "HALEAKALA")


def source_label(path: Path, assets: Path) -> str:
    parts = path.relative_to(assets).parts
    return "Base" if parts[0] == "Base" else parts[1] if parts[0] == "DLC" else parts[0]


def inventory(assets: Path) -> str:
    if not assets.is_dir():
        raise FileNotFoundError(assets)
    names: dict[str, set[str]] = defaultdict(set)
    name_tags: dict[str, str] = {}
    types: dict[str, set[str]] = defaultdict(set)
    localizations: dict[str, str] = {}
    volcano_features: dict[str, set[str]] = defaultdict(set)
    natural_wonders: dict[str, set[str]] = defaultdict(set)
    parsed = 0
    for path in sorted(assets.rglob("*.xml")):
        if "/Data/" not in path.as_posix() and "/Text/en_US/" not in path.as_posix():
            continue
        try:
            root = ET.parse(path).getroot()
        except ET.ParseError:
            continue
        parsed += 1
        origin = source_label(path, assets)
        for section in root.iter():
            if section.tag == "NamedVolcanoes":
                for row in section.findall("Row"):
                    key = row.get("NamedVolcanoType")
                    if key:
                        names[key].add(origin)
                        name_tags[key] = row.get("Name", "")
            elif section.tag == "Types":
                for row in section.findall("Row"):
                    if row.get("Kind") == "KIND_NAMED_VOLCANO":
                        types[row.get("Type", "")].add(origin)
            elif section.tag == "Features_XP2":
                for row in section.findall("Row"):
                    if row.get("Volcano", "").lower() == "true":
                        volcano_features[row.get("FeatureType", "")].add(origin)
            elif section.tag == "Features":
                for row in section.findall("Row"):
                    if row.get("NaturalWonder", "").lower() == "true":
                        natural_wonders[row.get("FeatureType", "")].add(origin)
            if "/Text/en_US/" in path.as_posix() and section.tag == "Row":
                tag = section.get("Tag")
                value = section.findtext("Text")
                if tag and value:
                    localizations[tag] = value.strip()

    art: dict[str, set[str]] = defaultdict(set)
    art_files = 0
    for path in sorted(assets.rglob("*.artdef")):
        art_files += 1
        body = path.read_text(errors="replace")
        for value in re.findall(r'<m_(?:EntryName|Name) text="([^"]+)"', body):
            if any(key in value.upper() for key in KEYS[:4]):
                art[value].add(path.relative_to(assets).as_posix())

    # The BLP header gives the small package-data section. Shared texture data
    # can be large, so this scans metadata only and never extracts any art.
    packages: dict[str, list[str]] = {}
    blp_count = 0
    for path in sorted(assets.rglob("*.blp")):
        blp_count += 1
        with path.open("rb") as stream:
            header = stream.read(28)
            if len(header) < 28 or header[:6] != b"CIVBLP":
                continue
            offset, size = struct.unpack_from("<II", header, 8)
            if size > 256 * 1024 * 1024:
                raise ValueError(f"Unexpected package metadata size: {path}")
            stream.seek(offset)
            body = stream.read(size)
        hits = sorted({match.decode("ascii", errors="ignore")
                       for match in re.findall(rb"[\x20-\x7e]{5,}", body)
                       if any(key.encode() in match.upper() for key in KEYS)})
        if hits:
            packages[path.relative_to(assets).as_posix()] = hits

    lines = [
        "# Installed Civ VI volcano inventory",
        "",
        "Read-only scan of installed Data XML, English localization, ArtDefs, and BLP",
        "package metadata. IDs and localized names are gameplay labels; they do not",
        "establish distinct geometry. The installed files are local prototype inputs.",
        "",
        f"Scanned {parsed} Data/English Text XML files, {art_files} ArtDefs, and {blp_count} BLP packages.",
        "",
        "## Ordinary named volcanoes",
        "",
        f"{len(names)} `NamedVolcanoes` rows, {len(types)} `KIND_NAMED_VOLCANO` types."
        " Names have no matching per-name terrain ArtDef entries.",
        "By data source: " + ", ".join(f"{source} {sum(source in origins for origins in names.values())}"
                                          for source in sorted(set().union(*names.values()))) + ".",
        "",
        "| Name | ID | DLC/data source |",
        "| --- | --- | --- |",
    ]
    for key in sorted(names, key=lambda item: localizations.get(name_tags.get(item, ""), item).casefold()):
        title = localizations.get(name_tags.get(key, ""), f"[{name_tags.get(key, 'no English text')}]")
        lines.append(f"| {title.replace('|', '/')} | `{key}` | {', '.join(sorted(names[key]))} |")
    if set(names) != set(types):
        lines += ["", f"Type/table difference: rows without Types={sorted(set(names)-set(types))}; "
                  f"Types without rows={sorted(set(types)-set(names))}."]
    lines += [
        "", "## Volcano gameplay features", "",
        "`Features_XP2.Volcano=true` flags these features:", "",
    ]
    for key in sorted(volcano_features):
        lines.append(f"- `{key}` ({', '.join(sorted(volcano_features[key]))}); "
                     f"NaturalWonder={key in natural_wonders}.")
    lines += [
        "", "Eyjafjallajökull, Kilimanjaro, and Vesuvius are natural-wonder features and should be",
        "reserved for the deferred natural-wonder renderer work.",
        "", "## Art bindings and package evidence", "",
        "The ordinary `FEATURE_VOLCANO` TerrainStyle ArtDef binds one element",
        "(`ART_DEF_TERRAIN_ELEMENT_FEATURE_VOLCANO_01`) and one terrain asset",
        "(`ART_DEF_TERRAIN_ASSET_FEATURE_VOLCANO_01`). The Expansion2 terrain",
        "element package exposes `Feature_Volcano_HM_0/1`, `HBlend_0/1`, and",
        "`ID_0/1` fields: two resolution levels of one named ordinary form,",
        "rather than separate Fuji, Asama, or Haleakalā variants.",
        "Eyjafjallajökull, Kilimanjaro, and Vesuvius have separate `NWON_*`",
        "terrain elements/fields.",
        "The ordinary feature is valid on snow mountains, but the installed",
        "ArtDefs and BLP metadata show no separate ordinary snow-volcano",
        "height field or material. The Lab snow treatment is deferred.",
        "", "Relevant ArtDef entries:", "",
    ]
    for key in sorted(art):
        if key.startswith("ART_DEF_TERRAIN_") or key.startswith("FEATURE_"):
            lines.append(f"- `{key}`: {', '.join(sorted(art[key]))}")
    lines += ["", "BLP package metadata strings containing volcano/name keys:", ""]
    for path, hits in packages.items():
        lines.append(f"- `{path}`: {', '.join(f'`{value}`' for value in hits)}")
    lines += [
        "", "The BLP scan reads package metadata, not all texture blob payloads.",
        "No distinct ordinary named-volcano terrain element or asset appears",
        "in the installed ArtDefs or BLP package metadata. The individual",
        "names are available for design inspiration; their unique shapes would",
        "have to be authored or derived in C3X Lab.", "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, default=DEFAULT_ASSETS)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("CIV6_INVENTORY.md"))
    args = parser.parse_args()
    args.output.write_text(inventory(args.assets))
    print(args.output)


if __name__ == "__main__":
    main()

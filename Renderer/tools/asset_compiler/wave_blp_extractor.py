"""Decode the installed single-entry coastline-wave package; keep licensed art local.

This deliberately bounded profile validates pointers, dimensions and complete mip
payloads. It does not claim to decode the source engine's spline or shader code.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import struct
import sys
import xml.etree.ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from Renderer.tools.asset_compiler import civblp_probe as probe
from Renderer.tools.asset_compiler.c3x_asset_compiler import make_dds_dx10_header
from Renderer.tools.asset_compiler.indexed_static_package import IndexedStaticPackage

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ASSETS = Path.home() / "Library/Application Support/Steam/steamapps/common/Sid Meier's Civilization VI/Civ6.app/Contents/Assets"
DEFAULT_OUTPUT = ROOT / "Renderer/lab/out/waves/source"


def digest(data):
    return hashlib.sha256(data).hexdigest()


class WavePackage(IndexedStaticPackage):
    def _infer_package_base(self, temp_base, target):
        # Unlike landmark libraries this package has only one allocated string.
        # Do not relax the shared multi-string decoder's validation threshold.
        candidates = []
        for pointer, allocation in enumerate(self.allocations, 1):
            if (allocation["stripe"] == 0 and allocation["parent_pointer"] == 0
                    and allocation["size"] == len(target) + 9
                    and probe.resolve_type_name(allocation["type_pointer"], self.package,
                        self.allocations, {1: temp_base}) == "char"):
                for position in probe.raw_occurrences(self.package, target.encode() + b"\0"):
                    base = position - 8 - allocation["target_offset"]
                    candidates.append((base, pointer))
        if len(candidates) != 1:
            raise ValueError("Wave single-string stripe is ambiguous")
        base, pointer = candidates[0]
        if base < 0:
            raise ValueError("Wave stripe precedes package")
        return base, pointer, ["single allocated name; validated by typed entry graph and payloads"]


def rgba_dds(width, height, mips, payload):
    # R16G16 and RGBA8 both have four bytes/texel; preserve source RGBA8 format.
    header = bytearray(make_dds_dx10_header(dict(width=width, height=height,
                                                mip_count=mips, dxgi_format=35)))
    struct.pack_into("<I", header, 128, 28)
    return bytes(header) + payload


def decode(source):
    package = WavePackage(source, "WaveTest")
    entry = package.unique_allocation("CoastlineWaves::PackageEntry")
    raw = package.bytes_for(entry)
    if len(raw) != 128 or package.direct_string(struct.unpack_from("<Q", raw, 0x38)[0]) != "WaveTest":
        raise ValueError("Unsupported wave entry layout")
    columns, rows, pages, delays_per_page = struct.unpack_from("<4I", raw, 0x48)
    if (columns, rows, pages, delays_per_page) != (8, 2, 16, 512):
        raise ValueError("Unsupported crest atlas layout")
    atlas, auxiliary, delays = struct.unpack_from("<3Q", raw, 0x58)
    table = package.bytes_for(delays)
    if package.type_name(delays) != "float" or len(table) != pages * delays_per_page * 4:
        raise ValueError("Invalid wave delay table")
    if struct.unpack_from("<Q", raw, 0x78)[0] != pages * delays_per_page:
        raise ValueError("Wave vector count disagrees with allocation")
    textures = {}
    for role, pointer in (("crest", atlas), ("auxiliary", auxiliary)):
        if not 1 <= pointer <= len(package.allocations):
            raise ValueError("Wave texture pointer escapes allocation graph")
        allocation = package.allocations[pointer - 1]
        parent = allocation["parent_pointer"]
        if package.type_name(pointer) != "BLP::TextureEntry" or not parent:
            raise ValueError("Wave texture does not reference typed array")
        record = package.array_element(parent, allocation["target_offset"])
        fmt, height, width, depth, array_size, mips = struct.unpack_from("<6H", record, 0x58)
        offset, size = struct.unpack_from("<2Q", record, 0x20)
        expected = sum(max(1, width >> level) * max(1, height >> level) * 4 for level in range(mips))
        if (fmt, depth, array_size) != (28, 1, 1) or size != expected:
            raise ValueError("Unsupported embedded texture layout/mip length")
        if (width, height, mips) != ((1024, 1024, 11) if role == "crest" else (512, 256, 10)):
            raise ValueError("Unexpected wave texture dimensions")
        payload = package.big_data(offset, size)
        textures[role] = dict(width=width, height=height, mip_count=mips,
            dxgi_format=fmt, payload_bytes=size, payload_sha256=digest(payload),
            entry_pointer=pointer, big_data_offset=offset, payload=payload)
    return package, dict(columns=columns, rows=rows, pages=pages,
                         delays_per_page=delays_per_page), textures, table


def artdef(path):
    values = {}
    tree = ET.parse(path)
    for element in tree.findall(".//m_Values/Element"):
        key = element.find("m_ParamName")
        if key is None:
            continue
        value = {child.tag: child.attrib.get("text", child.text) for child in element
                 if child.tag != "m_ParamName"}
        values.setdefault(key.attrib["text"], []).append(value)
    return values


def extract(assets=DEFAULT_ASSETS, output=DEFAULT_OUTPUT):
    relative = Path("Base/Platforms/Windows/BLPs/Wave.blp")
    package, layout, textures, table = decode(assets / relative)
    output.mkdir(parents=True, exist_ok=True)
    metadata = {}
    for role, texture in textures.items():
        data = rgba_dds(texture["width"], texture["height"], texture["mip_count"], texture["payload"])
        (output / (role + ".dds")).write_bytes(data)
        metadata[role] = {key: value for key, value in texture.items() if key != "payload"}
        metadata[role].update(texture=role + ".dds", sha256=digest(data))
    (output / "crest-delays.f32").write_bytes(table)
    # Generic data contract, separate from source-specific audit/provenance.
    # No runtime behavior is enabled merely by recovering a source texture.
    normalized = dict(schema="c3x.coastal_wave_assets.v1", enabled=False,
        atlas=dict(texture="crest.dds", columns=layout["columns"], rows=layout["rows"],
                   variants=layout["pages"], across_texels=128, along_texels=512),
        auxiliary=dict(texture="auxiliary.dds"),
        crest_delays=dict(file="crest-delays.f32", samples_per_variant=layout["delays_per_page"],
                          encoding="little-endian float32", inactive="FLT_MAX"),
        channels=dict(crest_rgb="crest intensity; C3X interpretation", crest_alpha="preserved; unresolved",
                      auxiliary_rgb="fine foam detail; C3X interpretation", auxiliary_alpha="constant one"),
        animation="absolute presentation time and stable contour instance ID; binding pending")
    (output / "wave.json").write_text(json.dumps(normalized, indent=2) + "\n")
    delays = struct.unpack("<8192f", table)
    active = [value for value in delays if value < 1e30]
    if any(not 0 <= value <= 1 for value in active):
        raise ValueError("Unexpected active crest delay")
    report = dict(schema="c3x.wave_source_audit.v1", source=relative.as_posix(),
        source_sha256=digest(package.data), entry="WaveTest", layout=layout, textures=metadata,
        delays=dict(file="crest-delays.f32", sha256=digest(table), count=len(delays),
                    active=len(active), inactive=len(delays)-len(active), minimum=min(active),
                    maximum=max(active), inactive_encoding="FLT_MAX; preserve as sentinel"),
        artdefs={name: artdef(assets / "Base/ArtDefs" / name)
                 for name in ("Wave.artdef", "Water.artdef", "WaterMaterials.artdef")},
        unresolved=["source shader equations and alpha meaning", "spline generation/assignment",
                    "distance and time units in engine", "exact use of crest delays during crash"])
    # Inventory all installed wave/foam names, including false positives and
    # deferred wonder/flood art. Inventory is not a claim that all belong to surf.
    report["related_files"] = [dict(path=p.relative_to(assets).as_posix(), bytes=p.stat().st_size,
                                     sha256=digest(p.read_bytes()))
        for p in sorted(assets.rglob("*")) if p.is_file()
        and ("wave" in p.name.lower() or "foam" in p.name.lower())]
    water_relative = Path("Base/Platforms/Windows/BLPs/Water.blp")
    water_data = (assets / water_relative).read_bytes()
    names = [s.decode("ascii") for s in re.findall(rb"[ -~]{6,}", water_data)]
    report["water_package_inventory"] = dict(source=water_relative.as_posix(), sha256=digest(water_data),
        evidence="printable resource/type inventory only, not parameter decoding",
        names=[name for name in names if any(token in name for token in
               ("Water/", "WhiteCap", "LeanMap", "Water_Bumps", "Density_"))])
    (output / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, default=DEFAULT_ASSETS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = extract(args.assets, args.output)
    print(json.dumps({key: result[key] for key in ("layout", "delays")}, indent=2))

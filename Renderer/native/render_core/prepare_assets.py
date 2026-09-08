"""Build the current local hill/cliff pack from curated source inputs.

Only bundle paths and the gloss DDS view format are rewritten. Geometry and
compressed texture payloads remain exact. No historical handoff is required.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "Renderer/packs/TerrainProfileSources/current"
OUTPUT = ROOT / "Renderer/packs/TerrainProfileR1"


def local(path, root=ROOT):
    path = Path(path).resolve()
    path.relative_to(root.resolve())
    return path


def bundle_paths(data):
    if len(data) < 24 or data[:8] != b"C3XVEG1\0" or struct.unpack_from("<4I", data, 8) != (1, 24, 6, 1):
        raise ValueError("Expected the six-body, four-channel cliff bundle")
    cursor, paths = 24, []
    for _ in range(24):
        if cursor + 4 > len(data):
            raise ValueError("Truncated cliff path table")
        length, = struct.unpack_from("<I", data, cursor)
        cursor += 4
        if not 0 < length <= 4096 or cursor + length > len(data):
            raise ValueError("Invalid cliff path length")
        paths.append(data[cursor:cursor + length].decode())
        cursor += length
    return paths, cursor


def rebind(data, names, cursor):
    rewritten = bytearray(data[:24])
    for name in names:
        encoded = name.encode()
        rewritten += struct.pack("<I", len(encoded)) + encoded
    return bytes(rewritten) + data[cursor:]


def import_sources(height, cliffs, destination=SOURCE):
    """Preserve a selected normalized bundle; refuse differing existing inputs.

    Incoming texture names are repository-relative. Preserved names are relative
    to the new source directory, so old fixture trees can be retired.
    """
    destination = local(destination)
    data = local(cliffs).read_bytes()
    paths, cursor = bundle_paths(data)
    names = [f"textures/cliff_{index//4}_{index%4}.dds" for index in range(24)]
    payloads = {"height.dds": local(height).read_bytes(),
                "cliffs.bin": rebind(data, names, cursor)}
    for path, name in zip(paths, names):
        payloads[name] = local(ROOT / path).read_bytes()
    # Check every existing target before writing any source bytes.
    for name, payload in payloads.items():
        target = local(destination / name, destination)
        if target.exists() and target.read_bytes() != payload:
            raise ValueError("Refusing to replace different preserved input: " + name)
    for name, payload in payloads.items():
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.write_bytes(payload)
    print(f"Preserved {len(payloads)} current hill/cliff inputs")


def prepare(source=SOURCE, output=OUTPUT):
    source, output = local(source), local(output)
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("Generated output must not overlap preserved source inputs")
    records = []

    def copy(name, destination, view_format=None):
        path = local(source / name, source)
        original = path.read_bytes()
        data = original
        if view_format is not None:
            if len(data) < 148 or data[:4] != b"DDS " or data[84:88] != b"DX10":
                raise ValueError("Invalid cliff DDS: " + name)
            changed = bytearray(data)
            struct.pack_into("<I", changed, 128, view_format)
            data = bytes(changed)
        target = local(output / destination, output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        records.append({"path": destination, "sha256": hashlib.sha256(data).hexdigest(),
                        "source": path.relative_to(ROOT).as_posix(),
                        "source_sha256": hashlib.sha256(original).hexdigest()})
        return data

    copy("height.dds", "height.dds")
    data = copy("cliffs.bin", "cliffs.bin")
    paths, cursor = bundle_paths(data)
    names = []
    for index, path in enumerate(paths):
        name = f"textures/cliff_{index//4}_{index%4}.dds"
        copy(path, name, 71 if index % 4 == 3 else None)
        names.append(name)
    rewritten = rebind(data, names, cursor)
    (output / "cliffs.bin").write_bytes(rewritten)
    records[1]["sha256"] = hashlib.sha256(rewritten).hexdigest()
    record = {"schema": "c3x.terrain_profile.v1", "revision": 1,
              "files": records, "source_mesh_bytes_preserved": True}
    (output / "manifest.json").write_text(json.dumps(record, indent=2) + "\n")
    print(f"Prepared {len(records)} current assets in {output.relative_to(ROOT)}")
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--import-height", type=Path)
    parser.add_argument("--import-cliffs", type=Path)
    args = parser.parse_args()
    if bool(args.import_height) != bool(args.import_cliffs):
        parser.error("Provide both --import-height and --import-cliffs")
    if args.import_height:
        import_sources(args.import_height, args.import_cliffs)
    prepare(output=args.output)

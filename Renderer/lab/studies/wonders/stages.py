"""Construction stages of a Civ VI wonder, sampled from its own build movie (Lab only).

A WonderMovie.blp model is a build movie: every piece (block, brick, scaffold
pole, crane) is a rigid mesh on its own bone of the movie skeleton, and props
(cranes, brick and wood piles) are attached to bones. The movie animation
(e.g. WON_Pyramids_build) moves, raises, scales in and removes those bones,
so posing the skeleton at a fraction of the movie gives that stage of the
construction site.

The importer keeps each piece's mesh index; the mesh -> bone table is the
base model's int32 array with one entry per mesh (FGX behaviour data), read
here straight from the package. The animation is the base model's
AnimationDesc entry in the package's BLP::AnimationEntry table, converted on
the Windows VM by Renderer/tools/asset_compiler/CONVERT_CIV6_ANIMATION.bat
into a source-independent .c3anim and sampled on macOS.

    python3 Renderer/lab/studies/wonders/stages.py extract --source pyramids
    python3 Renderer/lab/studies/wonders/stages.py convert --source pyramids
    python3 Renderer/lab/studies/wonders/stages.py probe --source pyramids [--at 0 .25 .5 .75 1]
"""
from __future__ import annotations

import argparse
import json
import math
import struct
import sys
from functools import lru_cache
from pathlib import Path, PureWindowsPath

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from Renderer.lab.studies.wonders import kits
from Renderer.lab.studies.districts import kits as dk
from Renderer.tools.asset_compiler import clutter_blp_extractor as cbe
from Renderer.tools.asset_compiler import normalized_animation as na
from Renderer.tools.asset_compiler import normalized_skin as ns
from Renderer.tools.asset_compiler.artdef_graph_resolver import DEFAULT_ASSETS_ROOT
from Renderer.tools.asset_compiler.indexed_static_package import IndexedStaticPackage

STAGES = kits.OUT / "stages"
# WonderMovie sources are 100 units per tile (the importer's scale for these packages).
TRANSLATION_SCALE = 0.01


def folder(key: str) -> Path:
    return STAGES / key


def package_path(key: str) -> Path:
    record = kits.record(key)
    kits.use(key)
    resolved = kits.compiler(key).resolve(record)
    return Path(resolved) if Path(resolved).is_absolute() else DEFAULT_ASSETS_ROOT / resolved


def extract(key: str) -> dict:
    """Write the build animation's raw Granny payload and the mesh -> bone table."""
    entry = kits.record(key)["entry"]
    saved = cbe.TYPE_CITY_BLOCK
    cbe.TYPE_CITY_BLOCK = "WonderMoviePackageEntry"
    try:
        try:
            package = IndexedStaticPackage(package_path(key), entry)
        except ValueError:
            # Some Workshop packages need the relaxed temp-stripe inference.
            package = IndexedStaticPackage(package_path(key), entry, minimum_temp_support=2)
        _landmark, _user, base = cbe.landmark_base_model(package)
    finally:
        cbe.TYPE_CITY_BLOCK = saved
    mesh_array = package.pointer_fields(base, cbe.TYPE_MESH)[0][1]
    mesh_count = package.allocations[mesh_array - 1]["element_count"]
    tables = [pointer for _offset, pointer in package.pointer_fields(base, "int32")
              if package.allocations[pointer - 1]["element_count"] == mesh_count]
    if len(tables) != 1:
        raise ValueError(f"{key}: expected one per-mesh bone table, found {len(tables)}")
    raw = package.bytes_for(tables[0])
    mesh_bones = list(struct.unpack(f"<{mesh_count}i", raw))
    descriptors = package.pointer_fields(base, "FGXModelFramework::BehaviorDesc::AnimationDesc")
    tables = [pointer for pointer in range(1, len(package.allocations) + 1)
              if package.type_name(pointer) == "BLP::AnimationEntry"]
    if len(descriptors) != 1 or len(tables) != 1:
        raise ValueError(f"{key}: expected one animation descriptor and table")
    described = package.bytes_for(descriptors[0][1])
    indices = sorted({struct.unpack_from("<I", described, offset + 4)[0] for offset in range(0, len(described), 8)})
    clips = []
    target = folder(key)
    target.mkdir(parents=True, exist_ok=True)
    for index in indices:
        record = package.array_element(tables[0], index)
        name = package.string_value(struct.unpack_from("<Q", record, 8)[0])
        offset, size = struct.unpack_from("<QQ", record, 0x20)
        payload = package.big_data(offset, size)
        if payload[:16].hex() != "e59b495e6f631f141e13eba990beedc4":
            raise ValueError(f"{key}: animation {name} is not a raw Granny payload")
        (target / f"{name}.fgx").write_bytes(payload)
        clips.append(name)
    info = {"key": key, "entry": entry, "mesh_bones": mesh_bones, "clips": clips}
    (target / "stages.json").write_text(json.dumps(info) + "\n")
    return {"key": key, "meshes": mesh_count, "clips": clips}


def _vm(relative_cwd: str, command: str) -> dict:
    """One VM command, retried on the transient Parallels transport failures."""
    from Renderer.lab import platform
    for _attempt in range(4):
        result = platform.native_command_result(relative_cwd, command, timeout_seconds=900)
        if "PrlJob" not in result["output_tail"]:
            break
    return result


def convert(key: str) -> None:
    """Convert each extracted clip on the Windows VM (shared checkout): its
    transform tracks to .c3anim, and the per-bone visibility vector tracks (which
    that converter does not read) to a text timeline."""
    info = json.loads((folder(key) / "stages.json").read_text())
    for name in info["clips"]:
        # The batch runs in Renderer/tools/asset_compiler; paths are relative to it.
        relative = "..\\..\\..\\" + str(PureWindowsPath((folder(key) / f"{name}.fgx").relative_to(ROOT)))
        output = relative.removesuffix(".fgx") + ".c3anim"
        result = _vm("Renderer/tools/asset_compiler",
                     f'CONVERT_CIV6_ANIMATION.bat "{relative}" "{output}" {TRANSLATION_SCALE}')
        if result["status"] != "pass":
            raise ValueError(f"{key}: conversion of {name} failed: {result['output_tail'][-600:]}")
        study = "..\\..\\" + str(PureWindowsPath((folder(key) / name).relative_to(ROOT / "Renderer/lab")))
        result = _vm("Renderer/lab/studies/wonders", f'EXPORT_VISIBILITY.bat "{study}.fgx" "{study}.visibility.txt"')
        if result["status"] != "pass":
            raise ValueError(f"{key}: visibility export of {name} failed: {result['output_tail'][-600:]}")


@lru_cache(maxsize=None)
def movie(key: str):
    """(info, clip, skeleton in normalized_skin form, track group index)."""
    info = json.loads((folder(key) / "stages.json").read_text())
    _asset, document, _transform = kits.landmarks(key)[0]
    binding = max(document["animation_binding"], key=lambda item: item["skeleton"])
    skeleton = json.loads((dk.SOURCES / document["components"]["skeletons"][binding["skeleton"]]).read_text())
    for bone in skeleton["bones"]:
        bone["local"] = bone["rest"]
    names = {bone["name"] for bone in skeleton["bones"]}
    # A wonder may carry more than one clip (Ruhr Valley's cart loop): the build
    # movie is the clip whose track group drives most of the movie skeleton.
    best = None
    for name in info["clips"]:
        if not (folder(key) / f"{name}.c3anim").is_file():
            continue
        clip = na.load_clip(folder(key) / f"{name}.c3anim")
        for index, group in enumerate(clip.groups):
            score = sum(track.name in names for track in group.tracks)
            if best is None or score > best[0]:
                best = (score, name, clip, index)
    _score, name, clip, group = best
    info = {**info, "clip": name}
    return info, clip, skeleton, group, binding["skeleton"]


@lru_cache(maxsize=None)
def visibility(key: str) -> dict:
    """bone name -> [(frame, value)] changes of its "<bone>_visibility" track."""
    name = movie(key)[0]["clip"]
    tracks = {}
    for line in (folder(key) / f"{name}.visibility.txt").read_text().splitlines()[1:]:
        # A bone name may itself hold a tab (the Oracle's "2\t"): split from both ends.
        group, rest = line.split("\t", 1)
        track, _dimension, changes = rest.rsplit("\t", 2)
        tracks[track.removesuffix("_visibility")] = [(int(frame), float(value.split(",")[0]))
                                                       for frame, value in (item.split(":") for item in changes.split())]
    return tracks


def _level(changes: list, frame: int) -> bool:
    value = changes[0][1]
    for at, level in changes:
        if at > frame:
            break
        value = level
    return value > .5


@lru_cache(maxsize=None)
def last_frame(key: str) -> int:
    """The movie's finished frame. Many movies swap the finished, lit building in
    on their very last frame (the Temple of Artemis' "Temple_Of_Artemis" piece
    carries the night lights); the exact end sample of others evaluates nearly
    empty (a curve endpoint), and then the frame before it is the finished wonder.
    Decided by the vertices each frame shows."""
    info, clip, skeleton, _group, skeleton_index = movie(key)
    _asset, document, _transform = kits.landmarks(key)[0]
    tracks = visibility(key)
    frames = clip.frame_count

    def shown(frame: int) -> int:
        total = 0
        for binding in document["draw_bindings"]:
            if binding["skeleton"] != skeleton_index or "worked" not in binding["states"]:
                continue
            changes = tracks.get(skeleton["bones"][info["mesh_bones"][binding["mesh"]]]["name"])
            if not changes or _level(changes, frame):
                geometry = document["components"]["geometry"][binding["geometry"]]
                total += len(kits._geometry(str(dk.SOURCES / geometry))["vertices"])
        return total
    return frames - 1 if shown(frames - 1) >= .5 * shown(frames - 2) else frames - 2


def visible(key: str, fraction: float, bone_name: str) -> bool:
    changes = visibility(key).get(bone_name)
    if not changes:
        return True
    frames = movie(key)[1].frame_count
    return _level(changes, min(round(fraction * (frames - 1)), last_frame(key)))


@lru_cache(maxsize=64)
def skin_matrices(key: str, fraction: float) -> tuple:
    info, clip, skeleton, group, _ = movie(key)
    pose = ns.sample_pose(skeleton, clip, group, fraction * clip.duration, False)
    worlds = ns.world_matrices(skeleton, pose)
    return tuple(ns._multiply(bone["inverse_bind_matrix"], world)
                 for bone, world in zip(skeleton["bones"], worlds))


def piece_matrix(key: str, fraction: float, mesh: int) -> tuple | None:
    """The skin matrix of a piece's bone at a fraction of the movie, or None when
    the movie hides the piece then (its bone scaled to nothing)."""
    info, _clip, skeleton = movie(key)[:3]
    bone = info["mesh_bones"][mesh]
    if not visible(key, fraction, skeleton["bones"][bone]["name"]):
        return None
    matrix = skin_matrices(key, fraction)[bone]
    size = abs(matrix[0] * (matrix[5] * matrix[10] - matrix[6] * matrix[9])
               - matrix[1] * (matrix[4] * matrix[10] - matrix[6] * matrix[8])
               + matrix[2] * (matrix[4] * matrix[9] - matrix[5] * matrix[8]))
    return None if size < 1e-6 else matrix


def _scale(matrix) -> float:
    return abs(matrix[0] * (matrix[5] * matrix[10] - matrix[6] * matrix[9])
               - matrix[1] * (matrix[4] * matrix[10] - matrix[6] * matrix[8])
               + matrix[2] * (matrix[4] * matrix[9] - matrix[5] * matrix[8]))


@lru_cache(maxsize=64)
def world_matrices(key: str, fraction: float) -> tuple:
    info, clip, skeleton, group, _ = movie(key)
    return ns.world_matrices(skeleton, ns.sample_pose(skeleton, clip, group, fraction * clip.duration, False))


def ready(key: str) -> bool:
    info = folder(key) / "stages.json"
    if not info.is_file():
        return False
    clips = json.loads(info.read_text())["clips"]
    return any((folder(key) / f"{name}.c3anim").is_file() and (folder(key) / f"{name}.visibility.txt").is_file()
               for name in clips)


def posed_pieces(key: str, fraction: float):
    """(model index, document, binding, transform) of the movie at a fraction:
    the main model's pieces on their bones (hidden ones left out), then each
    prop attached to a bone the movie shows at that moment, posed with it."""
    _asset, document, _transform = kits.landmarks(key)[0]
    skeleton_index = movie(key)[4]
    for binding in document["draw_bindings"]:
        if binding["skeleton"] != skeleton_index:
            continue
        matrix = piece_matrix(key, fraction, binding["mesh"])
        if matrix is not None:
            yield 0, document, binding, matrix
    skeleton = movie(key)[2]
    worlds = world_matrices(key, fraction)
    found = kits.compiler(key)
    for point in document["attachment_points"]:
        if point["binding_status"] != "resolved" or point["skeleton"] != skeleton_index:
            continue
        if not visible(key, fraction, skeleton["bones"][point["bone"]]["name"]) or \
                _scale(worlds[point["bone"]]) < 1e-6:
            continue
        for _asset, child, transform in dk.components(found.manifest, point["component_asset"],
                                                      list(worlds[point["bone"]]), []):
            for binding in child["draw_bindings"]:
                yield 1, child, binding, transform


@lru_cache(maxsize=None)
def span(key: str) -> tuple[float, float]:
    """The part of the movie that builds the wonder, as fractions: from the first
    frame showing a tenth of the busiest frame's pieces to the first frame from
    which exactly the finished set shows. Some movies raise the finished pieces
    themselves (the Pyramids); others build with construction pieces and swap in
    the finished building at the end (the Great Library). C3X build progress
    maps onto this span."""
    info, clip, skeleton, _group, skeleton_index = movie(key)
    _asset, document, _transform = kits.landmarks(key)[0]
    bones = sorted({info["mesh_bones"][binding["mesh"]] for binding in document["draw_bindings"]
                    if binding["skeleton"] == skeleton_index})
    last = clip.frame_count - 1
    final = {bone for bone in bones if visible(key, 1.0, skeleton["bones"][bone]["name"])}
    if not final:
        raise ValueError(f"{key}: the build movie's last frame shows no pieces")
    counts, exact = [], []
    for frame in range(clip.frame_count):
        shown = {bone for bone in bones if visible(key, frame / last, skeleton["bones"][bone]["name"])}
        counts.append(len(shown))
        exact.append(shown == final)
    start = next(frame for frame, count in enumerate(counts) if count >= .1 * max(counts))
    end = next(frame for frame in range(last + 1) if all(exact[frame:]))
    return start / last, end / last


def progress(key: str, built: float) -> float:
    """Movie fraction for a C3X build progress (0 = just started, 1 = complete)."""
    start, end = span(key)
    return start + max(0.0, min(1.0, built)) * (end - start)


def posed_triangles(key: str, fraction: float) -> tuple[list, list]:
    """Triangles and material colours of the movie at a fraction, for probes."""
    triangles, colours = [], []
    for index, document, binding, matrix in posed_pieces(key, fraction):
        if "worked" not in binding["states"]:
            continue
        mesh = kits._geometry(str(dk.SOURCES / document["components"]["geometry"][binding["geometry"]]))
        points = [kits._apply(matrix, vertex["position"]) for vertex in mesh["vertices"]]
        indices = mesh["topology"]["indices"]
        colour = kits.PALETTE[(binding["material"] + 6 * index) % len(kits.PALETTE)]
        for k in range(0, len(indices) - 2, 3):
            triangles.append([points[i] for i in indices[k:k + 3]])
            colours.append(colour)
    return triangles, colours


def probe(key: str, fractions, built: bool = False) -> Path:
    """A strip of point previews of the build movie at the given fractions
    (or C3X build progress values when `built`)."""
    from Renderer.lab.studies.resources import audit
    tiles = []
    for fraction in fractions:
        triangles, colours = posed_triangles(key, progress(key, fraction) if built else fraction)
        tiles.append(kits._splat(triangles, colours, 300))
    target = folder(key) / ("progress.png" if built else "probe.png")
    audit.write_png(target, 300 * len(tiles), 300, [b"".join(tile[2][row] for tile in tiles) for row in range(300)])
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("action", choices=("extract", "convert", "prepare", "probe"))
    parser.add_argument("--source", action="append", required=True)
    parser.add_argument("--at", nargs="*", type=float, default=[0.0, .25, .5, .75, 1.0])
    parser.add_argument("--built", action="store_true", help="--at values are C3X build progress")
    args = parser.parse_args()
    for key in args.source:
        if args.action == "extract":
            print(json.dumps(extract(key)), flush=True)
        elif args.action == "convert":
            convert(key)
            print(key, "converted", flush=True)
        elif args.action == "prepare":
            # Extract and convert what is missing; a source without a per-piece bone
            # table (no build movie to stage) is reported and skipped.
            try:
                if not (folder(key) / "stages.json").is_file():
                    extract(key)
                if not ready(key):
                    convert(key)
                print(key, "ready", span(key), flush=True)
            except (ValueError, KeyError, IndexError) as exc:
                print(f"{key} skipped: {exc!r}", flush=True)
        else:
            print(probe(key, args.at, args.built), span(key), flush=True)
    return 0


if __name__ == "__main__":
    from Renderer.lab.studies.wonders import stages as module
    raise SystemExit(module.main())

"""Civ VI and Workshop natural-wonder TerrainAssets compiled for the Lab kits.

A TerrainAsset is an ordinary landmark entry: an optional mesh, a skeleton with
one bone per scattered prop, and attachments to tilebase pieces. The shared
compound_landmark_importer needs six small allowances for this kind; they are
applied to an in-memory copy here, so the shared file stays unchanged:

1. pick the bone-name array whose element count equals the bone count;
2. accept the TerrainAsset attachment schema (no ConnectionType/ResourceType,
   TerrainFollowMode "None");
3. allow skeleton-only model records with no meshes;
4. allow a material table emptied by omitted dummy draws;
5. treat PERMANENT attachments with an empty asset as source-absent;
6. accept clutter models that carry no attachment-point list.
"""
from __future__ import annotations

import hashlib
import json
import math
import struct
import types
from functools import lru_cache
from pathlib import Path

from Renderer.tools.asset_compiler import compound_landmark_importer as cli
from Renderer.tools.asset_compiler.artdef_graph_resolver import DEFAULT_ASSETS_ROOT, _package_index, _resolve_package
from Renderer.tools.asset_compiler.improvement_asset_importer import _asset_id
from Renderer.tools.asset_compiler.indexed_static_package import IndexedStaticPackage

ROOT = Path(__file__).resolve().parents[4]
SOURCES = ROOT / "Renderer/packs/NaturalWonderSourcesLab"
WORKSHOP = Path.home() / "Library/Application Support/Steam/steamapps/workshop/content/289070"
BASE_SHARED = DEFAULT_ASSETS_ROOT / "Base/Platforms/Windows/BLPs/SHARED_DATA"
BASE_TILEBASES = DEFAULT_ASSETS_ROOT / "Base/Platforms/Windows/BLPs/landmarks/tilebases.blp"
IDENTITY = [1.0 if row == column else 0.0 for row in range(4) for column in range(4)]
PATCHES = (
    ("""        bone_fields = package.pointer_fields(pointer, TYPE_GRANNY_BONE)
        name_fields = package.pointer_fields(pointer, TYPE_STRING)
        if len(bone_fields) != 1 or len(name_fields) != 1:""",
     """        bone_fields = package.pointer_fields(pointer, TYPE_GRANNY_BONE)
        name_fields = package.pointer_fields(pointer, TYPE_STRING)
        if len(bone_fields) == 1 and len(name_fields) > 1:
            _count = package.allocations[bone_fields[0][1] - 1]["element_count"]
            name_fields = [field for field in name_fields
                           if package.allocations[field[1] - 1]["element_count"] == _count
                           and len(package.bytes_for(field[1])) == 8 * _count]
        if len(bone_fields) != 1 or len(name_fields) != 1:"""),
    ("""    required = {
        "ConnectionType",""",
     """    if "ConnectionType" not in values and "ResourceType" not in values:
        values.setdefault("ConnectionType", "NONE")
        values.setdefault("ResourceType", {"value": "DON'T CARE", "root": "Resource"})
    if values.get("TerrainFollowMode") == "None":
        values["TerrainFollowMode"] = "Pivot Height"
    required = {
        "ConnectionType","""),
    ("""            if count < 1 or first_mesh + count > mesh_count:
                raise ValueError("Compound model has an invalid mesh range")""",
     """            if (count < 1 and not skeletons) or first_mesh + count > mesh_count:
                raise ValueError("Compound model has an invalid mesh range")"""),
    ("    if uv_addresses and isinstance(uv_addresses[0], tuple):",
     "    if not uv_addresses or isinstance(uv_addresses[0], tuple):"),
    ("""and component["runtime_selection"]["cull"] in ("optional", "no_mtn_corner_opt")""",
     """and component["runtime_selection"]["cull"] in ("optional", "no_mtn_corner_opt", "permanent")"""),
    ("""    fields = package.pointer_fields(landmark_user_data, TYPE_ATTACHMENT_LIST)
    if len(fields) != 1:
        raise ValueError("Compound landmark must have exactly one attachment-point list")""",
     """    fields = package.pointer_fields(landmark_user_data, TYPE_ATTACHMENT_LIST)
    if not fields:
        return [], {"list": None, "data": None, "count": 0, "points": []}
    if len(fields) != 1:
        raise ValueError("Compound landmark must have exactly one attachment-point list")"""),
)


@lru_cache(maxsize=None)
def importer():
    source = Path(cli.__file__).read_text()
    for old, new in PATCHES:
        if source.count(old) != 1:
            raise ValueError("compound_landmark_importer changed; update the natural-wonder Lab patch")
        source = source.replace(old, new)
    module = types.ModuleType("natural_wonder_lab_importer")
    module.__file__ = cli.__file__
    exec(compile(source, "natural_wonder_lab_importer", "exec"), module.__dict__)
    return module


def load(path: Path):
    return json.loads(path.read_text())


def content_root(item: str | None) -> Path:
    return DEFAULT_ASSETS_ROOT if item is None else WORKSHOP / item


class Compiler:
    def __init__(self) -> None:
        self.civ6_packages = _package_index(DEFAULT_ASSETS_ROOT)
        self.mod_packages: dict[str, list] = {}
        # Package-bytes caches are keyed by relative path, which Workshop items share.
        self.package_bytes: dict[str, dict] = {}
        self.opened: dict[Path, IndexedStaticPackage] = {}
        self.texture_cache: dict = {}
        self.manifest = load(SOURCES / "manifest.json") if (SOURCES / "manifest.json").is_file() else \
            {"schema": "c3x.asset_pack.v0", "name": SOURCES.name, "assets": {}, "roots": {}, "failures": {},
             "source_policy": "Local licensed-source import; derived art is not redistributable."}
        self.failures: list = []

    def _mod_index(self, item: str) -> list:
        if item not in self.mod_packages:
            self.mod_packages[item] = [dict(entry, content_root="MOD") for entry in _package_index(WORKSHOP / item)
                                       if "/windows/" in entry["relative"].lower()]
        return self.mod_packages[item]

    def _open(self, path: Path, entry: str) -> IndexedStaticPackage:
        if path not in self.opened:
            try:
                self.opened[path] = IndexedStaticPackage(path, entry)
            except ValueError:
                self.opened[path] = IndexedStaticPackage(path, entry, minimum_temp_support=2)
        return self.opened[path]

    @staticmethod
    def _shared(path: Path) -> list[Path]:
        parts = path.parts
        blps = [part.lower() for part in parts].index("blps")
        local = Path(*parts[:blps + 1]) / "SHARED_DATA"
        return [local] + ([BASE_SHARED] if local != BASE_SHARED else [])

    def root(self, item: str | None, package: str, entry: str) -> str:
        key = f"{item or 'civ6'}|{package}|{entry}"
        if key in self.manifest["roots"]:
            return self.manifest["roots"][key]
        asset = self.ensure(content_root(item) / package, entry, item)
        self.manifest["roots"][key] = asset
        return asset

    def ensure(self, path: Path, entry: str, item: str | None, stack: tuple = ()) -> str:
        if item is None:
            asset_id = _asset_id(path.relative_to(DEFAULT_ASSETS_ROOT).as_posix(), entry)
        else:
            asset_id = "mod/" + hashlib.sha256(f"{item}\0{path.relative_to(WORKSHOP / item)}\0{entry}".encode()).hexdigest()[:16]
        if asset_id in self.manifest["assets"]:
            return asset_id
        if asset_id in stack:
            raise ValueError(f"component cycle at {entry}")
        package = self._open(path, entry)
        manifest_asset, evidence = importer()._compile_asset(
            package, self._shared(path), SOURCES, entry, asset_id, 100.0, self.texture_cache,
            terrain_edit_policy="preserve_unresolved", omit_empty_material_draws=True)
        document_path = SOURCES / manifest_asset["landmark"]
        document = load(document_path)
        points = {point["id"]: point for point in evidence["attachments"]["points"]}
        for point in document["attachment_points"]:
            if point["binding_status"] != "component_unresolved":
                continue
            terminal = points[point["id"]]["component_source"]
            candidates = []
            if item is not None:
                found = _resolve_package(self._mod_index(item), terminal["package"], "MOD", terminal["entry"],
                                         self.package_bytes.setdefault(item, {}))
                if found["status"] == "resolved":
                    candidates.append((WORKSHOP / item / found["package_path"], item))
            found = _resolve_package(self.civ6_packages, terminal["package"], "Base", terminal["entry"],
                                     self.package_bytes.setdefault("civ6", {}))
            if found["status"] == "resolved":
                candidates.append((DEFAULT_ASSETS_ROOT / found["package_path"], None))
            candidates.append((BASE_TILEBASES, None))
            error = None
            for candidate, owner in candidates:
                try:
                    point["component_asset"] = self.ensure(candidate, terminal["entry"], owner, stack + (asset_id,))
                    point["binding_status"] = "resolved"
                    error = None
                    break
                except (OSError, ValueError, KeyError, TypeError, struct.error) as exc:
                    error = exc
            if error is not None:
                point["binding_status"] = "component_compile_unresolved"
                self.failures.append({"parent": entry, "child": terminal["entry"], "reason": str(error)[:200]})
        document_path.write_text(json.dumps(document, indent=1, sort_keys=True) + "\n")
        self.manifest["assets"][asset_id] = {**manifest_asset, "source_entry": entry}
        return asset_id

    def save(self) -> None:
        SOURCES.mkdir(parents=True, exist_ok=True)
        (SOURCES / "manifest.json").write_text(json.dumps(self.manifest, indent=1, sort_keys=True) + "\n")


def _skeleton_worlds(skeleton: dict) -> list:
    from Renderer.preview.render_improvement_sheet import _skeleton_worlds as worlds
    return worlds(skeleton)


def components(manifest: dict, asset_id: str, transform: list[float], found: list) -> list:
    landmark = load(SOURCES / manifest["assets"][asset_id]["landmark"])
    found.append((asset_id, landmark, transform))
    worlds = [_skeleton_worlds(load(SOURCES / path)) for path in landmark["components"]["skeletons"]]
    for point in landmark["attachment_points"]:
        if point["binding_status"] == "resolved":
            components(manifest, point["component_asset"],
                       cli._matrix_multiply(worlds[point["skeleton"]][point["bone"]], transform), found)
    return found


def yaw_scale(transform: list[float]) -> tuple[float, float] | None:
    scale = math.hypot(transform[0], transform[1])
    if scale <= 0 or any(abs(transform[i]) > 1e-3 * scale for i in (2, 6, 8, 9)) or \
            abs(transform[10] - scale) > 1e-3 * scale or abs(transform[5] - transform[0]) > 1e-3 * scale or \
            abs(transform[4] + transform[1]) > 1e-3 * scale:
        return None
    return math.atan2(transform[1], transform[0]), scale


def placed_meshes(compiler: Compiler, asset_id: str, state: str = "worked") -> list[dict]:
    """Every draw of the asset tree: mesh vertices in the component's own frame
    plus its placement (pivot, yaw, uniform scale) in the wonder's frame, in tile units."""
    out = []
    for component, landmark, transform in components(compiler.manifest, asset_id, IDENTITY, []):
        fitted = yaw_scale(transform)
        linear = IDENTITY if fitted else transform[:12] + [0.0, 0.0, 0.0, 1.0]
        for binding in landmark["draw_bindings"]:
            if state not in binding["states"] and "default" not in binding["states"]:
                continue
            material = load(SOURCES / landmark["components"]["materials"][binding["material"]])
            channels = material.get("channels", {})
            if not isinstance(channels.get("base_color"), dict):
                continue
            opacity = channels.get("opacity")
            geometry = landmark["components"]["geometry"][binding["geometry"]]
            yaw, scale = fitted or (0.0, 1.0)
            normal, detail = channels.get("normal_0"), channels.get("normal_1")
            out.append({"geometry": geometry, "colour": SOURCES / channels["base_color"]["texture"],
                        "opacity": SOURCES / opacity["texture"] if isinstance(opacity, dict) else None,
                        "normal": SOURCES / normal["texture"] if isinstance(normal, dict) else None,
                        "detail": SOURCES / detail["texture"] if isinstance(detail, dict) else None,
                        "linear": linear, "fitted": bool(fitted), "pivot": list(transform[12:15]),
                        "yaw": yaw, "scale": scale, "component": component})
    return out

#!/usr/bin/env python3
"""Extract Civ VI unit attack-effect timing and FX sockets into local-only JSON.

For every unit recipe and compound composition in the C3X roster this tool
reads the installed Civ VI unit packages (read-only) and decodes, per source
model:

* ``TimelineSet::Timeline`` / ``TimelineSet::Trigger`` records (effect, sound,
  marker, event and impact triggers with their start times),
* the ``DDT::StateGraph`` that names each state and binds it to an animation
  slot and a timeline,
* the attachment-point arrays (socket name hashes, bone indices and local
  matrices).

All names are recovered by FNV-1a 32-bit hash matching against VFX package
strings, Wwise event names, VFX.artdef elements, state-graph names and model
socket names; unmatched hashes are kept as hex. The output is Civ VI-derived
and must stay in the ignored ``Renderer/packs`` tree. Runtime code consumes
only its generic timing/socket values, never unit names.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import math
import os
import re
import struct
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from Renderer.tools.asset_compiler import civblp_probe
from Renderer.tools.asset_compiler.compound_landmark_importer import _decode_skeletons
from Renderer.tools.asset_compiler.indexed_static_package import IndexedStaticPackage
from Renderer.tools.asset_compiler.unit_family_asset_importer import (
    _initial_entry,
    _physical_package,
)
from Renderer.tools.asset_compiler.unit_member_resolver import resolve_unit


SCHEMA = "c3x.unit_effect_timing.v0"
RENDERER_ROOT = Path(__file__).resolve().parents[2]
TOOLS_ROOT = Path(__file__).resolve().parent
CIV6_ROOT_ENV = "C3X_CIV6_ROOT"
DEFAULT_CIV6_ROOT = (
    Path.home()
    / "Library/Application Support/Steam/steamapps/common"
    / "Sid Meier's Civilization VI/Civ6.app/Contents/Assets"
)
DEFAULT_OUTPUT = RENDERER_ROOT / "packs" / "UnitEffectTiming" / "timing.json"
PACKS_ROOT = RENDERER_ROOT / "packs"
UNIT_PACKAGE = "units/units.blp"
SOURCE_UNITS_PER_TILE = 100.0
RECIPE_STRATEGIES = (
    TOOLS_ROOT / "unit_early_strategy.json",
    TOOLS_ROOT / "unit_family_strategy.json",
    TOOLS_ROOT / "unit_roster_strategy.json",
    TOOLS_ROOT / "unit_roster_expansion_strategy.json",
    RENDERER_ROOT / "lab" / "studies" / "units" / "settler_carrier_strategy.json",
)
STRATEGY_SCHEMA = "c3x.source_unit_family_strategy.v0"
COMPOSITION_PACK = "CompoundUnitRosterLab"
COMPOSITION_SOURCE_SETS = TOOLS_ROOT / "compound_unit_roster_sets.json"
COMPOSITION_SCHEMA = "c3x.source_compound_unit_sets.v0"

TYPE_MODEL_ENTRY = "ModelPackageEntry"
TYPE_USER_DATA = "BLP::BLPPtr<FGXModel::IUserData>"
TYPE_BASE_MODEL = "ModelPackageEntry::BaseModelData_Entry"
TYPE_ATTACHMENT_LIST = "AttachmentPointList"
TYPE_ATTACHMENT_DATA = "AttachmentPointCookData"
TYPE_ANIMATION_ENTRY = "BLP::AnimationEntry"
TYPE_ANIMATION_DESC = "FGXModelFramework::BehaviorDesc::AnimationDesc"
TYPE_SLOT_BINDING = "FGXModelFramework::BehaviorDesc::SlotBinding"
TYPE_GRAPH_ROOT = "DDT::StateGraph::Root"
TYPE_GRAPH_SOURCE = "DDT::StateGraph::SourceNode"
TYPE_GRAPH_DESTINATION = "DDT::StateGraph::DestinationNode"
TYPE_GRAPH_ANIMATION = "DDT::StateGraph::AnimationGraphNode"
TYPE_BLOB = "uint8"

# ModelPackageEntry::BaseModelData_Entry is 448 bytes. Array fields are
# (pointer, count, capacity) qword triples at these offsets.
BASE_MODEL_BYTES = 448
BASE_STATE_GRAPH = 160
BASE_TRIGGERS = 176
BASE_TIMELINES = 200
BASE_ANIMATIONS = 224
BASE_SOCKET_BONES = 248
BASE_SOCKET_MATRICES = 272
BASE_SOCKET_NAMES = 320
BASE_SLOT_BINDINGS = 368

TRIGGER_RECORD = struct.Struct("<IffiIII")
TIMELINE_RECORD = struct.Struct("<QIIfI")
SOCKET_NAME_RECORD = struct.Struct("<II")
ANIMATION_DESC_RECORD = struct.Struct("<II")
ANIMATION_ENTRY_BYTES = 64
ANIMATION_ENTRY_NAME = 0x08
ANIMATION_ENTRY_HASH = 0x30
STATE_LEAF_BYTES = 40
DESTINATION_NAME = 48
DESTINATION_HASH = 56

TRIGGER_KINDS = {1: "effect", 2: "sound", 3: "marker", 4: "event", 5: "impact"}
RELEASE_KINDS = ("effect", "event", "impact")
LEAF_ANIMATION_SLOT = 1
LEAF_TIMELINE = 2
FIRE_EVENTS = ("FIRE", "HIT")
# Gameplay event names are hashed engine constants; only names whose FNV-1a
# hash matches a trigger are ever reported.
EVENT_VOCABULARY = ("END", "FIRE", "HIT", "IMPACT", "LAUNCH", "OFF", "ON", "START", "STOP")
ATTACK_STATE = re.compile(r"^ATTACK(?:_[A-Z0-9]+)?$")
ACTION_STATE = re.compile(r"^(?:ATTACK|HERO|ACTION)(?:_[A-Z0-9]+)?$")
EMPTY_NAME_HASH = 0x811C9DC5
ROUND = 6
CONVENTIONS = {
    "time": "start_s is seconds from the start of the attack timeline",
    "normalized": (
        "start_s / attack duration_s, the runtime's normalized action progress; "
        "authored pre-roll starts clamp to 0 and values above 1 fire after the timeline ends"
    ),
    "duration_source": (
        "attack_timeline when the selected Civ VI timeline has a duration, "
        "otherwise the converted runtime clip duration"
    ),
    "hash": "FNV-1a 32-bit over the source name; unmatched names are kept as hex",
    "socket_local": (
        "row-major row-vector transform relative to the bound bone; translation in tiles "
        "(source units / 100); rotation_xyzw uses the normalized-skeleton orientation convention"
    ),
    "bone_index": (
        "index into the component's source skeleton; payload_bone_index is the same bone "
        "in the normalized pack skeleton and C3XANM1/C3XRIG1 palette"
    ),
    "trigger_types": {str(key): value for key, value in sorted(TRIGGER_KINDS.items())},
}


def fnv1a_32(text: str) -> int:
    value = 0x811C9DC5
    for byte in text.encode("utf-8"):
        value ^= byte
        value = (value * 0x01000193) & 0xFFFFFFFF
    return value


def hex32(value: int) -> str:
    return f"0x{value:08x}"


def _round(value: float) -> float:
    rounded = round(float(value), ROUND)
    return 0.0 if rounded == 0.0 else rounded


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --- Pure record decoders -------------------------------------------------


def decode_trigger(raw: bytes) -> dict[str, Any]:
    """Decode one 28-byte ``TimelineSet::Trigger`` record."""
    if len(raw) != TRIGGER_RECORD.size:
        raise ValueError(f"Trigger record must be {TRIGGER_RECORD.size} bytes")
    kind, start, duration, socket_index, socket_hash, data_hash, extra_hash = (
        TRIGGER_RECORD.unpack(raw)
    )
    # Authored starts may sit a few milliseconds before zero; a duration of -1
    # marks an effect that persists until the timeline stops it.
    if not (math.isfinite(start) and math.isfinite(duration)) or start < -1.0 or duration < -1.0:
        raise ValueError("Trigger has a non-finite or out-of-range time")
    if socket_index < -1:
        raise ValueError("Trigger has an invalid socket index")
    return {
        "type": kind,
        "kind": TRIGGER_KINDS.get(kind, f"type_{kind}"),
        "start_s": start,
        "duration_s": None if duration < 0.0 else duration,
        "socket_index": None if socket_index < 0 else socket_index,
        "socket_hash": socket_hash or None,
        "data_hash": data_hash,
        "extra_hash": extra_hash,
    }


def decode_timeline(raw: bytes, trigger_count: int) -> dict[str, Any]:
    """Decode one 24-byte ``TimelineSet::Timeline`` record."""
    if len(raw) != TIMELINE_RECORD.size:
        raise ValueError(f"Timeline record must be {TIMELINE_RECORD.size} bytes")
    reserved, first, count, duration, padding = TIMELINE_RECORD.unpack(raw)
    if reserved or padding:
        raise ValueError("Timeline record has non-zero reserved bytes")
    if not math.isfinite(duration) or duration < 0.0:
        raise ValueError("Timeline has a non-finite or negative duration")
    if first + count > trigger_count:
        raise ValueError("Timeline trigger range is outside the trigger array")
    return {"first_trigger": first, "trigger_count": count, "duration_s": duration}


def decode_slot_binding(value: int) -> tuple[int, int]:
    """Return (AnimationDesc index, state-graph animation slot)."""
    return value >> 16, value & 0xFFFF


def decode_state_leaf(raw: bytes) -> tuple[int, int] | None:
    """Return (leaf kind, index) for a 40-byte selector leaf, else None."""
    if len(raw) != STATE_LEAF_BYTES or any(raw[:16]):
        return None
    words = struct.unpack("<10i", raw)
    return words[4], words[8]


def normalized_progress(start_s: float, duration_s: float | None) -> float | None:
    """Normalized action progress; authored pre-roll starts clamp to zero."""
    if not duration_s or duration_s <= 0.0:
        return None
    return _round(max(0.0, start_s) / duration_s)


def ambient_timelines(states: list[dict[str, Any]]) -> set[int]:
    """Timelines layered under many states (idle loops, rotors) are ambient.

    A state's own timeline is referenced by one or two states; the shared
    ambient layer is referenced by at least three and a quarter of them.
    """
    counts: dict[int, int] = {}
    for state in states:
        for timeline in set(state["timelines"]):
            counts[timeline] = counts.get(timeline, 0) + 1
    return {
        timeline for timeline, count in counts.items()
        if count >= 3 and count * 4 >= len(states)
    }


def state_timeline(state: dict[str, Any], ambient: set[int]) -> int | None:
    own = [timeline for timeline in state["timelines"] if timeline >= 0 and timeline not in ambient]
    if own:
        return own[0]
    valid = [timeline for timeline in state["timelines"] if timeline >= 0]
    return valid[0] if valid else None


def _state_rank(state: dict[str, Any], release_counts: dict[int, int]) -> tuple:
    name = state["name"]
    timeline = state["timeline"]
    return (
        0 if name == "ATTACK" else 1 if ATTACK_STATE.match(name) else 2,
        -release_counts.get(timeline, 0),
        timeline if timeline is not None else 1 << 30,
        name,
    )


def select_attack_state(
    states: list[dict[str, Any]],
    clip_slots: set[int],
    release_counts: dict[int, int],
    fire_timelines: set[int],
) -> tuple[dict[str, Any] | None, str]:
    """Choose the state whose timeline times the attack clip.

    ``states`` carry ``name``, ``animation_slots`` and the resolved
    ``timeline``. ``release_counts`` maps timeline -> non-sound trigger count
    and ``fire_timelines`` holds timelines carrying FIRE/HIT events.
    """
    usable = [state for state in states if state["timeline"] is not None]
    bound = [state for state in usable if clip_slots & set(state["animation_slots"])]
    if bound:
        return min(bound, key=lambda state: _state_rank(state, release_counts)), "runtime_attack_clip"
    attack = [state for state in usable if ATTACK_STATE.match(state["name"])]
    with_releases = [state for state in attack if release_counts.get(state["timeline"], 0)]
    if with_releases:
        chosen = min(with_releases, key=lambda state: _state_rank(state, release_counts))
        return chosen, "state_graph_attack" if chosen["name"] == "ATTACK" else "attack_state_content"
    firing = [
        state for state in usable
        if ACTION_STATE.match(state["name"]) and state["timeline"] in fire_timelines
    ]
    if firing:
        return min(firing, key=lambda state: _state_rank(state, release_counts)), "action_state_fire_event"
    plain = [state for state in usable if state["name"] == "ATTACK"]
    if plain:
        return plain[0], "state_graph_attack"
    return None, "no_attack_timeline"


def share_sibling_state(choices: list[dict[str, Any] | None]) -> list[dict[str, Any] | None]:
    """Civ VI drives every member of a unit through one state graph state.

    Nodes whose own runtime clip is not bound adopt the attack state of the
    first sibling node that binds its runtime clip to an ATTACK state.
    """
    anchor = next(
        (
            choice for choice in choices
            if choice and choice["selection"] == "runtime_attack_clip" and ATTACK_STATE.match(choice["state"])
        ),
        None,
    )
    if anchor is None:
        return choices
    return [
        choice if choice and choice["selection"] == "runtime_attack_clip"
        else {"state": anchor["state"], "selection": "sibling_node_state", "primary_model": None}
        for choice in choices
    ]


def socket_transform(matrix: list[float] | tuple[float, ...]) -> dict[str, Any]:
    """Decompose a row-major row-vector affine socket matrix in source units.

    The quaternion uses the normalized-skeleton convention: its column-vector
    rotation matrix is the transpose of the stored row-vector 3x3 block.
    """
    if len(matrix) != 16 or any(not math.isfinite(value) for value in matrix):
        raise ValueError("Socket matrix must contain 16 finite values")
    if (matrix[3], matrix[7], matrix[11], matrix[15]) != (0.0, 0.0, 0.0, 1.0):
        raise ValueError("Socket matrix is not an affine row-vector transform")
    rows = [list(matrix[row * 4 : row * 4 + 3]) for row in range(3)]
    scale = [math.sqrt(sum(value * value for value in row)) for row in rows]
    if min(scale) <= 1.0e-9:
        raise ValueError("Socket matrix has a collapsed axis")
    rotation_row = [[value / scale[index] for value in row] for index, row in enumerate(rows)]
    column = [[rotation_row[c][r] for c in range(3)] for r in range(3)]
    quaternion = _quaternion_from_column_matrix(column)
    translation = [value / SOURCE_UNITS_PER_TILE for value in matrix[12:15]]
    tile_matrix = list(matrix[:12]) + translation + [1.0]
    return {
        "translation": [_round(value) for value in translation],
        "rotation_xyzw": [_round(value) for value in quaternion],
        "scale": [_round(value) for value in scale],
        "matrix": [_round(value) for value in tile_matrix],
    }


def _quaternion_from_column_matrix(r: list[list[float]]) -> list[float]:
    candidates = [
        1.0 + r[0][0] - r[1][1] - r[2][2],
        1.0 - r[0][0] + r[1][1] - r[2][2],
        1.0 - r[0][0] - r[1][1] + r[2][2],
        1.0 + r[0][0] + r[1][1] + r[2][2],
    ]
    k = max(range(4), key=lambda index: candidates[index])
    q = [0.0, 0.0, 0.0, 0.0]
    q[k] = math.sqrt(max(0.0, candidates[k])) / 2.0
    divisor = 4.0 * q[k]
    if k == 3:
        q[0] = (r[2][1] - r[1][2]) / divisor
        q[1] = (r[0][2] - r[2][0]) / divisor
        q[2] = (r[1][0] - r[0][1]) / divisor
    else:
        a, b = (k + 1) % 3, (k + 2) % 3
        q[a] = (r[a][k] + r[k][a]) / divisor
        q[b] = (r[b][k] + r[k][b]) / divisor
        q[3] = (r[b][a] - r[a][b]) / divisor
    length = math.sqrt(sum(value * value for value in q))
    q = [value / length for value in q]
    leading = q[3] if abs(q[3]) > 1.0e-9 else next(value for value in q if abs(value) > 1.0e-9)
    return [-value for value in q] if leading < 0.0 else q


# --- Hash-verified name tables -------------------------------------------


def _package_strings(path: Path) -> set[str]:
    data = path.read_bytes()
    header = civblp_probe.parse_file_header(
        data[:28], len(data), allow_declared_size_mismatch=True
    )
    package = data[header["package_data"]["offset"] : header["big_data"]["offset"]]
    return {match.group(1).decode("ascii") for match in re.finditer(rb"([\x20-\x7e]{2,})\x00", package)}


def _first_by_hash(names: Iterable[str]) -> dict[int, str]:
    table: dict[int, str] = {}
    for name in sorted(names):
        table.setdefault(fnv1a_32(name), name)
    return table


class NameTables:
    """FNV-1a lookups for effect, sound, event and VFX ArtDef names."""

    def __init__(self, civ6_root: Path, contents: Iterable[str]) -> None:
        self.sources: list[dict[str, str]] = []
        effects: set[str] = set()
        sounds: set[str] = set()
        artdefs: list[Path] = []
        for content in sorted(set(contents)):
            blps = civ6_root / content / "Platforms" / "Windows" / "BLPs"
            for path in sorted(blps.glob("*.blp")) if blps.is_dir() else []:
                if path.name.lower().startswith("vfx"):
                    effects |= _package_strings(path)
                    self._record(civ6_root, path, "effect_names")
            soundbanks = civ6_root / content / "Platforms" / "Windows" / "audio" / "SoundbanksInfo.xml"
            if soundbanks.is_file():
                text = soundbanks.read_text(encoding="utf-8", errors="replace")
                sounds |= set(re.findall(r'<Event Id="\d+" Name="([^"]+)"', text))
                self._record(civ6_root, soundbanks, "sound_event_names")
            artdef = civ6_root / content / "ArtDefs" / "VFX.artdef"
            if artdef.is_file():
                artdefs.append(artdef)
                self._record(civ6_root, artdef, "vfx_artdef")
        self.effects = _first_by_hash(effects)
        self.sounds = _first_by_hash(sounds)
        self.events = _first_by_hash(EVENT_VOCABULARY)
        self.collections: dict[int, str] = {}
        self.elements: dict[tuple[int, int], str] = {}
        for artdef in artdefs:
            self._read_vfx_artdef(artdef)

    def _record(self, civ6_root: Path, path: Path, role: str) -> None:
        self.sources.append(
            {"path": path.relative_to(civ6_root).as_posix(), "role": role, "sha256": _sha256(path)}
        )

    def _read_vfx_artdef(self, path: Path) -> None:
        roots = ET.parse(path).getroot().find("m_RootCollections")
        for collection in [] if roots is None else roots.findall("Element"):
            name_node = collection.find("m_CollectionName")
            if name_node is None or not name_node.get("text"):
                continue
            collection_hash = fnv1a_32(name_node.get("text"))
            self.collections.setdefault(collection_hash, name_node.get("text"))
            for element in collection.findall("Element"):
                element_name = element.find("m_Name")
                if element_name is not None and element_name.get("text"):
                    key = (collection_hash, fnv1a_32(element_name.get("text")))
                    self.elements.setdefault(key, element_name.get("text"))


# --- Civ VI unit package reader -------------------------------------------


class UnitPackage:
    """Typed reads of one ``units.blp`` package (models, graphs, animations)."""

    def __init__(self, path: Path, civ6_root: Path, anchors: list[str]) -> None:
        self.path = path
        self.relative = path.relative_to(civ6_root).as_posix()
        self.package = IndexedStaticPackage(path, _initial_entry(path, anchors))
        self.entries = self._model_entries()
        self.animation_names = self._animation_names()
        self._graphs: dict[int, list[dict[str, Any]]] = {}
        self._models: dict[str, dict[str, Any]] = {}

    def qword(self, pointer: int, offset: int) -> int:
        return struct.unpack_from("<Q", self.package.bytes_for(pointer), offset)[0]

    def _model_entries(self) -> dict[str, int]:
        entries: dict[str, list[int]] = {}
        for pointer in range(1, len(self.package.allocations) + 1):
            if self.package.type_name(pointer) != TYPE_MODEL_ENTRY:
                continue
            raw = self.package.bytes_for(pointer)
            if len(raw) != 72:
                continue
            name = self.package.string_value(struct.unpack_from("<Q", raw, 0x38)[0])
            if name:
                entries.setdefault(name, []).append(pointer)
        return {name: pointers[0] for name, pointers in entries.items() if len(pointers) == 1}

    def _animation_names(self) -> list[str]:
        pointer = self.package.unique_allocation(TYPE_ANIMATION_ENTRY)
        raw = self.package.bytes_for(pointer)
        count = self.package.allocations[pointer - 1]["element_count"]
        if len(raw) != count * ANIMATION_ENTRY_BYTES:
            raise ValueError("Animation entry table has an unexpected record size")
        names = []
        for index in range(count):
            offset = index * ANIMATION_ENTRY_BYTES
            name_pointer = struct.unpack_from("<Q", raw, offset + ANIMATION_ENTRY_NAME)[0]
            name = self.package.string_value(name_pointer)
            declared = struct.unpack_from("<I", raw, offset + ANIMATION_ENTRY_HASH)[0]
            if not name or fnv1a_32(name) != declared:
                raise ValueError(f"Animation entry {index} name does not match its hash")
            names.append(name)
        return names

    def array(self, base: int, offset: int, expected_type: str) -> tuple[int, int]:
        pointer, count, capacity = struct.unpack_from("<3Q", self.package.bytes_for(base), offset)
        if not pointer:
            return 0, 0
        if self.package.type_name(pointer) != expected_type:
            raise ValueError(f"Base-model field +{offset} is not {expected_type}")
        if count != capacity or count != self.package.allocations[pointer - 1]["element_count"]:
            raise ValueError(f"Base-model field +{offset} has inconsistent counts")
        return pointer, count

    def model(self, name: str) -> dict[str, Any]:
        if name not in self._models:
            entry = self.entries.get(name)
            if entry is None:
                raise KeyError(f"{name} is not a unique model entry in {self.relative}")
            self._models[name] = decode_model(self, name, entry)
        return self._models[name]

    def state_graph(self, root: int) -> list[dict[str, Any]]:
        if root not in self._graphs:
            self._graphs[root] = decode_state_graph(self, root)
        return self._graphs[root]


def _model_records(source: UnitPackage, entry: int) -> tuple[int, int]:
    user_data = source.qword(entry, 0x20)
    if source.package.type_name(user_data) != TYPE_USER_DATA:
        raise ValueError("Model entry has no user-data record")
    bases = source.package.pointer_fields(user_data, TYPE_BASE_MODEL)
    if len(bases) != 1 or len(source.package.bytes_for(bases[0][1])) != BASE_MODEL_BYTES:
        raise ValueError("Model entry has no unique 448-byte base-model record")
    return user_data, bases[0][1]


def _decode_triggers(source: UnitPackage, base: int) -> tuple[list[dict], list[dict]]:
    trigger_pointer, trigger_count = source.array(base, BASE_TRIGGERS, "TimelineSet::Trigger")
    timeline_pointer, timeline_count = source.array(base, BASE_TIMELINES, "TimelineSet::Timeline")
    raw = source.package.bytes_for(trigger_pointer) if trigger_pointer else b""
    triggers = [
        decode_trigger(raw[i : i + TRIGGER_RECORD.size])
        for i in range(0, len(raw), TRIGGER_RECORD.size)
    ]
    raw = source.package.bytes_for(timeline_pointer) if timeline_pointer else b""
    timelines = [
        decode_timeline(raw[i : i + TIMELINE_RECORD.size], trigger_count)
        for i in range(0, len(raw), TIMELINE_RECORD.size)
    ]
    if len(triggers) != trigger_count or len(timelines) != timeline_count:
        raise ValueError("Timeline/trigger arrays do not match their declared counts")
    return triggers, timelines


def _decode_animation_slots(source: UnitPackage, base: int) -> dict[int, list[str]]:
    desc_pointer, _ = source.array(base, BASE_ANIMATIONS, TYPE_ANIMATION_DESC)
    raw = source.package.bytes_for(desc_pointer) if desc_pointer else b""
    animations = []
    for reserved, index in ANIMATION_DESC_RECORD.iter_unpack(raw):
        if reserved or index >= len(source.animation_names):
            raise ValueError("Animation descriptor references an invalid animation entry")
        animations.append(source.animation_names[index])
    binding_pointer, _ = source.array(base, BASE_SLOT_BINDINGS, TYPE_SLOT_BINDING)
    raw = source.package.bytes_for(binding_pointer) if binding_pointer else b""
    slots: dict[int, list[str]] = {}
    for (value,) in struct.iter_unpack("<I", raw):
        animation, slot = decode_slot_binding(value)
        if animation >= len(animations):
            raise ValueError("Slot binding references an invalid animation descriptor")
        slots.setdefault(slot, []).append(animations[animation])
    return slots


def _decode_sockets(
    source: UnitPackage, user_data: int, base: int, bones: list[str]
) -> list[dict[str, Any]]:
    bone_pointer, count = source.array(base, BASE_SOCKET_BONES, "int32")
    matrix_pointer, matrix_count = source.array(base, BASE_SOCKET_MATRICES, "FGXMatrixRow")
    name_pointer, _ = source.array(base, BASE_SOCKET_NAMES, "Types::pair<uint32,uint32>")
    if count != matrix_count:
        raise ValueError("Socket bone and matrix arrays disagree")
    if not count:
        return []
    cooked = set()
    for _offset, pointer in source.package.pointer_fields(user_data, TYPE_ATTACHMENT_LIST):
        for _field, data in source.package.pointer_fields(pointer, TYPE_ATTACHMENT_DATA):
            for index in range(source.package.allocations[data - 1]["element_count"]):
                record = source.package.array_element(data, index)
                name = source.package.string_value(struct.unpack_from("<Q", record, 0)[0])
                if name:
                    cooked.add(name)
    by_hash = _first_by_hash(cooked)
    aliases: dict[int, list[str]] = {}
    for name_hash, index in SOCKET_NAME_RECORD.iter_unpack(source.package.bytes_for(name_pointer)):
        if index >= count:
            raise ValueError("Socket name references an invalid socket index")
        aliases.setdefault(index, []).append(by_hash.get(name_hash, hex32(name_hash)))
    bone_indices = struct.unpack(f"<{count}i", source.package.bytes_for(bone_pointer))
    matrices = source.package.bytes_for(matrix_pointer)
    sockets = []
    for index in range(count):
        bone = bone_indices[index]
        if bone < 0 or (bones and bone >= len(bones)):
            raise ValueError("Socket references a bone outside the model skeleton")
        sockets.append(
            {
                "index": index,
                "names": sorted(aliases.get(index, [])),
                "name_hashes": {
                    name: hex32(fnv1a_32(name))
                    for name in aliases.get(index, [])
                    if not name.startswith("0x")
                },
                "bone_index": bone,
                "bone": bones[bone] if bones else None,
                "local": socket_transform(struct.unpack_from("<16f", matrices, index * 64)),
            }
        )
    return sockets


def decode_model(source: UnitPackage, name: str, entry: int) -> dict[str, Any]:
    user_data, base = _model_records(source, entry)
    triggers, timelines = _decode_triggers(source, base)
    skeletons, _evidence = _decode_skeletons(
        source.package, base, SOURCE_UNITS_PER_TILE, allow_unvalidated=True
    )
    bones = [bone["name"] for bone in skeletons[0]["bones"]] if skeletons else []
    root = struct.unpack_from("<Q", source.package.bytes_for(base), BASE_STATE_GRAPH)[0]
    if root and source.package.type_name(root) != TYPE_GRAPH_ROOT:
        raise ValueError(f"{name} state-graph field is not a graph root")
    return {
        "name": name,
        "package": source.relative,
        "graph_root": root or None,
        "triggers": triggers,
        "timelines": timelines,
        "animation_slots": _decode_animation_slots(source, base),
        "bones": bones,
        "skeleton_count": len(skeletons),
        "sockets": _decode_sockets(source, user_data, base, bones),
    }


def _graph_leaves(source: UnitPackage, pointer: int, seen: set[int]) -> list[tuple[int, int]]:
    if pointer in seen or source.package.type_name(pointer) != TYPE_BLOB:
        return []
    seen.add(pointer)
    raw = source.package.bytes_for(pointer)
    leaf = decode_state_leaf(raw)
    if leaf is not None:
        return [leaf]
    leaves = []
    for offset in range(0, len(raw) - 7, 8):
        child = struct.unpack_from("<Q", raw, offset)[0]
        if 0 < child <= len(source.package.allocations):
            leaves.extend(_graph_leaves(source, child, seen))
    return leaves


def _destination_state(source: UnitPackage, pointer: int) -> dict[str, Any]:
    raw = source.package.bytes_for(pointer)
    name = source.package.string_value(struct.unpack_from("<Q", raw, DESTINATION_NAME)[0])
    if not name or struct.unpack_from("<I", raw, DESTINATION_HASH)[0] != fnv1a_32(name):
        raise ValueError(f"State-graph destination {pointer} name does not match its hash")
    node = struct.unpack_from("<Q", raw, 0)[0]
    if source.package.type_name(node) != TYPE_GRAPH_ANIMATION:
        raise ValueError(f"State {name} has no animation-graph node")
    leaves: list[tuple[int, int]] = []
    seen: set[int] = set()
    node_raw = source.package.bytes_for(node)
    for offset in range(0, len(node_raw) - 7, 8):
        child = struct.unpack_from("<Q", node_raw, offset)[0]
        if 0 < child <= len(source.package.allocations):
            leaves.extend(_graph_leaves(source, child, seen))
    return {
        "name": name,
        "animation_slots": [index for kind, index in leaves if kind == LEAF_ANIMATION_SLOT and index >= 0],
        "timelines": [index for kind, index in leaves if kind == LEAF_TIMELINE and index >= 0],
    }


def decode_state_graph(source: UnitPackage, root: int) -> list[dict[str, Any]]:
    """Walk Root/Source/Destination nodes; each destination is a named state."""
    pending = [root]
    seen: set[int] = set()
    destinations = []
    while pending:
        pointer = pending.pop()
        if pointer in seen:
            continue
        seen.add(pointer)
        node_type = source.package.type_name(pointer)
        if node_type == TYPE_GRAPH_DESTINATION:
            destinations.append(pointer)
        raw = source.package.bytes_for(pointer)
        for offset in (0, 8):
            child = struct.unpack_from("<Q", raw, offset)[0] if len(raw) >= offset + 8 else 0
            if source.package.type_name(child) in (TYPE_GRAPH_SOURCE, TYPE_GRAPH_DESTINATION):
                pending.append(child)
    states = [_destination_state(source, pointer) for pointer in sorted(destinations)]
    ambient = ambient_timelines(states)
    for state in states:
        state["timeline"] = state_timeline(state, ambient)
    return states


# --- Roster discovery -----------------------------------------------------


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _strategy_units(paths: Iterable[Path]) -> list[dict[str, Any]]:
    units = []
    for path in paths:
        document = _load_json(path)
        if document.get("schema") != STRATEGY_SCHEMA:
            raise ValueError(f"{path.name} is not a unit strategy")
        for unit in document["units"]:
            units.append({**unit, "source_content": unit.get("source_content") or document["source_content"]})
    return units


def _match_strategy(units: list[dict[str, Any]], slug: str, civ3_ids: list[str]) -> dict[str, Any] | None:
    matches = [unit for unit in units if unit["slug"] == slug and set(unit["civ3_ids"]) & set(civ3_ids)]
    if len(matches) > 1:
        raise ValueError(f"Ambiguous strategy for {slug}")
    return matches[0] if matches else None


def _action_source(
    actions: dict[str, Any], action: str, node: str | None = None
) -> tuple[str | None, str | None]:
    """Return (source animation, alias target); an aliased action has no clip of its own."""
    record = actions.get(action)
    if not isinstance(record, dict):
        return None, None
    if "alias" in record:
        return None, record["alias"]
    if node is not None:
        return record.get("nodes", {}).get(node), None
    return record.get("source"), None


def _recipe_units(packs_root: Path, strategies: list[dict[str, Any]]) -> list[dict[str, Any]]:
    units = []
    for path in sorted(packs_root.glob("*/units/*recipe.json")):
        recipe = _load_json(path)
        slug = path.name[: -len("_recipe.json")]
        civ3_ids = recipe.get("civ3_ids") or []
        strategy = _match_strategy(strategies, slug, civ3_ids) if civ3_ids else None
        actions = {**(strategy or {}).get("actions", {}), **(strategy or {}).get("additional_actions", {})}
        units.append(
            {
                "slug": slug,
                "civ3_ids": civ3_ids,
                "pack": path.parents[1].name,
                "pack_dir": path.parents[1],
                "document": path.relative_to(path.parents[1]).as_posix(),
                "strategy": strategy,
                "nodes": [
                    {
                        "node": "unit",
                        "variation": None,
                        "exclude_roles": (strategy or {}).get("exclude_roles", []),
                        "components": recipe.get("components", []),
                        "attack": _action_source(actions, "attack"),
                        "runtime_clip": recipe.get("actions", {}).get("attack"),
                    }
                ],
            }
        )
    return units


def _composition_units(packs_root: Path, source_sets_path: Path) -> list[dict[str, Any]]:
    source_sets = _load_json(source_sets_path)
    if source_sets.get("schema") != COMPOSITION_SCHEMA:
        raise ValueError("Unsupported compound-unit source-set schema")
    by_slug = {item["slug"]: item for item in source_sets["compositions"]}
    units = []
    for path in sorted((packs_root / COMPOSITION_PACK / "units").glob("*_composition.json")):
        composition = _load_json(path)
        slug = path.name[: -len("_composition.json")]
        source = by_slug.get(slug)
        nodes = []
        for node in [] if source is None else [source["parent"], *source["children"]]:
            runtime = composition["actions"].get("attack", {}).get("node_clips", {}).get(node["id"])
            nodes.append(
                {
                    "node": node["id"],
                    "variation": node["variation"],
                    "exclude_roles": node.get("exclude_roles", []),
                    "components": composition["nodes"].get(node["id"], {}).get("components", []),
                    "attack": _action_source(source["actions"], "attack", node["id"]),
                    "runtime_clip": runtime,
                }
            )
        strategy = None if source is None else {
            **source,
            "source_content": source.get("source_content") or source_sets["source_content"],
        }
        units.append(
            {
                "slug": slug,
                "civ3_ids": composition.get("civ3_ids") or [],
                "pack": COMPOSITION_PACK,
                "pack_dir": packs_root / COMPOSITION_PACK,
                "document": path.relative_to(path.parents[1]).as_posix(),
                "strategy": strategy,
                "nodes": nodes,
            }
        )
    return units


def _resolve_node_components(
    civ6_root: Path, unit: dict[str, Any], node: dict[str, Any]
) -> list[dict[str, Any]]:
    """Attach a Civ VI model entry to each runtime component of a node."""
    components = node["components"]
    if components and all(component.get("source_entry") for component in components):
        return [{**component, "source_package": UNIT_PACKAGE} for component in components]
    strategy = unit["strategy"]
    recipe = resolve_unit(
        civ6_root,
        strategy["source_artdef"],
        "Any",
        node["variation"],
        strategy.get("member_index"),
        content=strategy["source_content"],
    )
    selected = [item for item in recipe["selected_components"] if item["role"] not in node["exclude_roles"]]
    if [item["role"] for item in selected] != [component["role"] for component in components]:
        raise ValueError(f"{unit['slug']}/{node['node']} resolved roles differ from the runtime recipe")
    return [
        {**component, "source_entry": item["source_entry"], "source_package": item["source_package"]}
        for component, item in zip(components, selected)
    ]


# --- Per-unit extraction --------------------------------------------------


class SourceSet:
    """Lazily opened unit packages per content root, with Base fallback."""

    def __init__(self, civ6_root: Path, anchors: dict[str, list[str]] | None = None) -> None:
        self.civ6_root = civ6_root
        self.anchors = anchors or {}
        self.packages: dict[Path, UnitPackage] = {}

    def _package(self, content: str, logical: str, entry: str) -> UnitPackage | None:
        path = _physical_package(self.civ6_root, content, logical)
        if path not in self.packages:
            if (entry.encode("ascii") + b"\0") not in path.read_bytes():
                return None
            anchors = [entry, *self.anchors.get(f"{content}/{logical}", [])]
            self.packages[path] = UnitPackage(path, self.civ6_root, anchors)
        return self.packages[path]

    def model(self, content: str, logical: str, entry: str) -> dict[str, Any]:
        """Read a model from the content package, inheriting Base like the importers."""
        for candidate in dict.fromkeys((content, "Base")):
            package = self._package(candidate, logical, entry)
            if package is not None and entry in package.entries:
                return package.model(entry)
        raise KeyError(f"{entry} is not a model entry in {content} or Base unit packages")

    def package_for(self, model: dict[str, Any]) -> UnitPackage:
        return next(item for item in self.packages.values() if item.relative == model["package"])


def _release_counts(model: dict[str, Any], names: NameTables) -> tuple[dict[int, int], set[int]]:
    counts: dict[int, int] = {}
    firing: set[int] = set()
    for index, timeline in enumerate(model["timelines"]):
        triggers = model["triggers"][timeline["first_trigger"] : timeline["first_trigger"] + timeline["trigger_count"]]
        counts[index] = sum(trigger["kind"] in RELEASE_KINDS for trigger in triggers)
        if any(
            trigger["kind"] == "event" and names.events.get(trigger["data_hash"]) in FIRE_EVENTS
            for trigger in triggers
        ):
            firing.add(index)
    return counts, firing


def _clip_slots(model: dict[str, Any], clip: str | None) -> set[int]:
    return {slot for slot, animations in model["animation_slots"].items() if clip in animations}


def _bound_animations(state: dict[str, Any], model: dict[str, Any]) -> set[str]:
    return {name for slot in state["animation_slots"] for name in model["animation_slots"].get(slot, [])}


def _socket_alias(socket: dict[str, Any] | None, socket_hash: int | None) -> str | None:
    """The exact socket name a trigger hashed; sockets may carry several aliases."""
    if socket_hash is None:
        return None
    wanted = hex32(socket_hash)
    exact = None if socket is None else next(
        (name for name, value in socket["name_hashes"].items() if value == wanted), None
    )
    return exact or wanted


SELECTION_ORDER = (
    "runtime_attack_clip",
    "state_graph_attack",
    "attack_state_content",
    "action_state_fire_event",
)


def choose_node_state(
    models: list[dict[str, Any]], sources: SourceSet, names: NameTables, clip: str | None
) -> dict[str, Any] | None:
    """Pick one state name for the node; prefer the model that binds the clip."""
    best = None
    for position, model in enumerate(models):
        if model["graph_root"] is None:
            continue
        states = sources.package_for(model).state_graph(model["graph_root"])
        counts, firing = _release_counts(model, names)
        state, method = select_attack_state(states, _clip_slots(model, clip), counts, firing)
        if state is None:
            continue
        candidate = (SELECTION_ORDER.index(method), -counts.get(state["timeline"], 0), position)
        if best is None or candidate < best[0]:
            best = (candidate, {"state": state["name"], "selection": method, "primary_model": model["name"]})
    return None if best is None else best[1]


def _resolved_trigger(trigger: dict[str, Any], model: dict[str, Any], names: NameTables) -> dict[str, Any]:
    result: dict[str, Any] = {
        "type": trigger["type"],
        "kind": trigger["kind"],
        "start_s": _round(trigger["start_s"]),
    }
    if trigger["duration_s"] is None:
        result["persistent"] = True
    elif trigger["duration_s"]:
        result["duration_s"] = _round(trigger["duration_s"])
    data = trigger["data_hash"]
    lookups = {"effect": names.effects, "sound": names.sounds, "event": names.events}
    if trigger["kind"] in lookups:
        name = "" if data == EMPTY_NAME_HASH else lookups[trigger["kind"]].get(data)
        result[trigger["kind"]] = name
        if name is None:
            result[trigger["kind"] + "_hash"] = hex32(data)
    elif trigger["kind"] == "impact":
        collection = names.collections.get(trigger["extra_hash"])
        element = names.elements.get((trigger["extra_hash"], data))
        result["impact"] = {"collection": collection, "element": element}
        if collection is None or element is None:
            result["impact"].update({"collection_hash": hex32(trigger["extra_hash"]), "element_hash": hex32(data)})
    else:
        result["value"] = data
    if trigger["socket_index"] is not None:
        index = trigger["socket_index"]
        socket = model["sockets"][index] if index < len(model["sockets"]) else None
        result["socket_index"] = index
        result["socket"] = _socket_alias(socket, trigger["socket_hash"])
        if socket is not None:
            result["bone_index"] = socket["bone_index"]
            result["bone"] = socket["bone"]
    return result


@functools.lru_cache(maxsize=None)
def _pack_manifest(pack_dir: Path) -> dict[str, Any]:
    path = pack_dir / "manifest.json"
    return _load_json(path) if path.is_file() else {}


def _payload_skeleton(pack_dir: Path, asset: str | None) -> tuple[str | None, list[str] | None]:
    """The converted component's normalized skeleton path and bone names."""
    if not asset:
        return None, None
    record = _pack_manifest(pack_dir).get("assets", {}).get(asset, {})
    component_path = pack_dir / record.get("component", "")
    if not record.get("component") or not component_path.is_file():
        return None, None
    skeleton = _load_json(component_path).get("skeleton")
    if not skeleton or not (pack_dir / skeleton).is_file():
        return None, None
    return skeleton, [bone["name"] for bone in _load_json(pack_dir / skeleton)["bones"]]


def _runtime_clip_duration(pack_dir: Path, clip_id: str | None) -> float | None:
    record = _pack_manifest(pack_dir).get("animations", {}).get(clip_id) if clip_id else None
    return None if not record or "duration" not in record else _round(record["duration"])


def _component_record(
    unit: dict[str, Any],
    component: dict[str, Any],
    model: dict[str, Any],
    state_name: str | None,
    sources: SourceSet,
    names: NameTables,
) -> dict[str, Any]:
    skeleton_path, payload_bones = _payload_skeleton(unit["pack_dir"], component.get("asset"))
    sockets = [dict(socket) for socket in model["sockets"]]
    for socket in sockets:
        socket.pop("name_hashes")
        if payload_bones is not None and socket["bone"] in payload_bones:
            socket["payload_bone_index"] = payload_bones.index(socket["bone"])
    record: dict[str, Any] = {
        "asset": component.get("asset"),
        "role": component.get("role"),
        "source_model": model["name"],
        "package": model["package"],
        "skeleton_bones": len(model["bones"]),
        **({"source_skeletons": model["skeleton_count"]} if model["skeleton_count"] > 1 else {}),
        "sockets": sockets,
        "payload_skeleton": None if skeleton_path is None else {
            "path": skeleton_path,
            "bone_order": "source_order" if payload_bones == model["bones"] else "differs",
        },
        "timeline": None,
        "triggers": [],
    }
    if model["graph_root"] is None or state_name is None:
        return record
    states = {state["name"]: state for state in sources.package_for(model).state_graph(model["graph_root"])}
    state = states.get(state_name)
    if state is None or state["timeline"] is None or state["timeline"] >= len(model["timelines"]):
        return record
    timeline = model["timelines"][state["timeline"]]
    triggers = model["triggers"][timeline["first_trigger"] : timeline["first_trigger"] + timeline["trigger_count"]]
    record["timeline"] = {
        "index": state["timeline"],
        "duration_s": _round(timeline["duration_s"]),
        "bound_animations": sorted(_bound_animations(state, model)),
    }
    record["triggers"] = sorted(
        (_resolved_trigger(trigger, model, names) for trigger in triggers),
        key=lambda item: (item["start_s"], item["type"], json.dumps(item, sort_keys=True)),
    )
    return record


def _alternates(
    models: list[dict[str, Any]], chosen: str | None, sources: SourceSet, names: NameTables
) -> list[dict[str, Any]]:
    """Other ATTACK* states with releases (for example the opposite broadside)."""
    result: dict[str, dict[str, Any]] = {}
    for model in models:
        if model["graph_root"] is None:
            continue
        counts, _firing = _release_counts(model, names)
        for state in sources.package_for(model).state_graph(model["graph_root"]):
            if state["name"] == chosen or not ATTACK_STATE.match(state["name"]) or state["timeline"] is None:
                continue
            releases = counts.get(state["timeline"], 0)
            if not releases or state["timeline"] >= len(model["timelines"]):
                continue
            entry = result.setdefault(state["name"], {
                "state": state["name"],
                "timeline": state["timeline"],
                "duration_s": _round(model["timelines"][state["timeline"]]["duration_s"]),
                "bound_animations": [],
                "release_count": 0,
            })
            entry["release_count"] += releases
            entry["bound_animations"] = sorted(
                set(entry["bound_animations"]) | _bound_animations(state, model)
            )
    return [result[name] for name in sorted(result)]


def _prepare_node(unit: dict[str, Any], node: dict[str, Any], sources: SourceSet) -> list[tuple[dict, dict]]:
    content = unit["strategy"]["source_content"]
    components = _resolve_node_components(sources.civ6_root, unit, node)
    return [
        (component, sources.model(content, component["source_package"], component["source_entry"]))
        for component in components
    ]


def _primary_record(records: list[dict[str, Any]], primary_model: str | None) -> dict[str, Any] | None:
    timed = [record for record in records if record["timeline"]]
    named = [record for record in timed if record["source_model"] == primary_model]
    positive = [record for record in timed if record["timeline"]["duration_s"] > 0]
    return (named or positive or timed or [None])[0]


def _node_attack(
    unit: dict[str, Any],
    node: dict[str, Any],
    choice: dict[str, Any],
    records: list[dict[str, Any]],
    alternates: list[dict[str, Any]],
) -> dict[str, Any]:
    primary = _primary_record(records, choice["primary_model"])
    timeline = primary["timeline"] if primary else None
    runtime_duration = _runtime_clip_duration(unit["pack_dir"], node["runtime_clip"])
    from_timeline = timeline is not None and timeline["duration_s"] > 0
    duration = timeline["duration_s"] if from_timeline else runtime_duration
    for record in records:
        for trigger in record["triggers"]:
            trigger["normalized"] = normalized_progress(trigger["start_s"], duration)
            if trigger["normalized"] is not None and trigger["normalized"] > 1.0:
                trigger["after_timeline_end"] = True
    return {
        **choice,
        "primary_model": primary["source_model"] if primary else None,
        "source_animation": node["attack"][0],
        "timeline": timeline["index"] if timeline else None,
        "duration_s": duration,
        "duration_source": "attack_timeline" if from_timeline else "runtime_clip" if duration else None,
        "runtime_clip": node["runtime_clip"],
        "runtime_clip_duration_s": runtime_duration,
        "timeline_matches_runtime_clip": (
            None if runtime_duration is None or timeline is None
            else abs(timeline["duration_s"] - runtime_duration) <= 1.0 / 30.0
        ),
        "alternates": alternates,
    }


def _build_node(
    unit: dict[str, Any],
    node: dict[str, Any],
    pairs: list[tuple[dict, dict]],
    choice: dict[str, Any] | None,
    sources: SourceSet,
    names: NameTables,
) -> dict[str, Any]:
    state_name = None if choice is None else choice["state"]
    records = [
        _component_record(unit, component, model, state_name, sources, names)
        for component, model in pairs
    ]
    attack = None
    if choice is not None:
        alternates = _alternates([model for _component, model in pairs], state_name, sources, names)
        attack = _node_attack(unit, node, choice, records, alternates)
    result = {"node": node["node"], "attack": attack, "components": records}
    if node["attack"][1]:
        result["attack_alias"] = node["attack"][1]
    return result


def _releases(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    releases = []
    for node in nodes:
        for component in node["components"]:
            for trigger in component["triggers"]:
                if trigger["kind"] not in RELEASE_KINDS:
                    continue
                release = {
                    "node": node["node"],
                    "asset": component["asset"],
                    "source_model": component["source_model"],
                    **{key: value for key, value in trigger.items() if key != "type"},
                }
                sockets = {socket["index"]: socket for socket in component["sockets"]}
                payload = sockets.get(trigger.get("socket_index"), {}).get("payload_bone_index")
                if payload is not None:
                    release["payload_bone_index"] = payload
                releases.append(release)
    return sorted(
        releases,
        key=lambda item: (item["start_s"], item["node"], item["asset"] or "", json.dumps(item, sort_keys=True)),
    )


def extract_unit(unit: dict[str, Any], sources: SourceSet, names: NameTables) -> dict[str, Any]:
    prepared = [_prepare_node(unit, node, sources) for node in unit["nodes"]]
    # Units whose runtime recipe has no attack clip (aliased or absent) get no attack timing.
    timed = [bool(node["attack"][0]) for node in unit["nodes"]]
    choices = [
        choose_node_state([model for _component, model in pairs], sources, names, node["attack"][0])
        if has_clip else None
        for node, pairs, has_clip in zip(unit["nodes"], prepared, timed)
    ]
    choices = [choice if has_clip else None for has_clip, choice in zip(timed, share_sibling_state(choices))]
    nodes = [
        _build_node(unit, node, pairs, choice, sources, names)
        for node, pairs, choice in zip(unit["nodes"], prepared, choices)
    ]
    return {
        "slug": unit["slug"],
        "civ3_ids": unit["civ3_ids"],
        "roster": {"pack": unit["pack"], "document": unit["document"]},
        "source_unit": unit["strategy"].get("source_artdef"),
        "source_content": unit["strategy"]["source_content"],
        "source_models": sorted({component["source_model"] for node in nodes for component in node["components"]}),
        "nodes": nodes,
        "releases": _releases(nodes),
    }


def _coverage_reason(unit: dict[str, Any], record: dict[str, Any] | None) -> str | None:
    if not unit["civ3_ids"]:
        return "recipe has no native Civ III key"
    if unit["strategy"] is None:
        return "no source strategy matches the recipe"
    if record is None:
        return "source extraction failed"
    attacks = [node["attack"] for node in record["nodes"] if node["attack"]]
    if not attacks:
        aliases = sorted({node["attack"][1] for node in unit["nodes"] if node["attack"][1]})
        if aliases:
            return "runtime attack action aliases " + ", ".join(aliases)
        if not any(node["attack"][0] for node in unit["nodes"]):
            return "no attack action in the runtime recipe"
        return "no attack timeline in the state graph"
    if not record["releases"]:
        return "attack timeline has only sound/marker triggers"
    return None


def compile_unit_effect_timing(civ6_root: Path, packs_root: Path = PACKS_ROOT) -> dict[str, Any]:
    strategy_paths = [path for path in RECIPE_STRATEGIES if path.is_file()]
    strategies = _strategy_units(strategy_paths)
    roster = _recipe_units(packs_root, strategies) + _composition_units(packs_root, COMPOSITION_SOURCE_SETS)
    anchors: dict[str, list[str]] = {}
    for path in strategy_paths:
        for key, names in _load_json(path).get("package_anchors", {}).items():
            anchors.setdefault(key, []).extend(names)
    sources = SourceSet(civ6_root, anchors)
    contents = {"Base"} | {unit["strategy"]["source_content"] for unit in roster if unit["strategy"]}
    names = NameTables(civ6_root, contents)
    units: dict[str, Any] = {}
    covered, not_covered = [], []
    for unit in sorted(roster, key=lambda item: (item["pack"], item["slug"])):
        record, error = None, None
        if unit["civ3_ids"] and unit["strategy"] is not None:
            try:
                record = extract_unit(unit, sources, names)
            except (KeyError, ValueError, struct.error) as exc:
                error = str(exc)
        reason = _coverage_reason(unit, record)
        if record is not None:
            for key in unit["civ3_ids"]:
                units[key] = record
        label = {"slug": unit["slug"], "pack": unit["pack"], "civ3_ids": unit["civ3_ids"]}
        if reason is None:
            covered.append(label)
        else:
            not_covered.append({**label, "reason": reason, **({"error": error} if error else {})})
    return {
        "schema": SCHEMA,
        "source_policy": "Local-only Civ VI-derived metadata; keep under the ignored packs tree and never redistribute.",
        "conventions": CONVENTIONS,
        "sources": {
            "unit_packages": sorted(
                ({"path": package.relative, "sha256": _sha256(package.path)} for package in sources.packages.values()),
                key=lambda item: item["path"],
            ),
            "names": sorted(names.sources, key=lambda item: item["path"]),
        },
        "units": {key: units[key] for key in sorted(units)},
        "coverage": {
            "covered": sorted(covered, key=lambda item: (item["pack"], item["slug"])),
            "not_covered": sorted(not_covered, key=lambda item: (item["pack"], item["slug"])),
        },
    }


def default_civ6_root() -> Path:
    configured = os.environ.get(CIV6_ROOT_ENV)
    return Path(configured) if configured else DEFAULT_CIV6_ROOT


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--civ6-root", type=Path, default=None, help=f"Civ VI Assets folder (or ${CIV6_ROOT_ENV})")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    civ6_root = args.civ6_root or default_civ6_root()
    try:
        if not (civ6_root / "Base" / "Platforms" / "Windows" / "BLPs" / UNIT_PACKAGE).is_file():
            raise FileNotFoundError(f"No Civ VI unit package under {civ6_root}")
        result = compile_unit_effect_timing(civ6_root)
        _write_json(args.output, result)
    except (OSError, ValueError, KeyError, struct.error, ET.ParseError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    coverage = result["coverage"]
    print(
        f"Unit effect timing: {len(result['units'])} native keys, "
        f"{len(coverage['covered'])} roster units covered, {len(coverage['not_covered'])} not covered"
    )
    print(f"Output: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

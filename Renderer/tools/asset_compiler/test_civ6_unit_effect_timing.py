import math
import struct
import tempfile
import unittest
from pathlib import Path

from Renderer.tools.asset_compiler import civ6_unit_effect_timing as timing


def trigger_bytes(kind, start, duration, socket=-1, socket_hash=0, data=0, extra=0) -> bytes:
    return struct.pack("<IffiIII", kind, start, duration, socket, socket_hash, data, extra)


def state(name, slots, timelines, timeline=None):
    return {"name": name, "animation_slots": slots, "timelines": timelines, "timeline": timeline}


def row_rotation_from_quaternion(q):
    """The normalized-skeleton convention used by compound_landmark_importer."""
    x, y, z, w = q
    column = [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]
    return [[column[c][r] for c in range(3)] for r in range(3)]


class HashTests(unittest.TestCase):
    def test_fnv1a_matches_reference_vectors_and_civ6_names(self) -> None:
        self.assertEqual(0x811C9DC5, timing.fnv1a_32(""))
        self.assertEqual(0xE40C292C, timing.fnv1a_32("a"))
        self.assertEqual(0xBF9CF968, timing.fnv1a_32("foobar"))
        # Confirmed against units.blp AnimationEntry hashes and state-graph names.
        self.assertEqual(0xF86780EC, timing.fnv1a_32("ANIMATION_Artillery_AttackA"))
        self.assertEqual(0xCBFD4BD9, timing.fnv1a_32("FIRE"))
        self.assertEqual(timing.EMPTY_NAME_HASH, timing.fnv1a_32(""))


class RecordDecodingTests(unittest.TestCase):
    def test_trigger_record_fields(self) -> None:
        decoded = timing.decode_trigger(trigger_bytes(1, 0.494048, 1.0, 5, 0xA5990D5C, 0x2744E8D4))
        self.assertEqual("effect", decoded["kind"])
        self.assertAlmostEqual(0.494048, decoded["start_s"], places=6)
        self.assertEqual(1.0, decoded["duration_s"])
        self.assertEqual(5, decoded["socket_index"])
        self.assertEqual(0xA5990D5C, decoded["socket_hash"])
        self.assertEqual(0x2744E8D4, decoded["data_hash"])

    def test_trigger_without_socket_and_persistent_duration(self) -> None:
        decoded = timing.decode_trigger(trigger_bytes(1, -0.0023, -1.0, data=7))
        self.assertIsNone(decoded["socket_index"])
        self.assertIsNone(decoded["socket_hash"])
        self.assertIsNone(decoded["duration_s"])
        self.assertLess(decoded["start_s"], 0.0)

    def test_impact_trigger_keeps_collection_hash(self) -> None:
        decoded = timing.decode_trigger(trigger_bytes(5, 1.5333, 0.0, 0, 1, 1728460134, 1879711926))
        self.assertEqual("impact", decoded["kind"])
        self.assertEqual(1879711926, decoded["extra_hash"])

    def test_trigger_rejects_bad_records(self) -> None:
        with self.assertRaises(ValueError):
            timing.decode_trigger(b"\0" * 27)
        with self.assertRaises(ValueError):
            timing.decode_trigger(trigger_bytes(1, math.nan, 0.0))
        with self.assertRaises(ValueError):
            timing.decode_trigger(trigger_bytes(1, 0.0, 0.0, socket=-2))

    def test_timeline_record(self) -> None:
        raw = struct.pack("<QIIfI", 0, 6, 12, 2.0, 0)
        self.assertEqual(
            {"first_trigger": 6, "trigger_count": 12, "duration_s": 2.0},
            timing.decode_timeline(raw, 30),
        )
        with self.assertRaises(ValueError):
            timing.decode_timeline(raw, 17)
        with self.assertRaises(ValueError):
            timing.decode_timeline(struct.pack("<QIIfI", 1, 0, 0, 1.0, 0), 30)
        with self.assertRaises(ValueError):
            timing.decode_timeline(raw[:20], 30)

    def test_slot_binding_and_state_leaf(self) -> None:
        self.assertEqual((15, 40), timing.decode_slot_binding(0x000F0028))
        self.assertEqual((0, 8), timing.decode_slot_binding(8))
        leaf = struct.pack("<10i", 0, 0, 0, 0, 2, 13, 693, 105, 29, 0)
        self.assertEqual((2, 29), timing.decode_state_leaf(leaf))
        branch = struct.pack("<10i", 6832, 0, 6831, 0, 0, 0, 2, 6, 693, -114)
        self.assertIsNone(timing.decode_state_leaf(branch))
        self.assertIsNone(timing.decode_state_leaf(leaf + b"\0" * 8))


class StateSelectionTests(unittest.TestCase):
    def setUp(self) -> None:
        raw = [
            state("RUN", [0], [0, 1]),
            state("IDLE", [3, 4], [1, 4]),
            state("ATTACK", [4], [5, 1]),
            state("HERO", [9], [10, 1]),
            state("DEATH", [7], [8]),
            state("ATTACK_P", [28], [29, 1]),
            state("ATTACK_S", [34], [35, 1]),
        ]
        ambient = timing.ambient_timelines(raw)
        for item in raw:
            item["timeline"] = timing.state_timeline(item, ambient)
        self.states = raw

    def test_shared_layer_is_ambient_and_states_keep_own_timeline(self) -> None:
        self.assertEqual({1}, timing.ambient_timelines(self.states))
        timelines = {item["name"]: item["timeline"] for item in self.states}
        self.assertEqual({"RUN": 0, "IDLE": 4, "ATTACK": 5, "HERO": 10, "DEATH": 8,
                          "ATTACK_P": 29, "ATTACK_S": 35}, timelines)

    def test_runtime_clip_binding_wins(self) -> None:
        chosen, method = timing.select_attack_state(self.states, {28}, {29: 13, 35: 13}, {29, 35})
        self.assertEqual(("ATTACK_P", "runtime_attack_clip"), (chosen["name"], method))

    def test_clip_bound_to_several_states_prefers_attack(self) -> None:
        self.states[3]["animation_slots"] = [4]
        chosen, method = timing.select_attack_state(self.states, {4}, {10: 9}, set())
        self.assertEqual(("ATTACK", "runtime_attack_clip"), (chosen["name"], method))

    def test_clip_bound_to_non_attack_state_is_still_the_runtime_clip(self) -> None:
        chosen, method = timing.select_attack_state(self.states, {9}, {}, set())
        self.assertEqual(("HERO", "runtime_attack_clip"), (chosen["name"], method))

    def test_unbound_clip_uses_attack_state_with_releases(self) -> None:
        chosen, method = timing.select_attack_state(self.states, set(), {5: 3}, set())
        self.assertEqual(("ATTACK", "state_graph_attack"), (chosen["name"], method))
        chosen, method = timing.select_attack_state(self.states, set(), {35: 4, 29: 2}, set())
        self.assertEqual(("ATTACK_S", "attack_state_content"), (chosen["name"], method))

    def test_content_fallback_ignores_idle_states(self) -> None:
        chosen, method = timing.select_attack_state(self.states, set(), {4: 3, 10: 2}, {4, 10})
        self.assertEqual(("HERO", "action_state_fire_event"), (chosen["name"], method))
        chosen, method = timing.select_attack_state(self.states, set(), {4: 3}, {4})
        self.assertEqual(("ATTACK", "state_graph_attack"), (chosen["name"], method))
        chosen, method = timing.select_attack_state([self.states[1]], set(), {4: 3}, {4})
        self.assertIsNone(chosen)
        self.assertEqual("no_attack_timeline", method)

    def test_sibling_nodes_share_the_clip_bound_attack_state(self) -> None:
        bound = {"state": "ATTACK_P", "selection": "runtime_attack_clip", "primary_model": "Rider"}
        proxy = {"state": "ATTACK", "selection": "state_graph_attack", "primary_model": "Horse"}
        shared = timing.share_sibling_state([proxy, bound, None])
        self.assertEqual(["ATTACK_P", "ATTACK_P", "ATTACK_P"], [item["state"] for item in shared])
        self.assertEqual("sibling_node_state", shared[0]["selection"])
        self.assertIs(bound, shared[1])
        self.assertEqual([proxy, None], timing.share_sibling_state([proxy, None]))


class NormalizationTests(unittest.TestCase):
    def test_normalized_progress(self) -> None:
        self.assertEqual(0.247024, timing.normalized_progress(0.494048, 2.0))
        self.assertEqual(0.0, timing.normalized_progress(-0.0019, 1.5))
        self.assertEqual(2.00145, timing.normalized_progress(2.00145, 1.0))
        self.assertIsNone(timing.normalized_progress(0.5, 0.0))
        self.assertIsNone(timing.normalized_progress(0.5, None))

    def test_action_source_resolves_aliases_as_unclipped(self) -> None:
        actions = {"idle": {"source": "ANIMATION_Idle"}, "attack": {"alias": "idle"}}
        self.assertEqual((None, "idle"), timing._action_source(actions, "attack"))
        self.assertEqual(("ANIMATION_Idle", None), timing._action_source(actions, "idle"))
        nodes = {"attack": {"nodes": {"body": "ANIMATION_Body"}}}
        self.assertEqual(("ANIMATION_Body", None), timing._action_source(nodes, "attack", "body"))
        self.assertEqual((None, None), timing._action_source({}, "attack"))


class SocketTransformTests(unittest.TestCase):
    def test_identity(self) -> None:
        result = timing.socket_transform([1.0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 1.0])
        self.assertEqual([0.0, 0.0, 0.0, 1.0], result["rotation_xyzw"])
        self.assertEqual([0.0, 0.0, 0.0], result["translation"])
        self.assertEqual([1.0, 1.0, 1.0], result["scale"])

    def test_half_turn_muzzle_offset_converts_to_tiles(self) -> None:
        matrix = [-1.0, 0, 0, 0, 0, -1.0, 0, 0, 0, 0, 1.0, 0, 15.0, 0, 0, 1.0]
        result = timing.socket_transform(matrix)
        self.assertEqual([0.15, 0.0, 0.0], result["translation"])
        self.assertEqual([0.0, 0.0, 1.0, 0.0], result["rotation_xyzw"])
        self.assertEqual(0.15, result["matrix"][12])

    def test_quaternion_round_trips_normalized_skeleton_convention(self) -> None:
        angle = math.radians(35.0)
        q = [0.0, math.sin(angle / 2), 0.0, math.cos(angle / 2)]
        rows = row_rotation_from_quaternion(q)
        matrix = [*rows[0], 0.0, *rows[1], 0.0, *rows[2], 0.0, 1.0, 2.0, 3.0, 1.0]
        result = timing.socket_transform(matrix)
        for expected, actual in zip(q, result["rotation_xyzw"]):
            self.assertAlmostEqual(expected, actual, places=5)
        self.assertEqual([0.01, 0.02, 0.03], result["translation"])

    def test_rejects_projective_or_collapsed_matrices(self) -> None:
        with self.assertRaises(ValueError):
            timing.socket_transform([1.0] * 16)
        with self.assertRaises(ValueError):
            timing.socket_transform([0.0] * 15 + [1.0])


class TriggerResolutionTests(unittest.TestCase):
    def test_names_resolve_by_hash_and_unknowns_stay_hex(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            names = timing.NameTables(Path(directory), [])
        names.effects = {timing.fnv1a_32("FX_Muzzle"): "FX_Muzzle"}
        names.collections = {timing.fnv1a_32("TerrainVFX"): "TerrainVFX"}
        names.elements = {(timing.fnv1a_32("TerrainVFX"), timing.fnv1a_32("AoE_Hit")): "AoE_Hit"}
        model = {"sockets": [{
            "index": 0, "names": ["FX_Socket"], "bone_index": 7, "bone": "Barrel",
            "name_hashes": {"FX_Socket": timing.hex32(timing.fnv1a_32("FX_Socket"))},
        }]}
        effect = timing.decode_trigger(trigger_bytes(
            1, 0.5, 1.0, 0, timing.fnv1a_32("FX_Socket"), timing.fnv1a_32("FX_Muzzle")))
        resolved = timing._resolved_trigger(effect, model, names)
        self.assertEqual("FX_Muzzle", resolved["effect"])
        self.assertEqual(("FX_Socket", "Barrel", 7), (resolved["socket"], resolved["bone"], resolved["bone_index"]))
        event = timing.decode_trigger(trigger_bytes(4, 1.5, 0.0, data=timing.fnv1a_32("HIT")))
        self.assertEqual("HIT", timing._resolved_trigger(event, model, names)["event"])
        impact = timing.decode_trigger(trigger_bytes(
            5, 1.5, 0.0, data=timing.fnv1a_32("AoE_Hit"), extra=timing.fnv1a_32("TerrainVFX")))
        self.assertEqual({"collection": "TerrainVFX", "element": "AoE_Hit"},
                         timing._resolved_trigger(impact, model, names)["impact"])
        sound = timing.decode_trigger(trigger_bytes(2, 0.1, 0.0, data=0x12345678))
        resolved = timing._resolved_trigger(sound, model, names)
        self.assertIsNone(resolved["sound"])
        self.assertEqual("0x12345678", resolved["sound_hash"])
        empty = timing.decode_trigger(trigger_bytes(4, 0.2, 0.0, data=timing.EMPTY_NAME_HASH))
        self.assertEqual("", timing._resolved_trigger(empty, model, names)["event"])
        persistent = timing.decode_trigger(trigger_bytes(1, 0.0, -1.0, data=timing.fnv1a_32("FX_Muzzle")))
        self.assertTrue(timing._resolved_trigger(persistent, model, names)["persistent"])


if __name__ == "__main__":
    unittest.main()

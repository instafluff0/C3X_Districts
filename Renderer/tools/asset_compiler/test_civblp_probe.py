#!/usr/bin/env python3
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import civblp_probe as probe


class CivblpProbeTests(unittest.TestCase):
    def make_fixture(self) -> bytes:
        package_offset = 0x400
        package_size = 0x800
        package = bytearray(package_size)
        package_base = 0x80
        temp_base = 0x500

        allocations = []

        def allocation(stripe, offset, size, count, type_pointer, parent=0, user_data=0):
            allocations.append((stripe, offset, size, count, type_pointer, parent, user_data))
            return len(allocations)

        texture_array = allocation(0, 0, 104, 1, 7)
        target_pointer = allocation(0, 0x100, len(probe.DEFAULT_TARGET) + 1, len(probe.DEFAULT_TARGET) + 1, 8)
        texture_name_pointer = allocation(0, 0x140, 18, 18, 8)
        texture_class_pointer = allocation(0, 0x160, 18, 18, 8)
        material_pointer = allocation(0, 0x200, 128, 1, 6)
        material_type_pointer = allocation(1, 0x00, len(probe.MATERIAL_TYPE) + 1, len(probe.MATERIAL_TYPE) + 1, 8)
        texture_type_pointer = allocation(1, 0x40, len(probe.TEXTURE_TYPE) + 1, len(probe.TEXTURE_TYPE) + 1, 8)
        char_type_pointer = allocation(1, 0x80, len(probe.CHAR_TYPE) + 1, len(probe.CHAR_TYPE) + 1, 8)
        entry_map_pointer = allocation(1, 0x90, len(probe.ENTRY_MAP_TYPE), len(probe.ENTRY_MAP_TYPE), 8)
        texture_pointer = allocation(0, 0, 104, 1, 7, parent=texture_array)

        self.assertEqual((material_pointer, target_pointer, texture_pointer, entry_map_pointer), (5, 2, 10, 9))

        def put(offset, value):
            package[offset : offset + len(value)] = value

        put(package_base + 0x100, probe.DEFAULT_TARGET.encode("ascii") + b"\x00")
        put(package_base + 0x140, b"TEXTURE_SYNTHETIC\x00")
        put(package_base + 0x160, b"Synthetic_Class__\x00")
        texture_record = package_base
        struct.pack_into("<Q", package, texture_record + 0x00, texture_name_pointer)
        struct.pack_into("<Q", package, texture_record + 0x38, texture_class_pointer)
        material_record = package_base + 0x200
        struct.pack_into("<Q", package, material_record + 0x30, target_pointer)
        struct.pack_into("<Q", package, material_record + 0x40, texture_pointer)
        struct.pack_into("<Q", package, material_record + 0x48, 0)
        struct.pack_into("<Q", package, material_record + 0x70, 0x12345678)

        put(temp_base + 0x00, probe.MATERIAL_TYPE.encode("ascii") + b"\x00")
        put(temp_base + 0x40, probe.TEXTURE_TYPE.encode("ascii") + b"\x00")
        put(temp_base + 0x80, probe.CHAR_TYPE.encode("ascii") + b"\x00")
        put(temp_base + 0x90, probe.ENTRY_MAP_TYPE)
        table_offset = temp_base + 0x90 + len(probe.ENTRY_MAP_TYPE)
        for index, item in enumerate(allocations):
            stripe, offset, size, count, type_pointer, parent, user_data = item
            struct.pack_into(
                "<BB4sHIII4xQQ",
                package,
                table_offset + index * probe.ALLOCATION_SIZE,
                stripe,
                0,
                b"\x00" * 4,
                parent,
                offset,
                size,
                count,
                user_data,
                type_pointer,
            )

        file_size = package_offset + package_size
        header = bytearray(package_offset)
        header[:6] = b"CIVBLP"
        struct.pack_into("<H5I", header, 6, 2, package_offset, package_size, file_size, 1, file_size)
        return bytes(header + package)

    def test_probe_follows_typed_parent_allocation_to_texture_record(self) -> None:
        fixture = self.make_fixture()
        report = probe.probe_package_bytes(fixture, "synthetic.blp")

        self.assertEqual(report["schema"], "c3x.civblp_material_probe.v0")
        self.assertEqual(report["allocation_table"]["entry_count"], 10)
        self.assertEqual(report["material_record"]["allocation_pointer"], 5)
        self.assertEqual(report["material_record"]["entry_name_pointer"], 2)
        texture = report["material_record"]["candidate_texture_pointers"][0]
        self.assertEqual(texture["pointer"], 10)
        self.assertEqual(texture["candidate_record"]["bytes"], 104)
        self.assertEqual(
            [item["value"] for item in texture["candidate_record"]["strings"]],
            ["TEXTURE_SYNTHETIC", "Synthetic_Class__"],
        )
        self.assertTrue(any(item["name"] == "unknown_0x70" for item in report["material_record"]["unknown_qwords"]))

    def test_reads_direct_and_basic_string_allocations(self) -> None:
        direct = b"direct\x00"
        basic = struct.pack("<II", 6, 5) + b"basic\x00"
        data = direct + basic

        self.assertEqual(probe.read_allocated_string(data, 0, len(direct)), ("direct", 0))
        self.assertEqual(
            probe.read_allocated_string(data, len(direct), len(basic)),
            ("basic", len(direct) + 8),
        )
        self.assertIsNone(probe.read_allocated_string(b"unterminated", 0, 12))

    def test_temp_stripe_inference_requires_two_independent_type_names(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least two"):
            probe.infer_temp_stripe_base(b"", [], minimum_support=1)

    def test_report_is_deterministic_and_file_probe_skips_big_data(self) -> None:
        fixture = self.make_fixture() + b"not package metadata"
        fixture = bytearray(fixture)
        struct.pack_into("<I", fixture, 24, len(fixture))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "fixture.blp"
            path.write_bytes(fixture)
            first = probe.probe_file(path)
            second = probe.probe_file(path)

        self.assertEqual(json.dumps(first, sort_keys=True), json.dumps(second, sort_keys=True))
        self.assertIn("skipped big-data payload", first["read_policy"])

    def test_header_rejects_inconsistent_package_boundary(self) -> None:
        fixture = bytearray(self.make_fixture())
        struct.pack_into("<I", fixture, 16, 0x777)
        with self.assertRaisesRegex(ValueError, "does not end"):
            probe.probe_package_bytes(bytes(fixture), "broken.blp")

    def test_header_size_mismatch_is_opt_in_and_reported(self) -> None:
        fixture = self.make_fixture()
        declared = len(fixture) + 0x5000
        header = bytearray(fixture[: probe.FILE_HEADER_SIZE])
        struct.pack_into("<I", header, 24, declared)
        with self.assertRaisesRegex(ValueError, "declared file size"):
            probe.parse_file_header(bytes(header), len(fixture))
        parsed = probe.parse_file_header(
            bytes(header),
            len(fixture),
            allow_declared_size_mismatch=True,
        )
        self.assertEqual(declared, parsed["declared_file_bytes"])
        self.assertEqual(len(fixture), parsed["actual_file_bytes"])
        self.assertTrue(parsed["declared_size_mismatch_accepted"])

    def test_allocation_table_requires_zero_padding_and_known_stripe(self) -> None:
        fixture = bytearray(self.make_fixture())
        marker = fixture.index(probe.ENTRY_MAP_TYPE)
        table = marker + len(probe.ENTRY_MAP_TYPE)
        fixture[table] = 7
        with self.assertRaisesRegex(ValueError, "allocation-table candidate"):
            probe.probe_package_bytes(bytes(fixture), "broken.blp")


if __name__ == "__main__":
    unittest.main()

"""Read-only GOG unit-hook byte/stack audit; never patches or executes the game."""
import argparse
import csv
import hashlib
import json
import re
import struct
from pathlib import Path


def audit(path, csv_path=None):
    data = path.read_bytes()
    pe = struct.unpack_from("<I", data, 0x3c)[0]
    if data[:2] != b"MZ" or data[pe:pe+4] != b"PE\0\0" or struct.unpack_from("<H", data, pe+4)[0] != 0x14c:
        raise ValueError("expected a Windows x86 PE executable")
    count = struct.unpack_from("<H", data, pe+6)[0]
    optional = pe+24
    base = struct.unpack_from("<I", data, optional+28)[0]
    sections = optional+struct.unpack_from("<H", data, pe+20)[0]

    def read(address, size):
        rva = address-base
        for index in range(count):
            _virtual_size, start, raw_size, raw = struct.unpack_from("<4I", data, sections+index*40+8)
            if start <= rva and rva+size <= start+raw_size:
                return data[raw+rva-start:raw+rva-start+size]
        raise ValueError("audit address outside file-backed image")

    expected = {
        0x5cbf50: "81eccc000000", 0x5f88b0: "8b41085685c0",
        0x5f8940: "8b44240483ec14", 0x5f88c0: "c21800", 0x5f8957: "c22400",
        # Reduced call pushes palette, palette path, 2, 1, 1, Y, X, canvas, background.
        0x5cc361: "5250", 0x5cc36b: "6a02", 0x5cc374: "6a01", 0x5cc378: "6a01",
    }
    for address, text in expected.items():
        if read(address, len(text)//2).hex() != text:
            raise ValueError(f"unit-hook byte mismatch at {address:#010x}")
    calls = {0x5cc33b: 0x5f88b0, 0x5cc3a1: 0x5f8940,
             0x5cc861: 0x5f88b0, 0x5cc8bc: 0x5f88b0,
             0x5cc926: 0x5f8940, 0x5cc98d: 0x5f8940}
    for address, target in calls.items():
        instruction = read(address, 5)
        if instruction[0] != 0xe8 or address+5+struct.unpack_from("<i", instruction, 1)[0] != target:
            raise ValueError(f"unit body call mismatch at {address:#010x}")
    csv_path = csv_path or Path(__file__).resolve().parents[2]/"civ_prog_objects.csv"
    wanted = {"Unit_tick_anim": (0x5cbf50, 4), "Sprite_draw_unit_body_normal": (0x5f88b0, 6),
              "Sprite_draw_unit_body_reduced": (0x5f8940, 9)}
    rows = {}
    with csv_path.open(newline="") as stream:
        for row in csv.reader(stream, skipinitialspace=True):
            if len(row) < 6 or row[4] not in wanted:
                continue
            name = row[4]
            address, arguments = wanted[name]
            signature = re.search(r"\)\s*\((.*)\)\s*$", row[5])
            if (name in rows or int(row[1], 16) != address or not signature or
                    len(signature.group(1).split(","))-2 != arguments):
                raise ValueError(f"CSV unit-hook address/signature mismatch: {name}")
            rows[name] = {"capability": row[0].strip(), "gog_address": f"0x{address:08X}",
                          "stack_arguments": arguments, "signature": row[5]}
    if set(rows) != set(wanted):
        raise ValueError("CSV unit hooks missing")
    return {"schema": "c3x.unit_hook_audit.v1", "status": "pass",
        "executable_sha256": hashlib.sha256(data).hexdigest(),
        "build": "GOG address layout; other builds unverified",
        "byte_checks": {f"0x{a:08X}": text for a, text in expected.items()},
        "body_calls": {f"0x{a:08X}": f"0x{t:08X}" for a, t in calls.items()},
        "normal_stack_arguments": 6, "reduced_stack_arguments": 9,
        "csv_rows": rows, "csv_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
        "csv_inleads_ready": all(row["capability"] == "inlead" for row in rows.values()),
        "native_selection_and_hud": "outside intercepted primitives; retained by original Unit_tick_anim",
        "installed_hook_execution_verified": False, "csv_modified_by_audit": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, default=Path("Renderer/native/build/unit-audit-original.exe"))
    parser.add_argument("--output", type=Path, default=Path("Renderer/lab/out/verification/animation/unit-hook-audit.json"))
    args = parser.parse_args()
    result = audit(args.executable)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result))

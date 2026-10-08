from __future__ import annotations

import json
import struct
import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np

from Renderer.tools.asset_compiler import civ3_unit_action_timing as timing


def flc(path: Path, frames: int, frame_ms: int) -> None:
    header = bytearray(128)
    struct.pack_into("<IHHHH", header, 0, 128, 0xAF12, frames * 8, 4, 4)
    struct.pack_into("<I", header, 16, frame_ms)
    struct.pack_into("<2H", header, 96, 8, frames)
    path.write_bytes(bytes(header))


def wav(path: Path, level: float, silence_s: float) -> None:
    rate = 8000
    tone = (np.sin(np.arange(int(rate * .2)) * .3) * level * 32767).astype("<i2")
    samples = np.concatenate([np.zeros(int(rate * silence_s), "<i2"), tone])
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(rate)
        out.writeframes(samples.tobytes())


def amb(tracks) -> bytes:
    """Civ III AMB: prgm/kmap chunks, then MIDI (480 ticks, 1 s per beat)."""
    data = bytearray()
    for program, (name, sample, _) in enumerate(tracks, 1):
        body = struct.pack("<7I", program, 3, 200, 0, 127, 75, 250) + name.encode() + b"\0" + name.encode() + b"\0"
        data += b"prgm" + struct.pack("<I", len(body)) + body
        body = struct.pack("<3I", 2, 0, 0) + name.encode() + b"\0" + struct.pack("<5I", 1, 12, 127, 0, 1) + \
            sample.encode() + b"\0" + struct.pack("<I", 250)
        data += b"kmap" + struct.pack("<I", len(body)) + body
    data += b"glbl" + struct.pack("<I", 4) + b"\0\0\0\0"
    data += b"MThd" + struct.pack(">IHHH", 6, 1, len(tracks) + 1, 480)
    tempo = b"\x00\xff\x51\x03\x0f\x42\x40\x00\xff\x2f\x00"
    data += b"MTrk" + struct.pack(">I", len(tempo)) + tempo
    for program, (_, _, ticks) in enumerate(tracks, 1):
        delta = bytes([0x80 | (ticks >> 7), ticks & 0x7F]) if ticks > 127 else bytes([ticks])
        track = b"\x00\xc0" + bytes([program]) + delta + b"\x90\x3c\x40\x10\x80\x3c\x40\x00\xff\x2f\x00"
        data += b"MTrk" + struct.pack(">I", len(track)) + track
    return bytes(data)


class Civ3UnitActionTimingTests(unittest.TestCase):
    def test_attack_length_cues_and_sync_point(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            unit = root / "Art/Units/Gun"
            unit.mkdir(parents=True)
            flc(unit / "GunAttackA.flc", 15, 66)
            flc(unit / "GunBomb.flc", 20, 83)
            wav(unit / "GunCreak.wav", .3, 0)
            wav(unit / "GunBoom.wav", .9, 0)
            wav(unit / "GunBlast.wav", .8, .5)
            # The quiet creak starts first; the loud boom follows 120 ticks (0.25 s) in.
            (unit / "GunAttack.amb").write_bytes(amb([("Creak", "GunCreak.wav", 0), ("Boom", "GunBoom.wav", 120)]))
            (unit / "Gun.INI").write_text("[Animations]\nATTACK1=GunAttackA.flc\nVICTORY=GunBomb.flc\n"
                                          "[Sound Effects]\nATTACK1=GunAttack.amb\nVICTORY=GunBlast.wav\n")
            table = timing.build(root, output=root / "out.json")
            self.assertEqual(table, json.loads((root / "out.json").read_text()))
        attack = table["units"]["PRTO_Gun"]["ATTACK1"]
        self.assertEqual((15, 66, .99), (attack["frames"], attack["frame_ms"], attack["duration_s"]))
        self.assertEqual([["GunCreak.wav"], ["GunBoom.wav"]], [c["samples"] for c in attack["cues"]])
        self.assertAlmostEqual(.25, attack["sync_s"], delta=.02)
        # A plain wav is one cue at 0; its sync point is the blast after the silence.
        victory = table["units"]["PRTO_Gun"]["VICTORY"]
        self.assertAlmostEqual(1.66, victory["duration_s"])
        self.assertAlmostEqual(.5, victory["sync_s"], delta=.02)


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
"""Measure Civ III unit action timing into a generic table for unit packs.

For every PRTO key (as `civ3_unit_sprites` resolves it), the unit INI's
ATTACK1, VICTORY and DEATH entries are read: the FLC's own length (frames per
direction x header frame delay) and its sound cues. An `.amb` sound sequence is
a MIDI file with one track per sample: each note-on is a cue at that time (480
ticks per beat at the file's tempo), its sample resolved through the `prgm`
and `kmap` chunks to wav files. A plain `.wav` is one cue at 0. Each cue keeps
the sample's peak level, so a pack can align its effects with the loudest
moment (the gunshot) without naming any sample, and its onset: the first time
the sample's 20 ms loudness reaches half its maximum (a bomb sample's first
blast follows a whistle). Only numbers and file names
are written; no art or audio.

    python3 -m Renderer.tools.asset_compiler.civ3_unit_action_timing
    python3 -m Renderer.tools.asset_compiler.civ3_unit_action_timing --scenario-root PATH --output FILE

The Civ III root defaults to $C3X_CIV3_ROOT, else two levels above this checkout.
"""
from __future__ import annotations

import argparse
import configparser
import json
import os
import re
import struct
import wave
from pathlib import Path

import numpy as np

from Renderer.tools.asset_compiler.civ3_unit_sprites import game_roots, pedia_names

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = ROOT / "Renderer/inventory/civ3_unit_action_timing.json"
ACTIONS = ("ATTACK1", "VICTORY", "DEATH")


def flc_length(path: Path) -> dict:
    data = path.read_bytes()[:128]
    if len(data) < 128 or struct.unpack_from("<H", data, 4)[0] not in (0xAF12, 0xAF11):
        raise ValueError(f"not an FLC: {path}")
    total, = struct.unpack_from("<H", data, 6)
    speed, = struct.unpack_from("<I", data, 16)
    directions, length = struct.unpack_from("<2H", data, 96)
    frames = length or total
    return {"frames": frames, "frame_ms": speed, "duration_s": round(frames * speed / 1000, 4)}


def wav_level(path: Path) -> tuple[float, float] | None:
    """(peak level in [0, 1], onset seconds) of a PCM wav; None for other encodings."""
    try:
        with wave.open(str(path)) as audio:
            width, channels, rate = audio.getsampwidth(), audio.getnchannels(), audio.getframerate()
            raw = audio.readframes(audio.getnframes())
    except (wave.Error, EOFError, OSError):
        return None
    if width == 1:
        samples = np.frombuffer(raw, np.uint8).astype(float) - 128
    elif width == 2:
        samples = np.frombuffer(raw, "<i2").astype(float)
    else:
        return None
    samples = samples[:len(samples) // channels * channels].reshape(-1, channels).mean(1)
    if not len(samples) or not np.abs(samples).max():
        return 0.0, 0.0
    window = max(1, int(rate * .02))
    loudness = np.sqrt(np.convolve(samples * samples, np.ones(window) / window, "same"))
    onset = int(np.argmax(loudness >= .5 * loudness.max()))
    return round(float(np.abs(samples).max()) / (1 << (8 * width - 1)), 3), round(onset / rate, 4)


def _vlq(data: bytes, at: int):
    value = 0
    while True:
        byte = data[at]
        at += 1
        value = (value << 7) | (byte & 0x7F)
        if byte < 0x80:
            return value, at


def amb_cues(data: bytes) -> list[tuple[float, list[str]]]:
    """(seconds, wav names) per note-on of an AMB sound sequence."""
    programs, samples, tracks = {}, {}, []
    division, tempo, at = 480, 1_000_000, 0
    while at + 8 <= len(data):
        tag = data[at:at + 4]
        if tag in (b"prgm", b"kmap", b"glbl"):
            size, = struct.unpack_from("<I", data, at + 4)
            body = data[at + 8:at + 8 + size]
            if tag == b"prgm":
                names = [n.decode("latin-1").strip() for n in body[28:].split(b"\0")]
                programs[struct.unpack_from("<I", body, 0)[0]] = names[1] if len(names) > 1 else names[0]
            elif tag == b"kmap":
                text = [m.decode("latin-1") for m in re.findall(rb"[\x20-\x7e]{3,}", body)]
                if text:
                    samples[text[0].strip()] = [t for t in text[1:] if t.lower().endswith(".wav")]
            at += 8 + size
        elif tag == b"MThd":
            division, = struct.unpack_from(">H", data, at + 12)
            at += 14
        elif tag == b"MTrk":
            size, = struct.unpack_from(">I", data, at + 4)
            j, end, ticks, status, program, notes = at + 8, at + 8 + size, 0, 0, None, []
            while j < end:
                delta, j = _vlq(data, j)
                ticks += delta
                if data[j] == 0xFF:
                    kind = data[j + 1]
                    length, k = _vlq(data, j + 2)
                    if kind == 0x51:
                        tempo = int.from_bytes(data[k:k + length], "big")
                    j = k + length
                    continue
                if data[j] & 0x80:
                    status = data[j]
                    j += 1
                high = status & 0xF0
                if high in (0xC0, 0xD0):
                    program = data[j] if high == 0xC0 else program
                    j += 1
                else:
                    if high == 0x90 and data[j + 1] > 0:
                        notes.append(ticks)
                    j += 2
            tracks.append((program, notes))
            at = end
        else:
            at += 1
    seconds = tempo / 1e6 / division
    cues = [(round(t * seconds, 4), samples.get(programs.get(program, ""), []))
            for program, notes in tracks if program is not None for t in notes]
    return sorted(cues)


def action_timing(folder: Path, parser: configparser.ConfigParser, action: str) -> dict | None:
    files = {p.name.lower(): p for p in folder.iterdir()}
    flc = files.get(parser.get("Animations", action, fallback="").strip().lower())
    if flc is None:
        return None
    timing = flc_length(flc)
    sound = files.get(parser.get("Sound Effects", action, fallback="").strip().lower())
    cues = []
    if sound is not None and sound.suffix.lower() == ".amb":
        cues = amb_cues(sound.read_bytes())
    elif sound is not None:
        cues = [(0.0, [sound.name])]
    timing["cues"] = []
    for t, wavs in cues:
        levels = [v for v in (wav_level(files[w.lower()]) for w in wavs if w.lower() in files) if v is not None]
        level = max(levels) if levels else (None, 0.0)
        timing["cues"].append({"t": t, "samples": wavs, "peak": level[0], "onset_s": round(t + level[1], 4)})
    loud = [c for c in timing["cues"] if c["peak"] is not None]
    # The loudest cue's onset (earliest on a tie) is the action's sync point: a gun's report.
    timing["sync_s"] = min(loud, key=lambda c: (-c["peak"], c["onset_s"]))["onset_s"] if loud else None
    return timing


def build(civ3_root: Path, scenarios=(), output: Path = DEFAULT_OUTPUT) -> dict:
    roots = game_roots(civ3_root, scenarios)
    units = {}
    for prto, name in sorted(pedia_names(roots).items()):
        for root in roots:
            folder = root / "Art/Units" / name
            ini = next(folder.glob("*.[iI][nN][iI]"), None) if folder.is_dir() else None
            if ini is None:
                continue
            parser = configparser.ConfigParser(strict=False, interpolation=None)
            parser.optionxform = str
            parser.read_string(ini.read_text(errors="replace"))
            actions = {}
            for action in ACTIONS:
                try:
                    timing = action_timing(folder, parser, action)
                except (ValueError, IndexError, struct.error):
                    timing = None
                if timing is not None:
                    actions[action] = timing
            if actions:
                units[prto] = {"art": name, **actions}
            break
    if not units:
        raise ValueError("No Civ III unit art found; set C3X_CIV3_ROOT or --civ3-root")
    table = {"schema": 1,
             "method": "INI ATTACK1/VICTORY/DEATH: FLC frames per direction x header frame delay; AMB note-on cues "
                       "with sample peak and onset (20 ms loudness reaches half its maximum); "
                       "sync_s = loudest cue's onset (earliest on a tie)",
             "units": units}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(table, indent=1, sort_keys=True) + "\n")
    return table


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--civ3-root", type=Path, default=Path(os.environ.get("C3X_CIV3_ROOT", ROOT.parents[1])))
    parser.add_argument("--scenario-root", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    table = build(args.civ3_root, args.scenario_root, args.output)
    print(f"{len(table['units'])} Civ III unit action timings -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

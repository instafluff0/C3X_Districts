#!/usr/bin/env python3
"""Lab: every unit, Civ III | now | proposed, at map scale (Mac preview).

"now" is the pack's owner masks under the live look; "proposed" is the
transferred civ colours (transfer.py) under the proposed look (look.py
PROPOSED with the faint outline). Two directions per unit (SW, NE), noon,
on grass. Units without Civ III art keep their pack colours under the new
look. Each unit's row and statistics are cached, so a run can resume or
redo single units; the rows are then composed into contact sheets.

    python3 Renderer/lab/studies/unit_civ_colors/survey.py [--units PRTO_Tank,...] [--jobs 4] [--per-sheet 10]

Writes Renderer/lab/out/unit-civ-colors/survey/ (rows, per-unit JSON) and
survey-NN.png sheets plus survey.json (shares, fit and review flags).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import look as L  # noqa: E402
import transfer as T  # noqa: E402
from image_io import write_png  # noqa: E402
from Renderer.tools.asset_compiler import civ3_flc  # noqa: E402

OUT = T.OUT / "survey"
NOW = {"hour": 12, "masks": "current", "flat_owner": False}
PROPOSED = {"hour": 12, **L.PROPOSED, "outline": .35}
DIRECTIONS = (5, 1)
CELL_W, CELL_H, ZOOM = 150, 120, 2
FIT_MIN = .45   # sprite/model silhouette overlap below which the pack's colours are kept


def units() -> list:
    bindings = json.loads((T.PACK / "bindings.json").read_text())
    return sorted(v["key0"] for v in bindings.values() if isinstance(v, dict) and "idle" in v)


def bare(key: str) -> dict:
    """A unit without Civ III art: its views and pack masks only."""
    bindings = json.loads((T.PACK / "bindings.json").read_text())
    binding = next(v for v in bindings.values() if isinstance(v, dict) and v.get("key0") == key)
    parts = list(T.coverage.idle_parts(T.PACK, binding))
    textures = sorted({part["texture"] for _, part, *_ in parts})
    rgb = {t: T.coverage.dds_rgb(T.PACK / t) for t in textures}
    return {"key": key, "binding": binding, "parts": parts, "rgb": rgb,
            "size": {t: rgb[t].shape[:2] for t in textures},
            "views": {d: T.raster(parts, binding, d) for d in DIRECTIONS}, "sprites": {}}


def flags(stats: dict) -> list:
    out = []
    if "error" in stats:
        return ["error"]
    if stats.get("civ3_share") is None:
        return ["no Civ III art"]
    civ3, new = stats["civ3_share"], stats["new_share"]
    if stats.get("kept"):
        out.append("kept pack colours: " + ("weak fit" if stats["iou"] < FIT_MIN else "no consistent area"))
    elif abs(new - civ3) > .02 + .2 * civ3:
        out.append("share")
    return out


def survey_unit(key: str, civ: int) -> dict:
    T.CIVNESS = T.civness()
    civ_palette = civ3_flc.read_pcx_palette(T.CIV3_ROOT / "Art/Units/Palettes" / f"ntp{civ:02d}.pcx")
    owner = L.linear(T.owner_colour(civ_palette))
    grass = L.inverse_display(L.GRASS_DISPLAY[None, :].copy())[0]
    materials = L.Materials()
    bindings = json.loads((T.PACK / "bindings.json").read_text())
    binding = next(v for v in bindings.values() if isinstance(v, dict) and v.get("key0") == key)
    path = T.civ3_art([binding[f"key{i}"] for i in range(binding["key_count"])])
    stats = {"key": key}
    try:
        if path is None:
            result, proposed = bare(key), {**PROPOSED, "masks": "current"}
        else:
            result, proposed = T.transfer(key), PROPOSED
            stats.update({k: round(result[k], 4) for k in ("civ3_share", "current_share", "new_share", "iou", "tau")})
            if result["iou"] < FIT_MIN or result["new_share"] < .25 * result["civ3_share"]:
                # The 3D model is too unlike its sprite for placement to carry
                # over (a Huey for a Chinook), or no consistent area matches
                # Civ III's share (one-material hulls): keep the pack's colours.
                proposed = {**PROPOSED, "masks": "current"}
                stats["kept"] = True
            for direction in DIRECTIONS:   # sprites with fewer directions (missiles)
                if direction not in result["views"]:
                    result["views"][direction] = T.raster(result["parts"], result["binding"], direction)
        canvas = np.zeros((CELL_H, CELL_W * 3 * len(DIRECTIONS), 3))
        canvas[:] = L.GRASS_DISPLAY
        for slot, direction in enumerate(DIRECTIONS):
            x = slot * 3 * CELL_W + CELL_W // 2
            if path is not None and direction in result["sprites"]:
                shown, alpha, stats[f"civ3_{direction}"] = L.civ3_sprite(
                    result["sprites"][direction], civ3_flc.read(path).palette, civ_palette)
                L.place(canvas, shown, alpha, CELL_H // 2, x, CELL_W, CELL_H)
            for k, look in ((1, NOW), (2, proposed)):
                shown, alpha, cell, tone = L.render(result, direction, look, owner, materials, grass)
                stats[f"{'now' if k == 1 else 'proposed'}_{direction}"] = tone
                L.place(canvas, shown, alpha, CELL_H // 2, x + k * CELL_W, CELL_W, CELL_H, cell)
    except Exception as failure:   # one unit must not stop the survey
        stats["error"] = f"{type(failure).__name__}: {failure}"
        stats["trace"] = traceback.format_exc()[-2000:]
        canvas = np.full((CELL_H, CELL_W * 3 * len(DIRECTIONS), 3), .3)
    stats["flags"] = flags(stats)
    image = np.repeat(np.repeat((np.clip(canvas, 0, 1) * 255 + .5).astype(np.uint8), ZOOM, 0), ZOOM, 1)
    label = key[5:].replace("_", " ")
    if "civ3_share" in stats:
        label += f"  civ3 {stats['civ3_share']:.2f} now {stats['current_share']:.2f} new {stats['new_share']:.2f} fit {stats['iou']:.2f}"
    if stats["flags"]:
        label += "  " + " ".join(stats["flags"]).upper()
    image = np.concatenate([L.label_strip(label, image.shape[1]), image], 0)
    OUT.mkdir(parents=True, exist_ok=True)
    write_png(OUT / f"{key}.png", image)
    (OUT / f"{key}.json").write_text(json.dumps(stats, indent=1) + "\n")
    return stats


def read_png(path: Path) -> np.ndarray:
    """Minimal reader for the RGB PNGs write_png produces (zlib, filter 0)."""
    import struct
    import zlib
    data = path.read_bytes()
    width, height = struct.unpack(">II", data[16:24])
    chunks, offset = [], 8
    while offset < len(data):
        length, kind = struct.unpack(">I4s", data[offset:offset + 8])
        if kind == b"IDAT":
            chunks.append(data[offset + 8:offset + 8 + length])
        offset += 12 + length
    raw = np.frombuffer(zlib.decompress(b"".join(chunks)), np.uint8).reshape(height, width * 3 + 1)
    if raw[:, 0].any():
        raise ValueError(f"{path}: filtered PNG rows are not supported")
    return raw[:, 1:].reshape(height, width, 3)


def sheets(keys: list, per_sheet: int) -> list:
    header = np.concatenate([L.label_strip(text, CELL_W * ZOOM) for _ in DIRECTIONS
                             for text in ("civ3", "now", "proposed")], 1)
    written = []
    for n in range(0, len(keys), per_sheet):
        rows = [read_png(OUT / f"{key}.png") for key in keys[n:n + per_sheet] if (OUT / f"{key}.png").is_file()]
        if not rows:
            continue
        path = T.OUT / f"survey-{n // per_sheet + 1:02d}.png"
        write_png(path, np.concatenate([header] + rows, 0))
        written.append(path)
    return written


def vm_capture_running() -> bool:
    """A scripted game capture is running in the VM (it shares this CPU)."""
    # Only a prlctl process counts: a shell merely mentioning the script
    # (including a loop waiting on it) must not match.
    listing = subprocess.run(["ps", "-Ao", "comm=,args="], capture_output=True, text=True).stdout
    return any(line.strip().split(" ", 1)[0].endswith("prlctl") and "run_scripted_game_test" in line
               for line in listing.splitlines())


def report(stats: dict):
    print(stats["key"], {k: stats[k] for k in ("civ3_share", "new_share", "iou") if k in stats},
          stats["flags"], stats.get("error", ""), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--units", default="", help="comma list; default every unit in the pack")
    parser.add_argument("--civ", type=int, default=1)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--per-sheet", type=int, default=10)
    parser.add_argument("--sheets-only", action="store_true")
    parser.add_argument("--polite", action="store_true", help="pause while a VM game capture runs")
    args = parser.parse_args(argv)
    every = units()
    chosen = [k for k in args.units.split(",") if k] or every
    if not args.sheets_only:
        with ProcessPoolExecutor(args.jobs) as pool:
            pending = []
            for key in chosen:
                if args.polite and vm_capture_running():
                    # Host load distorts the VM's game captures: let them finish.
                    for future in pending:
                        report(future.result())
                    pending = []
                    while vm_capture_running():
                        time.sleep(5)
                while len(pending) >= args.jobs:
                    report(pending.pop(0).result())
                pending.append(pool.submit(survey_unit, key, args.civ))
            for future in pending:
                report(future.result())
    summary = {key: json.loads((OUT / f"{key}.json").read_text()) for key in every if (OUT / f"{key}.json").is_file()}
    (T.OUT / "survey.json").write_text(json.dumps(
        {key: {k: v for k, v in s.items() if k in ("civ3_share", "current_share", "new_share", "iou", "tau", "flags", "error", "kept")}
         for key, s in summary.items()}, indent=1) + "\n")
    print("flagged:", {key: s["flags"] for key, s in summary.items() if s.get("flags")})
    for path in sheets([key for key in every if key in summary], args.per_sheet):
        print(path)


if __name__ == "__main__":
    main()

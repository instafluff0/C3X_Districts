#!/usr/bin/env python3
"""Lab: combat effect sequences through the live unit path plus generic effect graphs.

    python3 Renderer/lab/studies/combat_effects/sequence.py render SCENARIO
    python3 Renderer/lab/studies/combat_effects/sequence.py compose SCENARIO [--outcome hit|miss|water]

`render` draws the scenario's units at each frame time with the unit
readability study's x64 fixture (production `prepare_real`/`draw_real`).
`compose` samples the generic effect graphs at each frame time and composites
them in scene-linear light before the production display transfer.

Nothing here names a unit's effects. A scenario only picks units and a camera.
Muzzle releases, their sockets and the impact set come from data: the local
Civ VI timing table (`Renderer/packs/UnitEffectTiming/timing.json`, from
`tools/asset_compiler/civ6_unit_effect_timing.py`), the importer map
`tools/asset_compiler/civ6_combat_effect_map.json`, and the effect pack
(`tools/asset_compiler/effect_graph_profiles.json`). Civ III directs timing:
releases follow the attack clip; the impact plays when the attack animation
returns (Civ III's hit/miss FLC and sound); a bomber's stick follows its fly-over.
Diagnostic only: no game, staging or packs are changed.
"""
from __future__ import annotations

import argparse
import json
import math
import struct
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE.parent / "unit_readability"))
import sheet  # noqa: E402
from image_io import write_png  # noqa: E402
from Renderer.tools.asset_compiler import effect_graph_compiler as graphs  # noqa: E402
from Renderer.tools.asset_compiler import unit_combat_bindings as combat_bindings  # noqa: E402
from Renderer.tools.asset_compiler import unit_owner_coverage as dds  # noqa: E402

OUT = ROOT / "Renderer/lab/out/combat-effects"
TIMING = ROOT / "Renderer/packs/UnitEffectTiming/timing.json"
EFFECT_MAP = ROOT / "Renderer/tools/asset_compiler/civ6_combat_effect_map.json"
PACK = "UnitAnimationFidelity"
Z_PIXELS = 150 * 128 / 224
CELL = (380, 250)
COLUMNS = 4
FIRE = (0.0, 1.0)          # tile-space direction from attacker to target (south-west)
WATER_DISPLAY = np.array([38, 92, 104], float) / 255

# A scenario picks units, a camera and native outcomes; timing comes from data.
SCENARIOS = {
    "artillery": {"attacker": "PRTO_Artillery", "target": "PRTO_Infantry", "style": "bombard"},
    "cannon": {"attacker": "PRTO_Cannon", "target": "PRTO_Pikeman", "style": "bombard"},
    "catapult": {"attacker": "PRTO_Catapult", "target": "PRTO_Spearman", "style": "bombard"},
    "battleship": {"attacker": "PRTO_Battleship", "target": "PRTO_Infantry", "style": "bombard",
                   "attacker_ground": "water"},
    "bomber": {"attacker": "PRTO_Bomber", "target": "PRTO_Infantry", "style": "bomb_run",
               "frames": (0.25, 0.6, 0.85, 1.05, 1.18, 1.32, 1.6, 2.2)},
    # Normal combat: both ships fire every round; the round's loser is hit and
    # the winner's tile takes near misses. The defender sinks after round two.
    "naval_battle": {"attacker": "PRTO_Battleship", "target": "PRTO_Battleship", "style": "duel",
                     "attacker_ground": "water", "target_ground": "water", "rounds": ("defender", "defender")},
}
SECONDS_PER_TILE = 0.4     # bomber fly-over speed in the Lab
DROP_TILES_BEFORE = 0.75   # Civ III queues the bomb FLC 25% into the step over the target
BOMB_RUN_LIFT = 2.0        # bomb-run altitude in Civ III sprite lifts (idle hover is 0.5); pending pack metadata
MISS_OVERSHOOT_TILE = 0.45  # a near miss on a ship lands beyond it along the line of fire
CIV3_TIMING = ROOT / "Renderer/inventory/civ3_unit_action_timing.json"
CIV3_SPRITES = ROOT / "Renderer/inventory/civ3_unit_sprite_sizes.json"


def tile_to_screen(x, y, z=0.0):
    return (x - y) * 64.0, (x + y) * 32.0 - z * Z_PIXELS


def cell_anchor(index):
    column, row = index % COLUMNS, index // COLUMNS
    return column * CELL[0] + 250, row * CELL[1] + 95


def target_of(anchor):
    dx, dy = tile_to_screen(2 * FIRE[0], 2 * FIRE[1])
    return int(anchor[0] + dx), int(anchor[1] + dy)


# ---- data: timing, sockets and the importer map -----------------------------

def combat_binding(key, pack=PACK):
    """A unit's attack effects from data (the pack builder's own resolver):
    releases (normalized progress, profile, socket point in model space),
    impact set and bearing."""
    root = ROOT / "Renderer/packs" / pack
    bindings = json.loads((root / "bindings.json").read_text())
    manifest = json.loads((root / "manifest.json").read_text())
    binding_key, binding = next((k, v) for k, v in bindings.items() if isinstance(v, dict) and v.get("key0") == key)
    assets = dds.components_by_binding(manifest, bindings)[binding_key]["attack"]
    combat = combat_bindings.resolve(json.loads(TIMING.read_text())["units"][key], json.loads(EFFECT_MAP.read_text()),
                                     assets, binding["attack"], lambda path: (root / path).read_bytes())
    for release in combat["releases"]:
        release["position"] = np.array(release["position"])
    return {**combat, "binding": binding, "impact_set": combat["impact_set"] or "shell"}


def civ3_action(key, action="ATTACK1"):
    return json.loads(CIV3_TIMING.read_text())["units"][key][action]


def attack_plan(key):
    """A unit's attack in Civ III time: duration, warped clip, release times."""
    combat, civ3 = combat_binding(key), civ3_action(key)
    duration = civ3["duration_s"]
    first = min((r["normalized"] for r in combat["releases"]), default=None)
    sync = civ3["sync_s"] and min(.95, civ3["sync_s"] / duration)
    releases = [{**r, "t": combat_bindings.native_phase(r["normalized"], sync, first) * duration}
                for r in combat["releases"]]
    return {**combat, "duration": duration, "clip": lambda p: combat_bindings.clip_phase(p, sync, first),
            "releases": releases}


def model_to_screen(point, binding, direction):
    angle = math.radians(binding.get("yaw_offset", 225.0) + (direction % 8) * 45)
    c, s, scale = math.cos(angle), math.sin(angle), binding["scale"]
    x = (point[0] * c - point[1] * s) * scale
    y = (point[0] * s + point[1] * c) * scale
    return tile_to_screen(x, y, (point[2] + binding["offset_z"]) * scale)


# ---- scenario timeline -------------------------------------------------------

def facing(toward, bearing):
    """Civ III direction (1-8) that puts a unit's line of fire `bearing` degrees
    off its bow onto the direction `toward`."""
    return (toward - round(bearing / 45) - 1) % 8 + 1


def firing(plan, start, anchor, direction, yaw, tag):
    """A unit's muzzle releases as effect events."""
    events = []
    for number, release in enumerate(plan["releases"]):
        sx, sy = model_to_screen(release["position"], plan["binding"], direction)
        events.append((release["profile"], start + release["t"], (anchor[0] + sx, anchor[1] + sy), yaw,
                       f"{tag}/{number}"))
    return events


def attacking(key, plan, t, direction, x, y, layer="attacker"):
    if 0 <= t < plan["duration"]:
        return (key, direction, 3, plan["clip"](t / plan["duration"]), x, y, layer)
    return (key, direction, 1, 0.0, x, y, layer)


def overshoot(point, towards):
    dx, dy = tile_to_screen(towards[0] * MISS_OVERSHOOT_TILE, towards[1] * MISS_OVERSHOOT_TILE)
    return point[0] + dx, point[1] + dy


def timeline(name, outcome="hit"):
    """Per frame: unit draws (key, direction, action, phase, x, y, layer[, dz])
    and the effect events live at that frame (profile, start, anchor, yaw, id)."""
    scenario = SCENARIOS[name]
    impact_sets = json.loads(graphs.DEFAULT_SOURCE.read_text())["impact_sets"]
    yaw, back = math.atan2(FIRE[1], FIRE[0]), math.atan2(-FIRE[1], -FIRE[0])
    if scenario["style"] == "bomb_run":
        key = scenario["attacker"]
        sprite = json.loads(CIV3_SPRITES.read_text())["units"][key]
        binding = combat_binding(key)["binding"]
        hover = json.loads((ROOT / "Renderer/native/environment_refresh/unit_quality.json").read_text())["hover"]
        dz = (BOMB_RUN_LIFT - hover["factor"]) * sprite["lift"] / (Z_PIXELS * binding["scale"])
        # The bomb FLC (VICTORY) is queued 25% into the step over the target;
        # its first blast is at the bomb sound's onset.
        blast = (2 - DROP_TILES_BEFORE) * SECONDS_PER_TILE + civ3_action(key, "VICTORY")["sync_s"]
        munition = combat_binding(key)["impact_set"]
        impact = impact_sets.get(munition + "_drop", impact_sets[munition])
        frames = scenario["frames"]
    else:
        plan = attack_plan(scenario["attacker"])
        impact = impact_sets[plan["impact_set"]]
        first = min((r["t"] for r in plan["releases"]), default=plan["duration"] * .15)
        frames = scenario.get("frames") or (first + .04, first + .3, first + .7, plan["duration"] * .9) + \
            tuple(plan["duration"] + d for d in (.03, .18, .5, 1.1))
    if scenario["style"] == "duel":
        defender = attack_plan(scenario["target"])
        rounds = scenario["rounds"]
        length = max(plan["duration"], defender["duration"]) + .3
        sink = len(rounds) * length
        frames = (first + .04, first + .35, plan["duration"] + .05, plan["duration"] + .4,
                  length + plan["duration"] + .1, sink + .2, sink + .8, sink + 1.6)
    result = []
    for index, t in enumerate(frames):
        anchor = cell_anchor(index)
        target = target_of(anchor)
        events, draws = [], []
        if scenario["style"] == "bomb_run":
            # Civ III flies the bomber from two tiles before the target to two past it.
            along = t / SECONDS_PER_TILE
            dx, dy = tile_to_screen(FIRE[0] * along, FIRE[1] * along)
            if along <= 3.2:  # it has left the frame afterwards
                draws.append((scenario["attacker"], 5, 2, along % 1, int(anchor[0] + dx), int(anchor[1] + dy), "air", dz))
            events.append((impact[outcome], blast, target, yaw, f"stick/{index}"))
            draws.append((scenario["target"], 1, 1, 0.0, *target, "target"))
        elif scenario["style"] == "bombard":
            direction = facing(5, plan["bearing"])
            draws.append(attacking(scenario["attacker"], plan, t, direction, *anchor))
            events += firing(plan, 0, anchor, direction, yaw, f"release/{index}")
            events.append((impact[outcome], plan["duration"], target, yaw, f"impact/{index}"))
            draws.append((scenario["target"], 1, 1, 0.0, *target, "target"))
        else:
            mine, theirs = facing(5, plan["bearing"]), facing(1, defender["bearing"])
            against = impact_sets[defender["impact_set"]]
            for number, loser in enumerate(rounds):
                start = number * length
                events += firing(plan, start, anchor, mine, yaw, f"a/{index}/{number}")
                events += firing(defender, start, target, theirs, back, f"d/{index}/{number}")
                if loser == "defender":
                    events.append((impact["ship"], start + plan["duration"], target, yaw, f"hit/{index}/{number}"))
                    events.append((against["water"], start + defender["duration"], overshoot(anchor, (-FIRE[0], -FIRE[1])),
                                   back, f"near/{index}/{number}"))
                else:
                    events.append((against["ship"], start + defender["duration"], anchor, back, f"hit/{index}/{number}"))
                    events.append((impact["water"], start + plan["duration"], overshoot(target, FIRE), yaw,
                                   f"near/{index}/{number}"))
            local = t % length if t < sink else -1
            draws.append(attacking(scenario["attacker"], plan, local, mine, *anchor))
            if t < sink:
                draws.append(attacking(scenario["target"], defender, local, theirs, *target, "target"))
            else:
                death = civ3_action(scenario["target"], "DEATH")["duration_s"]
                draws.append((scenario["target"], theirs, 6, min(.999, (t - sink) / death), *target, "target"))
        result.append({"t": t, "anchor": anchor, "target": target, "draws": draws, "events": events})
    return result


def render(name):
    """Each unit role renders to its own layer so `compose` can order units and
    effects by depth (the runtime depth-tests effects against units)."""
    OUT.mkdir(parents=True, exist_ok=True)
    frames = timeline(name)
    for layer in sorted({draw[6] for frame in frames for draw in frame["draws"]}):
        lines = [f"{key} {direction} {action} {phase:.4f} {x} {y}" + (f" {extra[1]:.4f}" if len(extra) > 1 else "")
                 for frame in frames for key, direction, action, phase, x, y, *extra in frame["draws"]
                 if extra[0] == layer]
        suffix = f"-{layer}"
        units = sheet.OUT / f"combat-{name}{suffix}.units"
        units.write_text("\n".join(lines) + "\n")
        command = (f'"{sheet.win(sheet.BUILD / "unit_sheet.exe")}" "{sheet.win(ROOT / "Renderer/packs" / PACK)}" '
                   f'"{sheet.win(units)}" "{sheet.win(sheet.OUT / ("combat-" + name + suffix + ".f16"))}" c8281e')
        print(sheet.vm(command, timeout=600).strip().splitlines()[-1])


# ---- compositing ---------------------------------------------------------------

def srgb_to_linear(c):
    return np.where(c <= .04045, c / 12.92, ((c + .055) / 1.055) ** 2.4)


def bc4(path):
    data = path.read_bytes()
    height, width = struct.unpack_from("<II", data, 12)
    blocks = np.frombuffer(data, np.uint8, (width // 4) * (height // 4) * 8, 148).reshape(-1, 8)
    out = np.zeros((height, width))
    for n, block in enumerate(blocks):
        a0, a1 = float(block[0]), float(block[1])
        table = [a0, a1] + ([a0 + (a1 - a0) * k / 7 for k in range(1, 7)] if a0 > a1 else
                            [a0 + (a1 - a0) * k / 5 for k in range(1, 5)] + [0, 255])
        bits = int.from_bytes(bytes(block[2:]), "little")
        by, bx = divmod(n, width // 4)
        for k in range(16):
            out[by * 4 + k // 4, bx * 4 + k % 4] = table[(bits >> (3 * k)) & 7] / 255
    return out


class Textures:
    def __init__(self):
        self.packs = {}
        for manifest in graphs.DEFAULT_TEXTURE_PACKS:
            for asset, entry in json.loads(manifest.read_text())["textures"].items():
                self.packs[asset] = (manifest.parent / entry["texture"], entry["format"])
        self.cache = {}

    def get(self, asset, alpha_asset=None):
        key = (asset, alpha_asset)
        if key not in self.cache:
            path, fmt = self.packs[asset]
            rgb = srgb_to_linear(dds.dds_rgb(path))
            alpha = dds.dds_alpha(path) if fmt.startswith("BC3") else np.ones(rgb.shape[:2])
            if alpha_asset:
                apath, afmt = self.packs[alpha_asset]
                alpha = bc4(apath) if afmt.startswith("BC4") else dds.dds_alpha(apath)
            self.cache[key] = (rgb, alpha)
        return self.cache[key]


def splat(canvas, particle, centre, angle, textures, light):
    """Composite one billboard (rotated quad) in scene-linear light."""
    rgb, alpha = textures.get(particle["texture"], particle.get("alpha_texture"))
    u0, v0, u1, v1 = particle["atlas_uv"]
    width, height = particle["size_tile"][0] * 128, particle["size_tile"][1] * 128
    # The pivot (sprite-space) sits at the particle position: shift the quad's centre.
    pu, pv = particle.get("pivot", [0.5, 0.5])
    c, s = math.cos(angle), math.sin(angle)
    centre = (centre[0] + (0.5 - pu) * width * c - (0.5 - pv) * height * s,
              centre[1] + (0.5 - pu) * width * s + (0.5 - pv) * height * c)
    reach = math.hypot(width, height) / 2 + 1
    x0, x1 = int(max(0, centre[0] - reach)), int(min(canvas.shape[1], centre[0] + reach + 1))
    y0, y1 = int(max(0, centre[1] - reach)), int(min(canvas.shape[0], centre[1] + reach + 1))
    if x0 >= x1 or y0 >= y1:
        return
    ys, xs = np.mgrid[y0:y1, x0:x1] + .5
    c, s = math.cos(angle), math.sin(angle)
    lx = ((xs - centre[0]) * c + (ys - centre[1]) * s) / width + .5
    ly = (-(xs - centre[0]) * s + (ys - centre[1]) * c) / height + .5
    inside = (lx >= 0) & (lx < 1) & (ly >= 0) & (ly < 1)
    if not inside.any():
        return
    u, v = u0 + lx * (u1 - u0), v0 + ly * (v1 - v0)

    def texel(image):
        h, w = image.shape[:2]
        return image[np.clip((v * h).astype(int), 0, h - 1), np.clip((u * w).astype(int), 0, w - 1)]
    colour, cover = texel(rgb) * np.array(particle["tint"]), texel(alpha) * inside * particle["opacity"]
    region = canvas[y0:y1, x0:x1]
    if particle["blend"] == "additive":
        region += colour * cover[..., None] * particle["intensity"]
    elif particle["blend"] == "premultiplied":
        region[:] = region * (1 - cover[..., None]) + colour * particle["opacity"] * inside[..., None] * particle["intensity"] * light
    else:
        region[:] = region * (1 - cover[..., None]) + colour * particle["intensity"] * light * cover[..., None]


def draw_effect(canvas, profile_id, profile, event_id, age_ms, anchor, yaw, textures, light=1.0):
    particles = graphs.sample_effect(profile_id, profile, event_id, int(age_ms), "normal")
    c, s = math.cos(yaw), math.sin(yaw)
    fx, fy = tile_to_screen(c, s)
    forward = math.atan2(fy, fx)
    placed = []
    for p in particles:
        x, y, z = p["position_tile"]
        dx, dy = tile_to_screen(x * c - y * s, x * s + y * c, z)
        placed.append((p["blend"] == "additive", anchor[1] + dy, p, (anchor[0] + dx, anchor[1] + dy)))
    for _, _, p, centre in sorted(placed, key=lambda item: (item[0], item[1])):
        angle = p["rotation"] + (forward if p["orientation"] == "emitter" else 0.0)
        splat(canvas, p, centre, angle, textures, light)


def compose(name, outcome="hit"):
    """Per cell, unit layers and effect events are painted far to near by
    screen depth (anchor y; a unit before effects at its own anchor). Flying
    units are painted last."""
    frames = timeline(name, outcome)
    grid = (math.ceil(len(frames) / COLUMNS) * CELL[1], COLUMNS * CELL[0])
    layers, exposure = {}, 1.0
    for layer in sorted({draw[6] for frame in frames for draw in frame["draws"]}):
        loaded, exposure = sheet.load(f"combat-{name}-{layer}")
        padded = np.zeros(grid + (4,))
        padded[:min(grid[0], loaded.shape[0]), :min(grid[1], loaded.shape[1])] = loaded[:grid[0], :grid[1]]
        layers[layer] = padded
    rows, cols = grid
    grass = sheet.inverse_display(sheet.GRASS_DISPLAY[None, None, :].copy(), exposure)[0, 0]
    water = sheet.inverse_display(WATER_DISPLAY[None, None, :].copy(), exposure)[0, 0]
    rgb = np.empty((rows, cols, 3))
    rgb[:] = sheet.inverse_display(np.array([[[88, 104, 44]]], float) / 255, exposure)
    ys, xs = np.mgrid[0:rows, 0:cols]
    attacker_ground = water if SCENARIOS[name].get("attacker_ground") == "water" else grass
    target_ground = water if outcome in ("water", "ship") or SCENARIOS[name].get("target_ground") == "water" else grass
    for frame in frames:
        for (cx, cy), ground in ((frame["anchor"], attacker_ground), (frame["target"], target_ground)):
            rgb[np.abs(xs + .5 - cx) / 64 + np.abs(ys + .5 - cy) / 32 <= 1] = ground
    compiled = graphs.compile_effect_graphs()["profiles"]
    textures = Textures()
    for index, frame in enumerate(frames):
        top, left = index // COLUMNS * CELL[1], index % COLUMNS * CELL[0]
        cell = (slice(top, top + CELL[1]), slice(left, left + CELL[0]))
        items = [(math.inf if draw[6] == "air" else draw[5], 0, draw[6]) for draw in frame["draws"]]
        items += [(anchor[1], 1, (profile, start, anchor, yaw, event_id))
                  for profile, start, anchor, yaw, event_id in frame["events"] if frame["t"] >= start]
        for _, kind, item in sorted(items, key=lambda entry: entry[:2]):
            if kind == 0:
                layer = layers[item][cell]
                rgb[cell] = layer[..., :3] + rgb[cell] * (1 - layer[..., 3:4])
            else:
                profile, start, anchor, yaw, event_id = item
                draw_effect(rgb, profile, compiled[profile], event_id, (frame["t"] - start) * 1000, anchor, yaw, textures)
    image = (np.clip(sheet.display(rgb * exposure), 0, 1) * 255 + .5).astype(np.uint8)
    strips = []
    for row in range(math.ceil(len(frames) / COLUMNS)):
        cells = []
        for column in range(COLUMNS):
            index = row * COLUMNS + column
            cell = image[row * CELL[1]:(row + 1) * CELL[1], column * CELL[0]:(column + 1) * CELL[0]]
            caption = f"{name} {outcome} T {frames[index]['t']:.2f}S" if index < len(frames) else ""
            cells.append(np.concatenate([sheet.label_strip(caption, CELL[0]), cell]))
        strips.append(np.concatenate(cells, 1))
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}-{outcome}.png"
    write_png(path, np.concatenate(strips))
    print(path)
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("render")
    r.add_argument("scenario", choices=sorted(SCENARIOS))
    c = sub.add_parser("compose")
    c.add_argument("scenario", choices=sorted(SCENARIOS))
    c.add_argument("--outcome", default="hit", choices=("hit", "miss", "water", "ship"))
    args = parser.parse_args(argv)
    if args.command == "render":
        render(args.scenario)
    else:
        compose(args.scenario, args.outcome)


if __name__ == "__main__":
    main()

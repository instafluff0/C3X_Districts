#!/usr/bin/env python3
"""PCX-style review sheets for the flat-ground city layouts.

Only normalized local art is read. Images and reports are disposable Lab output;
the production renderer and runtime pack are untouched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from Renderer.lab.studies.cities.build_layouts import wall_instances
from Renderer.lab.shared.cities.assets import component
from Renderer.preview.render_city_day_night_sheet import _draw_mesh
from Renderer.preview.render_feature_asset import DdsBc1Texture
from Renderer.preview.render_iso import Canvas
from Renderer.preview.render_textured_patch import DdsBc3Texture


ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "Renderer/lab/out/cities/design-sheets"
MAGENTA = (255, 0, 255)
TILE_PIXELS = 256
CELL = (380, 230)


class Bc3Texture:
    def __init__(self, texture: DdsBc3Texture, wrap_u: bool, wrap_v: bool):
        self.texture, self.wrap_u, self.wrap_v = texture, wrap_u, wrap_v

    def sample(self, u: float, v: float):
        u = u % 1 if self.wrap_u else max(0, min(1, u))
        v = v % 1 if self.wrap_v else max(0, min(1, v))
        return self.texture.sample_rgba(u, v)


class MaterialTexture:
    def __init__(self, base, opacity=None):
        self.base, self.opacity = base, opacity

    def sample(self, u: float, v: float):
        color = self.base.sample(u, v)
        if self.opacity is None:
            return color
        mask = self.opacity.sample(u, v)
        return (*color[:3], min(color[3], mask[3]))


@lru_cache(None)
def texture(path: str, address_u: str, address_v: str):
    file = (ROOT / path).resolve()
    if not (file.is_relative_to(ROOT / "Renderer/packs") or
            file.is_relative_to(ROOT / "Renderer/lab/out/cities")):
        raise ValueError("city texture must come from a normalized local pack")
    data = file.read_bytes()
    dxgi = struct.unpack_from("<I", data, 128)[0]
    wrap_u, wrap_v = address_u == "repeat", address_v == "repeat"
    if dxgi in (71, 72):
        return DdsBc1Texture(data, "wrap" if wrap_u else "clamp", "wrap" if wrap_v else "clamp")
    if dxgi in (77, 78):
        return Bc3Texture(DdsBc3Texture(data), wrap_u, wrap_v)
    raise ValueError(f"unsupported city color DDS format: {dxgi}")


def channel_texture(channel: dict):
    return texture(channel["texture"], channel.get("address_u", "clamp"),
                   channel.get("address_v", "clamp"))


def material_textures(material: dict):
    channels = material["channels"]
    base = channel_texture(channels["base_color"])
    opacity = channel_texture(channels["opacity"]) if "opacity" in channels else None
    emissive = channel_texture(channels["emissive"]) if "emissive" in channels else None
    return MaterialTexture(base, opacity), emissive


@lru_cache(None)
def prepared_asset(asset_id: str, pack_name: str):
    asset = component(asset_id, Path(pack_name))
    center = [(asset["lo"][i]+asset["hi"][i])/2 for i in (0, 1)]
    parts = []
    for mesh, material in asset["parts"]:
        vertices = [{**vertex, "position": [vertex["position"][0]-center[0],
                                           vertex["position"][1]-center[1],
                                           vertex["position"][2]]}
                    for vertex in mesh["vertices"]]
        base, emissive = material_textures(material)
        parts.append(({**mesh, "vertices": vertices}, base, emissive))
    return parts


def transformed(mesh: dict, instance: dict) -> dict:
    scale, rotation = instance["scale"], instance["rotation"]
    vertical_metric = instance.get("vertical_metric", .648266978876)
    c, s = math.cos(rotation), math.sin(rotation)
    ox, oy = instance["offset"]
    vertices = []
    for vertex in mesh["vertices"]:
        x, y, z = vertex["position"]
        nx, ny, nz = vertex["normal"]
        vertices.append({"position": [ox+scale*(x*c-y*s), oy+scale*(x*s+y*c),
                                      z*scale/vertical_metric],
                         "normal": [nx*c-ny*s, nx*s+ny*c, nz],
                         "uv0": vertex["uv0"]})
    return {"vertices": vertices, "topology": mesh["topology"]}


def render_cell(design: dict, size: int, walls: bool, capital: bool,
                cell: tuple[int, int] = CELL, tile_pixels: int = TILE_PIXELS):
    canvas = Canvas(*cell, MAGENTA)
    cx, cy = cell[0]//2, round(cell[1]*.64)
    half_width, half_height = tile_pixels//2, tile_pixels//4
    diamond = [(cx, cy-half_height), (cx+half_width, cy),
               (cx, cy+half_height), (cx-half_width, cy)]
    # The diamond is only a footprint guide; magenta remains the review ground.
    for a, b in zip(diamond, diamond[1:]+diamond[:1]):
        canvas.draw_line(a, b, (185, 35, 185))
    guide = list(canvas.pixels)
    depth = [-math.inf]*(cell[0]*cell[1])
    tier = design["tier_designs"][size]
    instances = list(tier["houses"])
    if not capital or not design.get("capital_replaces_centerpiece", False):
        instances.append(tier["base_centerpiece"])
    if capital:
        instances.append(tier["palace"])
    if walls:
        instances.extend(wall_instances(design["wall_kit"], size))
    for instance in instances:
        if "vertical_metric" in design and instance["pack"] != "Renderer/packs/CityAdjunctsNormalized":
            instance = {**instance, "vertical_metric": design["vertical_metric"]}
        for mesh, base, emissive in prepared_asset(instance["asset"], instance["pack"]):
            _draw_mesh(canvas, depth, transformed(mesh, instance), base, emissive,
                       (cx, cy), tile_pixels/(2*.72), 0.0, False)
    image = Image.new("RGB", cell)
    image.putdata(canvas.pixels)
    art_pixels = sum(actual != previous for actual, previous in zip(canvas.pixels, guide))
    if art_pixels < 80:
        raise ValueError(f"city cell is visually empty: {design['culture_name']} {design['era_name']} size {size}")
    return image, art_pixels


def font(size: int):
    for path in ("/System/Library/Fonts/Supplemental/Arial.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if Path(path).is_file():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def render_culture(layouts: dict, culture: int, output: Path, only_era: int | None = None):
    designs = [d for d in layouts["designs"] if d["culture"] == culture
               and (only_era is None or d["era"] == only_era)]
    variants = ((False, False, "Base"), (True, False, "Walls"),
                (False, True, "Capital"), (True, True, "Walls + capital"))
    focused = only_era is not None
    title = layouts["styles"][culture]
    evidence = []
    if focused:
        cell, tile_pixels = (760, 480), 512
        left, row_height, era_header, top = 154, cell[1]+8, 35, 95
        width = left+cell[0]*4
        height = top+len(designs)*(era_header+row_height*3)+20
        sheet = Image.new("RGB", (width, height), (33, 27, 42))
        draw = ImageDraw.Draw(sheet)
        draw.text((20, 13), title+" city designs", font=font(26), fill=(249, 240, 249))
        draw.text((20, 49), "Flat grassland  |  town within one tile; larger cities may sprawl  |  software material preview",
                  font=font(15), fill=(202, 186, 204))
        for column, (_, _, name) in enumerate(variants):
            draw.text((left+column*cell[0]+14, 73), name, font=font(17), fill=(250, 230, 247))
        for era_index, design in enumerate(designs):
            base_y = top+era_index*(era_header+row_height*3)
            draw.rectangle((0, base_y, width, base_y+era_header-2), fill=(67, 49, 72))
            draw.text((18, base_y+7), design["era_name"], font=font(18), fill=(250, 235, 248))
            for size, population in enumerate(("Town 1-6", "City 7-12", "Metro 13+")):
                y = base_y+era_header+size*row_height
                draw.text((14, y+15), population, font=font(16), fill=(242, 222, 239))
                for column, (walls, capital, _) in enumerate(variants):
                    image, art_pixels = render_cell(design, size, walls, capital,
                                                    cell=cell, tile_pixels=tile_pixels)
                    sheet.paste(image, (left+column*cell[0], y))
                    evidence.append({"culture": culture, "era": design["era"], "size": size,
                                     "walls": walls, "capital": capital,
                                     "houses": design["population_counts"][size],
                                     "art_pixels": art_pixels,
                                     "palace": design["palace"]["asset"] if capital else None})
    else:
        # Civ III's three population columns and four era rows stay intact.
        # Each population panel contains the four state variants in a 2x2
        # inset, so the complete culture remains readable on one sheet.
        cell, tile_pixels, caption = CELL, TILE_PIXELS, 23
        panel_width, panel_height = cell[0]*2, (cell[1]+caption)*2
        left, top = 112, 104
        width = left+panel_width*3+16
        height = top+panel_height*len(designs)+18
        sheet = Image.new("RGB", (width, height), (33, 27, 42))
        draw = ImageDraw.Draw(sheet)
        draw.text((18, 12), title+" city designs", font=font(26), fill=(249, 240, 249))
        draw.text((18, 48), "Era rows  |  population columns  |  four states per cell  |  flat grassland",
                  font=font(15), fill=(202, 186, 204))
        for size, population in enumerate(("Town 1-6", "City 7-12", "Metropolis 13+")):
            draw.text((left+size*panel_width+14, 75), population,
                      font=font(18), fill=(250, 230, 247))
        for era_index, design in enumerate(designs):
            row_y = top+era_index*panel_height
            draw.rectangle((0, row_y, left-6, row_y+panel_height-4), fill=(67, 49, 72))
            draw.text((14, row_y+16), design["era_name"],
                      font=font(18), fill=(250, 235, 248))
            for size in range(3):
                panel_x = left+size*panel_width
                for variant, (walls, capital, name) in enumerate(variants):
                    x = panel_x+(variant%2)*cell[0]
                    y = row_y+(variant//2)*(cell[1]+caption)
                    draw.rectangle((x, y, x+cell[0]-2, y+caption-1), fill=(57, 43, 65))
                    draw.text((x+9, y+3), name, font=font(15), fill=(250, 230, 247))
                    image, art_pixels = render_cell(design, size, walls, capital,
                                                    cell=cell, tile_pixels=tile_pixels)
                    sheet.paste(image, (x, y+caption))
                    evidence.append({"culture": culture, "era": design["era"], "size": size,
                                     "walls": walls, "capital": capital,
                                     "houses": design["population_counts"][size],
                                     "art_pixels": art_pixels,
                                     "palace": design["palace"]["asset"] if capital else None})
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)
    return {"image": output.resolve().relative_to(ROOT).as_posix(),
            "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            "size": [width, height], "cells": evidence}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layouts", type=Path, default=Path(__file__).with_name("layouts.json"))
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--culture", choices=("all", "american", "european", "mediterranean",
                                              "middle_eastern", "asian"), default="all")
    parser.add_argument("--era", choices=("all", "ancient", "medieval", "industrial", "modern"),
                        default="all")
    args = parser.parse_args()
    layouts = json.loads(args.layouts.read_text())
    if layouts["schema"] != "c3x.lab.city_design.v1" or len(layouts["designs"]) != 20:
        raise ValueError("city design inventory must cover five cultures and four eras")
    cultures = range(5) if args.culture == "all" else [
        ("american", "european", "mediterranean", "middle_eastern", "asian").index(args.culture)]
    era = None if args.era == "all" else ("ancient", "medieval", "industrial", "modern").index(args.era)
    images = []
    for culture in cultures:
        slug = ("american", "european", "mediterranean", "middle_eastern", "asian")[culture]
        image = args.output_dir / f"{slug}{'-'+args.era if era is not None else ''}.png"
        images.append(render_culture(layouts, culture, image, era))
        print(f"Rendered {image}", flush=True)
    report_name = "report.json" if args.culture == "all" and args.era == "all" else (
        "report-"+args.culture+("-"+args.era if era is not None else "")+".json")
    (args.output_dir/report_name).write_text(json.dumps({"schema": "c3x.lab.city_sheet.v1",
                                                   "layouts": args.layouts.resolve().relative_to(ROOT).as_posix(),
                                                   "images": images}, indent=2)+"\n")


if __name__ == "__main__":
    main()

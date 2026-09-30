#!/usr/bin/env python3
"""Render actual source materials/geometry plus seasonal HLSL on headless Metal.

This is a bounded Lab harness, not a production render-graph parity claim.
Source packs are read in place. All build intermediates are temporary and removed.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
OUT = ROOT / "Renderer/lab/out/seasons/programmatic"
sys.path.insert(0, str(ROOT))
from Renderer.lab.backends.cache import Cache, file_hash
from Renderer.lab.backends.compiler import shader_source, shaders


def run(args):
    result = subprocess.run([str(a) for a in args], cwd=ROOT, check=False)
    if result.returncode:
        raise RuntimeError(f"Lab command failed: {Path(str(args[0])).name}")


def portable(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def configured_pack(scenario=None, custom=None):
    from Renderer.definitions.definition_parser import parse_definition_file, merge_layers
    layers = [("default", parse_definition_file(ROOT / "Renderer/default.custom_rendering.txt", "default", ROOT))]
    if scenario:
        path = Path(scenario).resolve()
        layers.append(("scenario", parse_definition_file(path, "scenario", ROOT, path.parent)))
    candidate = Path(custom).resolve() if custom else ROOT / "custom.custom_rendering.txt"
    if candidate.exists():
        layers.append(("custom", parse_definition_file(candidate, "custom", ROOT, candidate.parent)))
    active = merge_layers(layers)
    assets = {a["id"]: a["values"] for a in active["assets"]}
    packs = {p["id"]: p["values"] for p in active["packs"]}
    rule = next(r for r in active["rules"] if r["id"] == "terrain.grassland.default")
    asset = assets[rule["values"]["asset"]]
    descriptor = packs[asset["pack"]]["path"]
    if descriptor["root"] == "mod":
        return (ROOT / descriptor["path"]).resolve()
    if descriptor["root"] == "scenario" and scenario:
        return (Path(scenario).resolve().parent / descriptor["path"]).resolve()
    raise ValueError("The Lab requires a project/scenario terrain pack")


def foliage_roles(output):
    """Offline source evidence -> generic semantic roles, never runtime name tests."""
    data = (ROOT / "Renderer/packs/BeautyStudies/beauty_objects.bin").read_bytes()
    at = 8
    def take(fmt):
        nonlocal at
        values = struct.unpack_from("<" + fmt, data, at)
        at += struct.calcsize("<" + fmt)
        return values
    def string():
        nonlocal at
        n, = take("I")
        value = data[at:at+n].decode()
        at += n
        return value
    version, nm, no, nr = take("4I")
    if version != 3:
        raise ValueError("Selected forest evidence layout changed")
    for _ in range(nm):
        for _ in range(7):
            string()
        take("2I")
    bodies = []
    for _ in range(no):
        name = string()
        kind, material, n = take("3I")
        at += n * 32
        if kind == 1:
            role = 2 if "pine" in name.lower() else 1
            bodies.append({"source_evidence": name, "role": role})
    if len(bodies) != 22:
        raise ValueError("Selected forest body ordering changed")
    bodies += [{"source_evidence": "normalized jungle body", "role": 3} for _ in range(10)]
    output.write_text(" ".join(str(b["role"]) for b in bodies) + "\n")
    return bodies


def body_end(source, start):
    opening = source.index("{", start)
    depth = 1
    i = opening + 1
    while depth:
        depth += (source[i] == "{") - (source[i] == "}")
        i += 1
    return i


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError("Production material adapter no longer matches: " + old[:80])
    return source.replace(old, new, 1)


def material_source(name):
    """Reuses current material equations; only explicit Lab bindings are adapted."""
    path = ROOT / f"Renderer/native/source_fidelity/{name}.hlsl"
    source = shader_source(path)
    source = source.replace("#define BEAUTY_VOLCANO_MATERIAL 1", "// Lab crop has no volcano")
    begin = source.index("// Binding adapter;")
    end = body_end(source, source.index("float q6_shadow_visibility", begin))
    source = source[:begin] + """float q6_shadow_visibility(Texture2D field,float3 world,float3 normal,
        float4 u,float4 v,float4 l,bool receive,bool contact) {
        return receive ? lab_shadow(world,normal) : 1;
    }
    float q6_world_visibility(Texture2D field,float4 world,float3 normal,bool water) {
        return lab_shadow(world.xyz,normal);
    }
""" + source[end:]
    source = source.replace("Texture2DArray ShadowField", "Texture2D ShadowField")
    source = re.sub(r"SamplerState (Wrap|Clamp) : register\(s[01]\);", "", source)
    prefix = 'SamplerState Wrap:register(s0);SamplerState Clamp:register(s1);\n'
    prefix += shader_source(HERE / "seasonal_policy.hlsl") + '\n' + shader_source(HERE / "lab_scene.hlsl") + '\n'
    if name == "terrain":
        source = replace_once(source, "float alpha = saturate(input.coast_coverage+10);",
            "float alpha = saturate(input.coast_coverage+10)*lab_land_coverage(input.world.xy);")
        needle = "#ifdef SANDBOX_TERRAIN_MATERIAL\n    // The sandbox retains"
        source = replace_once(source, needle,
            "lab_season_ground(albedo,geometric,specular_map,normalize(input.normal),input.world,0,height_detail);\n" + needle)
    elif name == "mountain":
        needle = "#ifdef SANDBOX_TERRAIN_MATERIAL\n#ifdef BEAUTY_VOLCANO_MATERIAL\n    albedo ="
        source = replace_once(source, needle,
            "lab_season_ground(albedo,normal,specular_map,geometric,input.world,rock_albedo_coverage,height_detail);\n" + needle)
        source = replace_once(source, "clip(coast_alpha - 0.001);",
            "coast_alpha *= lab_land_coverage(input.world.xy);\n    clip(coast_alpha - 0.001);")
    else:
        needle = "#ifdef BEAUTY_COMPOSED_SHADOWS\n    // Use the exact vector"
        source = replace_once(source, needle,
            "if(kind>.5 && kind<1.5)season_foliage(albedo,normal,gloss,geometric,input.world,input.uv);\n"
            "if(kind>1.5)lab_season_ground(albedo,normal,gloss,geometric,input.world,1);\n" + needle)
    source = source.replace("SunColorExposure.rgb", "season_key(SunColorExposure.rgb)")
    return prefix + source


def compile_lab(work):
    cache = Cache(work / "cache")
    translated = work / "shaders"
    inputs = {}
    for name in ("terrain", "mountain", "objects", "water", "shadow", "probe"):
        if name in ("water", "shadow", "probe"):
            path = HERE / f"{name}.hlsl"
        else:
            path = work / f"{name}.hlsl"
            path.write_text(material_source(name))
        stages = shaders(cache, path, 0, entries=("VSMain", "PSMain"), msl_version=20200)
        folder = translated / name
        folder.mkdir(parents=True)
        for entry, source in stages.items():
            shutil.copyfile(source, folder / f"{entry}.msl")
        inputs[name] = hashlib.sha256(shader_source(path).encode()).hexdigest()
        print("PASS seasonal HLSL translation: " + name, flush=True)
    # The shared dependency cache currently treats SDKSettings.json as a project
    # dependency on newer SDKs. Keep this disposable host build local to the study.
    scene = work / "scene.o"
    run(["clang++", "-std=c++17", "-O2", "-fobjc-arc", "-c", HERE / "scene.mm", "-o", scene])
    environment = work / "environment.o"
    run(["clang++", "-std=c++17", "-O2", "-c", ROOT / "Renderer/native/environment_runtime.cpp", "-o", environment])
    exe = work / "scene"
    run(["clang++", scene, environment, "-framework", "Foundation", "-framework", "Metal", "-o", exe])
    return exe, translated, inputs


def synthetic_map(path):
    """All five requested biomes in a controlled actual-geometry comparison."""
    width, height = 30, 32
    lines = [f"C3X_BIQ_TERRAIN_V3,{width},{height},{width*height//2},0,0"]
    # Wide raw-X bands remain separately visible at the default scene camera.
    codes = [2, 1, 0, 4, 3]
    for y in range(height):
        for x in range(y % 2, width, 2):
            base = codes[min(4, x // 6)]
            real = base
            if 4 < y < 8 and base not in (0, 4):
                real = 7
            if 21 < y < 26 and base not in (0, 4):
                real = 5
            river = 8 if base == 4 and x in (21, 22) else 0
            lines.append(f"{x},{y},{base},{real},0,0,{river}")
    path.write_text("\n".join(lines) + "\n")


def source_guard(pack, evidence):
    paths = set((pack / "natural_runtime").glob("*.dds"))
    paths.add(pack / "natural_runtime/natural.bin")
    paths.add(ROOT / "Renderer/packs/BeautyStudies/beauty_objects.bin")
    for name in ("snow_base_color", "snow_height", "snow_specular", "water/terrain/snow_decal_base"):
        path = pack / "textures" / (name + ".dds")
        if path.exists(): paths.add(path)
    for row in evidence["authored_winter_bodies"]:
        paths.update(ROOT / c["path"] for c in row["channels"])
    cliff = ROOT / "Renderer/packs/ShoreNormalized/cliff_runtime.bin"
    paths.add(cliff)
    data = cliff.read_bytes();count, = struct.unpack_from("<I", data, 12);at = 24
    for _ in range(count):
        n, = struct.unpack_from("<I", data, at);at += 4
        paths.add(cliff.parent / data[at:at+n].decode().replace("\\", "/"));at += n
    return {portable(p): file_hash(p) for p in sorted(paths)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", default="configured")
    parser.add_argument("--all-packs", action="store_true")
    parser.add_argument("--scenario-definitions")
    parser.add_argument("--custom-definitions")
    parser.add_argument("--case", choices=("test-biq", "biomes"), default="test-biq")
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=900)
    parser.add_argument("--hour", type=float, default=12)
    parser.add_argument("--camera-x", type=float)
    parser.add_argument("--camera-y", type=float)
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()
    try:
        from PIL import Image
    except ImportError:
        bundled = Path.home() / ".cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3"
        os.execv(str(bundled), [str(bundled), str(Path(__file__)), *sys.argv[1:]])
    OUT.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(OUT).free < 8*1024**3:
        raise ValueError("Less than 8 GiB free: remove known disposable outputs before a GPU Lab run")
    packs = [ROOT / "Renderer/packs" / p for p in ("TerrainNormalized", "Civ5EnvironmentSkin")] if args.all_packs else [
        configured_pack(args.scenario_definitions, args.custom_definitions) if args.pack == "configured" else (ROOT / args.pack).resolve()]
    for pack in packs:
        if not (pack / "natural_runtime/natural.bin").is_file():
            raise ValueError("Run the existing terrain asset preparation before this Lab: " + portable(pack))
    receipt = {"schema": "c3x.lab.seasonal_programmatic.v1", "backend": "headless Metal",
        "case": args.case, "production_parity": False, "packs": [], "vm_used": False,
        "limitations": ["Diagnostic water/field interpolation/shadow atlas; not the production render graph.",
            "Buildings and units are omitted from this scene harness."]}
    with tempfile.TemporaryDirectory(prefix="work-", dir=OUT) as directory:
        work = Path(directory)
        exe, programs, identities = compile_lab(work)
        receipt["material_adapters"] = identities
        if args.compile_only:
            print("PASS Metal host and all seasonal shader translations")
            return 0
        roles = work / "roles.txt"
        receipt["foliage_evidence"] = foliage_roles(roles)
        from Renderer.lab.studies.seasons.asset_inputs import prepare
        receipt["optional_art_inputs"] = prepare(work, receipt["foliage_evidence"])
        recipe = json.loads((HERE / "recipes.json").read_text())
        receipt["recipe_sha256"] = file_hash(HERE / "recipes.json")
        scene = work / "scene.csv"
        camera = (20, 75)
        if args.case == "test-biq":
            run(["node", HERE / "export_scene.js",
                ROOT / "Renderer/packs/RendererSourceStudies/maps/test.biq", scene])
        else:
            synthetic_map(scene)
            camera = (15, 16)
        camera = (args.camera_x if args.camera_x is not None else camera[0],
                  args.camera_y if args.camera_y is not None else camera[1])
        receipt["camera"] = list(camera)
        receipt["hour"] = args.hour
        receipt["scene_sha256"] = file_hash(scene)
        receipt["map_wrapping"] = [bool(int(v)) for v in scene.read_text().splitlines()[0].split(",")[4:6]]
        for pack in packs:
            destination = OUT / args.case / pack.name
            destination.mkdir(parents=True, exist_ok=True)
            identity = file_hash(pack / "natural_runtime/natural.bin")
            inputs_before = source_guard(pack, receipt["optional_art_inputs"])
            fall = recipe["pack_calibrations"].get(pack.name, recipe["default_fall_grass"])
            (work / "policy.txt").write_text(" ".join(str(v) for key in (recipe["snow_atlas"],fall,recipe["flower_atlas"]) for v in key) + "\n")
            run([exe, ROOT, pack, scene, programs, work, args.width, args.height, *camera, roles,args.hour])
            row = {"pack": portable(pack), "natural_payload_sha256": identity, "images": {},
                "fall_grass_tint": fall, "gpu_checks": json.loads((work / "gpu-checks.json").read_text()),
                "read_only_inputs": inputs_before}
            for index, name in enumerate(("summer", "fall", "winter", "spring")):
                raw = work / f"season-{index}.bgra"
                image = Image.frombytes("RGBA", (args.width, args.height), raw.read_bytes(), "raw", "BGRA").convert("RGB")
                target = destination / f"{name}.png"
                image.save(target, optimize=True)
                raw.unlink()
                row["images"][name] = {"path": portable(target), "sha256": file_hash(target)}
            raw = work / "fall-hue-blend.bgra"
            Image.frombytes("RGBA", (args.width,args.height),raw.read_bytes(),"raw","BGRA").convert("RGB").save(destination / "fall-hue-blend.png",optimize=True)
            raw.unlink()
            row["images"]["fall-hue-blend"] = {"path": portable(destination / "fall-hue-blend.png"), "sha256": file_hash(destination / "fall-hue-blend.png")}
            if source_guard(pack, receipt["optional_art_inputs"]) != inputs_before:
                raise ValueError("Read-only terrain pack changed during Lab rendering")
            receipt["packs"].append(row)
    (OUT / f"{args.case}-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print("PASS source-based seasonal scenes; temporary builds and raw frames removed", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

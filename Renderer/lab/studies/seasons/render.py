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
            "lab_season_ground(albedo,geometric,specular_map,normalize(input.normal),input.world,0,height_detail);\n"
            "if(input.material.y>2.5 && input.material.y<3.5)season_leaf_litter(albedo,input.world,"
            "smoothstep(.015,.15,lab_surface(input.world.xy).x)*lab_land_coverage(input.world.xy),input.uv);\n" + needle)
    elif name == "mountain":
        needle = "#ifdef SANDBOX_TERRAIN_MATERIAL\n#ifdef BEAUTY_VOLCANO_MATERIAL\n    albedo ="
        source = replace_once(source, needle,
            "lab_season_ground(albedo,normal,specular_map,geometric,input.world,rock_albedo_coverage,height_detail);\n" + needle)
        source = replace_once(source, "clip(coast_alpha - 0.001);",
            "coast_alpha *= lab_land_coverage(input.world.xy);\n    clip(coast_alpha - 0.001);")
    else:
        # A per-instance appearance value is added only to this Lab adapter.
        # Position, UV, opacity, normals and existing secondary metadata survive.
        source = source.replace("float2 secondary : TEXCOORD3;", "float2 secondary : TEXCOORD3;\n    float appearance : TEXCOORD4;\n    float4 crown : TEXCOORD5;")
        source = replace_once(source, "output.secondary = input.secondary;",
            "output.secondary = input.secondary;\n    output.appearance = input.appearance;\n    output.crown = input.crown;")
        needle = "#ifdef BEAUTY_COMPOSED_SHADOWS\n    // Use the exact vector"
        source = replace_once(source, needle,
            "float seasonal_leaf=0;\n"
            "if(kind>.5 && kind<1.5)seasonal_leaf=season_foliage(albedo,normal,gloss,geometric,input.world,input.uv,input.appearance,input.crown.w);\n"
            "if(kind>1.5)lab_season_ground(albedo,normal,gloss,geometric,input.world,1);\n" + needle)
        source = replace_once(source,"float diffuse = ndl;",
            "float diffuse = ndl;\n"
            "if(season_autumn_match())diffuse=season_crown_diffuse(diffuse,normal,input.crown.xyz,light_direction,seasonal_leaf);\n"
            "else if(season_autumn_beauty())diffuse=lerp(diffuse,saturate((dot(normal,light_direction)+.20)/1.20),seasonal_leaf*.55);")
        source = replace_once(source,"float roughness = lerp(0.91, 0.31, saturate(gloss));",
            "radiance+=season_crown_transmission(albedo,normal,input.crown.xyz,light_direction,SunColorExposure.rgb*Sun.w,shadow,seasonal_leaf);\n"
            "float roughness = lerp(0.91, 0.31, saturate(gloss));")
    source = source.replace("SunColorExposure.rgb", "season_key(SunColorExposure.rgb)")
    source = source.replace("Ambient.rgb", "season_ambient(Ambient.rgb)")
    return prefix + source


def compile_lab(work):
    cache = Cache(work / "cache")
    translated = work / "shaders"
    inputs = {}
    for name in ("terrain", "mountain", "objects", "water", "shadow", "probe", "winter_decals"):
        if name in ("water", "shadow", "probe", "winter_decals"):
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


def source_guard(pack, evidence, winter_wonderland=False):
    paths = set((pack / "natural_runtime").glob("*.dds"))
    paths.add(pack / "natural_runtime/natural.bin")
    paths.add(ROOT / "Renderer/packs/BeautyStudies/beauty_objects.bin")
    for name in ("snow_base_color", "snow_height", "snow_specular", "water/terrain/snow_decal_base", "water/terrain/snow_decal_height"):
        path = pack / "textures" / (name + ".dds")
        if path.exists(): paths.add(path)
    if winter_wonderland:
        paths.update(p for n in ("large","small","small_secondary","river")
            if (p := pack / f"textures/water/surface/{n}_lean0.dds").is_file())
    for row in evidence["authored_winter_bodies"]:
        paths.update(ROOT / c["path"] for c in row["channels"])
    if evidence.get("winter_exposure"):
        paths.update(ROOT / p for p in evidence["winter_exposure"]["inputs"])
        paths.update(ROOT / r["path"] for r in evidence["winter_exposure"]["masks"])
    if evidence.get("winter_decals"):
        paths.update(ROOT / r["path"] for r in evidence["winter_decals"]["inputs"])
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
    parser.add_argument("--zoom",type=int,choices=(64,128),default=128,help="Actual projection zoom; mesh proportions remain fixed")
    parser.add_argument("--camera-x", type=float)
    parser.add_argument("--camera-y", type=float)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--winter-focus", action="store_true", help="Keep only a bounded winter comparison image")
    parser.add_argument("--fall-focus", action="store_true", help="Refine autumn on original materials/trees; keep one bounded comparison image")
    parser.add_argument("--fall-beauty", action="store_true", help="Target-led autumn treatment with a corrected coast/water preview; keeps its own Summer baseline")
    parser.add_argument("--fall-target", action="store_true", help="Crown irradiance, tissue/value reskin and source-derived turf; original proportions unchanged")
    parser.add_argument("--winter-exposure", action="store_true", help="Bake/reuse Blender snow-exposure masks on the existing forest meshes")
    parser.add_argument("--winter-decals", action="store_true", help="Use the eleven authored snow-decal meshes over existing ground")
    parser.add_argument("--winter-wonderland", action="store_true", help="Layer wind drifts, deposition edges and an improved winter-only diagnostic shadow field")
    args = parser.parse_args()
    if args.fall_target:args.fall_beauty=True
    if args.fall_beauty:
        args.fall_focus=True
        if args.winter_wonderland or args.winter_exposure or args.winter_decals:
            parser.error("The corrected beauty harness uses a separate baseline; run saved Winter regression with --fall-focus")
    if args.winter_focus and args.all_packs:
        parser.error("Winter focus renders one explicitly selected pack")
    if args.fall_focus and (args.all_packs or args.winter_focus):
        parser.error("Fall focus renders one pack and is separate from winter focus")
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
        "harness_sha256":{p.name:file_hash(p) for p in (HERE/"scene.mm",HERE/"render.py",HERE/"lab_geometry.h")},
        "case": args.case, "production_parity": False, "packs": [], "vm_used": False,
        "limitations": ["Diagnostic water/field interpolation/shadow atlas; not the production render graph.",
            "Buildings and units are omitted from this scene harness."]}
    if args.winter_focus:
        receipt["winter_study"] = {"exposure_masks":args.winter_exposure, "authored_decals":args.winter_decals,
            "layered_snow":args.winter_wonderland,
            "geometry_policy": "Original configured forest bodies and placements; material and surface effects only."}
    if args.fall_focus:
        receipt["autumn_study"] = {"per_tree_palette":True, "existing_floor_litter":True,
            "diagnostic_shadow_and_water":True, "new_geometry":False}
        receipt["autumn_study"]["corrected_preview_harness"]=args.fall_beauty
        receipt["autumn_study"]["crown_irradiance_and_source_turf"]=args.fall_target
        receipt["autumn_study"]["tree_proportions_unchanged"]=True
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
        receipt["optional_art_inputs"] = prepare(work, receipt["foliage_evidence"], args.winter_exposure, args.winter_decals)
        from Renderer.lab.studies.seasons.autumn_inputs import prepare as prepare_autumn
        receipt["autumn_inputs"] = prepare_autumn(work, receipt["foliage_evidence"])
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
        receipt["zoom"] = args.zoom
        receipt["scene_sha256"] = file_hash(scene)
        receipt["map_wrapping"] = [bool(int(v)) for v in scene.read_text().splitlines()[0].split(",")[4:6]]
        for pack in packs:
            destination = OUT / "fall-focus" / args.case if args.fall_focus else OUT / "winter-focus" / args.case if args.winter_focus else OUT / args.case / pack.name
            if args.fall_target:
                base=OUT/"fall-focus"
                if pack.name!="Civ5EnvironmentSkin":base/=pack.name
                if args.hour!=12 or args.zoom!=128:base/=f"h{args.hour:g}-z{args.zoom}"
                destination=base/args.case
            destination.mkdir(parents=True, exist_ok=True)
            identity = file_hash(pack / "natural_runtime/natural.bin")
            quality = args.winter_wonderland or args.fall_focus
            inputs_before = source_guard(pack, receipt["optional_art_inputs"], quality)
            fall = recipe["pack_calibrations"].get(pack.name, recipe["default_fall_grass"])
            if args.fall_focus:
                fall = recipe["autumn_refined_grass"].get(pack.name,recipe["autumn_refined_grass"]["default"])
            if args.fall_beauty:fall=recipe["autumn_beauty_grass"]
            values = [v for key in (recipe["snow_atlas"],fall,recipe["flower_atlas"]) for v in key]
            values.append(recipe.get("winter_display_exposure",1.0))
            values.extend(recipe["winter_layers"] if args.winter_wonderland else [0,0,0,0])
            values.extend(recipe["autumn_beauty"] if args.fall_beauty else recipe["autumn_refinement"] if args.fall_focus else [0,0,0,0])
            if args.fall_target:values[-4:]=recipe["autumn_target"]
            values.extend(recipe["autumn_crown"] if args.fall_target else [0,0,0,0])
            (work / "policy.txt").write_text(" ".join(str(v) for v in values) + "\n")
            run([exe, ROOT, pack, scene, programs, work, args.width, args.height, *camera, roles,args.hour,args.zoom])
            row = {"pack": portable(pack), "natural_payload_sha256": identity, "images": {},
                "fall_grass_tint": fall, "gpu_checks": json.loads((work / "gpu-checks.json").read_text()),
                "read_only_inputs": inputs_before}
            for index, name in enumerate(("summer", "fall", "winter", "spring")):
                raw = work / f"season-{index}.bgra"
                if (args.winter_focus and index != 2) or (args.fall_focus and index != 1):
                    if index == 0:
                        if args.fall_beauty:
                            target=destination/("target-summer.png" if args.fall_target else "beauty-summer.png")
                            Image.frombytes("RGBA",(args.width,args.height),raw.read_bytes(),"raw","BGRA").convert("RGB").save(target,optimize=True)
                            row["images"]["summer"]={"path":portable(target),"sha256":file_hash(target)}
                        previous = OUT / args.case / pack.name / "summer.png"
                        if previous.is_file() and not args.fall_beauty:
                            image = Image.frombytes("RGBA", (args.width, args.height), raw.read_bytes(), "raw", "BGRA").convert("RGB")
                            original = Image.open(previous).convert("RGB")
                            if image.size == original.size:
                                if image.tobytes() != original.tobytes():
                                    raise ValueError("Seasonal focus changed the preceding Summer scene")
                                row["preceding_summer_pixel_exact"] = True
                    if index == 2 and args.fall_focus and args.winter_wonderland and args.winter_exposure and args.winter_decals:
                        previous = OUT / "winter-focus" / args.case / "wonderland.png"
                        if previous.is_file():
                            image = Image.frombytes("RGBA", (args.width,args.height),raw.read_bytes(),"raw","BGRA").convert("RGB")
                            original = Image.open(previous).convert("RGB")
                            if image.size == original.size:
                                from PIL import ImageChops
                                difference = ImageChops.difference(image,original)
                                histogram = difference.histogram()
                                maximum = max((i%256 for i,n in enumerate(histogram) if n),default=0)
                                changed = sum(any(pixel) for pixel in difference.getdata())
                                # A changed shader layout can round a few half-float
                                # samples across one display code. Reject visual drift.
                                tolerance = maximum <= 1 and changed <= args.width*args.height*.00001
                                row["preceding_winter_comparison"] = {"pixel_exact":changed==0,
                                    "maximum_rgb_code_difference":maximum,"changed_pixels":changed,
                                    "bounded_rounding_pass":tolerance}
                                if not tolerance:
                                    image.save(destination / "winter-regression.png",optimize=True)
                                    raise ValueError("Autumn refinement changed the preceding winter wonderland")
                    raw.unlink()
                    continue
                image = Image.frombytes("RGBA", (args.width, args.height), raw.read_bytes(), "raw", "BGRA").convert("RGB")
                winter_name = "wonderland.png" if args.winter_wonderland else "exposure-decals.png" if args.winter_decals else "exposure.png" if args.winter_exposure else "materials.png"
                target = destination / "target.png" if args.fall_target else destination / "beauty.png" if args.fall_beauty else destination / "refined.png" if args.fall_focus else destination / winter_name if args.winter_focus else destination / f"{name}.png"
                image.save(target, optimize=True)
                raw.unlink()
                row["images"][name] = {"path": portable(target), "sha256": file_hash(target)}
            raw = work / "fall-hue-blend.bgra"
            if not (args.winter_focus or args.fall_focus):
                Image.frombytes("RGBA", (args.width,args.height),raw.read_bytes(),"raw","BGRA").convert("RGB").save(destination / "fall-hue-blend.png",optimize=True)
            raw.unlink()
            if not (args.winter_focus or args.fall_focus):
                row["images"]["fall-hue-blend"] = {"path": portable(destination / "fall-hue-blend.png"), "sha256": file_hash(destination / "fall-hue-blend.png")}
            if source_guard(pack, receipt["optional_art_inputs"], quality) != inputs_before:
                raise ValueError("Read-only terrain pack changed during Lab rendering")
            receipt["packs"].append(row)
    winter_label = "wonderland" if args.winter_wonderland else "exposure-decals" if args.winter_decals else "exposure" if args.winter_exposure else "materials"
    receipt_path = destination.parent / f"{args.case}-target-receipt.json" if args.fall_target else OUT / "fall-focus" / f"{args.case}-beauty-receipt.json" if args.fall_beauty else OUT / "fall-focus" / f"{args.case}-receipt.json" if args.fall_focus else OUT / "winter-focus" / f"{args.case}-{winter_label}-receipt.json" if args.winter_focus else OUT / f"{args.case}-receipt.json"
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    print("PASS source-based seasonal scenes; temporary builds and raw frames removed", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

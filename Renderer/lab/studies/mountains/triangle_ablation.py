#!/usr/bin/env python3
"""Render one-at-a-time terrain material ablations at a fixed BIQ camera.

Each case gets an isolated shader tree under ignored Lab output. Game assets,
the scenario export, and the candidate DLL are shared and unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from Renderer.lab.platform import ROOT
from Renderer.lab.studies.terrain.test_biq import capture

OUT = ROOT / "Renderer/lab/out/mountains/triangle-ablation"
NATIVE = ROOT / "Renderer/native"
SOURCE_TREE = NATIVE
DLL = ROOT / "Renderer/lab/out/mountains/ground-handoff/v8-C3XRenderer.dll"
SCENE = ROOT / "Renderer/lab/out/mountains/ground-handoff/after-v8/test-biq.csv"
SHADERS = ("city_fidelity/terrain.hlsl", "city_fidelity/mountain.hlsl")


def replace_once(source: str, old: str, new: str, label: str) -> str:
    if source.count(old) < 1:
        raise ValueError(f"{label}: shader expression not found")
    return source.replace(old, new, 1)


def ablate(source: str, case: str) -> str:
    if case.startswith("grass-off-plus-"):
        return ablate(ablate(source, "grass-surface-off"),
                      case.removeprefix("grass-off-plus-"))
    if case == "edge-fade-on":
        if "Texture2D GroundSurfaceDetail" in source:
            return source
        return replace_once(source,
            "(desert_dune ? 0.62 : 1.0);",
            "(desert_dune ? 0.62 : 1.0);\n"
            "        if (!desert_dune && !plains_surface) {\n"
            "            float edge = min(input.material.w, min(input.biome.x, input.biome.y));\n"
            "            alpha *= smoothstep(0.0, 0.18, edge);\n"
            "        }",case)
    if case in ("edge-fade-soft", "edge-fade-minimal"):
        if "Texture2D GroundSurfaceDetail" in source:
            return source
        gain = "0.50" if case == "edge-fade-soft" else "0.25"
        return replace_once(source,
            "(desert_dune ? 0.62 : 1.0);",
            "(desert_dune ? 0.62 : 1.0);\n"
            "        if (!desert_dune && !plains_surface) {\n"
            "            float edge = min(input.material.w, min(input.biome.x, input.biome.y));\n"
            f"            alpha *= smoothstep(0.0, 0.26, edge) * {gain};\n"
            "        }",case)
    if case == "edge-fade-off":
        if "Texture2D GroundSurfaceDetail" in source:
            return source
        return replace_once(source,"alpha *= smoothstep(0.0, 0.18, edge);",
            "alpha *= 1.0;",case)
    if case in ("decal-all-off", "hill-decal-off", "surface-decal-off", "floor-decal-off",
                "grass-surface-off", "plains-surface-off", "desert-surface-off"):
        if "Texture2D GroundSurfaceDetail" in source:
            return source
        condition = {"decal-all-off": "input.material.y > 1.5",
                     "hill-decal-off": "input.material.y > 1.5 && input.material.y < 2.5",
                     "surface-decal-off": "input.material.y > 4.5",
                     "floor-decal-off": "input.material.y > 2.5 && input.material.y < 4.5",
                     "grass-surface-off": "input.material.y > 4.5 && input.material.y < 5.5",
                     "plains-surface-off": "input.material.y > 5.5 && input.material.y < 6.5",
                     "desert-surface-off": "input.material.y > 6.5"}[case]
        return replace_once(source, "    float3 geometric = normalize(input.normal);",
            f"    if ({condition}) clip(-1);\n    float3 geometric = normalize(input.normal);",case)
    if case == "material-class-view":
        if "Texture2D GroundSurfaceDetail" in source:
            return source
        return replace_once(source,
            "#ifdef BEAUTY_COMPOSED_SHADOWS\n    // Q6ShadowL",
            "albedo = input.material.y > 4.5 ? float3(1,0,0) : "
            "input.material.y > 3.5 ? float3(1,0,1) : "
            "input.material.y > 2.5 ? float3(0,0,1) : "
            "input.material.y > 1.5 ? float3(1,0.5,0) : float3(0.4,0.4,0.4);\n"
            "#ifdef BEAUTY_COMPOSED_SHADOWS\n    // Q6ShadowL",case)
    if case == "surface-subclass-view":
        if "Texture2D GroundSurfaceDetail" in source:
            return source
        return replace_once(source,
            "#ifdef BEAUTY_COMPOSED_SHADOWS\n    // Q6ShadowL",
            "albedo = input.material.y > 6.5 ? float3(1,0,0) : "
            "input.material.y > 5.5 ? float3(0,1,0) : "
            "input.material.y > 4.5 ? float3(0,0,1) : float3(0.4,0.4,0.4);\n"
            "#ifdef BEAUTY_COMPOSED_SHADOWS\n    // Q6ShadowL",case)
    if case.startswith("diagnostic-"):
        source = ablate(source, "flat-color-after-detail")
        operation = case.removeprefix("diagnostic-")
        if operation == "all-normal-flat":
            if "float ndl = saturate(dot(geometric, light_direction));" in source:
                return replace_once(source, "float ndl = saturate(dot(geometric, light_direction));",
                    "geometric = float3(0, 0, 1);\n    float ndl = saturate(dot(geometric, light_direction));", case)
            return replace_once(source, "float ndl = saturate(dot(normal, light_direction));",
                "normal = float3(0, 0, 1);\n    float ndl = saturate(dot(normal, light_direction));", case)
        if operation == "mesh-normal-flat":
            return replace_once(source,"float3 geometric = normalize(input.normal);",
                "float3 geometric = float3(0, 0, 1);",case)
        if operation == "detail-normal-off":
            if "Detail.y * (1 + 0.9 * grass_plains_detail));" in source:
                return replace_once(source,"Detail.y * (1 + 0.9 * grass_plains_detail));",
                    "0.0);",case)
            return replace_once(source,"0.075 * (1 + 0.9 * grass_plains_detail));",
                "0.0);",case)
        if operation == "shadow-off":
            if "shadow = lerp(lerp(1.0, shadow, 0.48), shadow, input.coast_inland);" in source:
                return replace_once(source,
                    "shadow = lerp(lerp(1.0, shadow, 0.48), shadow, input.coast_inland);",
                    "shadow = 1.0;",case)
            return replace_once(source,
                "received_shadow, coast_inland);",
                "received_shadow, coast_inland);\n    shadow = 1.0;",case)
        if operation == "cavity-off":
            if "float cavity = lerp(0.79, 1.0, smoothstep(0.02, 0.30, input.material.x));" in source:
                return replace_once(source,
                    "float cavity = lerp(0.79, 1.0, smoothstep(0.02, 0.30, input.material.x));",
                    "float cavity = 1.0;",case)
            return replace_once(source,
                "float cavity = lerp(ground_cavity, mountain_cavity, rock_albedo_coverage);",
                "float cavity = 1.0;",case)
        raise ValueError(case)
    if case == "control":
        return source
    if case == "final-albedo-flat":
        if "float ndl = saturate(dot(geometric, light_direction));" in source:
            return replace_once(source,
                "float ndl = saturate(dot(geometric, light_direction));",
                "albedo = float3(0.32, 0.37, 0.17);\n    float ndl = saturate(dot(geometric, light_direction));",case)
        return replace_once(source,
            "float ndl = saturate(dot(normal, light_direction));",
            "albedo = float3(0.32, 0.37, 0.17);\n    float ndl = saturate(dot(normal, light_direction));",case)
    if case == "both-grass-flat":
        if "Texture2D GroundSurfaceDetail" in source:
            return replace_once(source,
                "float3 grass = ground_surface_sample(GrassColor, input, 0.43, float2(.31,.17), false).rgb;",
                "float3 grass = float3(0.32, 0.37, 0.17);",case)
        return replace_once(source,"float3 grass = GrassColor.Sample(Wrap, uv0).rgb;",
            "float3 grass = float3(0.32, 0.37, 0.17);",case)
    if case in ("both-grass-four", "both-grass-highpass"):
        mountain = "Texture2D GroundSurfaceDetail" in source
        if mountain:
            old = "float3 grass = ground_surface_sample(GrassColor, input, 0.43, float2(.31,.17), false).rgb;"
            if case == "both-grass-four":
                new = "float3 grass = (ground_surface_sample(GrassColor, input, 0.43, float2(.31,.17), false).rgb + "
                new += "ground_surface_sample(GrassColor, input, 0.43, float2(.68,.28), false).rgb + "
                new += "ground_surface_sample(GrassColor, input, 0.43, float2(.44,.70), false).rgb + "
                new += "ground_surface_sample(GrassColor, input, 0.43, float2(.92,.88), false).rgb) * 0.25;"
            else:
                new = "float3 fine_grass = ground_surface_sample(GrassColor, input, 0.43, float2(.31,.17), false).rgb;\n"
                new += "    float3 low_grass = GrassColor.SampleBias(Wrap, input.world.xy * 0.43 + float2(.31,.17), 5).rgb;\n"
                new += "    float3 grass = saturate(float3(.32,.37,.17) + (fine_grass-low_grass)*.75);"
        else:
            old = "float3 grass = GrassColor.Sample(Wrap, uv0).rgb;"
            if case == "both-grass-four":
                new = "float3 grass = (GrassColor.Sample(Wrap,uv0).rgb + GrassColor.Sample(Wrap,uv0+float2(.37,.11)).rgb + "
                new += "GrassColor.Sample(Wrap,uv0+float2(.13,.53)).rgb + GrassColor.Sample(Wrap,uv0+float2(.61,.71)).rgb)*.25;"
            else:
                new = "float3 fine_grass = GrassColor.Sample(Wrap,uv0).rgb;\n"
                new += "        float3 low_grass = GrassColor.SampleBias(Wrap,uv0,5).rgb;\n"
                new += "        float3 grass = saturate(float3(.32,.37,.17) + (fine_grass-low_grass)*.75);"
        return replace_once(source,old,new,case)
    if case == "broad-tint-off":
        return replace_once(source,
            "albedo *= lerp(float3(0.88, 0.94, 0.97),\n",
            "albedo *= lerp(float3(1.0, 1.0, 1.0),\n", case).replace(
            "float3(1.09, 1.045, 0.91), broad);",
            "float3(1.0, 1.0, 1.0), broad);", 1)
    if case == "broad-contrast-off":
        return replace_once(source,
            "albedo *= 1 + clamp((broad - 0.30) * 0.55, -0.12, 0.15) * grass_plains_detail;",
            "albedo *= 1.0;", case)
    if case == "grain-off":
        return replace_once(source,
            "albedo *= 1 + clamp((grain - 0.303) * 0.9, -0.16, 0.22) * grass_plains_detail;",
            "albedo *= 1.0;", case)
    if case == "broad-normal-off":
        if "geometric = detail_normal_strength(geometric, input.world, broad_height,\n                                           0.35 * grass_plains_detail);" in source:
            return replace_once(source,
                "0.35 * grass_plains_detail);",
                "0.0);", case)
        return replace_once(source,
            "normal = detail_normal(normal, input.world, broad_height,\n        0.35 * grass_plains_detail);",
            "normal = detail_normal(normal, input.world, broad_height,\n        0.0);", case)
    if case == "source-color-flat":
        return replace_once(source,"float3 grass = GrassColor.Sample(Wrap, uv0).rgb;",
                            "float3 grass = float3(0.32, 0.37, 0.17);",case)
    if case == "plains-weight-off":
        return replace_once(source, "float3 base = lerp(lerp(grass, plains, plains_weight), tundra, tundra_weight);",
            "plains_weight = 0.0;\n    float3 base = lerp(lerp(grass, plains, plains_weight), tundra, tundra_weight);", case)
    if case == "hill-band-off":
        return replace_once(source, "albedo = lerp(base, hill, rocky_band * 0.90);",
            "albedo = base;", case)
    if case == "base-color-flat":
        return replace_once(source, "albedo = lerp(base, hill, rocky_band * 0.90);",
            "albedo = float3(0.32, 0.37, 0.17);", case)
    if case == "flat-color-after-detail":
        return replace_once(source,
            "albedo *= 1 + clamp((grain - 0.303) * 0.9, -0.16, 0.22) * grass_plains_detail;",
            "albedo = float3(0.32, 0.37, 0.17);",case)
    if case == "biome-weights-view":
        return replace_once(source,
            "albedo *= 1 + clamp((grain - 0.303) * 0.9, -0.16, 0.22) * grass_plains_detail;",
            "albedo = float3(plains_weight, plains_weight, plains_weight);",case)
    if case == "grass-texture-only":
        return replace_once(source, "albedo = lerp(base, hill, rocky_band * 0.90);",
            "albedo = grass;", case)
    raise ValueError(case)


def shader_tree(case: str) -> Path:
    root = OUT / "shaders" / case
    target = root / "Renderer/native"
    for shader in SOURCE_TREE.rglob("*.hlsl"):
        relative = shader.relative_to(SOURCE_TREE).as_posix()
        path = target / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        source = shader.read_text()
        if relative in SHADERS and not (relative.endswith("mountain.hlsl") and
                case in ("base-color-flat", "hill-band-off", "plains-weight-off",
                         "source-color-flat")):
            source = ablate(source, case)
        path.write_text(source)
    if case != "control":
        cached = OUT / "shaders/control/Renderer/native"
        for shader_cache in cached.rglob("*.cso"):
            relative = shader_cache.relative_to(cached).as_posix()
            if relative.startswith(SHADERS[0] + ".") or relative.startswith(SHADERS[1] + "."):
                continue
            destination = target / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(shader_cache, destination)
    return root


def main() -> None:
    global OUT, DLL, SOURCE_TREE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dll", type=Path, default=DLL)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--shader-base", type=Path, default=NATIVE)
    parser.add_argument("cases", nargs="*", default=["control", "broad-tint-off",
        "broad-contrast-off", "grain-off", "broad-normal-off", "source-color-flat"])
    args = parser.parse_args()
    OUT, DLL = args.output.resolve(), args.dll.resolve()
    SOURCE_TREE = args.shader_base.resolve()
    if not DLL.is_file() or not SCENE.is_file():
        raise FileNotFoundError("Saved Lab candidate or unchanged BIQ export missing")
    OUT.mkdir(parents=True, exist_ok=True)
    images = {}
    for case in args.cases:
        root = shader_tree(case)
        image = capture("frame", (20, 79), 256, 12, DLL, SCENE,
                        OUT / "captures" / case, root)
        images[case] = str(image.relative_to(ROOT))
        print(f"{case}: {image}", flush=True)
    (OUT / "receipt.json").write_text(json.dumps({
        "dll_sha256": hashlib.sha256(DLL.read_bytes()).hexdigest(),
        "scene_sha256": hashlib.sha256(SCENE.read_bytes()).hexdigest(),
        "images": images,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()

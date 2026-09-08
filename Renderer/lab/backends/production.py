"""Translate the current production passes without editing their HLSL sources.

This is shared shader preparation, not a replacement scene renderer. A successful
compile does not establish geometry, render-graph or cross-backend pixel parity.
"""
from pathlib import Path
import re
import shutil
from .compiler import ROOT, shaders, run


def programs():
    """Entry points used by the accepted city-fidelity production profile."""
    result = {}
    profile = ROOT / "Renderer/native/city_fidelity"
    for name in ("terrain", "mountain", "objects"):
        result[name] = (profile / (name + ".hlsl"), "VSNative", "PSFeature")
        result[name + "-reflection"] = (profile / (name + ".hlsl"), "VSReflection", "PSReflection")
    result["hydrology"] = (profile / "hydrology.hlsl", "VSIntegrated", "PSIntegrated")
    result["feature"] = (profile / "feature.hlsl", "VSIntegratedFeature", "PSIntegratedFeature")
    for name in ("hydrology", "feature"):
        result[name + "-reflection"] = (profile / (name + ".hlsl"), "VSReflection", "PSReflection")
    for name, vertex, pixel in (
        ("city", "VSNativeCity", "PSNativeCity"),
        ("city-emission", "VSNativeCity", "PSNativeCityEmission"),
        ("city-reflection", "VSNativeCityReflection", "PSNativeCityReflection"),
        ("city-reflection-emission", "VSNativeCityReflection", "PSNativeCityReflectionEmission"),
    ):
        result[name] = (profile / "city.hlsl", vertex, pixel)
    return result


def prepare(cache, output, *, metal=None):
    """Cache translations; optionally compile every resulting entry on Metal."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    selected = programs()
    # The runtime embeds this shader as one literal. Extract exactly its contents,
    # not a separately maintained approximation of the material response.
    header = ROOT / "Renderer/native/environment_refresh/unit_shader.h"
    matches = re.findall(r'R"C3XUNIT\((.*?)\)C3XUNIT"', header.read_text(), re.S)
    if len(matches) != 1:
        raise ValueError("Expected exactly one production unit material shader")
    unit_source = output / "unit.hlsl"
    if not unit_source.exists() or unit_source.read_text() != matches[0]:
        unit_source.write_text(matches[0])
    selected["unit"] = (unit_source, "VS", "PS")
    translated = {}
    for name, (source, vertex, pixel) in selected.items():
        directory = output / name
        directory.mkdir(exist_ok=True)
        stages = shaders(cache, source, 0, entries=("VSMain", "PSMain"),
                         entrypoints={"VSMain": vertex, "PSMain": pixel})
        for entry, path in stages.items():
            shutil.copyfile(path, directory / (entry + ".msl"))
        if metal is not None:
            run([metal, "--validate-shaders", directory, *stages])
        translated[name] = directory
        print("PASS production shader " + name + (" on Metal" if metal else " translation"), flush=True)
    return translated

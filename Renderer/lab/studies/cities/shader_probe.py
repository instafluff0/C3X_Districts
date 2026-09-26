#!/usr/bin/env python3
"""Compare isolated city material stages without changing checked-in shaders."""

from __future__ import annotations

import argparse
import json
import os

from Renderer.lab.studies.cities.build_layouts import ROOT
from Renderer.lab.studies.cities.isolate_pack import isolate
from Renderer.lab.studies.cities.test_biq_gallery import OUT, PACK, render


SHADER = ROOT / "Renderer/lab/out/cities/test-biq/root/Renderer/native/city_fidelity/city.hlsl"
SOURCE = ROOT / "Renderer/lab/out/cities/test-biq/all-designs-pack/city.bin"
BASE = "float3 base=q8_surface_sample(city_base_texture_0,p.uv,repeat_uv).rgb;"


def replace(path, data: bytes) -> None:
    temporary = path.with_name(path.name + ".shader-probe")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def variant(original: str, stage: str) -> str:
    if stage == "baseline":
        return original
    if original.count(BASE) != 1:
        raise ValueError("city material source changed")
    if stage == "unlit-base":
        return original.replace(BASE, BASE +
            "\n if(!emission_only)return q6_scene_output(float4(base,1));")
    if stage == "unlit-mip-zero":
        replacement = ("float3 base=(repeat_uv?"
                       "city_base_texture_0.SampleLevel(material_sampler,p.uv,0):"
                       "city_base_texture_0.SampleLevel(decal_sampler,p.uv,0)).rgb;"
                       "\n if(!emission_only)return q6_scene_output(float4(base,1));")
        return original.replace(BASE, replacement)
    if stage == "normal":
        marker = " float ao=1;"
        if original.count(marker) != 1:
            raise ValueError("city normal source changed")
        return original.replace(marker,
            " if(!emission_only)return q6_scene_output(float4(n*.5+.5,1));\n" + marker)
    if stage == "mip-bias":
        marker = "if(repeat_uv)return source.Sample(material_sampler,uv);\n return source.Sample(decal_sampler,uv);"
        if original.count(marker) != 1:
            raise ValueError("city sampler source changed")
        return original.replace(marker,
            "if(repeat_uv)return source.SampleBias(material_sampler,uv,-.75);\n"
            " return source.SampleBias(decal_sampler,uv,-.75);")
    raise ValueError(stage)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale", type=float, choices=(1.0, 2.0), default=1.0)
    args = parser.parse_args()
    suffix = "" if args.scale == 1.0 else "-2x"
    original_shader = SHADER.read_bytes()
    original_pack = SOURCE.read_bytes()
    house = isolate(original_pack, 4, 0, 0, False, args.scale)
    records = []
    try:
        replace(PACK, house)
        for stage in ("baseline", "unlit-base", "unlit-mip-zero", "normal", "mip-bias"):
            replace(SHADER, variant(original_shader.decode("utf-8"), stage).encode())
            result = render(4, 0, 0, (20, 64), "fidelity-shader" + suffix + "-" + stage,
                            capital=False, walls=False)
            records.append({"stage": stage, "image": result["image"]})
    finally:
        replace(SHADER, original_shader)
        replace(PACK, original_pack)
    (OUT / ("shader-probe" + suffix + ".json")).write_text(json.dumps(records, indent=2) + "\n")
    cards = "".join(f'<article><h2>{r["stage"]}</h2><img src="{r["image"]}"></article>'
                    for r in records)
    (OUT / ("shader-probe" + suffix + ".html")).write_text(
        '<!doctype html><meta charset="utf-8"><title>City material stages</title>'
        '<style>body{font:16px system-ui;background:#211b29;color:white;margin:20px}'
        'main{display:grid;grid-template-columns:repeat(auto-fit,minmax(520px,1fr));gap:20px}'
        'article{background:#372b3e;padding:12px}img{width:100%;height:auto}'
        'h2{margin:0 0 8px}</style><h1>City material stages</h1><main>' + cards + '</main>')
    print("Wrote", OUT / ("shader-probe" + suffix + ".html"))


if __name__ == "__main__":
    main()

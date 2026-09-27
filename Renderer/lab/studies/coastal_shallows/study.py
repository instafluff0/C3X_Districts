#!/usr/bin/env python3
"""Compare a coast-only shallow-water adjustment against the sandbox renderer."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from Renderer.lab.platform import native_command_result

OUT = ROOT / "Renderer/lab/out/coastal-shallows"
SANDBOX = ROOT / "Renderer/sandbox"
GROUND_COMPILER = ROOT / "Renderer/native/source_fidelity/ground_compiler.h"
RENDERER_CPP = ROOT / "Renderer/native/c3x_renderer.cpp"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def candidate_shader(source: str, *, rich: bool = False) -> str:
    # Keep the sandbox's family weights, normals, reflection and foam intact.
    # This changes only coast optical depth and transmitted shallow color.
    old = """    refracted += light * (float3(.012, .058, .046) * coast_family +
                          float3(.006, .026, .023) * sea_family);"""
    new = """    float clear_coast = coast_family * (1 - smoothstep(.20, .41, depth));
    refracted += light * (float3(.012, .058, .046) * coast_family +
                          float3(.006, .026, .023) * sea_family +
                          float3(.004, .026, .021) * clear_coast);"""
    if rich:
        new = new.replace(".20, .41", ".25, .55").replace(".004, .026, .021", ".010, .050, .040")
    if source.count(old) != 1:
        raise ValueError("Sandbox water-family source changed; inspect before rebasing")
    source = source.replace(old, new)
    old = """    float coverage = 1 - exp(-depth * lerp(2.3, 3.2,
        smoothstep(.10, .32, depth)));"""
    new = """    float coverage = 1 - exp(-depth * lerp(2.3 - .55 * clear_coast, 3.2,
        smoothstep(.10, .32, depth)));"""
    if rich:
        new = new.replace(".55 * clear_coast", "1.15 * clear_coast")
    if source.count(old) != 1:
        raise ValueError("Sandbox water-coverage source changed; inspect before rebasing")
    return source.replace(old, new)


def candidate_bed_shader(source: str, *, rich: bool = False) -> str:
    # The accepted bed already carries authored beach-alpha grain and projected
    # rock/crack decals. Let that existing detail survive a little farther
    # into coast shallows without inserting new textures or geometry.
    old = """ float3 absorption=exp(-input.hydrology_data.w*float3(9,5,3));
 return color*clamp(.86+(height-.426)*.55,.65,1.1)*absorption;"""
    new = """ float coast_shelf=(1-smoothstep(.11,.44,input.hydrology_data.w))
  *(1-smoothstep(.34,.68,input.surface_coordinate));
 color*=1+beach_grain*.18*coast_shelf;
 float3 absorption=exp(-input.hydrology_data.w*float3(9,5,3)
  *lerp(1,.90,coast_shelf));
 return color*clamp(.86+(height-.426)*.55,.65,1.1)*absorption;"""
    if rich:
        new = (new.replace(".11,.44", ".16,.53")
               .replace("beach_grain*.18", "beach_grain*.27")
               .replace("lerp(1,.90", "lerp(1,.78"))
    if source.count(old) != 1:
        raise ValueError("Sandbox bed material source changed; inspect before rebasing")
    return source.replace(old, new)


def lagoon_shaders(water: str, bed: str) -> tuple[str, str]:
    """Expose more of the existing bed atlas beneath coast-family water."""
    water = candidate_shader(water, rich=True)
    for old, new in ((".25, .55, depth", ".25, .68, depth"),
                     (".010, .050, .040", ".025, .100, .075"),
                     ("1.15 * clear_coast", "1.55 * clear_coast")):
        if water.count(old) != 1:
            raise ValueError("Lagoon water source changed: " + old)
        water = water.replace(old, new)
    bed = candidate_bed_shader(bed, rich=True)
    for old, new in ((".16,.53,input.hydrology_data.w", ".19,.65,input.hydrology_data.w"),
                     ("beach_grain*.27", "beach_grain*.32"),
                     ("lerp(1,.78,coast_shelf)", "lerp(1,.42,coast_shelf)")):
        if bed.count(old) != 1:
            raise ValueError("Lagoon bed source changed: " + old)
        bed = bed.replace(old, new)
    return water, bed


def scattered_bed_shader(source: str) -> str:
    """Scatter the same five source atlas cells sparsely in coast-family beds."""
    anchor = "float4 sample_coast_clutter(float2 world_position)"
    if source.count(anchor) != 1:
        raise ValueError("Scattered seabed insertion anchor changed")
    helper = """float3 scattered_water_uv_and_mask(float2 world_position)
{
    float2 projected = world_position / 1.75 + float2(0.07, 0.19);
    float2 cell = floor(projected);
    float2 jitter = float2(
        macro_decal_hash(cell + float2(17.0, 29.0)),
        macro_decal_hash(cell + float2(53.0, 11.0))) - 0.5;
    float2 local = frac(projected) - 0.5 - jitter * 0.30;
    float turn = floor(macro_decal_hash(cell + float2(7.0, 61.0)) * 4.0);
    float2 rotated = turn < 0.5 ? local :
        (turn < 1.5 ? float2(-local.y, local.x) :
        (turn < 2.5 ? -local : float2(local.y, -local.x)));
    float scale = lerp(0.62, 0.82,
        macro_decal_hash(cell + float2(37.0, 43.0)));
    float2 local_uv = rotated / scale + 0.5;
    float inside = step(0.0, local_uv.x) * step(local_uv.x, 1.0) *
                   step(0.0, local_uv.y) * step(local_uv.y, 1.0);
    float occupancy = 1.0 - step(0.34,
        macro_decal_hash(cell + float2(19.0, 7.0)));
    float mask = inside * occupancy *
        projected_decal_edge_fade(saturate(local_uv));
    float variant = floor(macro_decal_hash(cell + float2(71.0, 23.0)) * 5.0);
    return float3(ocean_clutter_atlas_uv(saturate(local_uv), variant), mask);
}

float4 sample_scattered_water_clutter(float2 world_position)
{
    float3 placement = scattered_water_uv_and_mask(world_position);
    float4 value = water_decal_base_texture.Sample(decal_sampler, placement.xy);
    value.a *= placement.z;
    return value;
}

float sample_scattered_water_clutter_height(float2 world_position)
{
    float3 placement = scattered_water_uv_and_mask(world_position);
    return water_decal_height_texture.Sample(decal_sampler, placement.xy).r *
           placement.z;
}

"""
    source = source.replace(anchor, helper + anchor)
    old = " float4 ocean=sample_water_clutter(world);"
    new = """ float coast_family=1-smoothstep(.34,.63,saturate(input.surface_coordinate));
 float4 ocean=lerp(sample_water_clutter(world),
  sample_scattered_water_clutter(world),coast_family);"""
    if source.count(old) != 2:
        raise ValueError("Scattered seabed color anchors changed")
    source = source.replace(old, new)
    old = "float height=(sample_water_clutter_height(world)-.5)*ocean.a;"
    new = """float height=(lerp(sample_water_clutter_height(world),
  sample_scattered_water_clutter_height(world),coast_family)-.5)*ocean.a;"""
    if source.count(old) != 1:
        raise ValueError("Scattered seabed normal anchor changed")
    source = source.replace(old, new)
    old = " bed*=1+(sample_water_clutter_height(world)-.5)*authored.a*.30;"
    new = """ float coast_family=1-smoothstep(.34,.63,saturate(input.surface_coordinate));
 float clutter_height=lerp(sample_water_clutter_height(world),
  sample_scattered_water_clutter_height(world),coast_family);
 bed*=1+(clutter_height-.5)*authored.a*.30;"""
    if source.count(old) != 1:
        raise ValueError("Scattered seabed material anchor changed")
    return source.replace(old, new)


def continuous_bed_shader(source: str) -> str:
    """Use the shallows material's continuous alpha structure in coast beds."""
    changes = (
        (" float3 bed=shallow_bed_texture.Sample(material_sampler,uv).rgb;",
         " float4 shallow=shallow_bed_texture.Sample(material_sampler,uv);\n"
         " float3 bed=shallow.rgb;"),
        (" float4 authored=q3_authored_bed_detail(input);",
         """ float coast_family=1-smoothstep(.34,.63,saturate(input.surface_coordinate));
 float4 authored=q3_authored_bed_detail(input);
 authored.a*=1-coast_family;
 float2 second_uv=float2(uv.y,-uv.x)*.47+float2(.19,.37);
 float second_alpha=shallow_bed_texture.Sample(material_sampler,second_uv).a;
 float shelf=coast_family*(1-smoothstep(.16,.58,input.hydrology_data.w));
 float structure=clamp((shallow.a-.34)*3.0+(second_alpha-.34)*1.6,-.42,.58);
 bed*=1+structure*.85*shelf;"""),
        (" return q3_margin_normal(input,height,.045);",
         """ float coast_family=1-smoothstep(.34,.63,saturate(input.surface_coordinate));
 float2 bed_uv=world*q3_source_repeat(.75);
 float bed_alpha=shallow_bed_texture.Sample(material_sampler,bed_uv).a;
 float3 decal_normal=q3_margin_normal(input,height,.045);
 float3 continuous_normal=q3_margin_normal(input,bed_alpha,.025);
 return normalize(lerp(decal_normal,continuous_normal,coast_family));"""),
    )
    for old, new in changes:
        if source.count(old) != 1:
            raise ValueError("Continuous bed anchor changed: " + old)
        source = source.replace(old, new)
    return source


def reef_field_bed_shader(source: str) -> str:
    """Give the clean shelf sparse, aperiodic rock forms with authored grain."""
    anchor = "float3 q3_authored_bed_normal(PixelInput input) {"
    helper = """float q3_lab_rock_noise(float2 p) {
 float2 c=floor(p),f=frac(p);f=f*f*(3-2*f);
 return lerp(lerp(macro_decal_hash(c),macro_decal_hash(c+float2(1,0)),f.x),
  lerp(macro_decal_hash(c+float2(0,1)),macro_decal_hash(c+1),f.x),f.y);
}
float q3_lab_rock_field(float2 world) {
 float2 p=world*q3_source_repeat(.54);
 float2 warp=float2(q3_lab_rock_noise(p*.47+float2(3.1,7.7)),
  q3_lab_rock_noise(p*.47+float2(9.3,1.4)))-.5;
 float2 q=p+warp*.74;
 float broad=q3_lab_rock_noise(q);
 float middle=q3_lab_rock_noise(float2(q.y,-q.x)*2.13+float2(4.2,1.8));
 return smoothstep(.57,.70,broad*.77+middle*.23);
}
float q3_lab_rock_coverage(PixelInput input,float2 world) {
 float depth=input.hydrology_data.w;
 float coast=1-smoothstep(.34,.63,saturate(input.surface_coordinate));
 return q3_lab_rock_field(world)*coast*smoothstep(.035,.16,depth)
  *(1-smoothstep(.39,.62,depth));
}
"""
    if source.count(anchor) != 1:
        raise ValueError("Rock-field helper anchor changed")
    source = source.replace(anchor, helper + anchor)
    old = " return normalize(lerp(decal_normal,continuous_normal,coast_family));"
    new = """ float3 base_normal=normalize(lerp(decal_normal,continuous_normal,coast_family));
 float rock_mask=q3_lab_rock_coverage(input,world);
 float2 rock_uv=world*q3_source_repeat(1.12)+float2(.21,.37);
 float rock_height=cliff_height_texture.Sample(material_sampler,rock_uv).r;
 float3 rock_normal=q3_margin_normal(input,rock_height*rock_mask,.14);
 return normalize(lerp(base_normal,rock_normal,rock_mask));"""
    if source.count(old) != 1:
        raise ValueError("Rock-field normal anchor changed")
    source = source.replace(old, new)
    old = " color*=lerp(1,fine_cavity,margin);"
    new = """ color*=lerp(1,fine_cavity,margin);
 float rock_mask=q3_lab_rock_coverage(input,world);
 float2 rock_uv=world*q3_source_repeat(1.12)+float2(.21,.37);
 float3 rock_grain=cliff_base_texture.Sample(material_sampler,rock_uv).rgb;
 float rock_height=cliff_height_texture.Sample(material_sampler,rock_uv).r;
 float rock_mean=cliff_height_texture.SampleBias(material_sampler,rock_uv,3).r;
 float3 submerged_rock=rock_grain*float3(.58,.83,1.16);
 color=lerp(color,submerged_rock,rock_mask*.78);
 color*=1+clamp((rock_height-rock_mean)*1.3,-.20,.25)*rock_mask;"""
    if source.count(old) != 1:
        raise ValueError("Rock-field material anchor changed")
    return source.replace(old, new)


def reef_forms_bed_shader(source: str) -> str:
    """Irregular source-rock placements, masking the source decal's soft plate."""
    anchor = "float3 q3_authored_bed_normal(PixelInput input) {"
    helper = """float3 q3_lab_reef_placement(float2 world) {
 float2 p=world/1.65,c=floor(p);
 float best=100,variant=0,angle=0;
 float2 chosen=0;
 [unroll] for(int y=-1;y<=1;y++) {
  [unroll] for(int x=-1;x<=1;x++) {
   float2 cell=c+float2(x,y);
   float r0=macro_decal_hash(cell+float2(17,31));
   float r1=macro_decal_hash(cell+float2(43,11));
   float r2=macro_decal_hash(cell+float2(71,47));
   float active=step(.44,r2);
   float2 center=cell+.5+(float2(r0,r1)-.5)*.90;
   float scale=lerp(.38,.67,frac(r0*13.7+r1*5.1));
   float2 delta=(p-center)/scale;
   float metric=dot(delta,delta);
   if(active>.5 && metric<best) {
    best=metric;chosen=delta;
    variant=floor(frac(r0*7.31+r1*3.97)*5);
    angle=frac(r2*17.13)*6.2831853;
   }
  }
 }
 float2 rotated=float2(chosen.x*cos(angle)-chosen.y*sin(angle),
  chosen.x*sin(angle)+chosen.y*cos(angle));
 float2 local=rotated*.5+.5;
 float inside=step(best,1.0)*step(0,local.x)*step(local.x,1)
  *step(0,local.y)*step(local.y,1);
 return float3(ocean_clutter_atlas_uv(saturate(local),variant),inside);
}
float4 q3_lab_reef_sample(float2 world) {
 float3 placement=q3_lab_reef_placement(world);
 float4 rock=water_decal_base_texture.Sample(decal_sampler,placement.xy);
 float2 relief=water_decal_height_texture.Sample(decal_sampler,placement.xy).rg;
 // The second packed channel isolates rock interiors in this source atlas;
 // its physical meaning remains unresolved, so this is a Lab art mask.
 rock.a*=placement.z*smoothstep(.06,.34,relief.g);
 return rock;
}
float q3_lab_reef_coverage(PixelInput input,float2 world) {
 float depth=input.hydrology_data.w;
 float coast=1-smoothstep(.34,.63,saturate(input.surface_coordinate));
 return q3_lab_reef_sample(world).a*coast*smoothstep(.035,.16,depth)
  *(1-smoothstep(.40,.63,depth));
}
"""
    if source.count(anchor) != 1:
        raise ValueError("Reef-forms helper anchor changed")
    source = source.replace(anchor, helper + anchor)
    old = " return normalize(lerp(decal_normal,continuous_normal,coast_family));"
    new = """ float3 base_normal=normalize(lerp(decal_normal,continuous_normal,coast_family));
 float reef=q3_lab_reef_coverage(input,world);
 float2 rock_uv=world*q3_source_repeat(1.12)+float2(.21,.37);
 float rock_height=cliff_height_texture.Sample(material_sampler,rock_uv).r;
 float3 rock_normal=q3_margin_normal(input,rock_height*reef,.15);
 return normalize(lerp(base_normal,rock_normal,reef));"""
    if source.count(old) != 1:
        raise ValueError("Reef-forms normal anchor changed")
    source = source.replace(old, new)
    old = " color*=lerp(1,fine_cavity,margin);"
    new = """ color*=lerp(1,fine_cavity,margin);
 float4 reef=q3_lab_reef_sample(world);
 float rock_mask=reef.a*coast_family*smoothstep(.035,.16,input.hydrology_data.w)
  *(1-smoothstep(.40,.63,input.hydrology_data.w));
 float2 rock_uv=world*q3_source_repeat(1.12)+float2(.21,.37);
 float3 rock_grain=cliff_base_texture.Sample(material_sampler,rock_uv).rgb;
 float rock_height=cliff_height_texture.Sample(material_sampler,rock_uv).r;
 float rock_mean=cliff_height_texture.SampleBias(material_sampler,rock_uv,3).r;
 float3 submerged_rock=lerp(reef.rgb,rock_grain,.30)*float3(.46,.73,1.05);
 color=lerp(color,submerged_rock,rock_mask*.80);
 color*=1+clamp((rock_height-rock_mean)*1.5,-.20,.25)*rock_mask;"""
    if source.count(old) != 1:
        raise ValueError("Reef-forms material anchor changed")
    return source.replace(old, new)


def refine() -> None:
    baseline = OUT / "shader-baseline/Renderer/native/city_fidelity/hydrology.hlsl"
    candidate = OUT / "shader-candidate/Renderer/native/city_fidelity/hydrology.hlsl"
    if not baseline.is_file() or not candidate.is_file():
        raise ValueError("Run prepare first")
    candidate.write_text(candidate_bed_shader(baseline.read_text()))
    rich_root = OUT / "shader-rich"
    for path in (OUT / "shader-baseline").rglob("*.hlsl"):
        destination = rich_root / path.relative_to(OUT / "shader-baseline")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    rich_water = rich_root / "Renderer/sandbox/water_surface.hlsl"
    rich_water.write_text(candidate_shader(
        (OUT / "shader-baseline/Renderer/sandbox/water_surface.hlsl").read_text(), rich=True))
    rich_hydrology = rich_root / "Renderer/native/city_fidelity/hydrology.hlsl"
    rich_hydrology.write_text(candidate_bed_shader(baseline.read_text(), rich=True))
    lagoon_root = OUT / "shader-lagoon"
    for path in (OUT / "shader-baseline").rglob("*.hlsl"):
        destination = lagoon_root / path.relative_to(OUT / "shader-baseline")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    lagoon_water, lagoon_bed = lagoon_shaders(
        (OUT / "shader-baseline/Renderer/sandbox/water_surface.hlsl").read_text(),
        baseline.read_text())
    (lagoon_root / "Renderer/sandbox/water_surface.hlsl").write_text(lagoon_water)
    (lagoon_root / "Renderer/native/city_fidelity/hydrology.hlsl").write_text(lagoon_bed)
    bed_only_root = OUT / "shader-bed-only"
    for path in (OUT / "shader-baseline").rglob("*.hlsl"):
        destination = bed_only_root / path.relative_to(OUT / "shader-baseline")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    bed_only_water = bed_only_root / "Renderer/sandbox/water_surface.hlsl"
    body = bed_only_water.read_text()
    old = "    alpha = lerp(alpha, 1, foam * .65);"
    if body.count(old) != 1:
        raise ValueError("Bed-only diagnostic anchor changed")
    bed_only_water.write_text(body.replace(old, old +
        "\n    alpha *= 1 - coast_family; // Lab diagnostic: show authored bed alone"))
    clear_root = OUT / "shader-clearwater"
    for path in lagoon_root.rglob("*.hlsl"):
        destination = clear_root / path.relative_to(lagoon_root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    clear_water = clear_root / "Renderer/sandbox/water_surface.hlsl"
    body = clear_water.read_text()
    if body.count(old) != 1:
        raise ValueError("Clearwater surface anchor changed")
    clear_water.write_text(body.replace(old, old +
        "\n    float bed_window = coast_family * (1 - smoothstep(.18, .58, depth));"
        "\n    alpha *= 1 - .62 * bed_window; // Lab: reveal the authored bed"))
    no_clutter_root = OUT / "shader-no-clutter"
    for path in clear_root.rglob("*.hlsl"):
        destination = no_clutter_root / path.relative_to(clear_root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    no_clutter_bed = no_clutter_root / "Renderer/native/city_fidelity/hydrology.hlsl"
    body = no_clutter_bed.read_text()
    old = " float4 authored=q3_authored_bed_detail(input);"
    if body.count(old) != 1:
        raise ValueError("Clutter diagnostic anchor changed")
    no_clutter_bed.write_text(body.replace(old,
        " float4 authored=float4(0,0,0,0); // Lab diagnostic: suppress atlas color"))
    scattered_root = OUT / "shader-scattered"
    for path in clear_root.rglob("*.hlsl"):
        destination = scattered_root / path.relative_to(clear_root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    scattered_bed = scattered_root / "Renderer/native/city_fidelity/hydrology.hlsl"
    scattered_bed.write_text(scattered_bed_shader(scattered_bed.read_text()))
    rockbeds_root = OUT / "shader-rockbeds"
    for path in scattered_root.rglob("*.hlsl"):
        destination = rockbeds_root / path.relative_to(scattered_root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    rockbeds_bed = rockbeds_root / "Renderer/native/city_fidelity/hydrology.hlsl"
    body = rockbeds_bed.read_text()
    for old, new in (("    value.a *= placement.z;",
                      "    value.rgb *= float3(.58, .74, .82);\n"
                      "    value.a = saturate(value.a * placement.z * 2.2);"),
                     (" return color*clamp(.86+(height-.426)*.55,.65,1.1)*absorption;",
                      " float3 bed_tint=lerp(1.0.xxx,float3(.68,1.04,1.18),coast_shelf);\n"
                      " return color*clamp(.86+(height-.426)*.55,.65,1.1)*absorption*bed_tint;")):
        if body.count(old) != 1:
            raise ValueError("Rockbeds material anchor changed: " + old)
        body = body.replace(old, new)
    rockbeds_bed.write_text(body)
    continuous_root = OUT / "shader-continuous"
    for path in clear_root.rglob("*.hlsl"):
        destination = continuous_root / path.relative_to(clear_root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    continuous_bed = continuous_root / "Renderer/native/city_fidelity/hydrology.hlsl"
    continuous_bed.write_text(continuous_bed_shader(continuous_bed.read_text()))
    aquamarine_root = OUT / "shader-aquamarine"
    for path in continuous_root.rglob("*.hlsl"):
        destination = aquamarine_root / path.relative_to(continuous_root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    aquamarine_water = aquamarine_root / "Renderer/sandbox/water_surface.hlsl"
    body = aquamarine_water.read_text()
    old = "float3(.025, .100, .075)"
    if body.count(old) != 1:
        raise ValueError("Aquamarine water anchor changed")
    aquamarine_water.write_text(body.replace(old, "float3(.020, .125, .140)"))
    aquamarine_bed = aquamarine_root / "Renderer/native/city_fidelity/hydrology.hlsl"
    body = aquamarine_bed.read_text()
    for old, new in (("bed*=1+structure*.85*shelf;", "bed*=1+structure*.22*shelf;"),
                     (" return color*clamp(.86+(height-.426)*.55,.65,1.1)*absorption;",
                      " float tint_strength=coast_shelf*smoothstep(.05,.22,input.hydrology_data.w);\n"
                      " float3 tint=lerp(1.0.xxx,float3(.48,1.12,1.80),tint_strength);\n"
                      " return color*clamp(.86+(height-.426)*.55,.65,1.1)*absorption*tint;")):
        if body.count(old) != 1:
            raise ValueError("Aquamarine bed anchor changed: " + old)
        body = body.replace(old, new)
    aquamarine_bed.write_text(body)
    no_margin_root = OUT / "shader-aquamarine-no-margin"
    for path in aquamarine_root.rglob("*.hlsl"):
        destination = no_margin_root / path.relative_to(aquamarine_root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    no_margin_bed = no_margin_root / "Renderer/native/city_fidelity/hydrology.hlsl"
    body = no_margin_bed.read_text()
    old = " float margin=(1-smoothstep(.08,.38,input.hydrology_data.w))*lerp(.25,1.0,q3_margin_patch(world));"
    if body.count(old) != 1:
        raise ValueError("Submerged margin detail anchor changed")
    no_margin_bed.write_text(body.replace(old, old.replace(";", "*(1-coast_family);")))
    clean_root = OUT / "shader-aquamarine-clean-bed"
    for path in no_margin_root.rglob("*.hlsl"):
        destination = clean_root / path.relative_to(no_margin_root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    clean_bed = clean_root / "Renderer/native/city_fidelity/hydrology.hlsl"
    body = clean_bed.read_text()
    old = " float3 bed=shallow.rgb;"
    # The highest mip supplies this texture's mean sand color without its four
    # baked rock clusters; the independent alpha still supplies fine detail.
    new = " float3 bed=lerp(shallow.rgb,shallow_bed_texture.SampleLevel(material_sampler,uv,10).rgb,coast_family);"
    # The coast-family value is normally declared below this line.
    if body.count(old) != 1:
        raise ValueError("Shallow-bed base color anchor changed")
    body = body.replace(old, new)
    declaration = " float coast_family=1-smoothstep(.34,.63,saturate(input.surface_coordinate));"
    if body.count(declaration) != 2:
        raise ValueError("Coast-family declaration contract changed")
    scene_start = body.index("float3 q3_scene_bed(PixelInput input) {")
    scene_end = body.index("void q3_shore_material", scene_start)
    scene = body[scene_start:scene_end]
    if scene.count(declaration) != 1:
        raise ValueError("Shallow-bed coast-family declaration changed")
    scene = scene.replace(declaration, "", 1).replace(
        " float4 shallow=shallow_bed_texture.Sample(material_sampler,uv);",
        declaration + "\n float4 shallow=shallow_bed_texture.Sample(material_sampler,uv);", 1)
    body = body[:scene_start] + scene + body[scene_end:]
    clean_bed.write_text(body)
    shelf_root = OUT / "shader-shelf-relief"
    for path in clean_root.rglob("*.hlsl"):
        destination = shelf_root / path.relative_to(clean_root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    shelf_bed = shelf_root / "Renderer/native/city_fidelity/hydrology.hlsl"
    body = shelf_bed.read_text()
    anchor = "float3 q3_authored_bed_normal(PixelInput input) {"
    helper = """// Filter the continuous source shallow pattern at two unrelated world scales.
// This is an inferred seabed height response, not recovered source geometry.
float q3_lab_shelf_height(float2 world) {
 float2 broad_uv=world*q3_source_repeat(.29)+float2(.17,.41);
 float2 middle_uv=float2(world.y,-world.x)*q3_source_repeat(.53)+float2(.63,.23);
 float broad=shallow_bed_texture.SampleBias(material_sampler,broad_uv,3).a;
 float middle=shallow_bed_texture.SampleBias(material_sampler,middle_uv,1).a;
 return (broad-.34)*.82+(middle-.34)*.30;
}
"""
    if body.count(anchor) != 1:
        raise ValueError("Shelf-relief helper anchor changed")
    body = body.replace(anchor, helper + anchor)
    old = """ float3 continuous_normal=q3_margin_normal(input,bed_alpha,.025);
 return normalize(lerp(decal_normal,continuous_normal,coast_family));"""
    new = """ float shelf=smoothstep(.03,.14,input.hydrology_data.w)
  *(1-smoothstep(.43,.68,input.hydrology_data.w));
 float sculpted=q3_lab_shelf_height(world)*shelf;
 float3 continuous_normal=q3_margin_normal(input,
  sculpted*.82+bed_alpha*.18,.42);
 return normalize(lerp(decal_normal,continuous_normal,coast_family));"""
    if body.count(old) != 1:
        raise ValueError("Shelf-relief normal anchor changed")
    body = body.replace(old, new)
    old = " color*=1+beach_grain*.32*coast_shelf;"
    new = """ color*=1+beach_grain*.32*coast_shelf;
 // Broad authored ridges carry a restrained sediment light/dark response.
 float shelf_height=q3_lab_shelf_height(world);
 color*=1+clamp(shelf_height*.45,-.12,.16)*coast_shelf;"""
    if body.count(old) != 1:
        raise ValueError("Shelf-relief material anchor changed")
    shelf_bed.write_text(body.replace(old, new))
    mesh_root = OUT / "shader-shelf-mesh"
    for path in clean_root.rglob("*.hlsl"):
        destination = mesh_root / path.relative_to(clean_root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    lit_mesh_root = OUT / "shader-shelf-mesh-lit"
    for path in mesh_root.rglob("*.hlsl"):
        destination = lit_mesh_root / path.relative_to(mesh_root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    lit_mesh_bed = lit_mesh_root / "Renderer/native/city_fidelity/hydrology.hlsl"
    body = lit_mesh_bed.read_text()
    old = """ float3 absorption=exp(-input.hydrology_data.w*float3(9,5,3)
  *lerp(1,.42,coast_shelf));"""
    new = """ // The isolated mesh carries broad elevation in q6_world.z. Use its
 // raised/depressed residual for local optical depth, as well as geometry.
 float depth=input.hydrology_data.w;
 float water=smoothstep(0,.14,-input.hydrology_data.x);
 float coast=1-smoothstep(.34,.63,saturate(input.surface_coordinate));
 float base=min(-2.5,-.75-depth*23)*water*coast;
 float elevation=input.q6_world.z*112-2.5-base;
 float visual_depth=max(.015,depth-clamp(elevation*.032,-.17,.17)*coast_shelf);
 float3 absorption=exp(-visual_depth*float3(9,5,3)
  *lerp(1,.42,coast_shelf));"""
    if body.count(old) != 1:
        raise ValueError("Shelf-mesh-light optical anchor changed")
    lit_mesh_bed.write_text(body.replace(old, new))
    control_root = OUT / "shader-shelf-mesh-control"
    for path in clean_root.rglob("*.hlsl"):
        destination = control_root / path.relative_to(clean_root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    reef_root = OUT / "shader-reef-field"
    for path in clean_root.rglob("*.hlsl"):
        destination = reef_root / path.relative_to(clean_root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    reef_bed = reef_root / "Renderer/native/city_fidelity/hydrology.hlsl"
    reef_bed.write_text(reef_field_bed_shader(reef_bed.read_text()))
    forms_root = OUT / "shader-reef-forms"
    for path in clean_root.rglob("*.hlsl"):
        destination = forms_root / path.relative_to(clean_root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    forms_bed = forms_root / "Renderer/native/city_fidelity/hydrology.hlsl"
    forms_bed.write_text(reef_forms_bed_shader(forms_bed.read_text()))
    ridges_root = OUT / "shader-reef-ridges"
    for path in forms_root.rglob("*.hlsl"):
        destination = ridges_root / path.relative_to(forms_root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    ridges_bed = ridges_root / "Renderer/native/city_fidelity/hydrology.hlsl"
    body = ridges_bed.read_text()
    old = "smoothstep(.06,.34,relief.g)"
    if body.count(old) != 1:
        raise ValueError("Reef-ridges relief anchor changed")
    ridges_bed.write_text(body.replace(old, "smoothstep(.30,.48,relief.g)"))
    relief_root = OUT / "shader-reef-relief"
    for path in ridges_root.rglob("*.hlsl"):
        destination = relief_root / path.relative_to(ridges_root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    relief_bed = relief_root / "Renderer/native/city_fidelity/hydrology.hlsl"
    body = relief_bed.read_text()
    old = """ float rock_height=cliff_height_texture.Sample(material_sampler,rock_uv).r;
 float3 rock_normal=q3_margin_normal(input,rock_height*reef,.15);"""
    new = """ float3 placement=q3_lab_reef_placement(world);
 float rock_relief=water_decal_height_texture.Sample(decal_sampler,placement.xy).g;
 float3 rock_normal=q3_margin_normal(input,rock_relief*reef,.32);"""
    if body.count(old) != 1:
        raise ValueError("Reef-relief normal anchor changed")
    body = body.replace(old, new)
    old = """ float rock_height=cliff_height_texture.Sample(material_sampler,rock_uv).r;
 float rock_mean=cliff_height_texture.SampleBias(material_sampler,rock_uv,3).r;"""
    new = """ float3 placement=q3_lab_reef_placement(world);
 float rock_relief=water_decal_height_texture.Sample(decal_sampler,placement.xy).g;
 float rock_height=cliff_height_texture.Sample(material_sampler,rock_uv).r;
 float rock_mean=cliff_height_texture.SampleBias(material_sampler,rock_uv,3).r;"""
    if body.count(old) != 1:
        raise ValueError("Reef-relief color anchor changed")
    body = body.replace(old, new)
    old = " color*=1+clamp((rock_height-rock_mean)*1.5,-.20,.25)*rock_mask;"
    new = """ color*=1+clamp((rock_height-rock_mean)*1.5,-.20,.25)*rock_mask;
 color*=1+clamp((rock_relief-.37)*1.6,-.22,.22)*rock_mask;"""
    if body.count(old) != 1:
        raise ValueError("Reef-relief contrast anchor changed")
    relief_bed.write_text(body.replace(old, new))
    contrast_root = OUT / "shader-reef-contrast"
    for path in relief_root.rglob("*.hlsl"):
        destination = contrast_root / path.relative_to(relief_root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    contrast_bed = contrast_root / "Renderer/native/city_fidelity/hydrology.hlsl"
    body = contrast_bed.read_text()
    changes = (
        ("float2 p=world/1.65,c=floor(p);", "float2 p=world/1.90,c=floor(p);"),
        ("float scale=lerp(.38,.67,frac(r0*13.7+r1*5.1));",
         "float scale=lerp(.30,.79,frac(r0*13.7+r1*5.1));"),
        ("float active=step(.44,r2);", "float active=step(.50,r2);"),
        ("float3 submerged_rock=lerp(reef.rgb,rock_grain,.30)*float3(.46,.73,1.05);",
         "float3 submerged_rock=lerp(float3(.075,.12,.15),float3(.29,.33,.32),"
         "smoothstep(.30,.50,rock_relief))*(.75+rock_grain*.85);"),
        ("color=lerp(color,submerged_rock,rock_mask*.80);",
         "color=lerp(color,submerged_rock,rock_mask*.88);"),
        ("color*=1+clamp((rock_relief-.37)*1.6,-.22,.22)*rock_mask;",
         "color*=1+clamp((rock_relief-.37)*2.2,-.24,.28)*rock_mask;"),
    )
    for old, new in changes:
        if body.count(old) != 1:
            raise ValueError("Reef-contrast anchor changed: " + old)
        body = body.replace(old, new)
    contrast_bed.write_text(body)
    lit_root = OUT / "shader-reef-lit"
    for path in relief_root.rglob("*.hlsl"):
        destination = lit_root / path.relative_to(relief_root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    lit_bed = lit_root / "Renderer/native/city_fidelity/hydrology.hlsl"
    body = lit_bed.read_text()
    changes = (
        ("float3 rock_normal=q3_margin_normal(input,rock_relief*reef,.32);",
         "float3 rock_normal=q3_margin_normal(input,rock_relief*reef,.72);"),
        ("float3 submerged_rock=lerp(reef.rgb,rock_grain,.30)*float3(.46,.73,1.05);",
         "float3 submerged_rock=lerp(float3(.20,.13,.09),float3(.45,.27,.16),"
         "smoothstep(.31,.50,rock_relief))*(.73+rock_grain*.95);"),
        ("color=lerp(color,submerged_rock,rock_mask*.80);",
         "color=lerp(color,submerged_rock,rock_mask*.94);"),
        ("color*=1+clamp((rock_relief-.37)*1.6,-.22,.22)*rock_mask;",
         "color*=1+clamp((rock_relief-.37)*2.0,-.22,.28)*rock_mask;"),
    )
    for old, new in changes:
        if body.count(old) != 1:
            raise ValueError("Reef-lit anchor changed: " + old)
        body = body.replace(old, new)
    lit_bed.write_text(body)
    stone_root = OUT / "shader-reef-stone"
    for path in lit_root.rglob("*.hlsl"):
        destination = stone_root / path.relative_to(lit_root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    stone_bed = stone_root / "Renderer/native/city_fidelity/hydrology.hlsl"
    body = stone_bed.read_text()
    changes = (
        (""" float2 relief=water_decal_height_texture.Sample(decal_sampler,placement.xy).rg;
 // The second packed channel isolates rock interiors in this source atlas;
 // its physical meaning remains unresolved, so this is a Lab art mask.
 rock.a*=placement.z*smoothstep(.30,.48,relief.g);""",
         """ // As in the hill material, source color separates stone from its
 // softer decal footprint. A filtered lookup carries the broad formation;
 // the native lookup keeps small irregular edges without isolated freckles.
 float3 broad=water_decal_base_texture.SampleBias(decal_sampler,placement.xy,3).rgb;
 float broad_ratio=broad.b/max(broad.g,.025);
 float fine_ratio=rock.b/max(rock.g,.025);
 float stone=smoothstep(.395,.455,broad_ratio)*.82
  +smoothstep(.38,.53,fine_ratio)*.18;
 rock.a*=placement.z*stone;"""),
        (""" float3 placement=q3_lab_reef_placement(world);
 float rock_relief=water_decal_height_texture.Sample(decal_sampler,placement.xy).g;
 float3 rock_normal=q3_margin_normal(input,rock_relief*reef,.72);""",
         """ float rock_height=cliff_height_texture.Sample(material_sampler,rock_uv).r;
 float3 rock_normal=q3_margin_normal(input,reef*(.60+.40*rock_height),.38);"""),
        (""" float3 placement=q3_lab_reef_placement(world);
 float rock_relief=water_decal_height_texture.Sample(decal_sampler,placement.xy).g;
 float rock_height=cliff_height_texture.Sample(material_sampler,rock_uv).r;
 float rock_mean=cliff_height_texture.SampleBias(material_sampler,rock_uv,3).r;
 float3 submerged_rock=lerp(float3(.20,.13,.09),float3(.45,.27,.16),smoothstep(.31,.50,rock_relief))*(.73+rock_grain*.95);
 color=lerp(color,submerged_rock,rock_mask*.94);
 color*=1+clamp((rock_height-rock_mean)*1.5,-.20,.25)*rock_mask;
 color*=1+clamp((rock_relief-.37)*2.0,-.22,.28)*rock_mask;""",
         """ float rock_height=cliff_height_texture.Sample(material_sampler,rock_uv).r;
 float rock_mean=cliff_height_texture.SampleBias(material_sampler,rock_uv,3).r;
 float rock_luma=dot(rock_grain,float3(.2126,.7152,.0722));
 // The continuous sand stays visible between authored stone interiors.
 float3 submerged_rock=lerp(rock_grain,rock_luma.xxx,.44)
  *float3(.54,.37,.24)*1.22;
 color=lerp(color,submerged_rock,rock_mask*.97);
 color*=1+clamp((rock_height-rock_mean)*1.9,-.25,.30)*rock_mask;"""),
    )
    for old, new in changes:
        if body.count(old) != 1:
            raise ValueError("Reef-stone anchor changed: " + old[:60])
        body = body.replace(old, new)
    stone_bed.write_text(body)
    grain_root = OUT / "shader-reef-stone-grain"
    for path in stone_root.rglob("*.hlsl"):
        destination = grain_root / path.relative_to(stone_root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    grain_bed = grain_root / "Renderer/native/city_fidelity/hydrology.hlsl"
    body = grain_bed.read_text()
    changes = (
        ("float2 p=world/1.65,c=floor(p);", "float2 p=world/1.95,c=floor(p);"),
        ("float active=step(.44,r2);", "float active=step(.54,r2);"),
        ("float2 center=cell+.5+(float2(r0,r1)-.5)*.90;",
         "float2 center=cell+.5+(float2(r0,r1)-.5)*.98;"),
        ("float scale=lerp(.38,.67,frac(r0*13.7+r1*5.1));",
         "float scale=lerp(.29,.81,frac(r0*13.7+r1*5.1));"),
        ("float3 rock_normal=q3_margin_normal(input,reef*(.60+.40*rock_height),.38);",
         "float3 rock_normal=q3_margin_normal(input,reef*(.60+.40*rock_height),.48);"),
        (""" float3 submerged_rock=lerp(rock_grain,rock_luma.xxx,.44)
  *float3(.54,.37,.24)*1.22;
 color=lerp(color,submerged_rock,rock_mask*.97);
 color*=1+clamp((rock_height-rock_mean)*1.9,-.25,.30)*rock_mask;""",
         """ float3 submerged_rock=lerp(rock_grain,rock_luma.xxx,.34)
  *float3(.65,.46,.31)*1.20;
 color=lerp(color,submerged_rock,rock_mask*.94);
 // Hill-like source detail sits within the broad formation; the desert-like
 // sand base remains continuous between sparse, world-stable accents.
 float3 placement=q3_lab_reef_placement(world);
 float3 broad=water_decal_base_texture.SampleBias(decal_sampler,placement.xy,3).rgb;
 float authored_grain=dot(reef.rgb-broad,float3(.2126,.7152,.0722));
 color*=1+clamp(authored_grain*1.6,-.18,.24)*rock_mask;
 color*=1+clamp((rock_height-rock_mean)*2.0,-.25,.30)*rock_mask;"""),
    )
    for old, new in changes:
        if body.count(old) != 1:
            raise ValueError("Reef-stone-grain anchor changed: " + old[:60])
        body = body.replace(old, new)
    grain_bed.write_text(body)
    window_root = OUT / "shader-reef-window"
    for path in lit_root.rglob("*.hlsl"):
        destination = window_root / path.relative_to(lit_root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    window_water = window_root / "Renderer/sandbox/water_surface.hlsl"
    body = window_water.read_text()
    old = "    alpha *= 1 - .62 * bed_window; // Lab: reveal the authored bed"
    new = old + "\n    alpha *= 1 - .25 * q3_lab_reef_coverage(input, world); // Lab: local clear water above rocks"
    if body.count(old) != 1:
        raise ValueError("Reef-window water alpha anchor changed")
    window_water.write_text(body.replace(old, new))
    detail_root = OUT / "shader-reef-detail"
    for path in lit_root.rglob("*.hlsl"):
        destination = detail_root / path.relative_to(lit_root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    detail_bed = detail_root / "Renderer/native/city_fidelity/hydrology.hlsl"
    body = detail_bed.read_text()
    changes = (
        (""" float2 relief=water_decal_height_texture.Sample(decal_sampler,placement.xy).rg;
 // The second packed channel isolates rock interiors in this source atlas;
 // its physical meaning remains unresolved, so this is a Lab art mask.
 rock.a*=placement.z*smoothstep(.30,.48,relief.g);""",
         " rock.a*=placement.z;"),
        (""" float3 placement=q3_lab_reef_placement(world);
 float rock_relief=water_decal_height_texture.Sample(decal_sampler,placement.xy).g;
 float3 rock_normal=q3_margin_normal(input,rock_relief*reef,.72);""",
         """ float source_luma=dot(q3_lab_reef_sample(world).rgb,float3(.30,.59,.11));
 float3 rock_normal=q3_margin_normal(input,source_luma*reef,.30);"""),
        (""" float3 placement=q3_lab_reef_placement(world);
 float rock_relief=water_decal_height_texture.Sample(decal_sampler,placement.xy).g;
 float rock_height=cliff_height_texture.Sample(material_sampler,rock_uv).r;
 float rock_mean=cliff_height_texture.SampleBias(material_sampler,rock_uv,3).r;
 float3 submerged_rock=lerp(float3(.20,.13,.09),float3(.45,.27,.16),smoothstep(.31,.50,rock_relief))*(.73+rock_grain*.95);
 color=lerp(color,submerged_rock,rock_mask*.94);
 color*=1+clamp((rock_height-rock_mean)*1.5,-.20,.25)*rock_mask;
 color*=1+clamp((rock_relief-.37)*2.0,-.22,.28)*rock_mask;""",
         """ float3 placement=q3_lab_reef_placement(world);
 float3 local_mean=water_decal_base_texture.SampleBias(decal_sampler,placement.xy,6).rgb;
 float3 source_form=(reef.rgb-local_mean)*3.2;
 color+=source_form*rock_mask*(.78+rock_grain*.32);"""),
    )
    for old, new in changes:
        if body.count(old) != 1:
            raise ValueError("Reef-detail anchor changed: " + old[:60])
        body = body.replace(old, new)
    detail_bed.write_text(body)
    composite_root = OUT / "shader-reef-composite"
    for path in detail_root.rglob("*.hlsl"):
        destination = composite_root / path.relative_to(detail_root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    composite_bed = composite_root / "Renderer/native/city_fidelity/hydrology.hlsl"
    body = composite_bed.read_text()
    old = """ float3 source_form=(reef.rgb-local_mean)*3.2;
 color+=source_form*rock_mask*(.78+rock_grain*.32);"""
    new = """ float relief=water_decal_height_texture.Sample(decal_sampler,placement.xy).g;
 float core=smoothstep(.22,.45,relief);
 color*=1-rock_mask*core*.28;
 float3 source_form=(reef.rgb-local_mean)*2.15;
 color+=source_form*rock_mask*(.78+rock_grain*.32);"""
    if body.count(old) != 1:
        raise ValueError("Reef-composite material anchor changed")
    composite_bed.write_text(body.replace(old, new))
    record = json.loads((OUT / "snapshot.json").read_text())
    record["baseline_hydrology_sha256"] = digest(baseline)
    record["candidate_hydrology_sha256"] = digest(candidate)
    record["rich_water_sha256"] = digest(rich_water)
    record["rich_hydrology_sha256"] = digest(rich_hydrology)
    record["lagoon_water_sha256"] = digest(lagoon_root / "Renderer/sandbox/water_surface.hlsl")
    record["lagoon_hydrology_sha256"] = digest(lagoon_root / "Renderer/native/city_fidelity/hydrology.hlsl")
    record["bed-only_water_sha256"] = digest(bed_only_water)
    record["bed-only_hydrology_sha256"] = digest(bed_only_root / "Renderer/native/city_fidelity/hydrology.hlsl")
    record["clearwater_water_sha256"] = digest(clear_water)
    record["clearwater_hydrology_sha256"] = digest(clear_root / "Renderer/native/city_fidelity/hydrology.hlsl")
    record["no-clutter_water_sha256"] = digest(no_clutter_root / "Renderer/sandbox/water_surface.hlsl")
    record["no-clutter_hydrology_sha256"] = digest(no_clutter_bed)
    record["scattered_water_sha256"] = digest(scattered_root / "Renderer/sandbox/water_surface.hlsl")
    record["scattered_hydrology_sha256"] = digest(scattered_bed)
    record["rockbeds_water_sha256"] = digest(rockbeds_root / "Renderer/sandbox/water_surface.hlsl")
    record["rockbeds_hydrology_sha256"] = digest(rockbeds_bed)
    record["continuous_water_sha256"] = digest(continuous_root / "Renderer/sandbox/water_surface.hlsl")
    record["continuous_hydrology_sha256"] = digest(continuous_bed)
    record["aquamarine_water_sha256"] = digest(aquamarine_water)
    record["aquamarine_hydrology_sha256"] = digest(aquamarine_bed)
    record["aquamarine-no-margin_water_sha256"] = digest(no_margin_root / "Renderer/sandbox/water_surface.hlsl")
    record["aquamarine-no-margin_hydrology_sha256"] = digest(no_margin_bed)
    record["aquamarine-clean-bed_water_sha256"] = digest(clean_root / "Renderer/sandbox/water_surface.hlsl")
    record["aquamarine-clean-bed_hydrology_sha256"] = digest(clean_bed)
    record["shelf-relief_water_sha256"] = digest(shelf_root / "Renderer/sandbox/water_surface.hlsl")
    record["shelf-relief_hydrology_sha256"] = digest(shelf_bed)
    record["shelf-mesh_water_sha256"] = digest(mesh_root / "Renderer/sandbox/water_surface.hlsl")
    record["shelf-mesh_hydrology_sha256"] = digest(mesh_root / "Renderer/native/city_fidelity/hydrology.hlsl")
    record["shelf-mesh-lit_water_sha256"] = digest(lit_mesh_root / "Renderer/sandbox/water_surface.hlsl")
    record["shelf-mesh-lit_hydrology_sha256"] = digest(lit_mesh_bed)
    record["shelf-mesh-control_water_sha256"] = digest(control_root / "Renderer/sandbox/water_surface.hlsl")
    record["shelf-mesh-control_hydrology_sha256"] = digest(control_root / "Renderer/native/city_fidelity/hydrology.hlsl")
    record["reef-field_water_sha256"] = digest(reef_root / "Renderer/sandbox/water_surface.hlsl")
    record["reef-field_hydrology_sha256"] = digest(reef_bed)
    record["reef-forms_water_sha256"] = digest(forms_root / "Renderer/sandbox/water_surface.hlsl")
    record["reef-forms_hydrology_sha256"] = digest(forms_bed)
    record["reef-ridges_water_sha256"] = digest(ridges_root / "Renderer/sandbox/water_surface.hlsl")
    record["reef-ridges_hydrology_sha256"] = digest(ridges_bed)
    record["reef-relief_water_sha256"] = digest(relief_root / "Renderer/sandbox/water_surface.hlsl")
    record["reef-relief_hydrology_sha256"] = digest(relief_bed)
    record["reef-contrast_water_sha256"] = digest(contrast_root / "Renderer/sandbox/water_surface.hlsl")
    record["reef-contrast_hydrology_sha256"] = digest(contrast_bed)
    record["reef-lit_water_sha256"] = digest(lit_root / "Renderer/sandbox/water_surface.hlsl")
    record["reef-lit_hydrology_sha256"] = digest(lit_bed)
    record["reef-stone_water_sha256"] = digest(stone_root / "Renderer/sandbox/water_surface.hlsl")
    record["reef-stone_hydrology_sha256"] = digest(stone_bed)
    record["reef-stone-grain_water_sha256"] = digest(grain_root / "Renderer/sandbox/water_surface.hlsl")
    record["reef-stone-grain_hydrology_sha256"] = digest(grain_bed)
    record["reef-window_water_sha256"] = digest(window_water)
    record["reef-window_hydrology_sha256"] = digest(window_root / "Renderer/native/city_fidelity/hydrology.hlsl")
    record["reef-detail_water_sha256"] = digest(detail_root / "Renderer/sandbox/water_surface.hlsl")
    record["reef-detail_hydrology_sha256"] = digest(detail_bed)
    record["reef-composite_water_sha256"] = digest(composite_root / "Renderer/sandbox/water_surface.hlsl")
    record["reef-composite_hydrology_sha256"] = digest(composite_bed)
    (OUT / "snapshot.json").write_text(json.dumps(record, indent=2) + "\n")


def prepare() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    source = SANDBOX / "water_surface.hlsl"
    binaries = ("client_x64.exe", "synthetic_host_x86.exe", "C3XReference_x64.dll")
    built = OUT / "build"
    for name in binaries:
        shutil.copy2((built if (built / name).is_file() else SANDBOX / "out") / name,
                     OUT / name)
    shutil.copy2(SANDBOX / "out/test-biq.csv", OUT / "scene.csv")
    for label in ("baseline", "candidate"):
        target = OUT / f"shader-{label}"
        for path in (ROOT / "Renderer/native").rglob("*.hlsl"):
            relative = path.relative_to(ROOT)
            destination = target / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
        variant = target / "Renderer/sandbox/water_surface.hlsl"
        variant.parent.mkdir(parents=True, exist_ok=True)
        body = source.read_text()
        variant.write_text(body if label == "baseline" else candidate_shader(body))
    (OUT / "snapshot.json").write_text(json.dumps({
        "sandbox_water_sha256": digest(source),
        "baseline_water_sha256": digest(OUT / "shader-baseline/Renderer/sandbox/water_surface.hlsl"),
        "candidate_water_sha256": digest(OUT / "shader-candidate/Renderer/sandbox/water_surface.hlsl"),
        "binaries": {name: digest(OUT / name) for name in binaries},
        "scene_sha256": digest(OUT / "scene.csv"),
        "status": "isolated sandbox-derived Lab experiment; production files unchanged",
    }, indent=2) + "\n")
    refine()


def build() -> None:
    """Compile current sandbox source into the ignored study folder."""
    OUT.mkdir(parents=True, exist_ok=True)
    source = (SANDBOX / "build_reference_x64.bat").read_text()
    source = source.replace('pushd "%~dp0"', 'pushd "%~dp0..\\..\\..\\sandbox"')
    source = source.replace("out\\", "..\\lab\\out\\coastal-shallows\\build\\")
    source = source.replace("if not exist out mkdir out", "if not exist ..\\lab\\out\\coastal-shallows\\build mkdir ..\\lab\\out\\coastal-shallows\\build")
    script = OUT / "build.bat"
    script.write_text(source)
    result = native_command_result("Renderer/sandbox",
                                   r'call "..\lab\out\coastal-shallows\build.bat"',
                                   timeout_seconds=300)
    if result["status"] != "pass":
        raise ValueError("Sandbox study build failed: " + result["output_tail"][-2000:])
    build_client()


def build_client() -> None:
    # The water study excludes synthetic units. The current unit-prewarm input
    # can fail before any sandbox water frame is drawn; this client-only copy
    # skips that unrelated step while keeping the compiled scene DLL intact.
    client = (SANDBOX / "client_x64.cpp").read_text()
    anchor = "int prewarm_result=prewarm(prepared_frame.hour,prepared_frame.season);"
    if client.count(anchor) != 1:
        raise ValueError("Sandbox client prewarm contract changed")
    client = client.replace(anchor,
        'char water_only[4]{};\n'
        '    int prewarm_result=GetEnvironmentVariableA("C3X_LAB_WATER_ONLY",'
        'water_only,sizeof(water_only)) && water_only[0]==\'1\' ? 0 : '
        'prewarm(prepared_frame.hour,prepared_frame.season);')
    client = client.replace('#include "../native/c3x_renderer_api.h"',
                            '#include "../../../../native/c3x_renderer_api.h"')
    client = client.replace('#include "exchange.h"',
                            '#include "../../../../sandbox/exchange.h"')
    (OUT / "build/client_water_only.cpp").write_text(client)
    compile_script = OUT / "build-client.bat"
    compile_script.write_text("\n".join((
        "@echo off", "setlocal", 'pushd "%~dp0..\\..\\..\\sandbox"',
        'set "VSWHERE=%ProgramFiles(x86)%\\Microsoft Visual Studio\\Installer\\vswhere.exe"',
        'for /f "usebackq delims=" %%i in (`"%VSWHERE%" -latest -prerelease -products * '
        '-requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) '
        'do set "C3X_SANDBOX_VS=%%i"',
        'if not defined C3X_SANDBOX_VS exit /b 1',
        'call "%C3X_SANDBOX_VS%\\VC\\Auxiliary\\Build\\vcvars64.bat" >nul',
        'if errorlevel 1 exit /b 1',
        'cl /nologo /std:c++17 /EHsc /O2 /W4 /WX /DC3X_SANDBOX_CLIENT /I..\\native '
        'reference_x64.cpp ..\\lab\\out\\coastal-shallows\\build\\client_water_only.cpp '
        '/Fo:..\\lab\\out\\coastal-shallows\\build\\obj\\ '
        '/Fe:..\\lab\\out\\coastal-shallows\\build\\client_x64.exe '
        '/link gdi32.lib user32.lib',
        'exit /b %errorlevel%', '',
    )))
    result = native_command_result("Renderer/sandbox",
                                   r'call "..\lab\out\coastal-shallows\build-client.bat"',
                                   timeout_seconds=180)
    if result["status"] != "pass":
        raise ValueError("Water-only study client build failed: " + result["output_tail"][-2000:])


def build_shelf_mesh() -> None:
    """Build one isolated DLL with separate bed geometry; restore both sources."""
    if not (OUT / "build.bat").is_file():
        raise ValueError("Run build first to create the isolated build script")
    original = GROUND_COMPILER.read_text()
    renderer_original = RENDERER_CPP.read_text()
    changes = (
        ('#include "../../lab/shared/natural/ground.h"',
         '#include "../../lab/shared/natural/ground.h"\n'
         '#include "../../lab/studies/coastal_shallows/seabed_relief.h"'),
        ('        bool underlay_surface = layer > 0.4f && layer < 0.6f;',
         '        bool underlay_surface = layer > 0.4f && layer < 0.6f;\n'
         '        bool bed_surface = layer > 3.5f && layer < 4.5f;'),
        ("""        float h = relief_sample[0] * input.relief_projection_scale;
        float signed_shore = point.signed_shore;""",
         """        float h = relief_sample[0] * input.relief_projection_scale;
        float bed_normal_x=0.f,bed_normal_y=0.f,bed_normal_z=1.f;
        if (bed_surface && input.pickup_profile) {
            auto bed_at=[&](float x,float y) {
                return coastal_lab::height(x,y,float(point.shore.distance),
                    float(point.shore.depth),point.surface_coordinate,
                    frame.world_width_tiles,frame.world_height_tiles,
                    frame.world_wrap_x!=0,frame.world_wrap_y!=0);
            };
            float bed_height=bed_at(world_u,world_v);
            relief_sample[0]=bed_height;
            h=bed_height*input.relief_projection_scale;
            constexpr float normal_step=.015f;
            float slope_x=(bed_at(world_u+normal_step,world_v)-
                           bed_at(world_u-normal_step,world_v))/(2*normal_step*64.f);
            float slope_y=(bed_at(world_u,world_v+normal_step)-
                           bed_at(world_u,world_v-normal_step))/(2*normal_step*64.f);
            float length=std::sqrt(1.f+slope_x*slope_x+slope_y*slope_y);
            bed_normal_x=-slope_x/length;
            bed_normal_y=-slope_y/length;
            bed_normal_z=1.f/length;
        }
        float signed_shore = point.signed_shore;"""),
        ("""        float normal_x = terrain_conforming_surface ? point.normal[0] : 0.0f;
        float normal_y = terrain_conforming_surface ? point.normal[1] : 0.0f;
        float normal_z = terrain_conforming_surface ? point.normal[2] : 1.0f;""",
         """        float normal_x = bed_surface && input.pickup_profile ? bed_normal_x :
            terrain_conforming_surface ? point.normal[0] : 0.0f;
        float normal_y = bed_surface && input.pickup_profile ? bed_normal_y :
            terrain_conforming_surface ? point.normal[1] : 0.0f;
        float normal_z = bed_surface && input.pickup_profile ? bed_normal_z :
            terrain_conforming_surface ? point.normal[2] : 1.0f;"""),
        ("""        if (input.world_ground) {
            float elevation=relief_sample[0]*(128.f/224.f*.82f),base=(u+v)*32.f;
            vertex.x=64.f+(u-v)*64.f;vertex.y=base-elevation;vertex.z=base+elevation*.75f;
            if(terrain_conforming_surface){vertex.normal_x=-point.normal_delta[0]/.012f;
                vertex.normal_y=point.normal_delta[1]/.012f;vertex.normal_z=1;}
        }
        return vertex;""",
         """        if (input.world_ground) {
            float elevation=relief_sample[0]*(128.f/224.f*.82f),base=(u+v)*32.f;
            vertex.x=64.f+(u-v)*64.f;vertex.y=base-elevation;vertex.z=base+elevation*.75f;
            if(terrain_conforming_surface){vertex.normal_x=-point.normal_delta[0]/.012f;
                vertex.normal_y=point.normal_delta[1]/.012f;vertex.normal_z=1;}
        }
        if (bed_surface && input.pickup_profile) {
            // The 2.5D renderer projects the water mask on the mean plane.
            // Keep bed coverage aligned while world height and lit normals
            // carry the relief underneath the flat water surface.
            float plane=input.world_ground ? (u+v)*32.f : ground_y;
            vertex.y=plane;
            vertex.z=plane-(input.world_ground ? 2.f : 2.f*frame.tile_width/128.f);
        }
        return vertex;"""),
        ("""    if (!input.pickup_profile) {
        append_ground_layer(destination.bed_vertices, 4.0f, input.flat_grid, &destination.bed_indices);""",
         """    if (input.pickup_profile &&
        std::abs(ground_point_at(.5f,.5f).shore.distance)<1.5f) {
        append_ground_layer(destination.bed_vertices, 4.0f, input.flat_grid,
                            &destination.bed_indices);
    }
    if (!input.pickup_profile) {
        append_ground_layer(destination.bed_vertices, 4.0f, input.flat_grid, &destination.bed_indices);"""),
        ('        bool record=input.retain_ground_grids && !input.world_ground && !input.prewarming && indices && !cached_grid;',
         '        bool record=input.retain_ground_grids && !input.world_ground && !input.prewarming && indices && !cached_grid && layer!=4.f;'),
    )
    patched = original
    for old, new in changes:
        if patched.count(old) != 1:
            raise ValueError("Shelf mesh source anchor changed: " + old[:65])
        patched = patched.replace(old, new)
    renderer_changes = (
        ("""                if(i==2 || i==3)continue;
                append(result->ground->meshes[i],result->ground_vertices[i],result->ground_indices[i]);""",
         """                if(i==3)continue; // Water alone retains the flat underlay mesh.
                append(result->ground->meshes[i],result->ground_vertices[i],result->ground_indices[i]);"""),
        ("""                if(pickup_profile && (layer==geometry_bed || layer==geometry_water)) {
                    if(!water_coverage)continue;""",
         """                if(pickup_profile && (layer==geometry_water ||
                    (layer==geometry_bed && prepared_ground->meshes[2].empty()))) {
                    if(!water_coverage)continue;"""),
    )
    renderer_patched = renderer_original
    for old, new in renderer_changes:
        if renderer_patched.count(old) != 1:
            raise ValueError("Shelf mesh draw-path anchor changed: " + old[:65])
        renderer_patched = renderer_patched.replace(old, new)
    original_hash = hashlib.sha256(original.encode()).hexdigest()
    renderer_hash = hashlib.sha256(renderer_original.encode()).hexdigest()
    try:
        GROUND_COMPILER.write_text(patched)
        RENDERER_CPP.write_text(renderer_patched)
        result = native_command_result("Renderer/sandbox",
                                       r'call "..\lab\out\coastal-shallows\build.bat"',
                                       timeout_seconds=600)
        if result["status"] != "pass":
            raise ValueError("Shelf mesh candidate build failed: " + result["output_tail"][-2500:])
        candidate = OUT / "build/C3XReference_x64.dll"
        if not candidate.is_file():
            raise ValueError("Shelf mesh build did not produce a DLL")
        shutil.copy2(candidate, OUT / "C3XReferenceShelf_x64.dll")
    finally:
        restore_error = None
        for path, expected, source in ((GROUND_COMPILER, patched, original),
                                       (RENDERER_CPP, renderer_patched, renderer_original)):
            if path.read_text() != expected:
                restore_error = f"{path.name} changed during isolated build; preserve its current contents"
            else:
                path.write_text(source)
        if restore_error:
            raise ValueError(restore_error)
    if digest(GROUND_COMPILER) != original_hash or digest(RENDERER_CPP) != renderer_hash:
        raise ValueError("Renderer sources were not restored byte for byte")
    record = json.loads((OUT / "snapshot.json").read_text())
    record["shelf_mesh_dll_sha256"] = digest(OUT / "C3XReferenceShelf_x64.dll")
    record["shelf_mesh_header_sha256"] = original_hash
    record["shelf_mesh_renderer_sha256"] = renderer_hash
    (OUT / "snapshot.json").write_text(json.dumps(record, indent=2) + "\n")
    print(OUT / "C3XReferenceShelf_x64.dll")


def build_mesh_control() -> None:
    """Compile the unmodified current source with the same build path."""
    if not (OUT / "build.bat").is_file():
        raise ValueError("Run build first to create the isolated build script")
    before = digest(GROUND_COMPILER)
    result = native_command_result("Renderer/sandbox",
                                   r'call "..\lab\out\coastal-shallows\build.bat"',
                                   timeout_seconds=600)
    if result["status"] != "pass":
        raise ValueError("Shelf mesh control build failed: " + result["output_tail"][-2500:])
    if digest(GROUND_COMPILER) != before:
        raise ValueError("Ground compiler changed during control build")
    candidate = OUT / "build/C3XReference_x64.dll"
    shutil.copy2(candidate, OUT / "C3XReferenceCurrent_x64.dll")
    record = json.loads((OUT / "snapshot.json").read_text())
    record["shelf_mesh_control_dll_sha256"] = digest(OUT / "C3XReferenceCurrent_x64.dll")
    (OUT / "snapshot.json").write_text(json.dumps(record, indent=2) + "\n")
    print(OUT / "C3XReferenceCurrent_x64.dll")


def build_form_probe() -> None:
    """Deliberately obvious bed-only bands to prove the visible material path."""
    source = OUT / "shader-shelf-mesh-control"
    target = OUT / "shader-form-probe"
    if not source.is_dir():
        raise ValueError("Run refine before the form probe")
    for path in source.rglob("*.hlsl"):
        destination = target / path.relative_to(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    bed = target / "Renderer/native/city_fidelity/hydrology.hlsl"
    body = bed.read_text()
    old = " return color*clamp(.86+(height-.426)*.55,.65,1.1)*absorption*tint;"
    new = """ // Diagnostic world-space ridge: deliberately simple and high contrast.
 // It is a route/scale probe, not proposed seabed art.
 float ridge=sin(world.x*4.2+world.y*1.2);
 float diagnostic=coast_shelf*.52*ridge;
 return color*clamp(.86+(height-.426)*.55,.65,1.1)*absorption*tint
  *(1+diagnostic);"""
    if body.count(old) != 1:
        raise ValueError("Form probe bed anchor changed")
    bed.write_text(body.replace(old, new))


def build_bed_hidden_probe() -> None:
    """Hide only the candidate bed draw to diagnose its water/depth coverage."""
    source = OUT / "shader-shelf-mesh-lit"
    target = OUT / "shader-bed-hidden-probe"
    for path in source.rglob("*.hlsl"):
        destination = target / path.relative_to(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    bed = target / "Renderer/native/city_fidelity/hydrology.hlsl"
    body = bed.read_text()
    old = " if(kind<4.5)return float4(q3_scene_bed(input)*q6_receiver_illumination(input,q3_authored_bed_normal(input),1,1),1);"
    new = " if(kind<4.5){clip(-1);return 0;}"
    if body.count(old) != 1:
        raise ValueError("Bed hidden probe anchor changed")
    bed.write_text(body.replace(old, new))


def build_desert_ripple_bed() -> None:
    """Reuse the desert's continuous source height under coast water only."""
    source = OUT / "shader-shelf-mesh-control"
    target = OUT / "shader-desert-ripple-bed"
    for path in source.rglob("*.hlsl"):
        destination = target / path.relative_to(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    bed = target / "Renderer/native/city_fidelity/hydrology.hlsl"
    body = bed.read_text()
    old = " float3 continuous_normal=q3_margin_normal(input,bed_alpha,.025);\n return normalize(lerp(decal_normal,continuous_normal,coast_family));"
    new = (" // Desert's continuous authored sand height, transferred beneath coast water.\n"
           " float2 sand_uv=world*q3_source_repeat(.26)+float2(.31,.17);\n"
           " float sand_height=desert_height_texture.Sample(material_sampler,sand_uv).r;\n"
           " float shelf=coast_family*(1-smoothstep(.28,.64,input.hydrology_data.w));\n"
           " float3 continuous_normal=q3_margin_normal(input,\n"
           "  lerp(bed_alpha,sand_height,.88),.42);\n"
           " return normalize(lerp(decal_normal,continuous_normal,shelf));")
    if body.count(old) != 1:
        raise ValueError("Desert ripple normal anchor changed")
    body = body.replace(old, new)
    old = " color*=1+beach_grain*.32*coast_shelf;"
    new = (" color*=1+beach_grain*.32*coast_shelf;\n"
           " float2 sand_uv=world*q3_source_repeat(.26)+float2(.31,.17);\n"
           " float sand_height=desert_height_texture.Sample(material_sampler,sand_uv).r;\n"
           " float sand_mean=desert_height_texture.SampleBias(material_sampler,sand_uv,3).r;\n"
           " // Source-height crest/cavity response, with no added stamps.\n"
           " color*=clamp(1+(sand_height-sand_mean)*2.4*coast_shelf,.72,1.23);")
    if body.count(old) != 1:
        raise ValueError("Desert ripple color anchor changed")
    bed.write_text(body.replace(old, new))


def build_desert_ripple_broad() -> None:
    """Add the source desert-hills height as a broad seabed relief scale."""
    build_desert_ripple_bed()
    source = OUT / "shader-desert-ripple-bed"
    target = OUT / "shader-desert-ripple-broad"
    for path in source.rglob("*.hlsl"):
        destination = target / path.relative_to(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    bed = target / "Renderer/native/city_fidelity/hydrology.hlsl"
    body = bed.read_text()
    old = """ float3 continuous_normal=q3_margin_normal(input,
  lerp(bed_alpha,sand_height,.88),.42);"""
    new = """ float2 broad_uv=float2(world.y,-world.x)*q3_source_repeat(.115)+float2(.43,.61);
 float broad_height=desert_hills_height_texture.Sample(material_sampler,broad_uv).r;
 float3 continuous_normal=q3_margin_normal(input,
  sand_height*.48+broad_height*.52,.63);"""
    if body.count(old) != 1:
        raise ValueError("Broad desert normal anchor changed")
    body = body.replace(old, new)
    old = " color*=clamp(1+(sand_height-sand_mean)*2.4*coast_shelf,.72,1.23);"
    new = """ float2 broad_uv=float2(world.y,-world.x)*q3_source_repeat(.115)+float2(.43,.61);
 float broad_height=desert_hills_height_texture.Sample(material_sampler,broad_uv).r;
 float broad_mean=desert_hills_height_texture.SampleBias(material_sampler,broad_uv,3).r;
 float surface=(sand_height-sand_mean)*1.5+(broad_height-broad_mean)*2.7;
 color*=clamp(1+surface*coast_shelf,.65,1.26);"""
    if body.count(old) != 1:
        raise ValueError("Broad desert color anchor changed")
    bed.write_text(body.replace(old, new))


def build_desert_direction_mix(broad: bool = False) -> None:
    """Blend two source-height dune flows, optionally retaining hills relief."""
    if broad:
        build_desert_ripple_broad()
    else:
        build_desert_ripple_bed()
    source = OUT / ("shader-desert-ripple-broad" if broad else "shader-desert-ripple-bed")
    target = OUT / ("shader-desert-direction-broad" if broad else "shader-desert-direction-mix")
    for path in source.rglob("*.hlsl"):
        destination = target / path.relative_to(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    bed = target / "Renderer/native/city_fidelity/hydrology.hlsl"
    body = bed.read_text()
    anchor = "float3 q3_authored_bed_normal(PixelInput input) {"
    helper = """float2 q3_coast_dune_height(float2 world) {
 // Both directions use the existing desert source height. Broad source noise
 // selects their flow continuously, so no tile owns a direction or stamp.
 float2 uv0=world*q3_source_repeat(.26)+float2(.31,.17);
 float2 uv1=float2(world.y,-world.x)*q3_source_repeat(.26)+float2(.71,.43);
 float2 a=float2(desert_height_texture.Sample(material_sampler,uv0).r,
  desert_height_texture.SampleBias(material_sampler,uv0,3).r);
 float2 b=float2(desert_height_texture.Sample(material_sampler,uv1).r,
  desert_height_texture.SampleBias(material_sampler,uv1,3).r);
 float2 flow_uv=world*float2(q3_source_repeat(.11),q3_source_repeat(.14))
  +float2(.19,.53);
 float flow=river_bank_noise_texture.Sample(material_sampler,flow_uv).r;
 return lerp(a,b,smoothstep(.39,.61,flow));
}
"""
    if body.count(anchor) != 1:
        raise ValueError("Directional dune helper anchor changed")
    head, tail = body.replace(anchor, helper + anchor).split("float3 q3_scene_bed(PixelInput input) {", 1)
    old = " float2 sand_uv=world*q3_source_repeat(.26)+float2(.31,.17);\n float sand_height=desert_height_texture.Sample(material_sampler,sand_uv).r;"
    if head.count(old) != 1:
        raise ValueError("Directional dune normal anchor changed")
    head = head.replace(old, " float sand_height=q3_coast_dune_height(world).x;")
    old = (" float2 sand_uv=world*q3_source_repeat(.26)+float2(.31,.17);\n"
           " float sand_height=desert_height_texture.Sample(material_sampler,sand_uv).r;\n"
           " float sand_mean=desert_height_texture.SampleBias(material_sampler,sand_uv,3).r;")
    if tail.count(old) != 1:
        raise ValueError("Directional dune color anchor changed")
    tail = tail.replace(old, " float2 dune=q3_coast_dune_height(world);\n float sand_height=dune.x,sand_mean=dune.y;")
    bed.write_text(head + "float3 q3_scene_bed(PixelInput input) {" + tail)


def build_desert_direction_patches() -> None:
    """Keep the preferred relief while changing flow over broad coast regions."""
    build_desert_direction_mix(broad=True)
    source = OUT / "shader-desert-direction-broad"
    target = OUT / "shader-desert-direction-patches"
    for path in source.rglob("*.hlsl"):
        destination = target / path.relative_to(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    bed = target / "Renderer/native/city_fidelity/hydrology.hlsl"
    body = bed.read_text()
    changes = (
        ("q3_source_repeat(.11),q3_source_repeat(.14)",
         "q3_source_repeat(.035),q3_source_repeat(.048)"),
        ("smoothstep(.39,.61,flow)", "smoothstep(.455,.545,flow)"),
    )
    for old, new in changes:
        if body.count(old) != 1:
            raise ValueError("Directional patch anchor changed: " + old)
        body = body.replace(old, new)
    bed.write_text(body)


def build_desert_domain_warp() -> None:
    """Bend one source dune field coherently instead of blending two fields."""
    build_desert_ripple_broad()
    source = OUT / "shader-desert-ripple-broad"
    target = OUT / "shader-desert-domain-warp"
    for path in source.rglob("*.hlsl"):
        destination = target / path.relative_to(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    bed = target / "Renderer/native/city_fidelity/hydrology.hlsl"
    body = bed.read_text()
    anchor = "float3 q3_authored_bed_normal(PixelInput input) {"
    helper = """float2 q3_coast_warped_sand_uv(float2 world) {
 // A broad, periodic source-noise vector bends one continuous dune field.
 // This remapping is inferred art direction; it does not place new decals.
 float2 flow_uv=world*float2(q3_source_repeat(.025),q3_source_repeat(.031));
 float2 bend=float2(
  river_bank_noise_texture.Sample(material_sampler,flow_uv+float2(.13,.47)).r,
  river_bank_noise_texture.Sample(material_sampler,flow_uv+float2(.61,.19)).r);
 return world*q3_source_repeat(.26)+float2(.31,.17)+(bend-.5)*.55;
}
"""
    if body.count(anchor) != 1:
        raise ValueError("Warped dune helper anchor changed")
    body = body.replace(anchor, helper + anchor)
    old = "float2 sand_uv=world*q3_source_repeat(.26)+float2(.31,.17);"
    if body.count(old) != 2:
        raise ValueError("Warped dune sample anchors changed")
    bed.write_text(body.replace(old, "float2 sand_uv=q3_coast_warped_sand_uv(world);"))


def build_desert_shore_guided() -> None:
    """Follow the prepared coast-distance field with source sand contours."""
    build_desert_ripple_broad()
    source = OUT / "shader-desert-ripple-broad"
    target = OUT / "shader-desert-shore-guided"
    for path in source.rglob("*.hlsl"):
        destination = target / path.relative_to(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    bed = target / "Renderer/native/city_fidelity/hydrology.hlsl"
    body = bed.read_text()
    anchor = "float3 q3_authored_bed_normal(PixelInput input) {"
    helper = """float2 q3_coast_shore_sand_uv(PixelInput input,float2 world) {
 // Source desert ridges run approximately along constant (u+v). Make that
 // source axis advance with the prepared signed shoreline distance, so dunes
 // broadly track local land rather than one global compass direction.
 float2 source_uv=world*q3_source_repeat(.26)+float2(.31,.17);
 float along=source_uv.x-source_uv.y;
 float outward=max(0,-input.hydrology_data.x);
 float2 flow_uv=world*float2(q3_source_repeat(.025),q3_source_repeat(.031));
 float bend=river_bank_noise_texture.Sample(material_sampler,
  flow_uv+float2(.13,.47)).r-.5;
 float across=outward*q3_source_repeat(.52)+.48+bend*.16;
 return float2((across+along)*.5,(across-along)*.5);
}
"""
    if body.count(anchor) != 1:
        raise ValueError("Shore-guided dune helper anchor changed")
    body = body.replace(anchor, helper + anchor)
    old = "float2 sand_uv=world*q3_source_repeat(.26)+float2(.31,.17);"
    if body.count(old) != 2:
        raise ValueError("Shore-guided dune sample anchors changed")
    bed.write_text(body.replace(old, "float2 sand_uv=q3_coast_shore_sand_uv(input,world);"))


def build_desert_shore_fine() -> None:
    """Refine shoreline-following dunes and attenuate them offshore."""
    build_desert_shore_guided()
    source = OUT / "shader-desert-shore-guided"
    target = OUT / "shader-desert-shore-fine"
    for path in source.rglob("*.hlsl"):
        destination = target / path.relative_to(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    bed = target / "Renderer/native/city_fidelity/hydrology.hlsl"
    body = bed.read_text()
    changes = (
        ("float2 source_uv=world*q3_source_repeat(.26)+float2(.31,.17);",
         "float2 source_uv=world*q3_source_repeat(.40)+float2(.31,.17);"),
        (" float across=outward*q3_source_repeat(.52)+.48+bend*.16;",
         """ float spacing=river_bank_noise_texture.Sample(material_sampler,
  flow_uv+float2(.61,.19)).r;
 float across=outward*q3_source_repeat(.80)*lerp(.72,1.24,spacing)
  +.48+bend*.24;"""),
        ("float shelf=coast_family*(1-smoothstep(.28,.64,input.hydrology_data.w));",
         "float shelf=coast_family*(1-smoothstep(.13,.49,input.hydrology_data.w));"),
        ("color*=clamp(1+surface*coast_shelf,.65,1.26);",
         """float dune_fade=coast_family*(1-smoothstep(.15,.52,input.hydrology_data.w));
 color*=clamp(1+surface*dune_fade,.65,1.26);"""),
    )
    for old, new in changes:
        if body.count(old) != 1:
            raise ValueError("Fine shore dune anchor changed: " + old)
        body = body.replace(old, new)
    bed.write_text(body)


def build_desert_broad_tuned() -> None:
    """Retain broad's soft relief with restrained shore flow and depth fade."""
    build_desert_ripple_broad()
    source = OUT / "shader-desert-ripple-broad"
    target = OUT / "shader-desert-broad-tuned"
    for path in source.rglob("*.hlsl"):
        destination = target / path.relative_to(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    bed = target / "Renderer/native/city_fidelity/hydrology.hlsl"
    body = bed.read_text()
    anchor = "float3 q3_authored_bed_normal(PixelInput input) {"
    helper = """float2 q3_coast_broad_tuned_uv(PixelInput input,float2 world) {
 // Keep most of the authored desert flow; bias its cross-ridge coordinate
 // toward local coast distance so contours turn gently with the shoreline.
 float2 base=world*q3_source_repeat(.33)+float2(.31,.17);
 float2 flow_uv=world*float2(q3_source_repeat(.025),q3_source_repeat(.031));
 float bend=river_bank_noise_texture.Sample(material_sampler,
  flow_uv+float2(.13,.47)).r-.5;
 float shore=max(0,-input.hydrology_data.x)*q3_source_repeat(.66)+.48+bend*.16;
 float across=lerp(base.x+base.y,shore,.32);
 float along=base.x-base.y;
 return float2((across+along)*.5,(across-along)*.5);
}
"""
    if body.count(anchor) != 1:
        raise ValueError("Tuned broad helper anchor changed")
    body = body.replace(anchor, helper + anchor)
    old = "float2 sand_uv=world*q3_source_repeat(.26)+float2(.31,.17);"
    if body.count(old) != 2:
        raise ValueError("Tuned broad sample anchors changed")
    body = body.replace(old, "float2 sand_uv=q3_coast_broad_tuned_uv(input,world);")
    changes = (
        ("float shelf=coast_family*(1-smoothstep(.28,.64,input.hydrology_data.w));",
         "float shelf=coast_family*(1-smoothstep(.18,.55,input.hydrology_data.w));"),
        ("color*=clamp(1+surface*coast_shelf,.65,1.26);",
         """float dune_fade=coast_shelf*(1-smoothstep(.28,.57,input.hydrology_data.w));
 color*=clamp(1+surface*dune_fade,.65,1.26);"""),
    )
    for old, new in changes:
        if body.count(old) != 1:
            raise ValueError("Tuned broad anchor changed: " + old)
        body = body.replace(old, new)
    bed.write_text(body)


def build_desert_broad_irregular() -> None:
    """Borrow the upper shelf's continuous irregular alpha to break sand bands."""
    build_desert_ripple_broad()
    source = OUT / "shader-desert-ripple-broad"
    target = OUT / "shader-desert-broad-irregular"
    for path in source.rglob("*.hlsl"):
        destination = target / path.relative_to(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    bed = target / "Renderer/native/city_fidelity/hydrology.hlsl"
    body = bed.read_text()
    changes = (
        ("float bed_alpha=shallow_bed_texture.Sample(material_sampler,bed_uv).a;",
         "float bed_alpha=shallow_bed_texture.SampleBias(material_sampler,bed_uv,2).a;"),
        ("float shelf=coast_family*(1-smoothstep(.28,.64,input.hydrology_data.w));",
         "float shelf=coast_family*(1-smoothstep(.18,.55,input.hydrology_data.w));"),
        ("sand_height*.48+broad_height*.52,.63);",
         "sand_height*.38+bed_alpha*.16+broad_height*.46,.55);"),
        ("float surface=(sand_height-sand_mean)*1.5+(broad_height-broad_mean)*2.7;",
         "float surface=(sand_height-sand_mean)*.85+(broad_height-broad_mean)*2.7;"),
        ("color*=clamp(1+surface*coast_shelf,.65,1.26);",
         """float dune_fade=coast_shelf*(1-smoothstep(.27,.57,input.hydrology_data.w));
 color*=clamp(1+surface*dune_fade,.65,1.26);"""),
    )
    for old, new in changes:
        if body.count(old) != 1:
            raise ValueError("Irregular broad anchor changed: " + old)
        body = body.replace(old, new)
    bed.write_text(body)


def build_desert_broad_mosaic() -> None:
    """Give broad coast regions either the irregular bed or clean control."""
    build_desert_broad_irregular()
    source = OUT / "shader-desert-broad-irregular"
    target = OUT / "shader-desert-broad-mosaic"
    for path in source.rglob("*.hlsl"):
        destination = target / path.relative_to(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    bed = target / "Renderer/native/city_fidelity/hydrology.hlsl"
    body = bed.read_text()
    anchor = "float3 q3_authored_bed_normal(PixelInput input) {"
    helper = """float q3_coast_irregular_region(float2 world) {
 // Two low-frequency source-noise views choose broad, continuous stretches;
 // no tile or decal owns a patch. The blend softens their boundaries.
 float2 uv0=world*float2(q3_source_repeat(.022),q3_source_repeat(.029))
  +float2(.19,.53);
 float2 uv1=world*float2(q3_source_repeat(.014),q3_source_repeat(.017))
  +float2(.61,.13);
 float broad=river_bank_noise_texture.Sample(material_sampler,uv0).r;
 float macro=river_bank_noise_texture.Sample(material_sampler,uv1).r;
 return smoothstep(.44,.56,broad*.75+macro*.25);
}
"""
    if body.count(anchor) != 1:
        raise ValueError("Mosaic region helper anchor changed")
    body = body.replace(anchor, helper + anchor)
    old = " return normalize(lerp(decal_normal,continuous_normal,shelf));"
    new = """ float3 irregular_normal=normalize(lerp(decal_normal,continuous_normal,shelf));
 float control_alpha=shallow_bed_texture.Sample(material_sampler,bed_uv).a;
 float3 control_detail=q3_margin_normal(input,control_alpha,.025);
 float3 control_normal=normalize(lerp(decal_normal,control_detail,coast_family));
 return normalize(lerp(control_normal,irregular_normal,
  q3_coast_irregular_region(world)));"""
    if body.count(old) != 1:
        raise ValueError("Mosaic normal anchor changed")
    body = body.replace(old, new)
    old = "color*=clamp(1+surface*dune_fade,.65,1.26);"
    new = "color*=clamp(1+surface*dune_fade*q3_coast_irregular_region(world),.65,1.26);"
    if body.count(old) != 1:
        raise ValueError("Mosaic color anchor changed")
    bed.write_text(body.replace(old, new))


def render(label: str, zoom: int = 128) -> None:
    if not (OUT / "snapshot.json").exists():
        raise ValueError("Run prepare first")
    output = OUT / (f"z{zoom}/{label}" if zoom != 128 else label)
    output.mkdir(parents=True, exist_ok=True)
    # Commands run from Renderer/native in the Windows VM. The asset root stays
    # at the checkout; only shader reads use the frozen study tree.
    shader_root = rf"..\..\Renderer\lab\out\coastal-shallows\shader-{label}"
    relative_output = f"z{zoom}\\{label}" if zoom != 128 else label
    capture = rf"..\lab\out\coastal-shallows\{relative_output}\frame"
    executable = r"..\lab\out\coastal-shallows\client_x64.exe"
    dll_name = ("C3XReferenceShelf_x64.dll" if label in ("shelf-mesh", "shelf-mesh-lit", "bed-hidden-probe") else
                "C3XReferenceCurrent_x64.dll" if label in ("shelf-mesh-control", "form-probe", "desert-ripple-bed", "desert-ripple-broad", "desert-direction-mix", "desert-direction-broad", "desert-direction-patches", "desert-domain-warp", "desert-shore-guided", "desert-shore-fine", "desert-broad-tuned", "desert-broad-irregular", "desert-broad-mosaic") else
                "C3XReference_x64.dll")
    dll = rf"..\lab\out\coastal-shallows\{dll_name}"
    scene = r"..\lab\out\coastal-shallows\scene.csv"
    batch = output / "render.bat"
    batch.write_text("\n".join((
        "@echo off", "setlocal",
        f'set "C3X_RENDERER_SHADER_SOURCE_ROOT={shader_root}"',
        f'set "C3X_SANDBOX_CAPTURE_SEQUENCE={capture}"',
        'set "C3X_SANDBOX_REPLAY_CLIP=1"',
        'set "C3X_SANDBOX_CLIP_FRAMES=2"',
        'set "C3X_LAB_WATER_ONLY=1"',
        'set "C3X_SANDBOX_UNITS=0"',
        'set "C3X_SANDBOX_WHOLE_WORLD=1"',
        'set "C3X_RENDERER_VISUAL_PROFILE="',
        'set "C3X_RENDERER_TRACE=2"',
        f'set "C3X_RENDERER_TRACE_FILE=..\\lab\\out\\coastal-shallows\\{relative_output}\\trace.log"',
        'set "C3X_RENDERER_PREVIEW_CUSTOM_DEFINITIONS=..\\..\\Renderer\\custom.custom_rendering.txt"',
        'set "C3X_RENDERER_SHARED_SCENE_SURFACE=1"',
        'set "C3X_RENDERER_WATER_MOTION=1"',
        'set "C3X_RENDERER_WAVES=1"',
        'set "C3X_SANDBOX_SHADOW_PATCHES=1"',
        f'"{executable}" "{dll}" ..\\.. ..\\default.custom_rendering.txt "{scene}" '
        '..\\lab\\out\\coastal-shallows\\' + relative_output +
        f'\\client-start.bmp 2240 1260 17 49 {zoom} 12',
        "exit /b %errorlevel%", "",
    )))
    command = rf'call "..\lab\out\coastal-shallows\{relative_output}\render.bat"'
    result = native_command_result("Renderer/native", command, timeout_seconds=600)
    frame = output / "frame-0000.jpg"
    if result["status"] != "pass" or not frame.is_file():
        raise ValueError(f"Sandbox capture failed (exit {result['returncode']}): "
                         f"{result['output_tail'][-1500:]}")
    (output / "result.json").write_text(json.dumps({
        "status": "pass", "frame_sha256": digest(frame),
        "tile_width": zoom, "center": [17, 49], "size": [2240, 1260],
        "shader_sha256": digest(OUT / f"shader-{label}/Renderer/sandbox/water_surface.hlsl"),
        "hydrology_sha256": digest(OUT / f"shader-{label}/Renderer/native/city_fidelity/hydrology.hlsl"),
        "scene_sha256": digest(OUT / "scene.csv"),
        "client_sha256": digest(OUT / "client_x64.exe"),
        "dll_sha256": digest(OUT / dll_name),
    }, indent=2) + "\n")
    print(frame)


def recover_rich_capture() -> None:
    """Record a complete rich JPEG capture after a VM transport timeout."""
    output = OUT / "rich"
    state = native_command_result("Renderer/sandbox",
        'tasklist /FO CSV /NH /FI "IMAGENAME eq client_x64.exe"', timeout_seconds=30)
    absent = "INFO: No tasks are running which match the specified criteria."
    if state["status"] != "pass" or state["output_tail"].strip() != absent:
        raise ValueError("Sandbox client is still running or its state is uncertain")
    from PIL import Image
    for index in range(3):
        path = output / f"frame-{index:04}.jpg"
        with Image.open(path) as frame:
            frame.verify()
    frame = output / "frame-0000.jpg"
    (output / "result.json").write_text(json.dumps({
        "status": "capture-only", "transport": "timed out; client later absent; three JPEGs verified",
        "frame_sha256": digest(frame),
        "shader_sha256": digest(OUT / "shader-rich/Renderer/sandbox/water_surface.hlsl"),
        "hydrology_sha256": digest(OUT / "shader-rich/Renderer/native/city_fidelity/hydrology.hlsl"),
        "scene_sha256": digest(OUT / "scene.csv"),
        "client_sha256": digest(OUT / "client_x64.exe"),
        "dll_sha256": digest(OUT / "C3XReference_x64.dll"),
    }, indent=2) + "\n")
    print(frame)


def review() -> None:
    from PIL import Image, ImageChops, ImageDraw, ImageEnhance, ImageStat
    snapshot = json.loads((OUT / "snapshot.json").read_text())
    receipts = {label: json.loads((OUT / label / "result.json").read_text())
                for label in ("baseline", "candidate")}
    for label, receipt in receipts.items():
        if receipt["frame_sha256"] != digest(OUT / label / "frame-0000.jpg"):
            raise ValueError(f"{label} frame changed after rendering")
        for field, expected in (("shader_sha256", snapshot[f"{label}_water_sha256"]),
                                ("scene_sha256", snapshot["scene_sha256"]),
                                ("client_sha256", snapshot["binaries"]["client_x64.exe"]),
                                ("dll_sha256", snapshot["binaries"]["C3XReference_x64.dll"])):
            if receipt[field] != expected:
                raise ValueError(f"{label} {field} does not match the frozen baseline")
        if receipt.get("hydrology_sha256", snapshot[f"{label}_hydrology_sha256"]) != snapshot[f"{label}_hydrology_sha256"]:
            raise ValueError(f"{label} hydrology shader changed")
    before = Image.open(OUT / "baseline/frame-0000.jpg").convert("RGB")
    after = Image.open(OUT / "candidate/frame-0000.jpg").convert("RGB")
    if before.size != after.size:
        raise ValueError("Mismatched frame sizes")
    # Full context plus a near-shore crop, both at their native pixels.
    context = Image.new("RGB", (before.width * 2, before.height + 32), "#14212a")
    context.paste(before, (0, 32))
    context.paste(after, (before.width, 32))
    draw = ImageDraw.Draw(context)
    draw.text((12, 9), "Sandbox baseline", fill="white")
    draw.text((before.width + 12, 9), "Coast clarity candidate", fill="white")
    context.save(OUT / "comparison.png")
    compact = Image.new("RGB", (before.width, before.height // 2 + 32), "#14212a")
    compact.paste(before.resize((before.width // 2, before.height // 2), Image.Resampling.LANCZOS), (0, 32))
    compact.paste(after.resize((after.width // 2, after.height // 2), Image.Resampling.LANCZOS),
                  (before.width // 2, 32))
    labels = ImageDraw.Draw(compact)
    labels.text((12, 9), "Sandbox baseline", fill="white")
    labels.text((before.width // 2 + 12, 9), "Coast clarity candidate", fill="white")
    compact.save(OUT / "context-review.png")
    box = (200, 120, 1350, 1040)
    close = Image.new("RGB", ((box[2] - box[0]) * 2, box[3] - box[1] + 32), "#14212a")
    close.paste(before.crop(box), (0, 32))
    close.paste(after.crop(box), (box[2] - box[0], 32))
    pen = ImageDraw.Draw(close)
    pen.text((12, 9), "Baseline", fill="white")
    pen.text((box[2] - box[0] + 12, 9), "Candidate", fill="white")
    close.save(OUT / "shore-close.png")
    box = (600, 300, 1250, 1000)
    detail = Image.new("RGB", ((box[2] - box[0]) * 2, box[3] - box[1] + 32), "#14212a")
    detail.paste(before.crop(box), (0, 32))
    detail.paste(after.crop(box), (box[2] - box[0], 32))
    detail_labels = ImageDraw.Draw(detail)
    detail_labels.text((8, 9), "Baseline", fill="white")
    detail_labels.text((box[2] - box[0] + 8, 9), "Candidate", fill="white")
    detail.save(OUT / "bed-detail.png")
    change = ImageChops.difference(before, after)
    change.save(OUT / "difference.png")
    ImageEnhance.Brightness(change).enhance(8).save(OUT / "difference-x8.png")
    regions = {"open_ocean": (0, 100, 350, 950),
               "sea_band": (350, 100, 650, 950),
               "coast_band": (650, 100, 1000, 950)}
    stats = {name: {"mean_abs_rgb": sum(ImageStat.Stat(change.crop(box)).mean) / 3,
                    "identical": change.crop(box).getbbox() is None}
             for name, box in regions.items()}
    if not stats["open_ocean"]["identical"]:
        raise ValueError("Open-ocean control changed")
    (OUT / "review.json").write_text(json.dumps({"regions": stats,
        "difference_display_gain": 8, "capture_format": "JPEG; per-pixel statistics are diagnostic",
        "baseline_frame_sha256": receipts["baseline"]["frame_sha256"],
        "candidate_frame_sha256": receipts["candidate"]["frame_sha256"]}, indent=2) + "\n")
    print(OUT / "comparison.png")
    print(OUT / "shore-close.png")
    print(OUT / "review.json")
    print("changed bounding box:", change.getbbox())


def review_rich() -> None:
    from PIL import Image, ImageChops, ImageDraw, ImageStat
    review()
    root = OUT / "rich"
    result = json.loads((root / "result.json").read_text())
    snapshot = json.loads((OUT / "snapshot.json").read_text())
    for field, expected in (("frame_sha256", digest(root / "frame-0000.jpg")),
                            ("shader_sha256", snapshot["rich_water_sha256"]),
                            ("hydrology_sha256", snapshot["rich_hydrology_sha256"]),
                            ("scene_sha256", snapshot["scene_sha256"]),
                            ("client_sha256", snapshot["binaries"]["client_x64.exe"]),
                            ("dll_sha256", snapshot["binaries"]["C3XReference_x64.dll"])):
        if result.get(field) != expected:
            raise ValueError("Rich candidate is stale: " + field)
    labels = ("baseline", "candidate", "rich")
    images = [Image.open(OUT / label / "frame-0000.jpg").convert("RGB") for label in labels]
    if len({item.size for item in images}) != 1:
        raise ValueError("Mismatched frame sizes")
    compact = Image.new("RGB", (1920, 392), "#14212a")
    close = Image.new("RGB", (1950, 732), "#14212a")
    for index, (label, frame) in enumerate(zip(labels, images)):
        compact.paste(frame.resize((640, 360), Image.Resampling.LANCZOS), (index * 640, 32))
        close.paste(frame.crop((600, 300, 1250, 1000)), (index * 650, 32))
        ImageDraw.Draw(compact).text((index * 640 + 8, 9), label, fill="white")
        ImageDraw.Draw(close).text((index * 650 + 8, 9), label, fill="white")
    compact.save(OUT / "three-way-context.png")
    close.save(OUT / "three-way-detail.png")
    difference = ImageChops.difference(images[0], images[2])
    regions = {"open_ocean": (0, 100, 350, 950),
               "sea_band": (350, 100, 650, 950),
               "coast_band": (650, 100, 1000, 950)}
    stats = {name: {"mean_abs_rgb": sum(ImageStat.Stat(difference.crop(box)).mean) / 3,
                    "identical": difference.crop(box).getbbox() is None}
             for name, box in regions.items()}
    if not stats["open_ocean"]["identical"]:
        raise ValueError("Open-ocean control changed in rich candidate")
    (OUT / "rich-review.json").write_text(json.dumps({"regions": stats,
        "capture_format": "JPEG; per-pixel statistics are diagnostic"}, indent=2) + "\n")
    print(OUT / "three-way-context.png")
    print(OUT / "three-way-detail.png")
    print(OUT / "rich-review.json")


def review_zoom() -> None:
    """Compare actual 256-pixel tiles without shrinking the shoreline crop."""
    from PIL import Image, ImageChops, ImageDraw, ImageEnhance, ImageFont, ImageStat
    root = OUT / "z256"
    snapshot = json.loads((OUT / "snapshot.json").read_text())
    frames = {}
    for label in ("baseline", "candidate", "rich", "lagoon", "bed-only", "clearwater", "no-clutter", "scattered", "rockbeds", "continuous", "aquamarine", "aquamarine-no-margin", "aquamarine-clean-bed", "reef-field", "reef-forms", "reef-ridges", "reef-relief", "reef-contrast", "reef-lit", "reef-window", "reef-detail", "reef-composite", "reef-stone", "reef-stone-grain"):
        folder = root / label
        if label in ("lagoon", "bed-only", "clearwater", "no-clutter", "scattered", "rockbeds", "continuous", "aquamarine", "aquamarine-no-margin", "aquamarine-clean-bed", "reef-field", "reef-forms", "reef-ridges", "reef-relief", "reef-contrast", "reef-lit", "reef-window", "reef-detail", "reef-composite", "reef-stone", "reef-stone-grain") and not (folder / "result.json").is_file():
            continue
        receipt = json.loads((folder / "result.json").read_text())
        frame = folder / "frame-0000.jpg"
        expected = {
            "frame_sha256": digest(frame),
            "shader_sha256": snapshot[f"{label}_water_sha256"],
            "hydrology_sha256": snapshot[f"{label}_hydrology_sha256"],
            "scene_sha256": snapshot["scene_sha256"],
            "client_sha256": snapshot["binaries"]["client_x64.exe"],
            "dll_sha256": snapshot["binaries"]["C3XReference_x64.dll"],
            "tile_width": 256,
        }
        if any(receipt.get(key) != value for key, value in expected.items()):
            raise ValueError(f"Stale zoomed {label} capture")
        frames[label] = Image.open(frame).convert("RGB")
    box = (300, 180, 1150, 1080)
    font = ImageFont.load_default(size=23)
    strip = Image.new("RGB", (1700, 948), "#14212a")
    for column, label in enumerate(("baseline", "rich")):
        strip.paste(frames[label].crop(box), (column * 850, 48))
        ImageDraw.Draw(strip).text((column * 850 + 12, 10),
                                   f"{label.title()} · 256 px tiles · native crop",
                                   fill="white", font=font)
    strip.save(root / "shoreline-before-rich.png")
    strip.paste(frames["candidate"].crop(box), (850, 48))
    ImageDraw.Draw(strip).rectangle((850, 0, 1700, 48), fill="#14212a")
    ImageDraw.Draw(strip).text((862, 10), "Candidate · 256 px tiles · native crop",
                               fill="white", font=font)
    strip.save(root / "shoreline-before-candidate.png")
    for label in ("lagoon", "bed-only", "clearwater", "no-clutter", "scattered", "rockbeds", "continuous", "aquamarine", "aquamarine-no-margin", "aquamarine-clean-bed", "reef-field", "reef-forms", "reef-ridges", "reef-relief", "reef-contrast", "reef-lit", "reef-window", "reef-detail", "reef-composite", "reef-stone", "reef-stone-grain"):
        if label not in frames:
            continue
        strip.paste(frames[label].crop(box), (850, 48))
        ImageDraw.Draw(strip).rectangle((850, 0, 1700, 48), fill="#14212a")
        ImageDraw.Draw(strip).text((862, 10),
                                   f"{label.title()} · 256 px tiles · native crop",
                                   fill="white", font=font)
        strip.save(root / f"shoreline-before-{label}.png")
    if "aquamarine" in frames and "clearwater" in frames:
        detail_box = (530, 120, 1080, 670)
        detail = Image.new("RGB", (1100, 598), "#14212a")
        for column, label in enumerate(("clearwater", "aquamarine")):
            detail.paste(frames[label].crop(detail_box), (column * 550, 48))
            ImageDraw.Draw(detail).text((column * 550 + 12, 10),
                                        f"{label.title()} · native coast detail",
                                        fill="white", font=font)
        detail.save(root / "stamps-vs-aquamarine.png")
    if "aquamarine-no-margin" in frames:
        detail_box = (530, 120, 1080, 670)
        detail = Image.new("RGB", (1100, 598), "#14212a")
        for column, label in enumerate(("aquamarine", "aquamarine-no-margin")):
            detail.paste(frames[label].crop(detail_box), (column * 550, 48))
            ImageDraw.Draw(detail).text((column * 550 + 12, 10),
                                        f"{label.title()} · native coast detail",
                                        fill="white", font=font)
        detail.save(root / "aquamarine-vs-no-margin.png")
    if "aquamarine-clean-bed" in frames:
        detail_box = (530, 120, 1080, 670)
        detail = Image.new("RGB", (1100, 598), "#14212a")
        for column, label in enumerate(("aquamarine", "aquamarine-clean-bed")):
            detail.paste(frames[label].crop(detail_box), (column * 550, 48))
            ImageDraw.Draw(detail).text((column * 550 + 12, 10),
                                        f"{label.title()} · native coast detail",
                                        fill="white", font=font)
        detail.save(root / "aquamarine-vs-clean-bed.png")
    if "reef-field" in frames:
        detail_box = (530, 120, 1080, 670)
        detail = Image.new("RGB", (1100, 598), "#14212a")
        for column, label in enumerate(("aquamarine-clean-bed", "reef-field")):
            detail.paste(frames[label].crop(detail_box), (column * 550, 48))
            ImageDraw.Draw(detail).text((column * 550 + 12, 10),
                                        f"{label.title()} · native coast detail",
                                        fill="white", font=font)
        detail.save(root / "clean-bed-vs-reef-field.png")
    if "reef-forms" in frames:
        detail_box = (530, 120, 1080, 670)
        detail = Image.new("RGB", (1100, 598), "#14212a")
        for column, label in enumerate(("aquamarine-clean-bed", "reef-forms")):
            detail.paste(frames[label].crop(detail_box), (column * 550, 48))
            ImageDraw.Draw(detail).text((column * 550 + 12, 10),
                                        f"{label.title()} · native coast detail",
                                        fill="white", font=font)
        detail.save(root / "clean-bed-vs-reef-forms.png")
    if "reef-ridges" in frames:
        detail_box = (530, 120, 1080, 670)
        detail = Image.new("RGB", (1100, 598), "#14212a")
        for column, label in enumerate(("aquamarine-clean-bed", "reef-ridges")):
            detail.paste(frames[label].crop(detail_box), (column * 550, 48))
            ImageDraw.Draw(detail).text((column * 550 + 12, 10),
                                        f"{label.title()} · native coast detail",
                                        fill="white", font=font)
        detail.save(root / "clean-bed-vs-reef-ridges.png")
    if "reef-relief" in frames:
        detail_box = (530, 120, 1080, 670)
        detail = Image.new("RGB", (1100, 598), "#14212a")
        for column, label in enumerate(("aquamarine-clean-bed", "reef-relief")):
            detail.paste(frames[label].crop(detail_box), (column * 550, 48))
            ImageDraw.Draw(detail).text((column * 550 + 12, 10),
                                        f"{label.title()} · native coast detail",
                                        fill="white", font=font)
        detail.save(root / "clean-bed-vs-reef-relief.png")
    if "reef-contrast" in frames:
        detail_box = (530, 120, 1080, 670)
        detail = Image.new("RGB", (1100, 598), "#14212a")
        for column, label in enumerate(("aquamarine-clean-bed", "reef-contrast")):
            detail.paste(frames[label].crop(detail_box), (column * 550, 48))
            ImageDraw.Draw(detail).text((column * 550 + 12, 10),
                                        f"{label.title()} · native coast detail",
                                        fill="white", font=font)
        detail.save(root / "clean-bed-vs-reef-contrast.png")
    if "reef-lit" in frames:
        detail_box = (530, 120, 1080, 670)
        detail = Image.new("RGB", (1100, 598), "#14212a")
        for column, label in enumerate(("aquamarine-clean-bed", "reef-lit")):
            detail.paste(frames[label].crop(detail_box), (column * 550, 48))
            ImageDraw.Draw(detail).text((column * 550 + 12, 10),
                                        f"{label.title()} · native coast detail",
                                        fill="white", font=font)
        detail.save(root / "clean-bed-vs-reef-lit.png")
        rock_box = (680, 170, 1000, 500)
        rock_detail = Image.new("RGB", (1280, 708), "#14212a")
        for column, label in enumerate(("aquamarine-clean-bed", "reef-lit")):
            crop = frames[label].crop(rock_box).resize(
                (640, 660), Image.Resampling.NEAREST)
            rock_detail.paste(crop, (column * 640, 48))
            ImageDraw.Draw(rock_detail).text((column * 640 + 12, 10),
                                             f"{label.title()} · 2× display",
                                             fill="white", font=font)
        rock_detail.save(root / "reef-lit-rocks-2x.png")
    if "reef-stone" in frames:
        detail_box = (530, 120, 1080, 670)
        detail = Image.new("RGB", (1100, 598), "#14212a")
        for column, label in enumerate(("aquamarine-clean-bed", "reef-stone")):
            detail.paste(frames[label].crop(detail_box), (column * 550, 48))
            ImageDraw.Draw(detail).text((column * 550 + 12, 10),
                                        f"{label.title()} · native coast detail",
                                        fill="white", font=font)
        detail.save(root / "clean-bed-vs-reef-stone.png")
    if "reef-stone-grain" in frames:
        detail_box = (530, 120, 1080, 670)
        detail = Image.new("RGB", (1100, 598), "#14212a")
        for column, label in enumerate(("reef-stone", "reef-stone-grain")):
            detail.paste(frames[label].crop(detail_box), (column * 550, 48))
            ImageDraw.Draw(detail).text((column * 550 + 12, 10),
                                        f"{label.title()} · native coast detail",
                                        fill="white", font=font)
        detail.save(root / "reef-stone-vs-grain.png")
    if "reef-window" in frames:
        detail_box = (530, 120, 1080, 670)
        detail = Image.new("RGB", (1100, 598), "#14212a")
        for column, label in enumerate(("reef-lit", "reef-window")):
            detail.paste(frames[label].crop(detail_box), (column * 550, 48))
            ImageDraw.Draw(detail).text((column * 550 + 12, 10),
                                        f"{label.title()} · native coast detail",
                                        fill="white", font=font)
        detail.save(root / "reef-lit-vs-window.png")
    if "reef-detail" in frames:
        detail_box = (530, 120, 1080, 670)
        detail = Image.new("RGB", (1100, 598), "#14212a")
        for column, label in enumerate(("aquamarine-clean-bed", "reef-detail")):
            detail.paste(frames[label].crop(detail_box), (column * 550, 48))
            ImageDraw.Draw(detail).text((column * 550 + 12, 10),
                                        f"{label.title()} · native coast detail",
                                        fill="white", font=font)
        detail.save(root / "clean-bed-vs-reef-detail.png")
    if "reef-composite" in frames:
        detail_box = (530, 120, 1080, 670)
        detail = Image.new("RGB", (1100, 598), "#14212a")
        for column, label in enumerate(("aquamarine-clean-bed", "reef-composite")):
            detail.paste(frames[label].crop(detail_box), (column * 550, 48))
            ImageDraw.Draw(detail).text((column * 550 + 12, 10),
                                        f"{label.title()} · native coast detail",
                                        fill="white", font=font)
        detail.save(root / "clean-bed-vs-reef-composite.png")
    difference = ImageChops.difference(frames["baseline"], frames["rich"])
    gain = ImageEnhance.Contrast(difference.crop(box)).enhance(8)
    gain.save(root / "rich-difference-x8.png")
    stats = {name: {
        "mean_abs_rgb": sum(ImageStat.Stat(ImageChops.difference(
            frames["baseline"].crop(box), frames[name].crop(box))).mean) / 3,
        "open_ocean_identical": ImageChops.difference(
            frames["baseline"].crop((0, 0, 200, 400)),
            frames[name].crop((0, 0, 200, 400))).getbbox() is None,
    } for name in frames if name != "baseline"}
    (root / "review.json").write_text(json.dumps({
        "tile_width": 256, "crop_pixels": box,
        "crop_resized": False, "regions": stats,
        "difference_display_gain": 8,
    }, indent=2) + "\n")
    print(root / "shoreline-before-rich.png")
    print(root / "shoreline-before-candidate.png")
    if "lagoon" in frames:
        print(root / "shoreline-before-lagoon.png")
    if "bed-only" in frames:
        print(root / "shoreline-before-bed-only.png")
    if "clearwater" in frames:
        print(root / "shoreline-before-clearwater.png")
    if "no-clutter" in frames:
        print(root / "shoreline-before-no-clutter.png")
    if "scattered" in frames:
        print(root / "shoreline-before-scattered.png")
    if "rockbeds" in frames:
        print(root / "shoreline-before-rockbeds.png")
    if "continuous" in frames:
        print(root / "shoreline-before-continuous.png")
    if "aquamarine" in frames:
        print(root / "shoreline-before-aquamarine.png")
        print(root / "stamps-vs-aquamarine.png")
    if "aquamarine-no-margin" in frames:
        print(root / "shoreline-before-aquamarine-no-margin.png")
        print(root / "aquamarine-vs-no-margin.png")
    if "aquamarine-clean-bed" in frames:
        print(root / "shoreline-before-aquamarine-clean-bed.png")
        print(root / "aquamarine-vs-clean-bed.png")
    if "reef-field" in frames:
        print(root / "shoreline-before-reef-field.png")
        print(root / "clean-bed-vs-reef-field.png")
    if "reef-forms" in frames:
        print(root / "shoreline-before-reef-forms.png")
        print(root / "clean-bed-vs-reef-forms.png")
    if "reef-ridges" in frames:
        print(root / "shoreline-before-reef-ridges.png")
        print(root / "clean-bed-vs-reef-ridges.png")
    if "reef-relief" in frames:
        print(root / "shoreline-before-reef-relief.png")
        print(root / "clean-bed-vs-reef-relief.png")
    if "reef-contrast" in frames:
        print(root / "shoreline-before-reef-contrast.png")
        print(root / "clean-bed-vs-reef-contrast.png")
    if "reef-lit" in frames:
        print(root / "shoreline-before-reef-lit.png")
        print(root / "clean-bed-vs-reef-lit.png")
        print(root / "reef-lit-rocks-2x.png")
    if "reef-stone" in frames:
        print(root / "shoreline-before-reef-stone.png")
        print(root / "clean-bed-vs-reef-stone.png")
    if "reef-stone-grain" in frames:
        print(root / "shoreline-before-reef-stone-grain.png")
        print(root / "reef-stone-vs-grain.png")
    if "reef-window" in frames:
        print(root / "shoreline-before-reef-window.png")
        print(root / "reef-lit-vs-window.png")
    if "reef-detail" in frames:
        print(root / "shoreline-before-reef-detail.png")
        print(root / "clean-bed-vs-reef-detail.png")
    if "reef-composite" in frames:
        print(root / "shoreline-before-reef-composite.png")
        print(root / "clean-bed-vs-reef-composite.png")
    print(root / "rich-difference-x8.png")
    print(root / "review.json")


def review_shelf(candidate: str = "shelf-relief") -> None:
    """Verify frozen inputs and compare the sculpted shelf at native zoom."""
    from PIL import Image, ImageChops, ImageDraw, ImageFont
    root = OUT / "z256"
    snapshot = json.loads((OUT / "snapshot.json").read_text())
    frames = {}
    for label in ("aquamarine-clean-bed", candidate):
        folder = root / label
        receipt = json.loads((folder / "result.json").read_text())
        frame = folder / "frame-0000.jpg"
        expected = {
            "frame_sha256": digest(frame),
            "shader_sha256": snapshot[f"{label}_water_sha256"],
            "hydrology_sha256": snapshot[f"{label}_hydrology_sha256"],
            "scene_sha256": snapshot["scene_sha256"],
            "client_sha256": snapshot["binaries"]["client_x64.exe"],
            "dll_sha256": (snapshot["shelf_mesh_dll_sha256"] if label == "shelf-mesh"
                           else snapshot["binaries"]["C3XReference_x64.dll"]),
            "tile_width": 256,
        }
        if any(receipt.get(key) != value for key, value in expected.items()):
            raise ValueError(f"Stale shelf capture: {label}")
        frames[label] = Image.open(frame).convert("RGB")
    before, after = frames.values()
    if before.size != after.size:
        raise ValueError("Mismatched shelf frame sizes")
    control = (0, 0, 200, 400)
    if ImageChops.difference(before.crop(control), after.crop(control)).getbbox():
        raise ValueError("Open-ocean control changed in shelf relief")
    crop = (530, 120, 1080, 670)
    review = Image.new("RGB", (1100, 598), "#14212a")
    font = ImageFont.load_default(size=23)
    for column, (label, frame) in enumerate(frames.items()):
        review.paste(frame.crop(crop), (column * 550, 48))
        ImageDraw.Draw(review).text((column * 550 + 12, 10),
                                    f"{label.title()} · native coast detail",
                                    fill="white", font=font)
    output = root / f"clean-bed-vs-{candidate}.png"
    review.save(output)
    print(output)


def review_mosaic() -> None:
    """Verify matched control/irregular/mosaic captures at both useful zooms."""
    from PIL import Image, ImageChops, ImageDraw, ImageEnhance, ImageStat
    labels = ("shelf-mesh-control", "desert-broad-irregular", "desert-broad-mosaic")
    reports = {}
    for zoom, box, output in (
        (256, (530, 120, 1080, 670), "control-irregular-mosaic-close.png"),
        (128, (450, 250, 1050, 850), "control-irregular-mosaic-gameplay.png"),
    ):
        root = OUT / f"z{zoom}" if zoom != 128 else OUT
        receipts = {label: json.loads((root / label / "result.json").read_text())
                    for label in labels}
        for label, receipt in receipts.items():
            expected = {
                "frame_sha256": digest(root / label / "frame-0000.jpg"),
                "shader_sha256": digest(OUT / f"shader-{label}/Renderer/sandbox/water_surface.hlsl"),
                "hydrology_sha256": digest(OUT / f"shader-{label}/Renderer/native/city_fidelity/hydrology.hlsl"),
                "tile_width": zoom,
            }
            if any(receipt.get(key) != value for key, value in expected.items()):
                raise ValueError(f"Stale mosaic capture: {zoom} {label}")
        for field in ("scene_sha256", "client_sha256", "dll_sha256", "center", "size"):
            if len({json.dumps(receipt[field]) for receipt in receipts.values()}) != 1:
                raise ValueError(f"Mismatched mosaic input: {zoom} {field}")
        frames = {label: Image.open(root / label / "frame-0000.jpg").convert("RGB")
                  for label in labels}
        if len({frame.size for frame in frames.values()}) != 1:
            raise ValueError("Mismatched mosaic frame sizes")
        control = frames[labels[0]]
        ocean = (0, 50, 250, 280)
        if any(ImageChops.difference(control.crop(ocean), frames[label].crop(ocean)).getbbox()
               for label in labels[1:]):
            raise ValueError(f"Open-ocean control changed at zoom {zoom}")
        # Asset reads use the live pack checkout. Bare sand witnesses catch
        # terrain drift without depending on animated vegetation or units.
        if zoom == 128:
            for dry_land in ((1100, 1030, 1200, 1100),
                             (1200, 1050, 1300, 1130)):
                for label in labels[1:]:
                    change = ImageChops.difference(control.crop(dry_land),
                                                   frames[label].crop(dry_land))
                    if sum(ImageStat.Stat(change).mean) / 3 > .05:
                        raise ValueError(f"Bare-sand witness drifted: {label}")
        width, height = box[2] - box[0], box[3] - box[1]
        sheet = Image.new("RGB", (width * len(labels), height + 30), "white")
        pen = ImageDraw.Draw(sheet)
        for index, label in enumerate(labels):
            sheet.paste(frames[label].crop(box), (index * width, 30))
            pen.text((index * width + 10, 8), label, fill="black")
        sheet.save(root / output)
        difference = ImageChops.difference(control.crop(box), frames[labels[-1]].crop(box))
        ImageEnhance.Brightness(difference).enhance(8).save(root / "mosaic-difference-x8.png")
        reports[str(zoom)] = {
            "frames": {label: receipts[label]["frame_sha256"] for label in labels},
            "open_ocean_identical": True,
            "mosaic_vs_control_mean_abs_rgb": ImageStat.Stat(difference).mean,
            "capture_format": "JPEG; per-pixel statistics are diagnostic",
        }
        print(root / output)
    (OUT / "mosaic-review.json").write_text(json.dumps(reports, indent=2) + "\n")
    print(OUT / "mosaic-review.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("build", "build-client", "build-shelf-mesh", "build-mesh-control", "prepare", "refine", "baseline", "candidate", "rich", "lagoon", "bed-only", "clearwater", "no-clutter", "scattered", "rockbeds", "continuous", "aquamarine", "aquamarine-no-margin", "aquamarine-clean-bed", "shelf-relief", "shelf-mesh", "shelf-mesh-lit", "shelf-mesh-control", "form-probe", "bed-hidden-probe", "desert-ripple-bed", "desert-ripple-broad", "desert-direction-mix", "desert-direction-broad", "desert-direction-patches", "desert-domain-warp", "desert-shore-guided", "desert-shore-fine", "desert-broad-tuned", "desert-broad-irregular", "desert-broad-mosaic", "reef-field", "reef-forms", "reef-ridges", "reef-relief", "reef-contrast", "reef-lit", "reef-stone", "reef-stone-grain", "reef-window", "reef-detail", "reef-composite", "recover-rich", "review", "review-rich", "review-zoom", "review-shelf", "review-mesh", "review-mosaic"))
    parser.add_argument("--zoom", type=int, choices=(128, 192, 256), default=128,
                        help="Sandbox tile width for a capture (default: 128)")
    args = parser.parse_args()
    if args.action == "build":
        build()
    elif args.action == "build-client":
        build_client()
    elif args.action == "build-shelf-mesh":
        build_shelf_mesh()
    elif args.action == "build-mesh-control":
        build_mesh_control()
    elif args.action == "form-probe":
        build_form_probe()
        render(args.action, args.zoom)
    elif args.action == "bed-hidden-probe":
        build_bed_hidden_probe()
        render(args.action, args.zoom)
    elif args.action == "desert-ripple-bed":
        build_desert_ripple_bed()
        render(args.action, args.zoom)
    elif args.action == "desert-ripple-broad":
        build_desert_ripple_broad()
        render(args.action, args.zoom)
    elif args.action == "desert-direction-mix":
        build_desert_direction_mix()
        render(args.action, args.zoom)
    elif args.action == "desert-direction-broad":
        build_desert_direction_mix(broad=True)
        render(args.action, args.zoom)
    elif args.action == "desert-direction-patches":
        build_desert_direction_patches()
        render(args.action, args.zoom)
    elif args.action == "desert-domain-warp":
        build_desert_domain_warp()
        render(args.action, args.zoom)
    elif args.action == "desert-shore-guided":
        build_desert_shore_guided()
        render(args.action, args.zoom)
    elif args.action == "desert-shore-fine":
        build_desert_shore_fine()
        render(args.action, args.zoom)
    elif args.action == "desert-broad-tuned":
        build_desert_broad_tuned()
        render(args.action, args.zoom)
    elif args.action == "desert-broad-irregular":
        build_desert_broad_irregular()
        render(args.action, args.zoom)
    elif args.action == "desert-broad-mosaic":
        build_desert_broad_mosaic()
        render(args.action, args.zoom)
    elif args.action == "prepare":
        prepare()
    elif args.action == "refine":
        refine()
    elif args.action == "review":
        review()
    elif args.action == "review-rich":
        review_rich()
    elif args.action == "review-zoom":
        review_zoom()
    elif args.action == "review-shelf":
        review_shelf()
    elif args.action == "review-mesh":
        review_shelf("shelf-mesh")
    elif args.action == "review-mosaic":
        review_mosaic()
    elif args.action == "recover-rich":
        recover_rich_capture()
    else:
        render(args.action, args.zoom)

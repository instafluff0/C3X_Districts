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
    dll = r"..\lab\out\coastal-shallows\C3XReference_x64.dll"
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
        "dll_sha256": digest(OUT / "C3XReference_x64.dll"),
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
    for label in ("baseline", "candidate", "rich", "lagoon", "bed-only", "clearwater", "no-clutter", "scattered", "rockbeds", "continuous", "aquamarine", "aquamarine-no-margin", "aquamarine-clean-bed", "reef-field", "reef-forms", "reef-ridges", "reef-relief", "reef-contrast", "reef-lit", "reef-window", "reef-detail", "reef-composite"):
        folder = root / label
        if label in ("lagoon", "bed-only", "clearwater", "no-clutter", "scattered", "rockbeds", "continuous", "aquamarine", "aquamarine-no-margin", "aquamarine-clean-bed", "reef-field", "reef-forms", "reef-ridges", "reef-relief", "reef-contrast", "reef-lit", "reef-window", "reef-detail", "reef-composite") and not (folder / "result.json").is_file():
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
    for label in ("lagoon", "bed-only", "clearwater", "no-clutter", "scattered", "rockbeds", "continuous", "aquamarine", "aquamarine-no-margin", "aquamarine-clean-bed", "reef-field", "reef-forms", "reef-ridges", "reef-relief", "reef-contrast", "reef-lit", "reef-window", "reef-detail", "reef-composite"):
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("build", "build-client", "prepare", "refine", "baseline", "candidate", "rich", "lagoon", "bed-only", "clearwater", "no-clutter", "scattered", "rockbeds", "continuous", "aquamarine", "aquamarine-no-margin", "aquamarine-clean-bed", "reef-field", "reef-forms", "reef-ridges", "reef-relief", "reef-contrast", "reef-lit", "reef-window", "reef-detail", "reef-composite", "recover-rich", "review", "review-rich", "review-zoom"))
    parser.add_argument("--zoom", type=int, choices=(128, 192, 256), default=128,
                        help="Sandbox tile width for a capture (default: 128)")
    args = parser.parse_args()
    if args.action == "build":
        build()
    elif args.action == "build-client":
        build_client()
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
    elif args.action == "recover-rich":
        recover_rich_capture()
    else:
        render(args.action, args.zoom)

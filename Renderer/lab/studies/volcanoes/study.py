#!/usr/bin/env python3
"""Controlled Lab probes of hidden volcano skin; no production shader edits.

The exposure probe discards later natural surfaces in the synthetic volcano's
fixed footprint. This diagnoses ownership; it is deliberately not a runtime fix.
"""
from pathlib import Path
import argparse
import os
import shutil
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from Renderer import renderer
from Renderer.lab import platform, preparation

OUT = ROOT / 'Renderer/lab/out/volcanoes/material-study'
MIRROR = OUT / 'root'
SCENE = 'Renderer/lab/shared/shaders/lighting/generated/scene_linear_v1.hlsl'
NATURAL = ('Renderer/lab/shared/shaders/relief/beauty_terrain.hlsl',
           'Renderer/lab/shared/shaders/relief/beauty_mountain.hlsl')
OLD_NORMAL = '''            material_normal = normalize(float3(
                material_normal.x + (volcano_height.x * 2.0 - 1.0) * 0.34,
                material_normal.y + (volcano_height.y * 2.0 - 1.0) * 0.34,
                material_normal.z));'''
GRADIENT = '''            // Lab hypothesis: R is detail height; G is not a Y normal.
            float2 ux=ddx(volcano_uv),uy=ddy(volcano_uv);
            float2 step_uv=max(float2(1.0/512,1.0/512),.5*(abs(ux)+abs(uy)));
            float2 gradient_uv=float2(
                volcano_height_texture.Sample(material_sampler,volcano_uv+float2(step_uv.x,0)).r-
                volcano_height_texture.Sample(material_sampler,volcano_uv-float2(step_uv.x,0)).r,
                volcano_height_texture.Sample(material_sampler,volcano_uv+float2(0,step_uv.y)).r-
                volcano_height_texture.Sample(material_sampler,volcano_uv-float2(0,step_uv.y)).r)/(2*step_uv);
            float3 px=ddx(input.q6_world.xyz),py=ddy(input.q6_world.xyz);
            float3 rx=cross(py,geometry_normal),ry=cross(geometry_normal,px);
            float det=dot(px,rx);
            float3 delta=(dot(gradient_uv,ux)*rx+dot(gradient_uv,uy)*ry)/(abs(det)>1e-9?det:1)*.04;
            delta*=min(1.0,.6/max(length(delta),.00001));
            material_normal=normalize(geometry_normal-delta*q4_volcano_coverage(input));'''


def setup():
    OUT.mkdir(parents=True, exist_ok=True)
    for p in (ROOT / 'Renderer/native').rglob('*'):
        if not p.is_file() or p.suffix not in ('.hlsl', '.cso'):
            continue
        q = MIRROR / p.relative_to(ROOT); q.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, q)
    for name in preparation.input_paths(ROOT):
        q=MIRROR/name; q.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/name,q)
    packs=MIRROR/'Renderer/packs'
    if not packs.exists():
        # Read-only shared art; this study never writes a linked pack file.
        shutil.copytree(ROOT/'Renderer/packs',packs,copy_function=os.link)
    for p in (ROOT/'Renderer').glob('*.txt'):
        shutil.copyfile(p,MIRROR/'Renderer'/p.name)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variants',nargs='+',default=['current','exposed'],
                        choices=['current','exposed','no-bc5','height-gradient','skin-probe','skin-shadow-probe','skin-lava-shadow-probe'])
    parser.add_argument('--cases',nargs='+',default=['gameplay'],choices=['detail','active','gameplay','coastal'])
    parser.add_argument('--zoom',type=int,default=224)
    parser.add_argument('--reuse-candidate',action='store_true')
    args=parser.parse_args()
    if not args.reuse_candidate:
        renderer.require_current_candidate();renderer.ensure_preview_tool()
    setup()
    candidate=OUT/'C3XRenderer.dll'; preview=OUT/'native_preview.exe'
    if not args.reuse_candidate:
        shutil.copyfile(ROOT/'Renderer/native/build/candidate/C3XRenderer.dll',candidate)
        shutil.copyfile(renderer.LAB/'.cache/native_preview.exe',preview)
        shutil.copyfile(renderer.LAB/'.cache/native-build.json',OUT/'candidate-build.json')
    if not candidate.is_file() or not preview.is_file():
        raise ValueError('Missing frozen study candidate/preview')
    staged=ROOT/'Renderer/bin/C3XRenderer.dll'; staged_hash=renderer.checksum(staged)
    native_run=platform.run_native_fixture
    def isolated(directory,command,run_id):
        batch=directory/'render.bat'; body=batch.read_text(); needle='C3XRenderer.dll" ..\\.. '
        if body.count(needle)!=1:raise ValueError('Unexpected fixture command')
        body=body.replace(needle,'C3XRenderer.dll" ..\\lab\\out\\volcanoes\\material-study\\root ')
        batch.write_text(body)
        return native_run(directory,command,run_id)
    platform.run_native_fixture=isolated
    receipt=OUT/f'render-{args.zoom}.json'
    previous=renderer.read(receipt) if receipt.exists() else {}
    records=previous.get('renders',[]) if previous.get('candidate_sha256')==renderer.checksum(candidate) else []
    try:
        for variant in args.variants:
            for name in NATURAL:
                source=(ROOT/name).read_text()
                if variant not in ('current','skin-probe','skin-shadow-probe','skin-lava-shadow-probe'):
                    key='Output shade(P input) {'
                    if source.count(key)!=1:raise ValueError('Natural entry point changed')
                    source=source.replace(key,key+'''
    // Fixed synthetic-fixture exposure control, never a production exclusion.
    if(input.world.z>.045 && all(abs(input.world.xy-float2(16.5,.5))<.80)) discard;
''')
                if variant.startswith('skin-'):
                    source='Texture2D StudyVolcanoColor : register(t69);\nTexture2D StudyLavaColor : register(t71);\n'+source
                    normal='normal' if name.endswith('beauty_mountain.hlsl') else 'geometric'
                    marker='    float ndl = saturate(dot('+normal+', light_direction));'
                    if source.count(marker)!=1:raise ValueError('Lighting insertion changed')
                    source=source.replace(marker,'''    // Fixed source-UV routing proof, not a generic runtime material binding.
    float2 volcano_offset=input.world.xy-float2(16.5,.5);
    float volcano_coverage=smoothstep(.025,.20,input.world.z)*
        (1-smoothstep(.60,.78,max(abs(volcano_offset.x),abs(volcano_offset.y))));
    float2 volcano_uv=.5+float2(volcano_offset.x,-volcano_offset.y)*.3875;
    albedo=lerp(albedo,StudyVolcanoColor.Sample(Clamp,volcano_uv).rgb,volcano_coverage);
'''+marker)
                if variant=='skin-lava-shadow-probe':
                    key='    albedo=lerp(albedo,StudyVolcanoColor.Sample(Clamp,volcano_uv).rgb,volcano_coverage);'
                    source=source.replace(key,key+'''
    // Existing static source art only: no emission, animation or smoke.
    // Align the static patch center with the macro crater low point.
    // This offset is a measured Lab placement, not decoded source metadata.
    float4 lava=StudyLavaColor.Sample(Clamp,volcano_uv+float2(.015,-.002));
    float lava_mask=smoothstep(.16,.52,max(lava.r,max(lava.g,lava.b)))*lava.a;
    albedo=lerp(albedo,lava.rgb,lava_mask*volcano_coverage);
''')
                (MIRROR/name).write_text(source)
            source=(ROOT/SCENE).read_text()
            if variant in ('no-bc5','height-gradient'):
                if source.count(OLD_NORMAL)!=1:raise ValueError('Volcano normal expression changed')
                source=source.replace(OLD_NORMAL,'            material_normal=geometry_normal;' if variant=='no-bc5' else GRADIENT)
            (MIRROR/SCENE).write_text(source)
            caster_name='Renderer/native/render_core/source_caster.hlsl'
            caster=(ROOT/caster_name).read_text()
            if variant in ('skin-shadow-probe','skin-lava-shadow-probe'):
                # Matched fixture proof: retain the raised volcano portion of
                # the unified surface even outside the mountain-only mask.
                caster=caster.replace('float boundary:TEXCOORD4;','float boundary:TEXCOORD4;float3 world:TEXCOORD5;')
                caster=caster.replace('o.boundary=i.material;return o;', 'o.boundary=i.material;o.world=i.world.xyz;return o;')
                key='  clip(smoothstep(.08,.72,i.coverage)-.45);return i.depth;'
                if caster.count(key)!=1:raise ValueError('Mountain caster mask changed')
                caster=caster.replace(key,'''  bool volcano_body=i.world.z>.045 && all(abs(i.world.xy-float2(16.5,.5))<.80);
  if(!volcano_body)clip(smoothstep(.08,.72,i.coverage)-.45);return i.depth;''')
            (MIRROR/caster_name).write_text(caster)
            preparation.generate(MIRROR)
            shader_hashes={name:renderer.checksum(MIRROR/name) for name in (*NATURAL,SCENE,caster_name)}
            for case in args.cases:
                destination=OUT/variant/f'{case}-z{args.zoom}'
                record=renderer.native_render('volcanoes',case,12,args.zoom,destination,candidate=candidate,preview=preview)
                from PIL import Image
                Image.open(ROOT/record['image']).save(destination/'preview.png')
                record.update(variant=variant,shaders=shader_hashes)
                records=[r for r in records if (r['variant'],r['case'],r['zoom'])!=(variant,case,args.zoom)]
                records.append(record)
            renderer.write(OUT/f'render-{args.zoom}.json',dict(renders=records,
                candidate_sha256=renderer.checksum(candidate),preview_sha256=renderer.checksum(preview),
                diagnostic_only=True))
    finally:
        platform.run_native_fixture=native_run
        if renderer.checksum(staged)!=staged_hash:raise ValueError('Staged DLL changed during study')


if __name__=='__main__':
    main()

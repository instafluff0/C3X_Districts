#!/usr/bin/env python3
"""Pin the r13 selected shader bodies and build a generic local natural pack.
No image resizing, texture transcoding, or changes to the live unit/city packs.
"""
from pathlib import Path
import hashlib, json, re, struct
ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
LAB=ROOT/'Renderer/lab/shared'
PACK=ROOT/'Renderer/packs/NaturalFidelityRuntime'
LOCAL_HILL_HEIGHT=ROOT/'Renderer/packs/HillierHillsSource/height.dds'

def function(s,name):
    start=s.rfind('\n',0,s.index(name+'('))+1
    a=s.index('{',start); depth=1; b=a+1
    while depth:
        depth+=(s[b]=='{')-(s[b]=='}'); b+=1
    return s[start:b]+'\n'

def terrain_boundaries(s):
    """Native world-continuous weights; selected source responses at interiors.

    The isolated Lab provider's scalar biome selector cannot describe a live
    four-way material boundary. Keep its channels and lighting equations, but
    supply independent interpolated weights from the retained world field.
    """
    s='#define BEAUTY_COAST_CLIFF 1\n'+s
    s=s.replace('    float4 material : TEXCOORD2;',
        '    float4 material : TEXCOORD2;\n    float2 biome : TEXCOORD3;\n    float coast_coverage : TEXCOORD4;')
    s=s.replace('    float3 world : TEXCOORD0;',
        '    float4 world : TEXCOORD0;',1)
    s=s.replace('    float coast_coverage : TEXCOORD4;\n};\nstruct Output',
        '    float coast_coverage : TEXCOORD4;\n    float coast_inland : TEXCOORD5;\n};\nstruct Output')
    s=s.replace('    output.world = input.world;',
        '    output.world = input.world.xyz;')
    s=s.replace('    output.material = input.material;',
        '    output.material = input.material;\n    output.biome = input.biome;\n    output.coast_coverage = input.coast_coverage;\n    output.coast_inland = max(0, input.world.w - 1);')
    s=s.replace('    float alpha = 1;', '''    // Retain the selected beach/water composition beneath this replacement.
    // The same coverage also clips its source-shadow caster triangles.
    float alpha = saturate(input.coast_coverage+10);
    if (input.material.y < 1.5) {
        float2 edge_uv=input.world.xy*Detail.x*2.05+float2(.31,.17);
        float edge_height=GrassColor.Sample(Wrap,edge_uv).a;
        float edge_mean=GrassColor.SampleBias(Wrap,edge_uv,3).a;
        float edge_grain=saturate(.5+(edge_height-edge_mean)*3);
        alpha=lerp(coast_edge_coverage(alpha,edge_grain),alpha,input.coast_inland);
    }
    clip(alpha-.001);''')
    s=s.replace('        alpha = decal.a;', '        alpha *= decal.a;')
    s=s.replace('SamplerState Wrap : register(s0);', '''Texture2D DesertColor : register(t19);
Texture2D DesertHeight : register(t20);
Texture2D DesertSpecular : register(t21);
SamplerState Wrap : register(s0);''')
    s=s.replace('float plains_weight = smoothstep(0.18, 0.88, input.material.w);',
        '''float desert_weight = saturate(input.biome.y);
        float tundra_weight = saturate(input.material.w / max(1-desert_weight, 0.00001));
        float plains_weight = saturate(input.biome.x / max(1-desert_weight-input.material.w, 0.00001));''')
    s=s.replace('        float tundra_weight = smoothstep(1.12, 1.82, input.material.w);\n','')
    key='        float base_s = lerp(lerp(grass_s, plains_s, plains_weight), tundra_s, tundra_weight);'
    s=s.replace(key,key+'''
        base = lerp(base, DesertColor.Sample(Wrap, uv0).rgb, desert_weight);
        base_h = lerp(base_h, DesertHeight.Sample(Wrap, uv0).r, desert_weight);
        base_s = lerp(base_s, DesertSpecular.Sample(Wrap, uv0).r, desert_weight);''')
    key='''    float shadow = q6_shadow_visibility(ShadowField, input.world, geometric,
        ShadowU, ShadowV, ShadowL, ShadowFlags.x > 0.5, true);'''
    assert s.count(key)==1
    s=s.replace(key,key+'''
    // Retain full inland shadows, but let broad coastal sky fill soften the
    // receiver through the same continuous shoreline field as its geometry.
    shadow = lerp(lerp(1.0, shadow, 0.48), shadow, input.coast_inland);''')
    return s

def shaders():
    """Prepare native bindings without rebuilding the unchanged texture pack."""
    for name,category in [('terrain','relief'),('mountain','relief'),('objects','objects')]:
        p=LAB/f'shaders/{category}/beauty_{name}.hlsl'
        s=p.read_text()
        if name=='terrain':s=terrain_boundaries(s)
        # Pixel equations remain selected source text. Only native register and
        # receiver storage ABI are adapted; never overwrite the Lab provider.
        s=s.replace('Texture2D ShadowField : register(t17);','Texture2DArray ShadowField : register(t17);')
        s=s.replace('cbuffer ShadowFrame : register(b1)', 'cbuffer ShadowFrame : register(b2)')
        s=s.replace('#include "../lighting/shadow_visibility_v1.hlsl"',(HERE/'shadow_adapter.hlsl').read_text().replace(
            '// C3X_SHARED_PAGED_SHADOW',(LAB/'shaders/lighting/paged_shadow_v1.hlsl').read_text()))
        if name=='mountain':s='#define BEAUTY_TERRAIN_TRANSITIONS 1\n'+s
        s='#define BEAUTY_COMPOSED_SHADOWS 1\n'+s
        s+='''\ncbuffer NativeViewport : register(b1) {
 float2 translation; float depth_translation; float padding;
 float2 inverse_size; float2 reserved;
 float4 natural_projection; // owner column/row, tile width, target height; zero disables
};
float3 native_project_position(float3 position, float3 world) {
 if(natural_projection.z<=0) return position;
 float dx=world.x-natural_projection.x,dy=world.y-natural_projection.y;
 float h=world.z*112-2.5;
 float base=(dx-dy+1)*natural_projection.z*.25;
 return float3((dx+dy)*natural_projection.z*.5,
     base-h*(natural_projection.z/224*.82),
     base+h*.0016*natural_projection.w);
}
P VSNative(V input) {
 input.position=native_project_position(input.position,input.world.xyz);
 P o=VSMain(input);
 o.position.xy=(floor(input.position.xy*256+0.5)/256+translation)*inverse_size*float2(2,-2)+float2(-1,1);
 o.position.z=clamp(0.5-(floor(input.position.z*256+0.5)/256+translation.y)/16384.0,0.001,0.999);
 return o;
}
'''
        (HERE/f'{name}.hlsl').write_text(s)

def build_pack(output=PACK):
    """Compile local source art into a separate output directory."""
    output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    # Shader freshness is checked by the workbench preparation cache; CPU code
    # by the candidate build. This record describes only inputs to the pack.
    pins={}
    # Generic binary payload: complete source tree vertices/recipes and channel
    # identities, with only tree material/body records retained.
    source_pack=ROOT/'Renderer/packs/BeautyStudies/beauty_objects.bin'
    data=source_pack.read_bytes();pos=8
    pins[str(source_pack.relative_to(ROOT))]=hashlib.sha256(data).hexdigest()
    def unpack(fmt):
        nonlocal pos
        value=struct.unpack_from('<'+fmt,data,pos);pos+=struct.calcsize('<'+fmt);return value
    def string():
        nonlocal pos
        n,=unpack('I');v=data[pos:pos+n].decode();pos+=n;return v
    version,nm,no,nr=unpack('4I');assert version==3
    mats=[([string() for _ in range(7)],*unpack('2I')) for _ in range(nm)]
    objs=[]
    for _ in range(no):
        label=string();kind,mat,n=unpack('3I');v=data[pos:pos+n*32];pos+=n*32
        objs.append((label,kind,mat,n,v))
    recipes=[unpack('IffIIIIff') for _ in range(nr)];assert pos==len(data)
    trees=[i for i,o in enumerate(objs) if o[1]==1];assert len(trees)==22 and nr==25 and sum(r[3] for r in recipes)==180
    materials=sorted({objs[i][2] for i in trees})
    assets=[]
    def asset(path):
        p=(ROOT/path).resolve();p.relative_to(ROOT)
        if p==output.resolve() or output.resolve() in p.parents:
            raise ValueError('Natural source must not overlap generated output')
        raw=p.read_bytes();h=hashlib.sha256(raw).hexdigest();dst=output/(h+p.suffix)
        if not dst.exists():
            dst.write_bytes(raw)
        elif dst.read_bytes()!=raw:
            raise ValueError('Preserving modified natural payload: '+dst.name)
        pins[str(p.relative_to(ROOT))]=h
        if dst.name not in assets:assets.append(dst.name)
        return assets.index(dst.name)
    def metadata(path):
        p=(ROOT/path).resolve();p.relative_to(ROOT)
        raw=p.read_bytes();pins[str(p.relative_to(ROOT))]=hashlib.sha256(raw).hexdigest()
        return json.loads(raw)
    def source(path):return asset('Renderer/packs/Civ5EnvironmentSkin/'+path)
    decal_pack=ROOT/'Renderer/packs/DecalsNormalized'
    decal_manifest=metadata('Renderer/packs/DecalsNormalized/manifest.json')
    def decal_channel(asset_id,role):
        entry=decal_manifest['assets'][asset_id]
        document=metadata((decal_pack/entry['decal']).relative_to(ROOT).as_posix())
        return asset((decal_pack/document['channels'][role]['texture']).relative_to(ROOT).as_posix())
    terrain=[]
    for family in ['grassland','grasshill_top','plains','plainshill_top']:
        terrain += [source(f'textures/{family}_{c}.dds') for c in ['base_color','height','specular']]
    terrain += [asset('Renderer/packs/DecalsNormalized/textures/decals/'+p) for p in ['base_color_c996c6a9d015eebe.dds','height_31eb0f0117ea3beb.dds']]
    # A loose authored hill field may be imported locally through the generic
    # R8 adapter. Runtime payloads remain source-independent; absent that local
    # experiment, retain the normalized baseline field exactly.
    hill_height = LOCAL_HILL_HEIGHT if LOCAL_HILL_HEIGHT.is_file() else (
        ROOT/'Renderer/packs/Civ5EnvironmentSkin/textures/relief/hills/standard/height_lod0.dds')
    terrain += [asset(hill_height.relative_to(ROOT).as_posix())]
    terrain += [source(f'textures/tundra_blend_{c}.dds') for c in ['base_color','height','specular']]
    terrain += [terrain[-1]]
    terrain += [source(f'textures/desert_{c}.dds') for c in ['base_color','height','specular']]
    terrain += [decal_channel('terrain/forest/floor_01',c) for c in ['base_color','height']]
    terrain += [decal_channel('terrain/jungle/floor_01',c) for c in ['base_color','height']]
    terrain += [decal_channel('terrain/plains/decal_01',c) for c in ['base_color','height']]
    terrain += [decal_channel('terrain/desert/dune/decal_01',c) for c in ['base_color','height']]
    terrain += [source('textures/relief_surface_detail.dds')]
    assert len(terrain)==31
    surface=[]
    surface_vertices=[]
    for biome,group in [(0,'terrain/grassland/surface'),(1,'terrain/plains/surface'),
                        (2,'terrain/desert/surface'),(2,'terrain/desert/dunes')]:
        rows=decal_manifest['decal_groups'][group]['placements']
        for placement in rows:
            entry=decal_manifest['assets'][placement['asset']]
            document=metadata((decal_pack/entry['decal']).relative_to(ROOT).as_posix())
            x0,y0,x1,y1=document['footprint']['bounds_xy']
            mesh=document['mesh'];first=len(surface_vertices)
            for index in mesh['indices']:
                vertex=mesh['vertices'][index]
                surface_vertices.append((*vertex['position'],*vertex['uv0']))
            surface.append((biome,float(placement['scale']),float(placement['scale_variation']),
                int(placement['count']),float(x1-x0),float(y1-y0),first,len(mesh['indices'])))
    assert surface and all(any(row[0]==biome for row in surface) for biome in range(3))
    assert surface_vertices and len(surface_vertices)%3==0
    mountain=terrain[:3]
    for family in ['mtn_base','mtn_top','mtn_snow']:
        mountain += [source(f'textures/{family}_{c}.dds') for c in ['base_color','height','specular']]
    macro=[[source(f'textures/relief/mountains/standard/variant_{i:02d}/{c}_lod0.dds') for c in ['height','blend']] for i in range(1,6)]
    mountain += [macro[1][0]]
    bindings=[]
    for i in materials:
        paths,tint,repeat=mats[i]
        bindings.append(([asset(p) if p else 0xffffffff for p in paths],tint,repeat))
    out=bytearray(b'C3XNAT3\0')
    out+=struct.pack('<6I',len(assets),len(bindings),len(trees),nr,len(surface),len(surface_vertices))
    for path in assets:
        b=path.encode();out+=struct.pack('<I',len(b))+b
    for row in [terrain,mountain,*macro]:out+=struct.pack('<'+'I'*len(row),*row)
    for channels,tint,repeat in bindings:out+=struct.pack('<9I',*channels,tint,repeat)
    for i in trees:
        _,_,mat,n,v=objs[i];out+=struct.pack('<2I',materials.index(mat),n)+v
    for r in recipes:out+=struct.pack('<IffIIIIff',trees.index(r[0]),*r[1:])
    for r in surface:out+=struct.pack('<IffIffII',*r)
    for vertex in surface_vertices:out+=struct.pack('<4f',*vertex)
    (output/'natural.bin').write_bytes(out)
    record={'schema':1,'authority':'source-fidelity-r13/inland','source_sha256':pins,
        'hill_height_source':'local-authored-overlay' if hill_height==LOCAL_HILL_HEIGHT else 'normalized-baseline',
        'trees':22,'recipes':25,'count_weight':180,'surface_recipes':len(surface),
        'surface_weight':sum(r[3] for r in surface),'surface_triangles':len(surface_vertices)//3,
        'texture_count':len(assets),
        'pack_sha256':hashlib.sha256(out).hexdigest(),'sampling':{'samples':4,'anisotropy':16,'render_scale':2,'mip_bias':-1,'reconstruction':'one scene-linear equal-area box'}}
    return record

def main():
    shaders()
    record=build_pack()
    (HERE/'provenance.json').write_text(json.dumps(record,indent=2)+'\n')
    print(f"PASS generic natural pack: {record['texture_count']} unchanged DDS payloads, 22 bodies, 25 tree recipes, {record['surface_recipes']} surface recipes, {record['surface_triangles']} exact decal triangles")
if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--shaders-only',action='store_true')
    args=parser.parse_args()
    shaders() if args.shaders_only else main()

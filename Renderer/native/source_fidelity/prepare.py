#!/usr/bin/env python3
"""Pin the r13 selected shader bodies and build a generic local natural pack.
No image resizing, texture transcoding, or changes to the live unit/city packs.
"""
from pathlib import Path
import hashlib, json, os, re, struct
ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
LAB=ROOT/'Renderer/terrain_lab/v2'
PACK=ROOT/'Renderer/packs/NaturalFidelityRuntime'

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
    s=s.replace('    float4 material : TEXCOORD2;',
        '    float4 material : TEXCOORD2;\n    float2 biome : TEXCOORD3;\n    float coast_coverage : TEXCOORD4;')
    s=s.replace('    output.material = input.material;',
        '    output.material = input.material;\n    output.biome = input.biome;\n    output.coast_coverage = input.coast_coverage;')
    s=s.replace('    float alpha = 1;', '''    // Retain the selected beach/water composition beneath this replacement.
    // The same coverage also clips its source-shadow caster triangles.
    float alpha = saturate(input.coast_coverage+10);
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
    return s

def main():
    PACK.mkdir(exist_ok=True)
    pins={}
    for name,category in [('terrain','relief'),('mountain','relief'),('objects','objects')]:
        p=LAB/f'shaders/{category}/beauty_{name}.hlsl'
        s=p.read_text();pins[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
        if name=='terrain':s=terrain_boundaries(s)
        # Pixel equations remain selected source text. Only native register and
        # receiver storage ABI are adapted; never overwrite the Lab provider.
        s=s.replace('Texture2D ShadowField : register(t17);','Texture2DArray ShadowField : register(t17);')
        s=s.replace('cbuffer ShadowFrame : register(b1)', 'cbuffer ShadowFrame : register(b2)')
        s=s.replace('#include "../lighting/shadow_visibility_v1.hlsl"',(HERE/'shadow_adapter.hlsl').read_text())
        s='#define BEAUTY_COMPOSED_SHADOWS 1\n'+s
        s+='''\ncbuffer NativeViewport : register(b1) {
 float2 translation; float depth_translation; float padding;
 float2 inverse_size; float2 reserved;
};
P VSNative(V input) {
 P o=VSMain(input);
 o.position.xy=(floor(input.position.xy*256+0.5)/256+translation)*inverse_size*float2(2,-2)+float2(-1,1);
 o.position.z=clamp(0.5-(floor(input.position.z*256+0.5)/256+translation.y)/16384.0,0.001,0.999);
 return o;
}
'''
        (HERE/f'{name}.hlsl').write_text(s)
    s=(LAB/'systems/relief/beauty_terrain.cpp').read_text()
    kernels='// Generated selected r13 numerical kernels; see prepare.py and provenance.json.\n#pragma once\nnamespace c3x_renderer { namespace fidelity {\n'
    kernels+='struct Tile { int source_x,source_y,column,row,real; };\nusing BiqWindowTile=Tile;\n'
    kernels+=s[s.index('struct Hill {'):s.index('constexpr std::array<Hill')]
    for f in ['clamp01','smooth01','normalize3','random_u32','random01','hill_support','composed_seed','composed_hill','composed_source_macro']:
        t=function(s,f)
        if f=='composed_source_macro': t='template<class HeightField>\n'+t
        kernels+=t+'\n'
    m=(LAB/'systems/relief/beauty_mountain.cpp').read_text()
    kernels+=function(m,'mountain_seed')
    kernels+='} }\n'; (HERE/'kernels.h').write_text(kernels)
    # Generic binary payload: complete source tree vertices/recipes and channel
    # identities, with only tree material/body records retained.
    data=(ROOT/'Renderer/packs/BeautyStudies/beauty_objects.bin').read_bytes();pos=8
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
        p=ROOT/path;raw=p.read_bytes();h=hashlib.sha256(raw).hexdigest();dst=PACK/(h+p.suffix)
        if not dst.exists():
            try: os.link(p,dst)
            except OSError: dst.write_bytes(raw)
        pins[str(p.relative_to(ROOT))]=h
        if dst.name not in assets:assets.append(dst.name)
        return assets.index(dst.name)
    def source(path):return asset('Renderer/packs/Civ5EnvironmentSkin/'+path)
    terrain=[]
    for family in ['grassland','grasshill_top','plains','plainshill_top']:
        terrain += [source(f'textures/{family}_{c}.dds') for c in ['base_color','height','specular']]
    terrain += [asset('Renderer/packs/DecalsNormalized/textures/decals/'+p) for p in ['base_color_c996c6a9d015eebe.dds','height_31eb0f0117ea3beb.dds']]
    terrain += [source('textures/relief/hills/standard/height_lod0.dds')]
    terrain += [source(f'textures/tundra_blend_{c}.dds') for c in ['base_color','height','specular']]
    terrain += [terrain[-1]]
    terrain += [source(f'textures/desert_{c}.dds') for c in ['base_color','height','specular']]
    mountain=terrain[:3]
    for family in ['mtn_base','mtn_top','mtn_snow']:
        mountain += [source(f'textures/{family}_{c}.dds') for c in ['base_color','height','specular']]
    macro=[[source(f'textures/relief/mountains/standard/variant_{i:02d}/{c}_lod0.dds') for c in ['height','blend']] for i in range(1,6)]
    mountain += [macro[1][0]]
    bindings=[]
    for i in materials:
        paths,tint,repeat=mats[i]
        bindings.append(([asset(p) if p else 0xffffffff for p in paths],tint,repeat))
    out=bytearray(b'C3XNAT1\0')
    out+=struct.pack('<4I',len(assets),len(bindings),len(trees),nr)
    for path in assets:
        b=path.encode();out+=struct.pack('<I',len(b))+b
    for row in [terrain,mountain,*macro]:out+=struct.pack('<'+'I'*len(row),*row)
    for channels,tint,repeat in bindings:out+=struct.pack('<9I',*channels,tint,repeat)
    for i in trees:
        _,_,mat,n,v=objs[i];out+=struct.pack('<2I',materials.index(mat),n)+v
    for r in recipes:out+=struct.pack('<IffIIIIff',trees.index(r[0]),*r[1:])
    (PACK/'natural.bin').write_bytes(out)
    for p in [LAB/'systems/relief/beauty_terrain.cpp',LAB/'systems/relief/beauty_mountain.cpp',LAB/'systems/objects/beauty_objects.cpp']:
        pins[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    record={'schema':1,'authority':'source-fidelity-r13/inland','source_sha256':pins,'trees':22,'recipes':25,'count_weight':180,'texture_count':len(assets),'pack_sha256':hashlib.sha256(out).hexdigest(),'sampling':{'samples':4,'anisotropy':16,'render_scale':2,'mip_bias':-1,'reconstruction':'one scene-linear equal-area box'}}
    (HERE/'provenance.json').write_text(json.dumps(record,indent=2)+'\n')
    print(f'PASS generic natural pack: {len(assets)} unchanged DDS payloads, 22 bodies, 25 recipes, weight 180')
if __name__=='__main__':main()

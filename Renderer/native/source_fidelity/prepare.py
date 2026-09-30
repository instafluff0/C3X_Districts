#!/usr/bin/env python3
"""Pin the r13 selected shader bodies and build a generic local natural pack.
No image resizing, texture transcoding, or changes to the live unit/city packs.
"""
from pathlib import Path
import hashlib, json, math, re, struct
ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
LAB=ROOT/'Renderer/lab/shared'
PACK=ROOT/'Renderer/packs/NaturalFidelityRuntime'
TERRAIN_PACK=ROOT/'Renderer/packs/Civ5EnvironmentSkin'
HILL_HEIGHT=TERRAIN_PACK/'textures/relief/hills/standard/height_lod0.dds'
TREE_HEIGHT_SCALE=.50
JUNGLE_HEIGHT_SCALE=.50

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
    s=s.replace('    float coast_coverage : TEXCOORD4;\n};\n#ifdef SANDBOX_TERRAIN_MATERIAL',
        '    float coast_coverage : TEXCOORD4;\n    float coast_inland : TEXCOORD5;\n};\n#ifdef SANDBOX_TERRAIN_MATERIAL')
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
    biome_detail='float grass_plains_detail = saturate(1 - tundra_weight)'
    if biome_detail not in s:
        raise ValueError('Grass/plains detail must be masked from native desert material')
    s=s.replace(biome_detail,
                'float grass_plains_detail = saturate(1 - tundra_weight - desert_weight)')
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
        if name=='terrain':s='#define BEAUTY_FLOODPLAIN_DECAL 1\n'+terrain_boundaries(s)
        if name in ('terrain','mountain'):
            s='#define BEAUTY_VOLCANO_MATERIAL 1\n'+s
            s=s.replace('#include "volcano_material.hlsl"',(LAB/'shaders/relief/volcano_material.hlsl').read_text())
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
 o.position.z=clamp(0.5-(floor(input.position.z*256+0.5)/256+depth_translation)/16384.0,0.001,0.999);
 return o;
}
'''
        if name=='objects':
            instance=(LAB/'shaders/objects/instance_geometry.hlsl').read_text()
            s+='\n'+instance
            (HERE/'instance_caster.hlsl').write_text('#define C3X_INSTANCE_CASTER 1\n'+instance)
        (HERE/f'{name}.hlsl').write_text(s)

def build_pack(output=PACK, terrain_pack=None):
    """Compile local source art into a separate output directory."""
    output=Path(output)
    terrain_pack=Path(terrain_pack) if terrain_pack is not None else TERRAIN_PACK
    terrain_pack=terrain_pack.resolve()
    terrain_pack.relative_to(ROOT)
    output.mkdir(parents=True,exist_ok=True)
    # Shader freshness is checked by the workbench preparation cache; CPU code
    # by the candidate build. This record describes only inputs to the pack.
    pins={}
    # Generic binary payload: selected source tree vertices/recipes and channel
    # identities, with a Lab vertical proportion adjustment to the tree bodies.
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
    # The older beauty study contains the forest only. Admit all ten desktop
    # jungle models from the normalized local import with their complete
    # source material stack, retaining the authored placement weights.
    vegetation='Renderer/packs/VegetationNormalized/'
    vegetation_manifest=metadata(vegetation+'manifest.json')
    jungle_placements=vegetation_manifest['features']['jungle']['placements']
    jungle_start=len(trees)
    for placement in jungle_placements:
        asset_id=placement['asset']
        entry=vegetation_manifest['assets'][asset_id]
        mesh=metadata(vegetation+entry['mesh'])
        material=metadata(vegetation+entry['material'])
        if material['alpha_mode']!='opaque' or len(mesh['topology']['indices'])%3:
            raise ValueError('Unsupported jungle model: '+asset_id)
        channels=[material['base_color']['texture'],material['lean_normal']['texture_0'],
                  material['lean_normal']['texture_1'],'',material['gloss']['texture'],'','']
        mat_index=len(mats)
        mats.append(([vegetation+path if path else '' for path in channels],0,0))
        body=bytearray()
        for index in mesh['topology']['indices']:
            vertex=mesh['vertices'][index]
            body+=struct.pack('<8f',*(vertex['position']+vertex['normal']+vertex['uv0']))
        object_index=len(objs)
        objs.append((asset_id,1,mat_index,len(mesh['topology']['indices']),bytes(body)))
        trees.append(object_index)
        flags=(1 if placement['allow_overlap'] else 0)|(2 if placement['show_decal'] else 0)
        recipes.append((object_index,float(placement['scale']),float(placement['scale_variation']),
                        int(placement['count']),int(placement['min_count']),int(placement['priority']),
                        flags,float(placement['width']),float(placement['low_end_reduction'])))
    if len(trees)!=32 or len(recipes)!=35 or sum(r[3] for r in recipes[25:])!=121:
        raise ValueError('Jungle source recipe count changed')
    materials=sorted({objs[i][2] for i in trees})
    def source(path):return asset((terrain_pack/path).relative_to(ROOT).as_posix())
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
    # Use this pack's standard normalized hill field for the Lab and runtime.
    # Loose authored imports remain available for isolated studies.
    terrain += [source('textures/relief/hills/standard/height_lod0.dds')]
    terrain += [source(f'textures/tundra_blend_{c}.dds') for c in ['base_color','height','specular']]
    terrain += [terrain[-1]]
    terrain += [source(f'textures/desert_{c}.dds') for c in ['base_color','height','specular']]
    terrain += [decal_channel('terrain/forest/floor_01',c) for c in ['base_color','height']]
    terrain += [decal_channel('terrain/jungle/floor_01',c) for c in ['base_color','height']]
    terrain += [decal_channel('terrain/plains/decal_01',c) for c in ['base_color','height']]
    terrain += [decal_channel('terrain/desert/dune/decal_01',c) for c in ['base_color','height']]
    terrain += [source('textures/relief_surface_detail.dds')]
    assert len(terrain)==31
    flood_pack=ROOT/'Renderer/packs/FloodplainNormalized'
    flood_manifest=metadata('Renderer/packs/FloodplainNormalized/manifest.json')
    flood_group=flood_manifest['decal_groups']['terrain/floodplain/grassland_surface']
    def flood_channel(role):
        entry=flood_manifest['assets'][flood_group['placements'][0]['asset']]
        document=metadata((flood_pack/entry['decal']).relative_to(ROOT).as_posix())
        return asset((flood_pack/document['channels'][role]['texture']).relative_to(ROOT).as_posix())
    floodplain=[flood_channel(role) for role in ('base_color','height','specular')]
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
    for placement in flood_group['placements']:
        entry=flood_manifest['assets'][placement['asset']]
        document=metadata((flood_pack/entry['decal']).relative_to(ROOT).as_posix())
        for role,expected in zip(('base_color','height','specular'),floodplain):
            channel=asset((flood_pack/document['channels'][role]['texture']).relative_to(ROOT).as_posix())
            if channel!=expected:raise ValueError('Floodplain decal material channels differ')
        x0,y0,x1,y1=document['footprint']['bounds_xy']
        mesh=document['mesh'];first=len(surface_vertices)
        for index in mesh['indices']:
            vertex=mesh['vertices'][index]
            surface_vertices.append((*vertex['position'],*vertex['uv0']))
        surface.append((3,float(placement['scale']),float(placement['scale_variation']),
            int(placement['count']),float(x1-x0),float(y1-y0),first,len(mesh['indices'])))
    assert surface and all(any(row[0]==biome for row in surface) for biome in range(4))
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
    out=bytearray(b'C3XNAT4\0')
    out+=struct.pack('<6I',len(assets),len(bindings),len(trees),len(recipes),len(surface),len(surface_vertices))
    for path in assets:
        b=path.encode();out+=struct.pack('<I',len(b))+b
    for row in [terrain,floodplain,mountain,*macro]:out+=struct.pack('<'+'I'*len(row),*row)
    for channels,tint,repeat in bindings:out+=struct.pack('<9I',*channels,tint,repeat)
    for i in trees:
        _,_,mat,n,v=objs[i]
        height_scale=TREE_HEIGHT_SCALE if i<jungle_start else JUNGLE_HEIGHT_SCALE
        body=bytearray(v)
        for vertex in range(n):
            at=vertex*32
            z,=struct.unpack_from('<f',body,at+8)
            nx,ny,nz=struct.unpack_from('<3f',body,at+12)
            nz/=height_scale
            length=math.sqrt(nx*nx+ny*ny+nz*nz)
            if not math.isfinite(length) or length<=0:
                raise ValueError('Invalid source tree normal')
            struct.pack_into('<f',body,at+8,z*height_scale)
            struct.pack_into('<3f',body,at+12,nx/length,ny/length,nz/length)
        out+=struct.pack('<2I',materials.index(mat),n)+body
    for r in recipes:out+=struct.pack('<IffIIIIff',trees.index(r[0]),*r[1:])
    for r in surface:out+=struct.pack('<IffIffII',*r)
    for vertex in surface_vertices:out+=struct.pack('<4f',*vertex)
    (output/'natural.bin').write_bytes(out)
    record={'schema':1,'authority':'source-fidelity-r13/inland','source_sha256':pins,
        'hill_height_source':'normalized-baseline',
        'trees':32,'recipes':35,'count_weight':301,'tree_height_scale':TREE_HEIGHT_SCALE,
        'jungle_bodies':10,'jungle_height_scale':JUNGLE_HEIGHT_SCALE,
        'surface_recipes':len(surface),
        'surface_weight':sum(r[3] for r in surface),'surface_triangles':len(surface_vertices)//3,
        'texture_count':len(assets),
        'pack_sha256':hashlib.sha256(out).hexdigest(),'sampling':{'samples':4,'anisotropy':16,'render_scale':2,'mip_bias':-1,'reconstruction':'one scene-linear equal-area box'}}
    return record

def main():
    shaders()
    record=build_pack()
    (HERE/'provenance.json').write_text(json.dumps(record,indent=2)+'\n')
    print(f"PASS generic natural pack: {record['texture_count']} unchanged DDS payloads, 32 bodies, 35 tree recipes, {record['surface_recipes']} surface recipes, {record['surface_triangles']} exact decal triangles")
if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--shaders-only',action='store_true')
    args=parser.parse_args()
    shaders() if args.shaders_only else main()

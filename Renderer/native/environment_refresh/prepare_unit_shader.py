"""Compile the selected BeautyStudies unit material closure for native sprites.

Keep the source GGX, sky fill, AO, gloss, emissive and rim equations verbatim.
Adapters supply native owner color, pose-local visibility and authoritative time.
The unresolved LEAN decode is deliberately absent from the unit entry point.
"""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
SOURCE=ROOT/'Renderer/lab/shared/shaders/objects/beauty_objects.hlsl'
def main():
    source=SOURCE.read_text()
    ggx=source[source.index('float ggx('):source.index('Output shade(')]
    response=source[source.index('    float sky = saturate(normal.z'):source.index('    output.color = float4(max(radiance, 0), 1);')]
    response=response.replace('Emissive.Sample(Clamp, input.uv).rgb','emission')
    # kind=2 is the generic non-foliage object material, shared by all units.
    body='''
Texture2D<float4> base : register(t0);
Texture2D<float> shadow_map : register(t1);
Texture2D<float4> ambient_occlusion : register(t2);
Texture2D<float4> gloss_texture : register(t3);
Texture2D<float4> emissive_texture : register(t4);
SamplerState sample_base : register(s0);SamplerState sample_emission : register(s1);
cbuffer Material : register(b0) {float4 tint,owner,sun,sun_color,moon,moon_color,ambient,channels;};
cbuffer BeautyFrame : register(b1) {float4 Sun,SunColorExposure,Ambient,View,Quality;};
struct Input {float3 p:POSITION;float3 n:NORMAL;float2 uv:TEXCOORD0;float3 shadow:TEXCOORD1;};
struct Output {float4 p:SV_Position;float3 n:NORMAL;float2 uv:TEXCOORD0;float3 shadow:TEXCOORD1;};
Output VS(Input i){Output o;o.p=float4(i.p,1);o.n=i.n;o.uv=i.uv;o.shadow=i.shadow;return o;}
'''+ggx+'''
float3 beauty_unit_response(float3 albedo,float3 normal,float ao,float gloss,float3 emission,float shadow) {
 float kind=2;float3 light_direction=Sun.xyz;float diffuse=saturate(dot(normal,light_direction));
 // Preserve the source's optional-emission gate without role-based dispatch.
 struct Metadata {float2 secondary;};Metadata input;input.secondary=float2(0,channels.z);
'''+response+'''
 return max(radiance,0);
}
float4 PS(Output i):SV_Target {
 clip(i.shadow.x);
 float4 b=base.Sample(sample_base,i.uv);if(ambient.w>.5)clip(b.a-.5);
 float3 albedo=b.rgb*tint.rgb;
 float mask=tint.w<.5?0:(tint.w<1.5?smoothstep(.06,.94,1-b.a):1);
 // Selected source modulation with Civ III's authoritative display color.
 // Multiplication retains atlas detail instead of replacing it by a luma ramp.
 albedo=lerp(albedo,albedo*(.45+owner.rgb*1.10),mask*owner.w);
 float3 n=normalize(i.n);int2 cell=int2(floor(i.shadow.yz*128));float occluded=0;
 [unroll]for(int oy=-1;oy<=1;++oy)[unroll]for(int ox=-1;ox<=1;++ox) {
  int2 q=cell+int2(ox,oy);
  if(all(q>=0) && all(q<128))occluded+=(shadow_map.Load(int3(q,0))>i.shadow.x+.006)?1.0/9:0;
 }
 float ao=channels.x>.5?lerp(.48,1,ambient_occlusion.Sample(sample_base,i.uv).r):1;
 float gloss=channels.y>.5?gloss_texture.Sample(sample_base,i.uv).r:.08;
 float3 emission=channels.z>.5?emissive_texture.Sample(sample_emission,i.uv).rgb:0;
 return float4(beauty_unit_response(albedo,n,ao,gloss,emission,1-occluded),1);
}
'''
    target=HERE/'unit_shader.h';target.write_text('#pragma once\nnamespace c3x_renderer {\ninline char const* unit_material_shader(){return R"C3XUNIT('+body+')C3XUNIT";}\n}\n')
    (HERE/'unit-shader-provenance.json').write_text(json.dumps({'authority':str(SOURCE.relative_to(ROOT)),
      'source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),'generated_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
      'ported':['GGX','sky-weighted ambient','source AO response','source gloss response','emissive response','rim fill','multiplicative owner-color response'],
      'adapters':['authoritative native environment and display color','existing pose-local self-shadow visibility','native sprite projection and clipping'],
      'lean_decode':False},indent=2)+'\n')
if __name__=='__main__':main()

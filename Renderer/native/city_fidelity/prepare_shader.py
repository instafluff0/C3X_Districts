"""Freeze the complete selected city shader with production input bindings."""
from pathlib import Path
import hashlib,json
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2];LAB=ROOT/'Renderer/lab/shared'

def main():
    pins={}
    def read(p):
        pins[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
        return p.read_text()
    source=read(HERE.parent/'environment_refresh/feature.hlsl')
    field='''
// Eight independently bounded city fields fit within one D3D11 constant
// buffer. The CPU selects intersecting cities per guarded block; overflow
// fails the candidate draw instead of truncating a city's emitting facades.
cbuffer NativeCityLights : register(b6) {
 float4 CityLightCounts;
 float4 Q8LocalEnvelopeLow4;float4 Q8LocalEnvelopeHigh4;
 float4 Q8LocalPositionRange[1024];float4 Q8LocalColorIntensity[1024];
 float4 Q8LocalDirectionOwner[1024];float4 Q8LocalBoxLow[256];float4 Q8LocalBoxHigh[256];
};
#define Q8_LOCAL_LIGHT_COUNT int(CityLightCounts.x)
#define Q8_LOCAL_BLOCKER_COUNT int(CityLightCounts.y)
#define Q8LocalEnvelopeLow Q8LocalEnvelopeLow4.xyz
#define Q8LocalEnvelopeHigh Q8LocalEnvelopeHigh4.xyz
#define Q8_LOCAL_Z_METRIC 0.648266978876
#define Q8_LOCAL_LIGHT_GAIN 4
'''+read(LAB/'shaders/lighting/local_facade_lights.hlsl').replace('environment_night_activation','CityLightCounts.z').replace('environment_emissive_scale','CityLightCounts.w')+'\n'
    marker='float q6_receiver_visibility(PixelInput input,float3 normal,float legacy_shadow) {'
    assert source.count(marker)==1
    source=source.replace(marker,field+marker)
    start=source.index('float3 q6_receiver_illumination(FeaturePixelInput input,')
    end=source.index('\n#ifdef Q4_COASTAL_ROCKS',start)
    source=source[:start]+'''float3 q6_receiver_illumination(FeaturePixelInput input,float3 normal,
 float legacy_shadow,float ambient_visibility) {
 float visibility=legacy_shadow;
 if(input.q6_world.w>.5 && Q6ShadowFlags.x>.5)
  visibility=q6_world_visibility(pickup_shadow_feature,input.q6_world,normal,false);
 return frame_illumination(normal,visibility,ambient_visibility);
}
'''+source[end:]
    source=source.replace('return frame_illumination(normal,','return q8_local_irradiance(input.q6_world,normal,ambient_visibility)+frame_illumination(normal,')
    marker='struct FeaturePixelInput\n{'
    assert source.count(marker)==1
    source=source.replace(marker,marker+'''
    float2 city_ao_uv : TEXCOORD3;
    float3 city_tangent : TEXCOORD4;
    float3 city_bitangent : TEXCOORD5;
    float2 city_emissive_uv : TEXCOORD6;
''').replace('FeaturePixelInput output;','FeaturePixelInput output=(FeaturePixelInput)0;')
    source=source.replace('Q6SceneOutput PSFeature(', 'Q6SceneOutput Q8LegacyPSFeature(')
    source=source.replace('return PSFeature(input).color;', 'return Q8LegacyPSFeature(input).color;')
    material=read(LAB/'shaders/objects/city_scene_material.hlsl')
    material=material.replace('q6_world_visibility(shallow_bed_texture,','q6_world_visibility(pickup_shadow_feature,')
    environment=read(LAB/'shaders/lighting/city_environment.hlsl')
    marker='float4 ground=city_base_texture_0.Sample(decal_sampler,p.uv);'
    assert material.count(marker)==1
    material=material.replace(marker,'float4 ground=p.material_index>61.5?q8_settlement_ground_sample(p):city_base_texture_0.Sample(decal_sampler,p.uv);')
    marker='  lit+=visibility*('
    assert material.count(marker)==1
    material=material.replace(marker,'  if(CityMaterialFlags.x>.5)lit+=q8_city_environment_specular(n,roughness,base,metalness,ao);\n'+marker)
    # A selected dielectric material must not acquire the unselected modern
    # environment experiment merely because it binds a metalness channel.
    material=material.replace('if((channels&16)!=0)metalness=', 'if(CityMaterialFlags.x>.5 && (channels&16)!=0)metalness=')
    source+='''
cbuffer CityMaterialFrame : register(b7) { float4 CityMaterialFlags; float4 CityAtlas; };
#define Q8_CITY_FEATURE_ENTRY PSNativeCityBody
#define Q8_CITY_AUXILIARY_AO 1
#define Q8_CITY_AO_STRENGTH 1
#define Q8_CITY_EXTRA_MATERIALS 1
#define Q8_CITY_SOURCE_SURFACE 1
#define Q8_CITY_SOURCE_SPECULAR 1
#define Q8_CITY_VIEW_DIRECTION normalize(float3(1,1,0.790569494147))
#define Q8_CITY_CHANNELS 0
#define Q8_CITY_SURFACE_DETAIL 0
#define Q8_CITY_WORLD_Z_TO_SOURCE 0.648266978876
#define Q8_CITY_SEPARATE_EMISSION 1
#define Q8_CITY_EMISSIVE_GAIN 8
#define Q8_SETTLEMENT_GAIN 1
#define Q8_SETTLEMENT_ATLAS CityAtlas
'''+environment+'\n'+read(LAB/'shaders/objects/settlement_ground.hlsl')+'\n'+material+'''
// The cached 168-byte native vertex retains every source channel. This is
// deliberately distinct from the 48-byte legacy feature cache layout.
struct NativeCityInput {
 float3 position:POSITION;float2 uv:TEXCOORD0;float3 normal:NORMAL;
 float2 ao:TEXCOORD1;float material:TEXCOORD2;
 float3 tangent:TEXCOORD3;float3 bitangent:TEXCOORD4;
 float3 world:TEXCOORD5;float2 emission:TEXCOORD6;
};
FeaturePixelInput VSNativeCity(NativeCityInput p) {
 IntegratedVertexInput i=(IntegratedVertexInput)0;
 i.position=p.position;
 FeaturePixelInput o=(FeaturePixelInput)0;
 o.position=float4(translated_position(i),translated_depth(i,true),1);
 o.uv=p.uv;o.geometry_normal=p.normal;o.material_index=p.material;
 o.city_ao_uv=p.ao;o.city_tangent=p.tangent;o.city_bitangent=p.bitangent;
 o.city_emissive_uv=p.emission;o.q6_world=float4(p.world,1);return o;
}
FeaturePixelInput VSNativeCityReflection(NativeCityInput p) {
 FeaturePixelInput o=VSNativeCity(p);
 float h=max(0,p.world.z-NativeReflection.z);
 o.position.y-=h*NativeReflection.x*4*c3x_inverse_viewport_size.y;
 float base=p.position.y+h*NativeReflection.x;
 o.position.z=clamp(.5-(floor((base-h*NativeReflection.y)*256+.5)/256+c3x_viewport_translation.y)/16384,.001,.999);
 return o;
}
float4 PSNativeCity(FeaturePixelInput p):SV_Target { return PSNativeCityBody(p).color; }
float4 PSNativeCityEmission(FeaturePixelInput p):SV_Target {
 clip(p.material_index-99.5);p.material_index+=100;return PSNativeCityBody(p).color;
}
float4 PSNativeCityReflection(FeaturePixelInput p):SV_Target {
 clip(p.q6_world.z-NativeReflection.z-.0001);return PSNativeCityBody(p).color;
}
float4 PSNativeCityReflectionEmission(FeaturePixelInput p):SV_Target {
 clip(p.q6_world.z-NativeReflection.z-.0001);
 clip(p.material_index-99.5);p.material_index+=100;return PSNativeCityBody(p).color;
}
'''
    (HERE/'city.hlsl').write_text(source)
    (HERE/'feature.hlsl').write_text(source)
    hydro=read(HERE.parent/'environment_refresh/hydrology.hlsl')
    marker='float q6_receiver_visibility(PixelInput input,float3 normal,float legacy_shadow) {'
    assert hydro.count(marker)==1
    hydro=hydro.replace(marker,field+marker)
    hydro=hydro.replace('return frame_illumination(normal,','return q8_local_irradiance(input.q6_world,normal,ambient_visibility)+frame_illumination(normal,')
    (HERE/'hydrology.hlsl').write_text(hydro)
    for name in ['terrain','mountain','objects']:
        natural=read(HERE.parent/'environment_refresh'/f'{name}.hlsl')
        marker='Output shade(P input) {'
        assert natural.count(marker)==1
        natural=natural.replace(marker,field+marker)
        normal='geometric' if name=='terrain' else 'normal'
        marker='    output.color = float4(max(radiance, 0)'
        assert natural.count(marker)==1
        natural=natural.replace(marker,f'    radiance+=albedo*q8_local_irradiance(float4(input.world,1),normalize({normal}*float3(1,-1,1/Q8_LOCAL_Z_METRIC)),1);\n'+marker)
        (HERE/f'{name}.hlsl').write_text(natural)
    # Retain the exact selected postprocess separately until the guarded block
    # adapter has passed comparison. It must not be replaced by a blur filter.
    glow=read(LAB/'shaders/common/hdr_glow_tiled.hlsl')
    glow=glow.replace('#include "../sampling/reconstruction_v1.hlsl"',read(LAB/'shaders/sampling/reconstruction_v1.hlsl'))
    # Native captures validity in premultiplied alpha rather than a second
    # render target. Reconstruction/highlight acceptance use that same channel.
    glow=glow.replace('Texture2D<float> map_validity','Texture2D<float4> map_validity').replace('map_validity.Load(int3(at,0))','map_validity.Load(int3(at,0)).a')
    glow=glow.replace('Texture2D<float> Validity','Texture2D<float4> Validity').replace('Validity.Load(int3(p,0))','Validity.Load(int3(p,0)).a')
    glow=glow.replace('int4 valid_rect;','int4 valid_rect;float4 NativeGlow;').replace('#define Q8_GLOW_GAIN 6.0','#define Q8_GLOW_GAIN NativeGlow.x')
    glow=glow.replace('#define Q8_OUTPUT_ORIGIN uint2(0,0)','#define Q8_OUTPUT_ORIGIN uint2(NativeGlow.yz)')
    (HERE/'hdr_glow.hlsl').write_text(glow)
    (HERE/'local_lights.hlsl').write_text(field)
    caster=read(HERE.parent/'render_core/source_caster.hlsl')
    caster=caster.replace('float PSCutout(Pixel i):SV_TARGET {','''Texture2D city_opacity:register(t34);
float PSCutout(Pixel i):SV_TARGET {
 if(i.material>=99.5) {
  uint w,h;city_opacity.GetDimensions(w,h);bool repeat=(int(round(i.material-100))&4)!=0;
  float2 uv=repeat?frac(i.uv):saturate(i.uv);
  clip(city_opacity.Load(int3(min(int2(uv*float2(w,h)),int2(w,h)-1),0)).a-.5);
  return i.depth;
 }
''')
    (HERE/'source_caster.hlsl').write_text(caster)

    (HERE/'shader-provenance.json').write_text(json.dumps({'authority':['r111-inland','r112-freshcanopy'],
        'source_sha256':pins,'bindings':'Native full vertex, shared shadow atlas, explicit material/environment and atlas constants',
        'runtime':['bounded per-block facade light selection','guarded linear reconstruction and HDR glow'],
        'pending':['user-run Civ III visual checkpoint','remaining palace styles and constrained-site coverage']},indent=2)+'\n')
if __name__=='__main__':main()

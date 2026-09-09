"""Native bindings for the selected static water and planar reflection paths."""
from pathlib import Path
import hashlib,json
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2];BASE=HERE.parent/'source_fidelity';LAB=ROOT/'Renderer/lab/shared'
FRAME='''
cbuffer NativeReflectionFrame : register(b5) {
 float4 NativeReflection; // world-height to native pixels, depth metric, plane Z, enabled
 float4 NativeReflectionTarget; // internal extent XY, sampling guard XY
};
'''
def main():
    pins={}
    def read(p):
        pins[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest();return p.read_text()
    water=read(LAB/'shaders/hydrology/water_natural.hlsl')
    water=water.replace('Q3_MATERIAL_WRAP_WIDTH*.5*q3_source_repeat(.6)','c3x_world_material.x*q3_source_repeat(.6)')
    water=water.replace('input.position.xy/Q3_REFLECTION_SIZE','(input.position.xy+NativeReflectionTarget.zw)/Q3_REFLECTION_SIZE')
    water=water.replace('float object_coverage=saturate(object.a)*inside;', 'inside*=NativeReflection.w;\n float object_coverage=saturate(object.a)*inside;')
    # The original source body is retained; only the native wrap period,
    # guarded block coordinates and explicit sky-only diagnostic are adapted.
    hydro='#define Q3_CONTINUOUS_RIVERS 1\n'+read(BASE/'hydrology.hlsl')
    marker='float4 q3_water_material(PixelInput input) {'
    assert marker in hydro
    hydro=hydro.replace(marker,FRAME+'\n#define Q3_NATURAL_WATER 1\n#define Q3_OBJECT_REFLECTION 1\n#define Q3_REFLECTION_SIZE NativeReflectionTarget.xy\n'+water+'\n'+marker)
    terrain_reflect='''
PixelInput VSReflection(IntegratedVertexInput input) {
 PixelInput o=VSIntegrated(input);
 float h=max(0,input.q6_world.z-NativeReflection.z);
 o.position.y-=h*NativeReflection.x*4*c3x_inverse_viewport_size.y;
 float base=input.position.y+h*NativeReflection.x;
 o.position.z=clamp(.5-(floor((base-h*NativeReflection.y)*256+.5)/256+c3x_viewport_translation.y)/16384,.001,.999);
 return o;
}
float4 PSReflection(PixelInput input):SV_Target {
 clip(input.q6_world.w-.5);clip(input.q6_world.z-NativeReflection.z-.0001);
 clip(input.hydrology_data.x);clip(3.5-input.surface_kind);
 return PSIntegrated(input);
}
'''
    (HERE/'hydrology.hlsl').write_text(hydro+terrain_reflect)
    feature=read(HERE.parent/'render_core/terrain_scene.hlsl')
    feature=read(LAB/'shaders/lighting/shadow_policy.hlsl')+'\n'+feature
    old='float alpha = frame_cast_shadow_strength() * 0.22 *'
    assert feature.count(old)==1
    # Native unit footprints use the captured strength directly. The legacy
    # feature floor would otherwise make resource shadows stronger at night.
    feature=feature.replace(old,'float alpha = environment_shadow_strength * c3x_dynamic_shadow_opacity *')
    feature=feature.replace('tile_object_weight * 0.07);','tile_object_weight * 0.07 * (1.0-step(0.175, material_fraction)));')
    # The retained feature provider predates the common paged receiver. Bind
    # the current filter here as well as in the natural shader generator.
    receiver=read(HERE.parent/'render_core/shadow_receiver.hlsl').replace(
        '// C3X_SHARED_PAGED_SHADOW',read(LAB/'shaders/lighting/paged_shadow_v1.hlsl'))
    start=feature.index('Texture2DArray pickup_shadow_terrain')
    end=feature.index('\n#endif',start)
    feature=feature[:start]+receiver+feature[end:]
    # This retained provider predates the shared cliff material. Bind that one
    # current body explicitly, without changing unrelated feature shaders.
    cliff=read(LAB/'shaders/relief/coast_rocks.hlsl')
    start=feature.index('float4 q4_coastal_rock(')
    end=feature.index('{',start)+1;depth=1
    while depth:
        depth+=(feature[end]=='{')-(feature[end]=='}');end+=1
    feature=feature[:start]+cliff.strip()+feature[end:]
    feature+=FRAME+'''
FeaturePixelInput VSReflection(PackedFeatureInput input) {
 FeaturePixelInput o=VSIntegratedFeature(input);
 float h=max(0,input.world.z-NativeReflection.z);
 o.position.y-=h*NativeReflection.x*4*c3x_inverse_viewport_size.y;
 float base=input.position.y+h*NativeReflection.x;
 o.position.z=clamp(.5-(floor((base-h*NativeReflection.y)*256+.5)/256+c3x_viewport_translation.y)/16384,.001,.999);
 return o;
}
float4 PSReflection(FeaturePixelInput input):SV_Target {
 clip(input.q6_world.w-.5);clip(input.q6_world.z-NativeReflection.z-.0001);
 return PSIntegratedFeature(input);
}
'''
    (HERE/'feature.hlsl').write_text(feature)
    for name in ['terrain','mountain','objects']:
        source=read(BASE/(name+'.hlsl'))+FRAME+'''
P VSReflection(V input) {
 P o=VSNative(input);
 float h=max(0,input.world.z-NativeReflection.z);
 o.position.y-=h*NativeReflection.x*4*inverse_size.y;
 float base=native_project_position(input.position,input.world.xyz).y+h*NativeReflection.x;
 o.position.z=clamp(.5-(floor((base-h*NativeReflection.y)*256+.5)/256+translation.y)/16384,.001,.999);
 return o;
}
float4 PSReflection(P input):SV_Target {
 clip(input.world.z-NativeReflection.z-.0001);
 return PSMain(input).color;
}
'''
        (HERE/(name+'.hlsl')).write_text(source)
    (HERE/'source_caster.hlsl').write_text(read(HERE.parent/'render_core/source_caster.hlsl'))
    (HERE/'provenance.json').write_text(json.dumps({'water':'water-natural-r6','reflection':'water-reflection-r5','source_sha256':pins},indent=2)+'\n')
if __name__=='__main__':main()

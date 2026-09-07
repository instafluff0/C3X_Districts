#ifndef Q3_WATER_EFFECTS_IMPL
#define Q3_WATER_EFFECTS_IMPL
// Lab prototype inspired by the technique breakdown in Alex Tardif's Water
// Walkthrough. Independent implementation using existing local pack textures.
// Fixed phase makes replay deterministic; runtime animation clock is pending.
#ifndef Q3_WATER_TIME
#define Q3_WATER_TIME 0.0
#endif
float q3_water_hash(float2 cell) {
    float period=Q3_MATERIAL_WRAP_WIDTH*.5*q3_source_repeat(1.2);
    if(period>0)cell-=floor(cell/period)*period;
    return macro_decal_hash(cell);
}
float q3_water_noise(float2 p) {
    float2 cell=floor(p),f=frac(p);f=f*f*(3-2*f);
    return lerp(lerp(q3_water_hash(cell),q3_water_hash(cell+float2(1,0)),f.x),
        lerp(q3_water_hash(cell+float2(0,1)),q3_water_hash(cell+1),f.x),f.y);
}
float3 q3_effect_normal(PixelInput input,float3 source_normal) {
    float2 world=q3_source_world(input);
    float time=Q3_WATER_TIME;
    float2 k0=float2(q3_source_repeat(1.38),q3_source_repeat(.52));
    float2 k1=float2(q3_source_repeat(-.72),q3_source_repeat(1.96));
    float2 k2=float2(q3_source_repeat(3.44),q3_source_repeat(2.22));
    float2 slopes=normalize(k0)*cos(dot(world,k0)*6.283185-time*1.12)*.12
        +normalize(k1)*cos(dot(world,k1)*6.283185-time*.79)*.055
        +normalize(k2)*cos(dot(world,k2)*6.283185-time*1.47)*.028;
    float2 drift=float2(time*.018,-time*.012);
    float2 small=water_small_lean0_texture.Sample(material_sampler,
        world*float2(q3_source_repeat(2.4),q3_source_repeat(3.06))+drift).rg*2-1;
    float depth_fade=smoothstep(0,.14,input.hydrology_data.w);
    return normalize(float3(source_normal.xy*.60-(slopes+small*.09)*depth_fade,1));
}
void q3_effect_color(PixelInput input,float3 normal,float3 illumination,
                    inout float3 tint,inout float alpha) {
    float2 world=q3_source_world(input);
    float time=Q3_WATER_TIME,d=max(0,-input.hydrology_data.x);
    float depth=max(0,input.hydrology_data.w);
    float2 noise_uv=world*q3_source_repeat(1.2);
    float patch=q3_water_noise(noise_uv);
    float2 foam_uv=world*float2(q3_source_repeat(3.4),q3_source_repeat(4.1))+float2(time*.025,-time*.012);
    float grain=water_foam_texture.Sample(material_sampler,foam_uv).a;
    // Broken shallow-water fronts approach the coast. Do not clamp at tile edges.
    float phase=d*27-time*1.3+patch*2.4;
    float front=pow(saturate(.5+.5*cos(phase)),12);
    float shore=smoothstep(.012,.055,d)*(1-smoothstep(.16,.30,d));
    float broken=smoothstep(.22,.70,grain*.70+patch*.48);
    float foam=front*shore*broken*.65;
    float contact=exp(-d*40)*smoothstep(.002,.025,d)*broken*.26;
    foam=saturate(foam+contact);
    float3 foam_light=float3(.65,.76,.77)*q6_receiver_illumination(input,float3(0,0,1),1,1);
    // More visible wave-facing sky response, still driven by the shared rig.
    float facing=saturate(normal.y*.8+normal.x*.3+.12);
    float3 sky=environment_ambient_color*float3(.055,.085,.105)*facing;
    tint+=sky*smoothstep(.05,.22,depth);
    // Compose foam over the existing water coverage without double premultiplication.
    float3 premult=tint*alpha*(1-foam)+foam_light*foam;
    alpha=alpha+(1-alpha)*foam;
    tint=premult/max(alpha,.0001);
}
#endif

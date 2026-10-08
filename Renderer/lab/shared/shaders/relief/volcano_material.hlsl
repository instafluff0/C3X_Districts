// Civ VI's dedicated volcano art on the shared mountain relief
// (lab/shared/natural/mountain_shape.h). owner.xy: position in the volcano
// element from its centre (crater rim near .055, foot near .29); owner.z: the
// volcano's coverage over neighbouring relief; owner.w: 2 x Civ III activity
// (0 dormant, 1 smoldering, 2 erupting) + the authored lava-channel mask.
// `rise` is the relief's rise above the ground in mountain units: ash meets
// the ground through the same kind of rise blend as mountain rock.
Texture2D VolcanoColor : register(t69);
Texture2D VolcanoActive : register(t71);
float volcano_coverage(float4 owner, float rise) {
    return owner.z*smoothstep(.01,.30,rise)*
        (1-smoothstep(.42,.50,max(abs(owner.x),abs(owner.y))));
}
float volcano_activity(float4 owner) { return floor(owner.w*.5+.25); }
float volcano_channel(float4 owner) { return saturate(owner.w-2*volcano_activity(owner)); }
float3 volcano_rock(float4 owner) {
    float3 rock=VolcanoColor.Sample(Clamp,.5+owner.xy).rgb;
    // Cooled lava streaks stay dark; an eruption buries the cone in fresh ash.
    rock*=lerp(1.0,.38,volcano_channel(owner));
    return rock*(volcano_activity(owner)>1.5?.55:1.0);
}
float3 volcano_albedo(float3 albedo, float4 owner, float rise) {
    return lerp(albedo,volcano_rock(owner),volcano_coverage(owner,rise));
}
// On relief the volcano replaces mountain rock: its footprint keeps the
// ground below the ash rather than grey stone (above the exact flat handoff).
float3 volcano_surface(float3 albedo, float3 ground, float4 owner, float rise) {
    float footprint=owner.z*smoothstep(.02,.10,rise)*(1-smoothstep(.42,.50,max(abs(owner.x),abs(owner.y))));
    return lerp(lerp(albedo,ground,footprint),volcano_rock(owner),volcano_coverage(owner,rise));
}
// Incandescent lava: dark red crust through orange to a yellow-white core.
float3 volcano_lava(float heat) {
    return lerp(lerp(float3(.28,.03,.005),float3(1.0,.30,.03),saturate(heat*2)),
                float3(1.0,.60,.20),saturate(heat*2-1));
}
// Static value noise for the lava crust, in element units.
float volcano_hash(float2 p) { return frac(sin(dot(p,float2(127.1,311.7)))*43758.5453); }
float volcano_noise(float2 x) {
    float2 i=floor(x),f=frac(x);f=f*f*(3-2*f);
    return lerp(lerp(volcano_hash(i),volcano_hash(i+float2(1,0)),f.x),
                lerp(volcano_hash(i+float2(0,1)),volcano_hash(i+1),f.x),f.y);
}
// Unlit lava: a crusted pool on the crater floor while the volcano is active,
// lighting the inner crater wall, and on an eruption the authored channels
// running from the rim down the cone. Bright seams mark where crust plates
// meet; a smoldering pool is mostly dark crust, an erupting one mostly molten.
float3 volcano_emission(float4 owner, float rise) {
    float activity=volcano_activity(owner);
    if(activity<.5)return 0;
    float coverage=volcano_coverage(owner,rise);
    float erupting=activity>1.5?1.0:0.0;
    float r=length(owner.xy);
    float2 q=owner.xy*70;
    float n=.65*volcano_noise(q)+.35*volcano_noise(q*2.7+5.3);
    float seam=1-smoothstep(.03,.12,abs(n-.5));
    float pool=1-smoothstep(.030,.052,r);
    float pool_heat=lerp(lerp(.10,.70,seam),lerp(.55,1.0,seam),erupting)*(1-.35*smoothstep(0,.05,r));
    float3 light=volcano_lava(pool_heat)*pool*lerp(1.2,1.8,erupting);
    float wall=exp(-((r-.05)/.012)*((r-.05)/.012))*(1-pool)*(1-smoothstep(.056,.066,r));
    light+=float3(1.0,.32,.05)*wall*lerp(.18,.55,erupting);
    // Flows leave the rim dark and narrow to the authored channels' cores.
    float flow=smoothstep(.55,.95,volcano_channel(owner))*smoothstep(.058,.075,r)*
        (1-smoothstep(.16,.26,r))*erupting;
    float flow_heat=saturate(.85-2.4*(r-.06))*lerp(.35,1.0,seam);
    light+=volcano_lava(flow_heat)*flow*(.35+.75*flow_heat);
    return light*coverage;
}

// Dedicated rock and static crater art share the ordinary scene lighting.
// Local offsets follow captured volcano tiles, including wrapped placements.
Texture2D VolcanoColor : register(t69);
Texture2D VolcanoLavaColor : register(t71);
float3 volcano_albedo(float3 albedo, float4 owner, float height) {
    float coverage=owner.z*smoothstep(.025,.20,height)*
        (1-smoothstep(.60,.78,max(abs(owner.x),abs(owner.y))));
    float2 uv=.5+float2(owner.x,-owner.y)*.3875;
    albedo=lerp(albedo,VolcanoColor.Sample(Clamp,uv).rgb,coverage);
    // Measured local art registration; not a recovered source-engine transform.
    float4 lava=VolcanoLavaColor.Sample(Clamp,uv+float2(.015,-.002));
    float mask=smoothstep(.16,.52,max(lava.r,max(lava.g,lava.b)))*lava.a;
    return lerp(albedo,lava.rgb,mask*coverage);
}

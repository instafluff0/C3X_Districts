// Dedicated dormant rock art shares the ordinary scene lighting.
// Local offsets follow captured volcano tiles, including wrapped placements.
Texture2D VolcanoColor : register(t69);
float3 volcano_albedo(float3 albedo, float4 owner, float height) {
    float coverage=owner.z*smoothstep(.025,.20,height)*
        (1-smoothstep(.60,.78,max(abs(owner.x),abs(owner.y))));
    float2 uv=.5+float2(owner.x,-owner.y)*.3875;
    return lerp(albedo,VolcanoColor.Sample(Clamp,uv).rgb,coverage);
}

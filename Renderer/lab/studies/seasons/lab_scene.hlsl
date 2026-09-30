// Headless Lab scene bindings. Production integration supplies its existing
// world pages/shadow field instead of these bounded diagnostic textures.
cbuffer LabScene : register(b3) {
    float4 LabField; // raw-X/Y origin and inverse span
    float4 LabU;
    float4 LabV;
    float4 LabL;
    float4 LabShadow; // U/V minimum and inverse span
    float4 LabDepth;  // light-depth minimum, inverse span, shadow size, unused
};
Texture2D LabSurface : register(t94); // shore distance, river distance, floodplain, unused
Texture2D LabBiomes : register(t95); // grassland, plains, desert, tundra
Texture2D LabShadowMap : register(t96);
float2 lab_field_uv(float2 p) {return (season_raw(p)-LabField.xy)*LabField.zw;}
float4 lab_surface(float2 p) {return LabSurface.SampleLevel(Clamp,lab_field_uv(p),0);}
float4 lab_biomes(float2 p) {return LabBiomes.SampleLevel(Clamp,lab_field_uv(p),0);}
float lab_land_coverage(float2 p) {return smoothstep(4.8,6.7,lab_surface(p).y);}
float lab_shadow(float3 p,float3 normal) {
    p+=normal*.014;
    float2 uv=(float2(dot(p,LabU.xyz),dot(p,LabV.xyz))-LabShadow.xy)*LabShadow.zw;
    if(any(uv<0)||any(uv>1))return 1;
    float receiver=dot(p,LabL.xyz);float visible=0;
    float nl=dot(normal,LabL.xyz);
    float2 gradient=-float2(dot(normal,LabU.xyz),dot(normal,LabV.xyz))/
        (abs(nl)>.12?nl:(nl<0?-.12:.12));
    [unroll]for(int y=-1;y<=1;y++)[unroll]for(int x=-1;x<=1;x++) {
        float b=LabShadowMap.SampleLevel(Clamp,uv+float2(x,y)/LabDepth.z,0).r;
        float adjusted=receiver+dot(gradient,float2(x,y)/(LabDepth.z*LabShadow.zw));
        visible+=step(b,adjusted+.008);
    }
    return visible/9;
}
void lab_season_ground(inout float3 albedo,inout float3 normal,inout float gloss,
    float3 geometric,float3 world,float extra_stone,float source_grain=.5) {
    float4 b=lab_biomes(world.xy);
    float flood=lab_surface(world.xy).z;
    float mineral=smoothstep(.50,.85,albedo.b/max(.001,albedo.g));
    float stone=max(extra_stone,mineral*(1-b.z));
    season_ground(albedo,normal,gloss,geometric,world,b,flood,stone,source_grain);
}

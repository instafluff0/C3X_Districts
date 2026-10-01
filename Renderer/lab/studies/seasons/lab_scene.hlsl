// Headless Lab scene bindings. Production integration supplies its existing
// world pages/shadow field instead of these bounded diagnostic textures.
cbuffer LabScene : register(b3) {
    float4 LabField; // raw-X/Y origin and inverse span
    float4 LabU;
    float4 LabV;
    float4 LabL;
    float4 LabShadow; // U/V minimum and inverse span
    float4 LabDepth;  // light-depth minimum, inverse span, shadow size, unused
    float4 LabCamera; // raw camera X/Y, map width/height
};
Texture2D LabSurface : register(t94); // shore distance, river distance, floodplain, unused
Texture2D LabBiomes : register(t95); // grassland, plains, desert, tundra
Texture2D LabShadowMap : register(t96);
Texture2D LabWinterShadowMap : register(t97);
float2 lab_field_uv(float2 p) {return (season_raw(p)-LabField.xy)*LabField.zw;}
float4 lab_surface(float2 p) {return LabSurface.SampleLevel(Clamp,lab_field_uv(p),0);}
float4 lab_biomes(float2 p) {return LabBiomes.SampleLevel(Clamp,lab_field_uv(p),0);}
float lab_land_coverage(float2 p) {
    float4 field=lab_surface(p);
    float river=smoothstep(4.8,6.7,field.y);
    // Correct the preview's source terrain/decal spill, for every season in
    // this harness mode. This is separate from the seasonal material policy.
    return river*(LabDepth.w>.5?smoothstep(-.015,.025,field.x):1);
}
float lab_shadow(float3 p,float3 normal) {
    if(season_quality() || LabDepth.w>.5){
        // Match the actual receiver plane rather than its bumped material
        // normal. Interpolating blocker depth created solid triangular marks
        // around relief and decal carriers in the earlier diagnostic atlas.
        float2 uv=(float2(dot(p,LabU.xyz),dot(p,LabV.xyz))-LabShadow.xy)*LabShadow.zw;
        if(any(uv<0)||any(uv>1))return 1;
        float2 texel=uv*LabDepth.z;
        float receiver=dot(p,LabL.xyz);
        float2 ux=ddx(texel),uy=ddy(texel);float zx=ddx(receiver),zy=ddy(receiver);
        float determinant=ux.x*uy.y-ux.y*uy.x;
        float2 gradient=0;
        if(abs(determinant)>1e-12)gradient=float2(zx*uy.y-zy*ux.y,zy*ux.x-zx*uy.x)/determinant;
        int2 center=int2(floor(texel));float visible=0,weights=0;
        [unroll]for(int y=-2;y<=2;y++)[unroll]for(int x=-2;x<=2;x++){
            int2 location=clamp(center+int2(x,y),int2(0,0),int2(LabDepth.zz)-1);
            float b=LabWinterShadowMap.Load(int3(location,0)).r;
            float plane=receiver+dot(gradient,float2(location)+.5-texel);
            float weight=(3-abs(x))*(3-abs(y));
            visible+=weight*step(b,plane+.0018);weights+=weight;
        }
        return visible/weights;
    }
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

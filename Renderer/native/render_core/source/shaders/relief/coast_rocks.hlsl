// Per-source-material batch: t25 base, t26 LEAN0, t27 LEAN1, t28 gloss.
// UV derivatives adapt the source slope channels; this is not a recovered
// source-engine LEAN equation. The original source channels remain bound.
float4 q4_coastal_rock(FeaturePixelInput input){
 float3 albedo=feature_base_texture_0.Sample(material_sampler,input.uv).rgb;
 float2 lean=feature_base_texture_1.Sample(material_sampler,input.uv).rg*2-1;
 float moment=feature_base_texture_2.Sample(material_sampler,input.uv).r;
 float gloss=feature_base_texture_3.Sample(material_sampler,input.uv).r;
 float3 n=normalize(input.geometry_normal);
 float3 world=input.q6_world.xyz*float3(1,-1,1);
 float3 dx=ddx(world),dy=ddy(world);
 float2 ux=ddx(input.uv),uy=ddy(input.uv);
 float det=ux.x*uy.y-ux.y*uy.x;
 if(abs(det)>1e-9){
  float3 tangent=normalize((dx*uy.y-dy*ux.y)/det);
  float3 bitangent=normalize((dy*ux.x-dx*uy.x)/det);
  n=normalize(n+(tangent*lean.x+bitangent*lean.y)*.35);
 }
 float3 color=albedo*q6_receiver_illumination(input,n,1,1);
 float roughness=clamp(1-gloss+max(0,moment-dot(lean,lean)*.25)*.25,.25,1);
 float3 view=normalize(float3(0,-.52,.86));
 float3 sunhalf=normalize(environment_sun_direction+view);
 float3 moonhalf=normalize(environment_moon_direction+view);
 float spec=lerp(100,8,roughness);
 color+=gloss*.045*(environment_sun_color*environment_sun_intensity*pow(saturate(dot(n,sunhalf)),spec)
       +environment_moon_color*environment_moon_intensity*pow(saturate(dot(n,moonhalf)),spec));
 return float4(color,1);
}

// Candidate-only flat/high material graph. Terrain draw bindings 116..121;
// feature draws retain their own resource materials in the same slot range.
#define q2_grass_high_color resource_base_texture_0
#define q2_grass_high_height resource_base_texture_1
#define q2_grass_high_specular resource_base_texture_2
#define q2_plains_high_color resource_base_texture_3
#define q2_plains_high_height resource_base_texture_4
#define q2_plains_high_specular resource_base_texture_5
float2 q2_continental_mix(PixelInput input) {
 float4 w=input.material_weights/max(.001,dot(input.material_weights,1)+input.material_tundra);
 float envelope=(input.surface_kind>.75&&input.surface_kind<1.25?1.0:0.0)
    *(1-smoothstep(.02,.2,input.authored_relief.y))*smoothstep(.985,1,input.geometry_normal.z);
 // The same actual displaced geometry drives color/height/specular. The
 // threshold is a Lab hypothesis; source engine high-layer masks are unknown.
 float h=max(0,input.q6_world.z*112-2.5)/max(.01,14*w.x+10*w.y);
 return w.xy*smoothstep(.32,.48,h)*envelope;
}
float q2_continental_height_delta(PixelInput input,float2 uv) {
 float2 w=q2_continental_mix(input);
 return w.x*(q2_grass_high_height.Sample(material_sampler,uv).r-height_texture.Sample(material_sampler,uv).r)
  +w.y*(q2_plains_high_height.Sample(material_sampler,uv).r-plains_height_texture.Sample(material_sampler,uv).r);
}
float q2_continental_specular_delta(PixelInput input) {
 float2 w=q2_continental_mix(input);
 return w.x*(q2_grass_high_specular.Sample(material_sampler,input.uv).r-specular_texture.Sample(material_sampler,input.uv).r)
  +w.y*(q2_plains_high_specular.Sample(material_sampler,input.uv).r-plains_specular_texture.Sample(material_sampler,input.uv).r);
}

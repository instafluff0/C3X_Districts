#ifndef Q2_SCENE_MATERIAL_V1
#define Q2_SCENE_MATERIAL_V1
// Include after the complete scene's texture declarations and PixelInput.
// Opt-in Q2_MATERIAL_RESPONSE: supplemental SOURCE detail on existing materials.
// This never replaces source relief, water, shore, macro albedo or illumination.
#ifndef Q2_SCENE_DETAIL
#define Q2_SCENE_DETAIL 1
#endif
float q2_source_height(PixelInput input,float2 uv) {
 float4 w=max(input.material_weights,0);float t=max(input.material_tundra,0);
 float total=max(.001,dot(w,1)+t);w/=total;t/=total;
 return dot(float4(height_texture.Sample(material_sampler,uv).r,
  plains_height_texture.Sample(material_sampler,uv).r,
  desert_height_texture.Sample(material_sampler,uv).r,
  marsh_height_texture.Sample(material_sampler,uv).r),w)
  +feature_base_texture_5.Sample(material_sampler,uv).r*t;
}
float q2_secondary_height(PixelInput input,float2 uv) {
 return .22*q2_source_height(input,uv*3)+.07*q2_source_height(input,uv*8);
}
float q2_base_detail_envelope(PixelInput input,float3 geometry_normal) {
 // Continuous masks preserve source-owned raised bodies and avoid a tile flag seam.
 return (input.surface_kind>.75&&input.surface_kind<1.25?1.0:0.0)
  *(1-smoothstep(.02,.45,saturate(input.authored_relief.y)))
  *smoothstep(.65,.98,geometry_normal.z);
}
void q2_material_form(PixelInput input,float2 world_position,float3 geometry_normal,
 inout float3 albedo,inout float3 material_normal) {
 if(!Q2_SCENE_DETAIL)return;
 float envelope=q2_base_detail_envelope(input,geometry_normal);if(envelope<=0)return;
 float h=q2_secondary_height(input,input.uv);
 float hx=q2_secondary_height(input,input.uv+float2(.002,0))-h;
 float hy=q2_secondary_height(input,input.uv+float2(0,.002))-h;
 float2 delta=clamp(float2(-hx-hy,-hx+hy)*8.485281,-.08,.08)*envelope;
 material_normal=normalize(float3(material_normal.xy+delta*geometry_normal.z,material_normal.z));
 // The zero-centered secondary field is subordinate to the selected source color.
 albedo*=1+(h-.145)*.065*envelope;
}
void q2_material_specular(PixelInput input,float2 world_position,float3 geometry_normal,
 inout float specular) {
 if(!Q2_SCENE_DETAIL)return;
 float envelope=q2_base_detail_envelope(input,geometry_normal);if(envelope<=0)return;
 float h=q2_secondary_height(input,input.uv);
 // Existing source specular remains authoritative; slight roughness variation only.
 specular*=clamp(1-(h-.145)*.04*envelope,.98,1.02);
}
#endif

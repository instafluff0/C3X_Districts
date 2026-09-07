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
#ifdef Q2_SURFACE_GRADIENT
// A physical surface gradient from the filtered source height. The old
// unnormalized one-source-texel difference nearly vanishes when a 4096-wide
// material is minified to gameplay scale. Evaluate across the visible footprint
// and divide by that interval, then transform through the actual surface basis.
// Height amplitude is a Lab interpretation in world units, not a recovered
// source-engine parameter. This branch is opt-in while visual QA is pending.
void q2_surface_gradient(PixelInput input,float3 n,inout float3 material_normal) {
 float2 ux=ddx(input.uv),uy=ddy(input.uv);
 float2 step_uv=max(height_texel,.5*(abs(ux)+abs(uy)));
 float2 g=float2(
  q2_source_height(input,input.uv+float2(step_uv.x,0))-q2_source_height(input,input.uv-float2(step_uv.x,0)),
  q2_source_height(input,input.uv+float2(0,step_uv.y))-q2_source_height(input,input.uv-float2(0,step_uv.y))) /(2*step_uv);
 float3 px=ddx(input.q6_world.xyz),py=ddy(input.q6_world.xyz);
 float3 rx=cross(py,n),ry=cross(n,px);
 float det=dot(px,rx);
 float3 gradient=(dot(g,ux)*rx+dot(g,uy)*ry)/(abs(det)>1e-9?det:1);
 float envelope=(input.surface_kind>.75&&input.surface_kind<1.25?1.0:0.0)
  *(1-smoothstep(.02,.45,saturate(input.authored_relief.y)));
 // Source base material remains active on hills, with tangent-correct response.
 float3 delta=gradient*.016;
 delta*=min(1.0,.55/max(length(delta),.00001));
 material_normal=normalize(material_normal-delta*envelope);
}
#endif
void q2_material_form(PixelInput input,float2 world_position,float3 geometry_normal,
 inout float3 albedo,inout float3 material_normal) {
#ifdef Q2_SURFACE_GRADIENT
 q2_surface_gradient(input,geometry_normal,material_normal);
#endif
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

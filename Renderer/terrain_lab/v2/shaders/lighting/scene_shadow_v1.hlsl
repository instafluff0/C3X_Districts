// Included after frozen texture, environment and pixel input declarations.
// Q0 world extension and wire5 common b1 are opt-in; ordinary linear is no-op.
#ifndef Q6_CAST_SHADOWS
#define Q6_CAST_SHADOWS 1
#endif
#ifndef Q6_SCENE_CONTACT
#define Q6_SCENE_CONTACT 1
#endif
#ifdef Q6_WORLD_SHADOWS
#include "frame_shadow_v1.hlsl"
#endif

float q6_receiver_visibility(PixelInput input,float3 normal,float legacy_shadow) {
#ifdef Q6_WORLD_SHADOWS
 if(input.q6_world.w>.5 && Q6ShadowFlags.x>.5) {
  bool water=input.surface_kind>3.5 && input.surface_kind<6.5;
  water=water || (input.surface_kind>8.5 && input.surface_kind<9.5);
  // Main t25 aliases a feature-only source binding; Q0 binds one common field.
  float visibility=q6_world_visibility(feature_base_texture_0,input.q6_world,normal,water);
  return visibility;
 }
#endif
 return legacy_shadow;
}
float3 q6_receiver_illumination(PixelInput input,float3 normal,
 float legacy_shadow,float ambient_visibility) {
 return frame_illumination(normal,q6_receiver_visibility(input,normal,legacy_shadow),ambient_visibility);
}
float3 q6_receiver_illumination(FeaturePixelInput input,float3 normal,
 float legacy_shadow,float ambient_visibility) {
#ifdef Q6_WORLD_SHADOWS
 if(input.q6_world.w>.5 && Q6ShadowFlags.x>.5) {
  // Feature t17 aliases the bed-only source binding.
  float visibility=q6_world_visibility(shallow_bed_texture,input.q6_world,normal,false);
  return frame_illumination(normal,visibility,ambient_visibility);
 }
#endif
 return frame_illumination(normal,legacy_shadow,ambient_visibility);
}

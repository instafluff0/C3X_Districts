#ifndef Q6_FRAME_SHADOW_V1
#define Q6_FRAME_SHADOW_V1
#include "shadow_visibility_v1.hlsl"
#ifndef Q6_CAST_SHADOWS
#define Q6_CAST_SHADOWS 1
#endif
#ifndef Q6_SCENE_CONTACT
#define Q6_SCENE_CONTACT 1
#endif
// Shared wire5/6 b1 for every shader namespace; texture binding is per draw.
cbuffer Q6SharedShadow : register(b1) {
 float4 Q6ShadowU; // xyz light right; w span in normalized tile units
 float4 Q6ShadowV; // xyz light up; w resolution
 float4 Q6ShadowL;
 float4 Q6ShadowOrigin;
 float4 Q6ShadowFlags; // enabled, tighter contact, reserved, reserved
};
float q6_world_visibility(Texture2D field,float4 world,float3 normal,bool water) {
 if(world.w<=.5 || Q6ShadowFlags.x<=.5)return 1;
 return q6_shadow_visibility(field,world.xyz-Q6ShadowOrigin.xyz,normal,
  Q6ShadowU,Q6ShadowV,Q6ShadowL,Q6_CAST_SHADOWS,
  !water && Q6_SCENE_CONTACT && Q6ShadowFlags.y>.5);
}
#endif

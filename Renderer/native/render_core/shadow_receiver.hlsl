// Binding adapter; the shared Lab provider owns filtering/contact/depth rules.
Texture2DArray pickup_shadow_terrain : register(t25);
Texture2DArray pickup_shadow_feature : register(t17);
// C3X_SHARED_PAGED_SHADOW
float q6_world_visibility(Texture2DArray field,float4 world,float3 normal,bool water) {
 return c3x_paged_visibility(field,world,normal,water,Q6ShadowU,Q6ShadowV,Q6ShadowL,Q6ShadowFlags);
}

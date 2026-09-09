// Binding adapter; the shared Lab provider owns filtering/contact/depth rules.
// C3X_SHARED_PAGED_SHADOW
float q6_world_visibility(Texture2DArray field,float4 world,float3 normal,bool water) {
 return c3x_paged_visibility(field,world,normal,water,ShadowU,ShadowV,ShadowL,ShadowFlags);
}
float q6_shadow_visibility(Texture2DArray field,float3 world,float3 normal,float4 u,float4 v,float4 l,bool receive,bool contact) {
 return receive?q6_world_visibility(field,float4(world,1),normal,!contact):1;
}

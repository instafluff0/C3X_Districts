#ifndef Q2_MATERIAL_RESPONSE_V1
#define Q2_MATERIAL_RESPONSE_V1
// Pure linear-material response; no light, clock, tone map or output encoding.
// Caller samples normalized source height at 1x/3x/8x with amplitudes 1/.22/.07.
// raw_dx/raw_dy are forward differences at source UV step .002, not derivatives
// divided by step. Normal is in the base lattice tangent frame (+Z up).
struct Q2MaterialDetailV1 {
 float3 albedo;
 float3 tangent_normal;
 float roughness;
};
Q2MaterialDetailV1 q2_material_detail_v1(float3 albedo,float height,
 float raw_dx,float raw_dy,float source_specular,bool enabled) {
 Q2MaterialDetailV1 result;
 result.albedo=albedo;
 result.tangent_normal=float3(0,0,1);
 if(enabled) {
  result.tangent_normal=normalize(float3(clamp(float2(-raw_dx-raw_dy,-raw_dx+raw_dy)*8.485281,-.28,.28),1));
  result.albedo*=1+(height-.645)*.065;
 }
 result.roughness=clamp(1-source_specular*.55+(enabled?(height-.645)*.04:0),.65,.98);
 return result;
}
#endif

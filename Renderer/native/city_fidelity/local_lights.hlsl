
// Eight independently bounded city fields fit within one D3D11 constant
// buffer. The CPU selects intersecting cities per guarded block; overflow
// fails the candidate draw instead of truncating a city's emitting facades.
cbuffer NativeCityLights : register(b6) {
 float4 CityLightCounts;
 float4 Q8LocalEnvelopeLow4;float4 Q8LocalEnvelopeHigh4;
 float4 Q8LocalPositionRange[1024];float4 Q8LocalColorIntensity[1024];
 float4 Q8LocalDirectionOwner[1024];float4 Q8LocalBoxLow[256];float4 Q8LocalBoxHigh[256];
};
#define Q8_LOCAL_LIGHT_COUNT int(CityLightCounts.x)
#define Q8_LOCAL_BLOCKER_COUNT int(CityLightCounts.y)
#define Q8LocalEnvelopeLow Q8LocalEnvelopeLow4.xyz
#define Q8LocalEnvelopeHigh Q8LocalEnvelopeHigh4.xyz
#define Q8_LOCAL_Z_METRIC 0.648266978876
#define Q8_LOCAL_LIGHT_GAIN 4
// Generic, bounded emissive-facade light proxies. Positions/colors are derived
// offline from normalized source materials, not recovered source light bindings.
#ifndef Q8_LOCAL_LIGHT_GAIN
#define Q8_LOCAL_LIGHT_GAIN 1
#endif
#ifndef Q8_LOCAL_OCCLUSION
#define Q8_LOCAL_OCCLUSION 1
#endif
bool q8_local_box_blocks(float3 start,float3 finish,float3 low,float3 high) {
 float3 ray=finish-start;
 float3 safe_ray=float3(ray.x<0?-max(abs(ray.x),1e-6):max(abs(ray.x),1e-6),
                        ray.y<0?-max(abs(ray.y),1e-6):max(abs(ray.y),1e-6),
                        ray.z<0?-max(abs(ray.z),1e-6):max(abs(ray.z),1e-6));
 float3 a=(low-start)/safe_ray,b=(high-start)/safe_ray;
 float3 entry=min(a,b),leave=max(a,b);
 float near_t=max(entry.x,max(entry.y,entry.z));
 float far_t=min(leave.x,min(leave.y,leave.z));
 return far_t>=max(near_t,.001) && near_t<.995;
}
float3 q8_local_irradiance(float4 world,float3 normal,float ambient_visibility) {
 if(world.w<.5 || CityLightCounts.z<=0 || Q8_LOCAL_LIGHT_GAIN<=0)return 0;
 float3 receiver_position=float3(world.x,-world.y,world.z*Q8_LOCAL_Z_METRIC);
 if(any(receiver_position<Q8LocalEnvelopeLow) || any(receiver_position>Q8LocalEnvelopeHigh))return 0;
 float3 light_sum=0;
 [loop]for(int i=0;i<Q8_LOCAL_LIGHT_COUNT;i++) {
  float3 to_light=Q8LocalPositionRange[i].xyz-receiver_position;
  float distance2=dot(to_light,to_light);
  float range=Q8LocalPositionRange[i].w;
  if(distance2>=range*range)continue;
  float3 direction=to_light*rsqrt(max(distance2,1e-8));
  float face=saturate(dot(Q8LocalDirectionOwner[i].xyz,-direction));
  float diffuse=saturate(dot(normal,direction));
  if(face*diffuse<=0)continue;
  bool blocked=false;
#if Q8_LOCAL_OCCLUSION
  [loop]for(int j=0;j<Q8_LOCAL_BLOCKER_COUNT;j++) {
   if(j==int(Q8LocalDirectionOwner[i].w))continue;
   if(q8_local_box_blocks(Q8LocalPositionRange[i].xyz,receiver_position,Q8LocalBoxLow[j].xyz,Q8LocalBoxHigh[j].xyz)) {blocked=true;break;}
  }
#endif
  if(blocked)continue;
  float normalized_distance=distance2/(range*range);
  float attenuation=pow(1-normalized_distance,2)/(1+8*normalized_distance);
  light_sum+=Q8LocalColorIntensity[i].rgb*Q8LocalColorIntensity[i].w*attenuation*face*diffuse;
 }
 return light_sum*(Q8_LOCAL_LIGHT_GAIN*CityLightCounts.z*CityLightCounts.w*ambient_visibility);
}


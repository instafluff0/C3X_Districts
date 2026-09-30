
// The scene-light adapter below replaces these regional declarations with
// a growable GPU field while keeping the shared material equations.
cbuffer NativeCityLights : register(b6) {
 float4 CityLightCounts;
 float4 Q8LocalEnvelopeLow4;float4 Q8LocalEnvelopeHigh4;
 float4 Q8LocalGrid;float4 Q8LocalGridInfo;
};
StructuredBuffer<float4> CityLightData : register(t127);

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
// Grid/list offsets share t127 with the complete light/blocker field.
// Zero mode retains the exact original loops (failure and diagnostic reference).
int q8_candidate_index(int offset,int entry) {
 int scalar=offset+entry;
 int base=int(Q8LocalGridInfo.z)+int(CityLightCounts.x);
 return int(CityLightData[base+scalar/4][scalar%4]);
}
float3 q8_local_irradiance(float4 world,float3 normal,float ambient_visibility) {
 if(world.w<.5 || CityLightCounts.z<=0 || Q8_LOCAL_LIGHT_GAIN<=0)return 0;
 float3 receiver_position=float3(world.x,-world.y,world.z*Q8_LOCAL_Z_METRIC);
 if(any(receiver_position<Q8LocalEnvelopeLow) || any(receiver_position>Q8LocalEnvelopeHigh))return 0;
 float3 light_sum=0;
 int light_offset=0,light_count=Q8_LOCAL_LIGHT_COUNT;
 bool indexed=Q8LocalGridInfo.w>.5;
 if(indexed) {
  int2 cell=int2(floor((receiver_position.xy-Q8LocalGrid.xy)*Q8LocalGrid.z));
  if(any(cell<0) || cell.x>=int(Q8LocalGrid.w) || cell.y>=int(Q8LocalGridInfo.x))return 0;
  float2 list=CityLightData[int(Q8LocalGridInfo.y)+cell.y*int(Q8LocalGrid.w)+cell.x].xy;
  light_offset=int(list.x);light_count=int(list.y);
 }
 [loop]for(int candidate=0;candidate<light_count;candidate++) {
  int i=indexed?q8_candidate_index(light_offset,candidate):candidate;
  float3 to_light=CityLightData[3*(i)+0].xyz-receiver_position;
  float distance2=dot(to_light,to_light);
  float range=CityLightData[3*(i)+0].w;
  if(distance2>=range*range)continue;
  float3 direction=to_light*rsqrt(max(distance2,1e-8));
  float face=saturate(dot(CityLightData[3*(i)+2].xyz,-direction));
  float diffuse=saturate(dot(normal,direction));
  if(face*diffuse<=0)continue;
  bool blocked=false;
#if Q8_LOCAL_OCCLUSION
  int blocker_offset=0,blocker_count=Q8_LOCAL_BLOCKER_COUNT;
  if(indexed) {
   float2 list=CityLightData[int(Q8LocalGridInfo.z)+i].xy;
   blocker_offset=int(list.x);blocker_count=int(list.y);
  }
  [loop]for(int blocker=0;blocker<blocker_count;blocker++) {
   int j=indexed?q8_candidate_index(blocker_offset,blocker):blocker;
   if(j==int(CityLightData[3*(i)+2].w))continue;
   if(q8_local_box_blocks(CityLightData[3*(i)+0].xyz,receiver_position,CityLightData[3*int(CityLightCounts.x)+2*(j)+0].xyz,CityLightData[3*int(CityLightCounts.x)+2*(j)+1].xyz)) {blocked=true;break;}
  }
#endif
  if(blocked)continue;
  float normalized_distance=distance2/(range*range);
  float attenuation=pow(1-normalized_distance,2)/(1+8*normalized_distance);
  light_sum+=CityLightData[3*(i)+1].rgb*CityLightData[3*(i)+1].w*attenuation*face*diffuse;
 }
 return light_sum*(Q8_LOCAL_LIGHT_GAIN*CityLightCounts.z*CityLightCounts.w*ambient_visibility);
}


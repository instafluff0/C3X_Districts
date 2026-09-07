// Complete source bodies in the shared terrain/water namespace. The caller has
// renamed the inherited PSFeature to Q8LegacyPSFeature before including terrain.
#ifndef Q8_CITY_FEATURE_ENTRY
#define Q8_CITY_FEATURE_ENTRY PSFeature
#endif
#ifndef Q8_CITY_CHANNELS
#define Q8_CITY_CHANNELS 0
#endif
#ifndef Q8_CITY_EMISSIVE_GAIN
#define Q8_CITY_EMISSIVE_GAIN 1.45
#endif
#ifndef Q8_CITY_SEPARATE_EMISSION
#define Q8_CITY_SEPARATE_EMISSION 0
#endif
#ifndef Q8_CITY_SURFACE_DETAIL
#define Q8_CITY_SURFACE_DETAIL 0
#endif
#ifndef Q8_CITY_WORLD_Z_TO_SOURCE
#define Q8_CITY_WORLD_Z_TO_SOURCE 1
#endif
float4 q8_surface_sample(Texture2D source,float2 uv,bool repeat_uv) {
 if(repeat_uv)return source.Sample(material_sampler,uv);
 return source.Sample(decal_sampler,uv);
}
Q6SceneOutput Q8_CITY_FEATURE_ENTRY(FeaturePixelInput p) {
 if(p.material_index<39.5)return Q8LegacyPSFeature(p);
 if(p.material_index>=59.5 && p.material_index<69.5) {
  float4 ground=city_base_texture_0.Sample(decal_sampler,p.uv);
  return q6_scene_output(float4(ground.rgb*q6_receiver_illumination(p,normalize(p.geometry_normal),1,1),ground.a));
 }
 int channels=(int)round(p.material_index-40);
 bool repeat_uv=(channels&4)!=0;
 float3 base=q8_surface_sample(city_base_texture_0,p.uv,repeat_uv).rgb;
 float3 emission=resource_base_texture_0.Sample(decal_sampler,p.uv).rgb;
 if(p.material_index>=79.5)
  return q6_scene_output(float4(emission*environment_night_activation*environment_emissive_scale*Q8_CITY_EMISSIVE_GAIN,1));
 float3 n=normalize(p.geometry_normal);
#if Q8_CITY_SURFACE_DETAIL
 if((channels&2)!=0) {
  // Source LEAN0 holds signed surface-direction detail. Its source-engine
  // scale and exact LEAN BRDF are not recovered. This diffuse-only adaptation
  // uses UV derivatives and preserves the geometric normal's tangent plane.
  float2 slope=q8_surface_sample(resource_base_texture_3,p.uv,repeat_uv).rg*2-1;
  float3 world=p.q6_world.xyz*float3(1,-1,Q8_CITY_WORLD_Z_TO_SOURCE);
  float3 dx=ddx(world),dy=ddy(world);
  float2 ux=ddx(p.uv),uy=ddy(p.uv);
  float determinant=ux.x*uy.y-ux.y*uy.x;
  if(abs(determinant)>1e-9) {
   float3 t=(dx*uy.y-dy*ux.y)/determinant;
   float3 b=(dy*ux.x-dx*uy.x)/determinant;
   t-=n*dot(t,n);b-=n*dot(b,n);
   if(dot(t,t)>1e-10 && dot(b,b)>1e-10)
    n=normalize(n+normalize(t)*slope.x+normalize(b)*slope.y);
  }
 }
#endif
 float ao=1;
#if Q8_CITY_CHANNELS
 if((channels&1)!=0)ao=q8_surface_sample(resource_base_texture_2,p.uv,repeat_uv).r;
 // normal_1 and gloss remain unbound until their source roles are established.
#endif
 float3 lit=base*q6_receiver_illumination(p,n,1,ao);
 if(!Q8_CITY_SEPARATE_EMISSION)
  lit+=emission*environment_night_activation*environment_emissive_scale*Q8_CITY_EMISSIVE_GAIN;
 return q6_scene_output(float4(lit,1));
}

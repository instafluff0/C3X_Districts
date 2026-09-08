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
#ifndef Q8_CITY_AUXILIARY_AO
#define Q8_CITY_AUXILIARY_AO 0
#endif
#ifndef Q8_CITY_AO_STRENGTH
#define Q8_CITY_AO_STRENGTH 1
#endif
#ifndef Q8_CITY_WORLD_Z_TO_SOURCE
#define Q8_CITY_WORLD_Z_TO_SOURCE 1
#endif
#ifndef Q8_CITY_SOURCE_SURFACE
#define Q8_CITY_SOURCE_SURFACE 0
#endif
#ifndef Q8_CITY_SOURCE_SPECULAR
#define Q8_CITY_SOURCE_SPECULAR 0
#endif
#ifndef Q8_CITY_EXTRA_MATERIALS
#define Q8_CITY_EXTRA_MATERIALS 0
#endif
float4 q8_surface_sample(Texture2D source,float2 uv,bool repeat_uv) {
 if(repeat_uv)return source.Sample(material_sampler,uv);
 return source.Sample(decal_sampler,uv);
}
#if Q8_CITY_SOURCE_SPECULAR
// Cooked two-lobe parameters established in the installed rigid-model shader.
// This partial adapter preserves shared Lab illumination. The source variance
// scale and environment-cube integration remain absent. Optional direct-only
// metalness is diagnostic until the environment response is implemented.
float3 q8_city_direct_specular(float3 light,float3 radiance,float3 n,float3 geometric,
 float3 tangent,float3 bitangent,float2 normal_xy,float3 roughness,float3 base,float metalness) {
 float3 halfway=normalize(light+Q8_CITY_VIEW_DIRECTION);
 float hz=dot(halfway,geometric);
 if(hz<=0)return 0;
 float2 offset=float2(dot(halfway,tangent),dot(halfway,bitangent))/hz-normal_xy;
 float2 inverse_variance=.5/max(roughness.rg,float2(1e-6,1e-6));
 float2 lobes=inverse_variance*exp(-min(inverse_variance*dot(offset,offset),256));
 float distribution=.25*(roughness.b+dot(lobes,float2(1.0/3.0,2.0/3.0)));
 float f0=.04*pow(1-saturate(sqrt(3.14159265*roughness.b)-.35),2);
 float3 reflectance=lerp(f0.xxx,base,metalness);
 float3 fresnel=reflectance+(1-reflectance)*pow(1-saturate(dot(halfway,light)),5);
 return radiance*(distribution*fresnel*saturate(dot(n,light)));
}
#endif
Q6SceneOutput Q8_CITY_FEATURE_ENTRY(FeaturePixelInput p) {
 if(p.material_index<39.5)return Q8LegacyPSFeature(p);
 if(p.material_index>=59.5 && p.material_index<69.5) {
  float4 ground=city_base_texture_0.Sample(decal_sampler,p.uv);
  return q6_scene_output(float4(ground.rgb*q6_receiver_illumination(p,normalize(p.geometry_normal),1,1),ground.a));
 }
 bool emission_only=(p.material_index>=79.5 && p.material_index<89.5)||p.material_index>=199.5;
 int channels=(int)round(p.material_index-(p.material_index>=199.5?200:p.material_index>=99.5?100:40));
 bool repeat_uv=(channels&4)!=0;
#if Q8_CITY_EXTRA_MATERIALS
 if((channels&32)!=0)clip(q8_surface_sample(resource_base_texture_5,p.uv,repeat_uv).a-.5);
#endif
 float3 base=q8_surface_sample(city_base_texture_0,p.uv,repeat_uv).rgb;
#if Q8_CITY_EXTRA_MATERIALS
 float3 emission=resource_base_texture_0.Sample(decal_sampler,p.city_emissive_uv).rgb;
#else
 float3 emission=resource_base_texture_0.Sample(decal_sampler,p.uv).rgb;
#endif
 if(emission_only)
  return q6_scene_output(float4(emission*environment_night_activation*environment_emissive_scale*Q8_CITY_EMISSIVE_GAIN,1));
 float3 n=normalize(p.geometry_normal);
#if Q8_CITY_SOURCE_SURFACE
 float3 geometric=n;
 float3 tangent=normalize(p.city_tangent),bitangent=normalize(p.city_bitangent);
 float2 normal_xy=float2(0,0);
 if((channels&2)!=0) {
  normal_xy=q8_surface_sample(resource_base_texture_3,p.uv,repeat_uv).rg*2-1;
  float normal_z=sqrt(max(0,1-dot(normal_xy,normal_xy)));
  n=normalize(tangent*normal_xy.x+bitangent*normal_xy.y+geometric*normal_z);
 }
#elif Q8_CITY_SURFACE_DETAIL
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
#if Q8_CITY_AUXILIARY_AO
 if((channels&1)!=0)ao=lerp(1,resource_base_texture_2.Sample(decal_sampler,p.city_ao_uv).r,Q8_CITY_AO_STRENGTH);
#elif Q8_CITY_CHANNELS
 if((channels&1)!=0)ao=q8_surface_sample(resource_base_texture_2,p.uv,repeat_uv).r;
 // normal_1 and gloss remain unbound until their source roles are established.
#endif
 float metalness=0;
#if Q8_CITY_EXTRA_MATERIALS
 if((channels&16)!=0)metalness=q8_surface_sample(resource_base_texture_4,p.uv,repeat_uv).r;
#endif
 float3 lit=base*(1-metalness)*q6_receiver_illumination(p,n,1,ao);
#if Q8_CITY_SOURCE_SPECULAR
 if((channels&8)!=0) {
  float3 roughness=q8_surface_sample(resource_base_texture_1,p.uv,repeat_uv).rgb;
  float visibility=1;
#ifdef Q6_WORLD_SHADOWS
  if(p.q6_world.w>.5 && Q6ShadowFlags.x>.5)
   visibility=q6_world_visibility(shallow_bed_texture,p.q6_world,n,false);
#endif
  lit+=visibility*(q8_city_direct_specular(environment_sun_direction,environment_sun_color*environment_sun_intensity,n,geometric,tangent,bitangent,normal_xy,roughness,base,metalness)
       +q8_city_direct_specular(environment_moon_direction,environment_moon_color*environment_moon_intensity,n,geometric,tangent,bitangent,normal_xy,roughness,base,metalness));
 }
#endif
 if(!Q8_CITY_SEPARATE_EMISSION)
  lit+=emission*environment_night_activation*environment_emissive_scale*Q8_CITY_EMISSIVE_GAIN;
 return q6_scene_output(float4(lit,1));
}

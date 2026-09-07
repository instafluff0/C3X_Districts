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
Q6SceneOutput Q8_CITY_FEATURE_ENTRY(FeaturePixelInput p) {
 if(p.material_index<39.5)return Q8LegacyPSFeature(p);
 float3 base=city_base_texture_0.Sample(decal_sampler,p.uv).rgb;
 float3 emission=resource_base_texture_0.Sample(decal_sampler,p.uv).rgb;
 float3 n=normalize(p.geometry_normal);
 float ao=1;
#if Q8_CITY_CHANNELS
 int channels=(int)round(p.material_index-40);
 if((channels&1)!=0)ao=resource_base_texture_2.Sample(decal_sampler,p.uv).r;
 // normal_1 and gloss remain unbound until their source roles are established.
#endif
 float3 lit=base*q6_receiver_illumination(p,n,1,ao);
 lit+=emission*environment_night_activation*environment_emissive_scale*Q8_CITY_EMISSIVE_GAIN;
 return q6_scene_output(float4(lit,1));
}

// Source-family dual-lobe material from the unit studio, driven by the shared
// native environment. No category-specific sun, camera or display transform.
float3 unit_microfacet_response(Output i,float3 albedo,float3 n,float shadow) {
 float ao=channels.x>.5?ambient_occlusion.Sample(sample_base,i.uv).r:1;
 float3 roughness=channels.y>.5?gloss_texture.Sample(sample_base,i.uv).rgb:float3(.1,.3,.1);
 float3 light=normalize(Sun.xyz),view=normalize(View.xyz);
 float ndl=saturate(dot(n,light));
 float3 fill=lerp(Ambient.rgb*float3(.285714,.2,.121212),Ambient.rgb,saturate(n.z*.5+.5));
 float3 radiance=albedo*(fill*ao+(SunColorExposure.rgb*Sun.w)*(ndl*shadow/3.14159265359));
 float3 geometric=normalize(i.n),halfway=normalize(light+view);
 float hz=dot(halfway,geometric);
 float2 xy=channels.w>.5?normal_texture.Sample(sample_base,i.uv).rg*2-1:float2(0,0);
 if(hz>0) {
  float2 offset=float2(dot(i.tangent,halfway),dot(i.bitangent,halfway))/max(hz,1e-6)-xy;
  float2 inv=.5/max(roughness.rg,float2(1e-6,1e-6));
  float2 lobes=inv*exp(-min(inv*dot(offset,offset),256));
  float distribution=.25*(roughness.b+dot(lobes,float2(1.0/3.0,2.0/3.0)));
  float f0=.04*pow(1-saturate(sqrt(3.14159265359*roughness.b)-.35),2);
  float fresnel=f0+(1-f0)*pow(1-saturate(dot(halfway,light)),5);
  radiance+=(SunColorExposure.rgb*Sun.w)*(distribution*fresnel*ndl*shadow);
 }
 if(channels.z>.5)radiance+=emissive_texture.Sample(sample_emission,i.uv).rgb;
 return max(radiance,0);
}

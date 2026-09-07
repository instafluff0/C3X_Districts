// Authored analytic environment fallback. This is not the source game's
// filtered cubearray or SH rig. It supplies the missing reflected sky/ground
// term from the shared sun/moon/ambient state without adding another clock.
// The two cooked lobe variances remain distinct; broad lobes approach the
// environment mean instead of retaining sharp horizon contrast.
float3 q8_city_environment_lobe(float3 reflected,float variance) {
 float spread=rsqrt(1+max(variance,0)*4);
 float sky_weight=saturate(.5+.5*reflected.z*spread);
 float3 illumination=environment_ambient_color
  +environment_sun_color*environment_sun_intensity
  +environment_moon_color*environment_moon_intensity;
 float3 ground=float3(.18,.16,.12);
 float3 sky=float3(.64,.82,1.0);
 return illumination*lerp(ground,sky,sky_weight);
}
float3 q8_city_environment_specular(float3 normal,float3 roughness,float3 base,float metalness,float ao) {
 float3 view=Q8_CITY_VIEW_DIRECTION;
 float3 reflected=reflect(-view,normal);
 float3 radiance=q8_city_environment_lobe(reflected,roughness.r)/3
  +q8_city_environment_lobe(reflected,roughness.g)*(2.0/3.0);
 float f0=.04*pow(1-saturate(sqrt(3.14159265*roughness.b)-.35),2);
 float3 reflectance=lerp(f0.xxx,base,metalness);
 float cosine=saturate(dot(normal,view));
 float3 fresnel=reflectance+(1-reflectance)*pow(1-cosine,5);
 // Installed rigid-model witness: broad roughness attenuates the directional
 // environment lobe, especially at grazing angles. Omitting this makes rough
 // tile/wood roofs look like polished metal. Cubearray filtering and SH remain
 // analytic approximations here; this attenuation is source-family evidence.
 float attenuation=saturate((.315-roughness.b)/.315)
  /(1+sqrt(3.14159265*roughness.b)*10*(1-sqrt(cosine)));
 float3 broad=q8_city_environment_lobe(normal,1)*roughness.b*reflectance;
 // AO bounds indirect reflection as well as diffuse light in deep recesses.
 return (radiance*fresnel*attenuation+broad)*ao;
}

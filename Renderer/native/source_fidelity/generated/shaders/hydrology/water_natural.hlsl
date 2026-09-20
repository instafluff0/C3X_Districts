#ifndef Q3_NATURAL_WATER_IMPL
#define Q3_NATURAL_WATER_IMPL
#ifndef Q3_NATURAL_COORD_SHIFT
#define Q3_NATURAL_COORD_SHIFT 0.0
#endif
#ifdef Q3_OBJECT_REFLECTION
Texture2D q3_object_reflection_texture : register(t121);
#endif
// Generic world-coherent surface motion; geometry and source textures remain
// immutable. The caller supplies a visible or frozen presentation sample.
#ifndef Q3_WATER_TIME
#define Q3_WATER_TIME 0.0
#endif
#ifndef Q3_WATER_DRIFT
#define Q3_WATER_DRIFT float3(0,0,0)
#endif
float q3_natural_hash(float2 cell) {
 float period=Q3_MATERIAL_WRAP_WIDTH*.5*q3_source_repeat(.6);
 if(period>0)cell-=floor(cell/period)*period;
 return macro_decal_hash(cell);
}
float q3_natural_noise(float2 p) {
 float2 c=floor(p),f=frac(p);f=f*f*(3-2*f);
 return lerp(lerp(q3_natural_hash(c),q3_natural_hash(c+float2(1,0)),f.x),
  lerp(q3_natural_hash(c+float2(0,1)),q3_natural_hash(c+1),f.x),f.y);
}
float3 q3_natural_normal(PixelInput input) {
 float2 world=q3_source_world(input)+Q3_NATURAL_COORD_SHIFT;
 float2 patch_uv=world*q3_source_repeat(.6);
 float2 warp=float2(q3_natural_noise(patch_uv),q3_natural_noise(patch_uv+float2(7,13)))-.5;
 float3 drift=Q3_WATER_DRIFT;
 float2 uv0=world*float2(q3_source_repeat(.40),q3_source_repeat(.54))+warp*.16+drift.xy;
 float2 uv1=world*float2(q3_source_repeat(.95),q3_source_repeat(1.24))+warp*.12+float2(.27,.61)+float2(-drift.y,drift.z);
 float2 a=water_large_lean0_texture.Sample(material_sampler,uv0).rg*2-1;
 float2 b=water_small_lean0_texture.Sample(material_sampler,uv1).rg*2-1;
 float2 secondary_uv=float2(world.y,-world.x)*float2(q3_source_repeat(1.4),q3_source_repeat(1.82))+warp*.1+float2(drift.z,-drift.x);
 float2 c=water_small_secondary_lean0_texture.Sample(material_sampler,secondary_uv).rg*2-1;
 // Rotate the crossing detail slope vector back to the world basis too.
 c=float2(-c.y,c.x);
 // Gentle world-coherent variation; fine ripples dominate at gameplay zoom.
 float envelope=lerp(.35,1,smoothstep(.20,.78,warp.x+.5));
 float2 slope=(a*.45+b*.35+c*.20)*envelope;
 slope*=lerp(.20,1,smoothstep(.015,.32,input.hydrology_data.w));
 return normalize(float3(-slope,1));
}
float q3_water_glint(float3 normal,float3 view,float3 light) {
 float3 half_vector=normalize(view+light);
 float2 along=normalize(light.xy+float2(.00001,0));
 float2 difference=normal.xy-half_vector.xy;
 float cross_error=dot(difference,float2(-along.y,along.x));
 float along_error=dot(difference,along);
 // Narrow across the light path, softer along it; individual ripple normals
 // break the reflected streak into moving facets. No screen-space sparkle mask.
 return exp2(-160*cross_error*cross_error-12*along_error*along_error)*saturate(light.z);
}
float4 q3_natural_water(PixelInput input) {
 float depth=max(0,input.hydrology_data.w);
 float3 normal=q3_natural_normal(input);
 float3 view=normalize(float3(.43,-.43,1));
#ifdef Q3_WATER_CAMERA
 // A finite reflection eye concentrates glints along the shared light's path.
 // This is a material approximation, not a change to map projection or anchors.
 float2 delta=q3_source_world(input)-Q3_WATER_CAMERA.xy;
 float2 raw=float2(delta.x+delta.y,delta.x-delta.y);
 raw-=round(raw/max(Q3_WATER_CAMERA.zw,1))*Q3_WATER_CAMERA.zw;
 delta=float2(raw.x+raw.y,raw.x-raw.y)*.5;
 const float eye_height=2.5;
 view=normalize(float3(float2(.43,-.43)*eye_height-delta,eye_height));
#endif
 // The volume is lit on the mean water plane. Fine slopes change reflected
 // light, not the diffuse shading of an opaque corrugated surface.
 float3 bulk_light=q6_receiver_illumination(input,float3(0,0,1),1,1);
 float3 body=lerp(float3(.023,.074,.096),float3(.003,.015,.040),smoothstep(.18,.43,depth))*bulk_light;
 // The shared rig supplies .04 at noon and .12 at night. Treat this small
 // reflectance control as the base response, not another multiplier on .02.
 float f0=saturate(environment_water_fresnel);
 float fresnel=f0+(1-f0)*pow(1-saturate(dot(normal,view)),5);
 float3 ray=reflect(-view,normal);
 float sky_band=smoothstep(.30,.90,ray.y);
 float3 sky_light=environment_ambient_color*.6+environment_sun_color*environment_sun_intensity*.6
  +environment_moon_color*environment_moon_intensity*.6;
 float3 sky=sky_light*lerp(float3(.16,.25,.36),float3(.42,.55,.68),sky_band);
#ifdef Q3_OBJECT_REFLECTION
 // Same authoritative camera and water plane as the mirrored render target.
 // Sampling uses the linear offscreen image, never the display-tonemapped PNG.
 float2 reflected_uv=input.position.xy/Q3_REFLECTION_SIZE;
 float2 distortion=normal.xy*float2(3.0,1.5)/Q3_REFLECTION_SIZE;
 float4 object=q3_object_reflection_texture.Sample(decal_sampler,reflected_uv+distortion);
 float inside=step(0,reflected_uv.x)*step(reflected_uv.x,1)*step(0,reflected_uv.y)*step(reflected_uv.y,1);
 float object_coverage=saturate(object.a)*inside;
 sky=sky*(1-object_coverage)+object.rgb*inside;
#endif
 float2 world=q3_source_world(input)+Q3_NATURAL_COORD_SHIFT;
 float2 micro=water_small_lean0_texture.Sample(material_sampler,
  world*float2(q3_source_repeat(3.4),q3_source_repeat(4.12))+float2(.71,.29)+float2(-Q3_WATER_DRIFT.z,Q3_WATER_DRIFT.y)).rg*2-1;
 float3 glint_normal=normalize(normal+float3(-micro*.03,0));
 // Shared radiance and direction drive a spatially concentrated reflection.
 float3 glint=(environment_sun_color*environment_sun_intensity*q3_water_glint(glint_normal,view,environment_sun_direction)
  +environment_moon_color*environment_moon_intensity*q3_water_glint(glint_normal,view,environment_moon_direction))
  *3*environment_water_specular*q6_receiver_visibility(input,normal,1);
 float reflection=saturate(fresnel);
 float coverage=1-exp(-depth*lerp(2.3,3.2,smoothstep(.10,.32,depth)));
 float alpha=coverage+(1-coverage)*reflection;
 float3 premult=body*coverage*(1-reflection)+sky*reflection+glint;
 return float4(premult/max(alpha,.0001),alpha);
}
#endif

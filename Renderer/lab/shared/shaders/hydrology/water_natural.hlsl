#ifndef Q3_NATURAL_WATER_IMPL
#define Q3_NATURAL_WATER_IMPL
#ifndef Q3_NATURAL_COORD_SHIFT
#define Q3_NATURAL_COORD_SHIFT 0.0
#endif
#ifdef Q3_OBJECT_REFLECTION
Texture2D q3_object_reflection_texture : register(t121);
#endif
// Static Lab experiment: periodic source-detail patches and view-dependent
// reflection. This is a generic adaptation, not recovered Civ VI equations.
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
 float2 uv0=world*float2(q3_source_repeat(.36),q3_source_repeat(.48))+warp*.16;
 float2 uv1=world*float2(q3_source_repeat(.72),q3_source_repeat(.94))+warp*.12+float2(.27,.61);
 float2 a=water_large_lean0_texture.Sample(material_sampler,uv0).rg*2-1;
 float2 b=water_small_lean0_texture.Sample(material_sampler,uv1).rg*2-1;
 float2 secondary_uv=float2(world.y,-world.x)*float2(q3_source_repeat(1.12),q3_source_repeat(1.46))+warp*.1;
 float2 c=water_small_secondary_lean0_texture.Sample(material_sampler,secondary_uv).rg*2-1;
 // Rotate the crossing detail slope vector back to the world basis too.
 c=float2(-c.y,c.x);
 // Broad calm lanes interrupt the source pattern without per-tile phases.
 float envelope=lerp(.16,1,smoothstep(.20,.78,warp.x+.5));
 float2 slope=(a*.40+b*.38+c*.22)*envelope;
 slope*=lerp(.20,1,smoothstep(.015,.32,input.hydrology_data.w));
 return normalize(float3(-slope,1));
}
float4 q3_natural_water(PixelInput input) {
 float depth=max(0,input.hydrology_data.w);
 float3 normal=q3_natural_normal(input);
 float3 view=normalize(float3(0,-.52,.86));
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
  world*float2(q3_source_repeat(3.4),q3_source_repeat(4.12))+float2(.71,.29)).rg*2-1;
 float sparkle=lerp(.22,1,smoothstep(.025,.16,length(micro)));
 float3 sunhalf=normalize(view+environment_sun_direction);
 float3 moonhalf=normalize(view+environment_moon_direction);
 float3 glint=(environment_sun_color*environment_sun_intensity*pow(saturate(dot(normal,sunhalf)),48)
  +environment_moon_color*environment_moon_intensity*pow(saturate(dot(normal,moonhalf)),48))
  *sparkle*.045*environment_water_specular*q6_receiver_visibility(input,normal,1);
 float reflection=saturate(fresnel);
 float coverage=1-exp(-depth*lerp(2.3,3.2,smoothstep(.10,.32,depth)));
 float alpha=coverage+(1-coverage)*reflection;
 float3 premult=body*coverage*(1-reflection)+sky*reflection+glint;
 return float4(premult/max(alpha,.0001),alpha);
}
#endif

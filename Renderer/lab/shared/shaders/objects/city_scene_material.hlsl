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
#ifdef Q8_CITY_LOOK
// Pack-selected readability response for lit bodies: (gain, contrast,
// saturation) offsets. Zero is the identity, so earlier packs are unchanged.
// Contrast is a power about a dim linear pivot: sunlit faces brighten and
// shaded faces deepen, as in Civ III's pre-lit city art.
float3 q8_city_look(float3 lit,float3 look,float3 base,float pale) {
 if(!any(look))return lit;
 float3 weights=float3(.2126,.7152,.0722);
 // Bright albedo (whitewash, white roofs, orange tile and thatch) already
 // reads; it takes less of the lift so it keeps detail beside timber, brick
 // and stone. The brightest channel catches saturated roofs as well as white.
 look.xy*=1-pale*smoothstep(.2,.65,max(base.r,max(base.g,base.b)));
 float luminance=max(dot(lit,weights),1e-5);
 float target=.1*pow(luminance/.1,1+look.y)*(1+look.x);
 // A soft shoulder keeps whitewash, plaster and glass detailed while darker
 // timber, thatch and stone receive the full lift.
 if(target>.42)target=.42+(target-.42)/(1+(target-.42)/.4);
 lit*=target/luminance;
 float grey=dot(lit,weights);
 return max(0,grey+(lit-grey)*(1+look.z));
}
#endif
#ifdef Q8_CITY_EMISSION_LOOK
// Pack-selected window response: (shoulder, gain offset). Zero is the
// identity. The shoulder keeps the brightest windows coloured instead of
// clipping to white while dim windows keep their glow.
float3 q8_city_emission(float3 emission,float2 look) {
 if(!any(look))return emission;
 emission*=1+look.y;
 if(look.x<=0)return emission;
 float luminance=max(dot(emission,float3(.2126,.7152,.0722)),1e-5);
 return emission*(1/(1+luminance/look.x));
}
#endif
#ifdef Q8_CITY_TIME
// Procedural attached effects on camera-facing quads: flame (90), smoke (91)
// and night light (92). p.uv is the quad coordinate (x -1..1 across, y 0..1
// up); seed and strength ride in the auxiliary coordinates. The phase is a
// pure function of the visual clock and the seed: no state, no catch-up.
float q8_effect_hash(float n){return frac(sin(n)*43758.5453);}
float q8_effect_noise(float2 x){
 float2 i=floor(x),f=frac(x);f=f*f*(3-2*f);float n=i.x+i.y*57;
 return lerp(lerp(q8_effect_hash(n),q8_effect_hash(n+1),f.x),lerp(q8_effect_hash(n+57),q8_effect_hash(n+58),f.x),f.y);
}
float4 q8_effect_over(float4 top,float4 under){
 // Premultiplied "over", returned straight for q6_scene_output.
 float a=top.a+under.a*(1-top.a);
 return float4((top.rgb*top.a+under.rgb*under.a*(1-top.a))/max(a,1e-4),a);
}
float4 q8_city_effect(FeaturePixelInput p){
 float kind=round(p.material_index-90),seed=clamp(p.city_ao_uv.x,0,64),strength=clamp(p.city_ao_uv.y,0,2);
 float t=Q8_CITY_TIME+seed*17.31,night=saturate(environment_night_activation);
 // Multisampled edges extrapolate the quad coordinate beyond the quad; keep
 // it inside so no pow() sees a negative base (NaN feeds bloom as stars).
 float2 q=float2(clamp(p.uv.x,-1,1),saturate(p.uv.y));
 // Quad units per pixel along each screen axis keep shapes round at any zoom.
 float2 pixel=float2(max(abs(ddx(p.uv.x)),1e-5),max(abs(ddy(p.uv.y)),1e-5));
 if(kind<.5){
  float v=(q.y-.3)/.7,flicker=.85+.15*sin(t*11+seed*6.3)+.12*(q8_effect_noise(float2(t*6,seed*9))-.5);
  float tip=.78*flicker,sway=(q8_effect_noise(float2(t*2.7+v*1.5,seed*5))-.5)*.4*saturate(v);
  float radius=.55*pow(saturate(1-v/max(tip,1e-3)),.8)*sqrt(saturate(v*5+.05));
  float body=saturate(1-abs(q.x-sway)/max(radius,1e-3))*step(0,v)*step(v,tip);
  float core=smoothstep(.35,1,body);
  // Kept near display range: brighter values bloom into a glow that hides
  // the flame's shape.
  float lift=(1.15+.45*night)*strength;
  // By day a saturated orange core separates the flame from tan roofs; at
  // night it whitens and the halo carries it.
  float3 hot=lerp(float3(1,.66,.24),float3(1,.86,.48),night);
  float4 flame=float4(lerp(float3(1,.3,.04),hot,core)*lift,saturate(body*2.2));
  float2 h=float2(q.x,(q.y-.38)*1.6);
  float4 halo=float4(float3(1,.5,.14)*lift*.6,exp(-dot(h,h)*5)*(.1+.16*night)*strength);
  float4 result=q8_effect_over(flame,halo);
  clip(result.a-.004);return result;
 }
 if(kind<1.5){
  // Billowing plume: puffs leave the mouth continuously, rise along a
  // wind-bent path, grow, erode and thin; a short stem keeps the plume on
  // its chimney. Units are quad half-widths on both axes so puffs stay round
  // at any zoom, and the upper-left of each puff is sunlit. At night the
  // smoke darkens and the furnace lights its base.
  float H=pixel.x/pixel.y,Y=q.y*H;
  float wind=.45+.3*q8_effect_noise(float2(seed*3,t*.1));
  float alpha=0,tone=0,weight=0;
  [unroll]for(int k=0;k<9;k++){
   float cycle=t/2.6+k/9.0,phase=frac(cycle);
   float h=pow(saturate(phase),.85)*.82,r=.3+.55*h;
   float2 c=float2(wind*pow(h,1.4)+(q8_effect_noise(float2(seed*7+k,floor(cycle)))-.5)*.25*h,h*H+r*.6);
   float2 d=(float2(q.x,Y)-c)/r;
   float erode=(.65*q8_effect_noise(float2(q.x*2.4+k*3.7+seed*11,Y*2.4-t*.8))
    +.35*q8_effect_noise(float2(q.x*5.1+k,Y*5.1-t*1.3))-.5)*.7;
   float a=smoothstep(1,.45,length(d)+erode)*smoothstep(0,.06,phase)*pow(saturate(1-phase),1.1)*.8;
   alpha=1-(1-alpha)*(1-a);
   tone+=(.13+.36*saturate(.5-.45*d.x+.55*d.y-erode)+.08*h)*a;weight+=a;
  }
  float stem=smoothstep(1,.35,abs(q.x-wind*pow(q.y,1.4))/(.18+.5*q.y))*smoothstep(.22,0,q.y)*smoothstep(0,.015,q.y)*.75;
  alpha=saturate((1-(1-alpha)*(1-stem))*strength)*lerp(1,.75,night);
  clip(alpha-.004);
  float grey=saturate((tone+.22*stem)/max(weight+stem,1e-4))*lerp(1,.18,night);
  float glow=exp(-q.y*14)*night*.7*strength;
  return float4(grey+glow,grey*.98+glow*.42,grey*.96+glow*.12,alpha);
 }
 float2 g=float2(q.x,(q.y-.5)*2);
 float glow=exp(-dot(g,g)*3)*night*strength;
 clip(glow-.004);
 return float4(float3(1,.72,.38)*(1.5+2*glow),saturate(glow*.85));
}
#endif
Q6SceneOutput Q8_CITY_FEATURE_ENTRY(FeaturePixelInput p) {
 if(p.material_index<39.5)return Q8LegacyPSFeature(p);
#ifdef Q8_CITY_TIME
 if(p.material_index>=89.5 && p.material_index<99.5) {
  float4 effect=q8_city_effect(p);
  // A non-finite sample would bloom into a star; drop it.
  if(any(!isfinite(effect)))discard;
  return q6_scene_output(float4(clamp(effect.rgb,0,4),saturate(effect.a)));
 }
#endif
 if(p.material_index>=59.5 && p.material_index<69.5) {
  float4 ground=city_base_texture_0.Sample(decal_sampler,p.uv);
  float3 ground_normal=normalize(p.geometry_normal);
  float3 ground_tangent=float3(1,0,0),ground_bitangent=float3(0,1,0);
  float2 ground_slope=float2(0,0);
  if(p.material_index>63.5)
   ground_normal=q8_settlement_ground_normal(p,ground_tangent,ground_bitangent,ground_slope);
  float3 ground_lit=ground.rgb*q6_receiver_illumination(p,ground_normal,1,1);
#if Q8_CITY_SOURCE_SPECULAR
  if(p.material_index>63.5) {
   float2 folded=frac(p.uv*.5)*2-1;
   float2 uv=q8_settlement_ground_uv(p,folded);
   float2 ux=q8_settlement_ground_gradient(p,folded,ddx(p.uv));
   float2 uy=q8_settlement_ground_gradient(p,folded,ddy(p.uv));
   float3 roughness=resource_base_texture_1.SampleGrad(decal_sampler,uv,ux,uy).rgb;
   ground_lit+=q8_city_direct_specular(environment_sun_direction,
    environment_sun_color*environment_sun_intensity,ground_normal,
    normalize(p.geometry_normal),ground_tangent,ground_bitangent,
    ground_slope,roughness,ground.rgb,0);
  }
#endif
  return q6_scene_output(float4(ground_lit,ground.a));
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
 if(emission_only) {
  float3 glow=emission*environment_night_activation*environment_emissive_scale*Q8_CITY_EMISSIVE_GAIN;
#ifdef Q8_CITY_EMISSION_LOOK
  glow=q8_city_emission(glow,Q8_CITY_EMISSION_LOOK);
#endif
  return q6_scene_output(float4(glow,1));
 }
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
#ifdef Q8_CITY_LOOK
 // The daylight response fades out at night so lit windows, not brightened
 // moonlit walls, carry the city after dark.
 lit=q8_city_look(lit,Q8_CITY_LOOK*(1-saturate(environment_night_activation)),base,Q8_CITY_LOOK_PALE);
#endif
 if(!Q8_CITY_SEPARATE_EMISSION)
  lit+=emission*environment_night_activation*environment_emissive_scale*Q8_CITY_EMISSIVE_GAIN;
 return q6_scene_output(float4(lit,1));
}

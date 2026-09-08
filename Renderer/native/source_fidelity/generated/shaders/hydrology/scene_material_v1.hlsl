#ifndef Q3_SCENE_MATERIAL_V1
#define Q3_SCENE_MATERIAL_V1
// Source-composed static material. Raw scene-linear RGB and straight coverage;
// Q6 wrapper premultiplies once. The Q0 CPU hook supplies exact continuous fields.
// hydrology_data = positive-land distance, beach width, rocky fraction, depth.
#ifndef Q3_MATERIAL_ORIGIN_X
#define Q3_MATERIAL_ORIGIN_X 0
#define Q3_MATERIAL_ORIGIN_Y 0
#define Q3_MATERIAL_WRAP_WIDTH 0
#endif
// Captured world dimensions/flags; b1 remains native viewport, b2 shadows.
cbuffer C3XWorldSettings : register(b3) {
 float4 c3x_world_dimensions; // width, height, wrap-x, wrap-y
 float4 c3x_world_material; // common repeat period, reserved
};
float2 q3_source_world(PixelInput input) {
 float2 world=input.macro_uv*2;
 float2 raw=float2(world.x+world.y,world.x-world.y);
 if(c3x_world_dimensions.z>.5)
  raw.x-=floor(raw.x/c3x_world_dimensions.x)*c3x_world_dimensions.x;
 if(c3x_world_dimensions.w>.5)
  raw.y-=floor(raw.y/c3x_world_dimensions.y)*c3x_world_dimensions.y;
 return float2(raw.x+raw.y,raw.x-raw.y)*.5;
}
float q3_source_repeat(float requested) {
 float period=c3x_world_material.x;
 return period>0?round(requested*period)/period:requested;
}
float4 q3_authored_bed_detail(PixelInput input) {
 float2 projected=q3_source_world(input)/1.0+float2(.29,.53);
 float variant=floor(macro_decal_hash(floor(projected))*4);
 float2 uv=coast_clutter_atlas_uv(frac(projected),variant);
 float4 detail=water_decal_base_texture.Sample(decal_sampler,uv);
 detail.a*=projected_decal_edge_fade(frac(projected))
  *(1-smoothstep(.24,.43,input.hydrology_data.w));
 return detail;
}
float3 q3_authored_bed_normal(PixelInput input) {
 float2 projected=q3_source_world(input)/1.0+float2(.29,.53);
 float variant=floor(macro_decal_hash(floor(projected))*4);
 float2 uv=coast_clutter_atlas_uv(frac(projected),variant);
 float dx=water_decal_height_texture.Sample(decal_sampler,uv+float2(.001,0)).r
  -water_decal_height_texture.Sample(decal_sampler,uv-float2(.001,0)).r;
 float dy=water_decal_height_texture.Sample(decal_sampler,uv+float2(0,.001)).r
  -water_decal_height_texture.Sample(decal_sampler,uv-float2(0,.001)).r;
 float support=q3_authored_bed_detail(input).a;
 return normalize(float3(-dx*14*support,-dy*14*support,1));
}
float3 q3_scene_bed(PixelInput input) {
 float sd=input.hydrology_data.x,rocky=saturate(input.hydrology_data.z);
 float2 uv=q3_source_world(input)*q3_source_repeat(.75);
 float3 sand=beach_base_texture.Sample(material_sampler,uv).rgb;
 float desert=saturate(input.material_weights.z);
 float3 desert_sand=desert_base_texture.Sample(material_sampler,input.uv).rgb;
 sand=lerp(sand,desert_sand,desert);
 float3 bed=shallow_bed_texture.Sample(material_sampler,uv).rgb;
 float2 world=q3_source_world(input);
 float4 authored=q3_authored_bed_detail(input);
 bed=lerp(bed,authored.rgb,authored.a);
 bed*=1+(sample_water_clutter_height(world)-.5)*authored.a*.30;
 float3 rock=cliff_base_texture.Sample(material_sampler,uv).rgb;
 float3 rim=lerp(sand,bed,authored.a*desert*.34);
 float3 color=lerp(rim,bed,smoothstep(0,.40,-sd));
#ifdef Q3_COAST_DETAIL
 color=lerp(rim,bed,smoothstep(0,lerp(.12,.34,desert),-sd));
#endif
 color=lerp(color,lerp(rock,bed,smoothstep(0,.70,-sd)),rocky);
 color*=lerp(.72,1.0,smoothstep(0,.32,-sd));
 float height=water_height_texture.Sample(material_sampler,uv).r;
 // Confirmed source height detail; no animated or inferred wave channels.
 // Spectral absorption tints the actual bed before coverage compositing.
 // This preserves authored contrast in shallows without a beige offshore plate.
 float3 absorption=exp(-input.hydrology_data.w*float3(14,7,3));
 return color*clamp(.86+(height-.426)*.55,.65,1.1)*absorption;
}
void q3_shore_material(PixelInput input,float2 world_position,inout float3 albedo,inout float3 material_normal) {
 float sd=input.hydrology_data.x,width=input.hydrology_data.y;
 float rocky=saturate(input.hydrology_data.z);
 float desert=saturate(input.material_weights.z);
 float3 sand=beach_base_texture.Sample(material_sampler,q3_source_world(input)*q3_source_repeat(.75)).rgb;
 float grain=dot(sand,float3(.2126,.7152,.0722));
 float blend=1-smoothstep(width*.25,width+.06,sd+(grain-.28)*.24);
 // Desert already supplies a continuous sand surface. Replacing it with the
 // separate beach material created a flat, darker ribbon beside the desert's
 // fine ripples. Preserve that sand family through the wet edge; mixed biome
 // weights reduce the beach replacement continuously rather than branching by
 // tile ownership.
 float beach_material=blend*(1-rocky)*(1-desert);
 albedo=lerp(albedo,sand,beach_material);
 float2 uv=q3_source_world(input)*q3_source_repeat(.75);
 float hx=beach_height_texture.Sample(material_sampler,uv+float2(.002,0)).r
  -beach_height_texture.Sample(material_sampler,uv-float2(.002,0)).r;
 float hy=beach_height_texture.Sample(material_sampler,uv+float2(0,.002)).r
  -beach_height_texture.Sample(material_sampler,uv-float2(0,.002)).r;
 float2 detail=clamp(float2(-hx-hy,-hx+hy)*12,-.18,.18)*beach_material;
 material_normal=normalize(float3(material_normal.xy+detail*material_normal.z,material_normal.z));
 float3 rock=cliff_base_texture.Sample(material_sampler,q3_source_world(input)*q3_source_repeat(.75)).rgb;
 albedo=lerp(albedo,rock,rocky*(1-smoothstep(.03,.22,sd)));
 albedo*=1-.28*(1-smoothstep(-.02,.12,sd));
}
float4 q3_water_material(PixelInput input) {
 float kind=input.surface_kind;
#ifdef Q3_BED_ONLY
 if(kind>4.5&&kind<5.5){clip(-1);return 0;}
#endif
 // Captured surf/foam is deferred, never retained as permanent pale geometry.
 if(kind>5.5&&kind<6.5){clip(-1);return 0;}
 float sd=input.hydrology_data.x,depth=max(0,input.hydrology_data.w);
 float3 normal=float3(0,0,1);
 float source_roughness=1;
#ifdef Q3_SOURCE_WATER_NORMALS
 if(kind>4.5&&kind<5.5){
  // Static source surface phase, sampled in the same wrapped world basis as
  // the bed. Source slopes/moments drive lighting; this is a C3X adaptation,
  // not a recovered source-engine LEAN or wave animation equation.
  float2 world=q3_source_world(input);
  float2 large_uv=world*float2(q3_source_repeat(.36),q3_source_repeat(.47));
  float2 small_uv=world*float2(q3_source_repeat(2.4),q3_source_repeat(3.05))+float2(.31,.17);
  float2 large=water_large_lean0_texture.Sample(material_sampler,large_uv).rg*2-1;
  float2 small=water_small_lean0_texture.Sample(material_sampler,small_uv).rg*2-1;
  float2 variance=water_large_lean1_texture.Sample(material_sampler,large_uv).rg
   +water_small_lean1_texture.Sample(material_sampler,small_uv).rg;
  float2 lean=large*.64+small*.24;
  normal=normalize(float3(-lean,1));
  source_roughness=rcp(1+dot(variance,float2(2,2)));
 }
#endif
#ifdef Q3_WATER_EFFECTS
 if(kind>4.5&&kind<5.5)normal=q3_effect_normal(input,normal);
#endif
 float3 illumination=q6_receiver_illumination(input,normal,1,1);
 if(kind>8.5&&kind<9.5){
  // The default keeps the frozen analytic distance. Q3_CONTINUOUS_RIVERS
  // consumes the opt-in shared corridor, including its terminal presentation.
  float distance_pixels=input.river_data.x;
#ifdef Q3_CONTINUOUS_RIVERS
  float2 world=q3_source_world(input);
  float2 uv=world*q3_source_repeat(.75);
  float noise=river_bank_noise_texture.Sample(material_sampler,
   world*float2(q3_source_repeat(.31),q3_source_repeat(.47))+float2(.19,.37)).r-.5;
  float sediment_noise=river_bank_noise_texture.Sample(material_sampler,
   world*float2(q3_source_repeat(.43),q3_source_repeat(.61))+float2(.61,.11)).r;
  float gravel_noise=river_bank_noise_texture.Sample(material_sampler,
   world*float2(q3_source_repeat(6.71),q3_source_repeat(8.93))+float2(.07,.73)).r;
  float grain=river_height_texture.Sample(material_sampler,uv).r;
  float water_width=5.8+noise*.8;
  float water=1-smoothstep(water_width-1.2,water_width,distance_pixels);
  float bank_width=12.2+noise*4.8+(sediment_noise-.5)*2.4+(grain-.5)*1.2;
  // Feather only the bank coverage: the sand retains its high-frequency
  // material detail while the underlying grassland/plains resolves gradually.
  float bank_feather=4.2+sediment_noise*.8;
  float bank_edge_distance=distance_pixels+(gravel_noise-.5)*.7;
  float bank=1-smoothstep(bank_width-bank_feather,bank_width+.6,bank_edge_distance);
  // Banks end at the optical shore; the water itself overlaps and dissolves
  // into the existing sea surface instead of ending in an offshore capsule.
  float land_bank=smoothstep(-.025,.065,sd);
  float outlet=smoothstep(-.20,.025,sd);
  bank*=outlet;clip(bank-.001);
  water=lerp(1,water,land_bank);
  float3 bed=river_base_texture.Sample(material_sampler,uv).rgb;
  float3 sand=beach_base_texture.Sample(material_sampler,uv).rgb;
  float3 fine_sand=beach_base_texture.Sample(material_sampler,
   world*float2(q3_source_repeat(2.17),q3_source_repeat(2.63))+float2(.37,.59)).rgb;
  float4 clutter=river_clutter_base_texture.Sample(decal_sampler,frac(world*.5+float2(.23,.41)));
  float bank_position=saturate((distance_pixels-water_width)/max(1,bank_width-water_width));
  float sediment=smoothstep(.30,.70,sediment_noise*.52+(noise+.5)*.32+grain*.16);
  float3 textured_sand=lerp(sand,fine_sand,.46);
  float3 dry_sand=lerp(bed,lerp(textured_sand,float3(.50,.39,.20),.12),.76)
   *(0.82+(grain-.5)*.26);
  float3 bank_soil=lerp(bed,float3(.22,.15,.07),.42)*(0.60+(grain-.5)*.18);
  float sand_patch=saturate(sediment*.86+smoothstep(.48,.94,bank_position)*.24);
  float3 dry=lerp(bank_soil,dry_sand,sand_patch)*lerp(.90,1.08,sediment);
  float gravel=smoothstep(.49,.76,gravel_noise)*clutter.a
   *smoothstep(.10,.58,bank_position);
  dry=lerp(dry,clutter.rgb*.82,gravel*.60);
  float pebble=smoothstep(.68,.84,gravel_noise)*smoothstep(.12,.68,bank_position);
  float3 pebble_color=lerp(float3(.16,.12,.07),textured_sand*1.18,sediment);
  dry=lerp(dry,pebble_color,pebble*.34);
  dry*=.91+gravel_noise*.16;
  float wet=1-smoothstep(water_width,bank_width-1.0,distance_pixels);
  float damp_breakup=saturate(.72+noise*.38+(gravel_noise-.5)*.20);
  float3 shore=lerp(dry,dry*lerp(.43,.58,damp_breakup),wet);
  float optical_depth=.10+.32*(1-smoothstep(0,5.5,max(0,distance_pixels)));
  float3 transmitted=bed*exp(-optical_depth*float3(8,4,2));
  float3 river=lerp(transmitted,float3(.018,.044,.052),1-exp(-optical_depth*5));
  float2 lean=river_lean0_texture.Sample(material_sampler,
    world*float2(q3_source_repeat(.92),q3_source_repeat(1.27))).rg*2-1;
  float3 river_normal=normalize(float3(-lean*.24,1));
  float3 water_light=q6_receiver_illumination(input,river_normal,1,1);
  float3 bank_light=q6_receiver_illumination(input,normalize(input.geometry_normal),1,1);
  return float4(lerp(shore*bank_light,river*water_light,water),bank);
#elif defined(Q3_STATIC_OPTICS_V2)
  // Keep the captured curve and navigable width. The source river bed remains
  // visible through shallow edges; narrow damp banks replace the sandy outline.
  float water=1-smoothstep(4.6,6.0,distance_pixels);
  float bank=1-smoothstep(6.0,7.4,distance_pixels);clip(bank-.001);
  float2 uv=q3_source_world(input)*q3_source_repeat(.75);
  float3 bed=river_base_texture.Sample(material_sampler,uv).rgb;
  float3 damp=lerp(bed,beach_base_texture.Sample(material_sampler,uv).rgb,.25)*.48;
  float optical_depth=.10+.32*(1-smoothstep(0,5.5,distance_pixels));
  float3 transmitted=bed*exp(-optical_depth*float3(8,4,2));
  float3 river=lerp(transmitted,float3(.009,.060,.075),1-exp(-optical_depth*5));
  return float4(lerp(damp,river,water)*illumination,bank);
#else
  float water=1-smoothstep(4.6,6.0,distance_pixels);
  float bank=1-smoothstep(6.0,9.0,distance_pixels);clip(bank-.001);
  float3 bed=river_base_texture.Sample(material_sampler,input.uv).rgb*.72;
  return float4(lerp(bed,float3(.065,.125,.155),water*.78)*illumination,bank*.96);
#endif
 }
 clip(-sd-.0001);
 if(kind<4.5)return float4(q3_scene_bed(input)*q6_receiver_illumination(input,q3_authored_bed_normal(input),1,1),1);
#ifdef Q3_NATURAL_WATER
 return q3_natural_water(input);
#endif
 // Optical absorption over separately shaded authored bed; no opaque water plate.
 float alpha=1-exp(-depth*3.2);
 float3 tint=lerp(float3(.023,.074,.096),float3(.003,.015,.040),smoothstep(.18,.43,depth))*illumination;
 float3 view=normalize(float3(0,-.52,.86));
 float fresnel=.02+.98*pow(1-saturate(dot(normal,view)),5);
 float3 reflection=environment_ambient_color*.2;
 float3 sunhalf=normalize(view+environment_sun_direction);
 float3 moonhalf=normalize(view+environment_moon_direction);
 float3 glint=environment_sun_color*environment_sun_intensity*pow(saturate(dot(normal,sunhalf)),180)
  +environment_moon_color*environment_moon_intensity*pow(saturate(dot(normal,moonhalf)),180);
 tint+=reflection*fresnel*environment_water_fresnel+glint*.12*source_roughness*environment_water_specular
  *q6_receiver_visibility(input,normal,1);
#ifdef Q3_WATER_EFFECTS
 q3_effect_color(input,normal,illumination,tint,alpha);
#endif
 return float4(tint,alpha);
}
#endif

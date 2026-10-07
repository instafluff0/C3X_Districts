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
float2 q3_source_world(PixelInput input) {
 float2 world=input.macro_uv*2+float2(Q3_MATERIAL_ORIGIN_X,Q3_MATERIAL_ORIGIN_Y);
 if(Q3_MATERIAL_WRAP_WIDTH>0){
  float rawx=world.x+world.y,rawy=world.x-world.y;
  rawx-=floor(rawx/max(1,Q3_MATERIAL_WRAP_WIDTH))*Q3_MATERIAL_WRAP_WIDTH;
  world=float2(rawx+rawy,rawx-rawy)*.5;
 }
 return world;
}
float q3_source_repeat(float requested) {
 float period=Q3_MATERIAL_WRAP_WIDTH*.5;
 return period>0?round(requested*period)/period:requested;
}
#ifdef Q3_WATER_EFFECTS
#include "water_effects.hlsl"
#endif
#ifdef Q3_NATURAL_WATER
#include "water_natural.hlsl"
#endif
float4 q3_authored_bed_detail(PixelInput input) {
 float2 world=q3_source_world(input);
 // Both confirmed source layers matter: broader ocean-rock forms plus cracks.
 float4 ocean=sample_water_clutter(world);
 float4 cracks=sample_coast_clutter(world);
 cracks.a*=1-smoothstep(.16,.55,input.hydrology_data.w);
 float alpha=cracks.a+ocean.a*(1-cracks.a);
 float3 color=(cracks.rgb*cracks.a+ocean.rgb*ocean.a*(1-cracks.a))/max(alpha,.0001);
 alpha*=1-smoothstep(.30,.60,input.hydrology_data.w);
 return float4(color,alpha);
}
// One world-anchored fine shore field continues from dry margins into shallows.
// Atlas cells are source-authored; scale, placement and shading are C3X choices.
float q3_surface_grain(float detail,float neighborhood) {
 // The installed terrain base-color alpha contains the fine material pattern;
 // its separate displacement height is nearly flat. Reconstruct local material
 // contrast from that pattern, without treating it as surface transparency.
 return clamp((detail-neighborhood)*3.5,-.45,.80);
}
float q3_margin_patch(float2 world) {
 float patch=river_bank_noise_texture.Sample(material_sampler,
  world*float2(q3_source_repeat(.63),q3_source_repeat(.81))+float2(.41,.13)).r;
 return smoothstep(.28,.72,patch);
}
float q3_margin_visibility(float height,float neighborhood) {
 return 1-.46*saturate((neighborhood-height)*9);
}
float4 q3_margin_detail(float2 world,out float height,out float cavity) {
 float2 projected=world*q3_source_repeat(1.785714)+float2(.29,.53);
 float variant=floor(macro_decal_hash(floor(projected))*4);
 // Four authored gravel patches; retain each cell's transparent perimeter.
 float2 cell=float2(fmod(variant,2),floor(variant*.5));
 float2 uv=(frac(projected)+cell)*.5;
 float4 detail=river_clutter_base_texture.Sample(decal_sampler,uv);
 float raw_height=river_clutter_height_texture.Sample(decal_sampler,uv).r;
 float mean_height=river_clutter_height_texture.SampleBias(decal_sampler,uv,2).r;
 detail.a*=projected_decal_edge_fade(frac(projected));
 height=lerp(.5,raw_height,detail.a);
 cavity=lerp(1,q3_margin_visibility(raw_height,mean_height),detail.a);
 return detail;
}
float3 q3_margin_normal(PixelInput input,float height,float strength) {
 float3 n=normalize(input.geometry_normal);
 float3 world=input.q6_world.xyz*float3(1,-1,1);
 float3 dx=ddx(world),dy=ddy(world);
 float3 r1=cross(dy,n),r2=cross(n,dx);
 float determinant=dot(dx,r1);
 float3 gradient=(ddx(height)*r1+ddy(height)*r2)*sign(determinant)/max(abs(determinant),.000001);
 gradient/=max(1,length(gradient)*.10);
 return normalize(n-gradient*strength);
}
float q3_coast_irregular_region(float2 world) {
 // Two low-frequency source-noise views choose broad, continuous stretches;
 // no tile or decal owns a patch. The blend softens their boundaries.
 float2 uv0=world*float2(q3_source_repeat(.022),q3_source_repeat(.029))
  +float2(.19,.53);
 float2 uv1=world*float2(q3_source_repeat(.014),q3_source_repeat(.017))
  +float2(.61,.13);
 float broad=river_bank_noise_texture.Sample(material_sampler,uv0).r;
 float macro=river_bank_noise_texture.Sample(material_sampler,uv1).r;
 return smoothstep(.44,.56,broad*.75+macro*.25);
}

float3 q3_authored_bed_normal(PixelInput input) {
 float2 world=q3_source_world(input);
 float4 ocean=sample_water_clutter(world);
 float4 cracks=sample_coast_clutter(world);
 float2 projected=world/1.20+float2(.29,.53);
 float variant=floor(macro_decal_hash(floor(projected))*4);
 float2 uv=coast_clutter_atlas_uv(frac(projected),variant);
 float crack_height=water_decal_height_texture.Sample(decal_sampler,uv).r-.5;
 float height=(sample_water_clutter_height(world)-.5)*ocean.a;
 float crack_coverage=cracks.a*(1-smoothstep(.16,.55,input.hydrology_data.w));
 height=lerp(height,crack_height,crack_coverage);
 height*=1-smoothstep(.30,.60,input.hydrology_data.w);
 float coast_family=1-smoothstep(.34,.63,saturate(input.surface_coordinate));
 float2 bed_uv=world*q3_source_repeat(.75);
 float bed_alpha=shallow_bed_texture.SampleBias(material_sampler,bed_uv,2).a;
 float3 decal_normal=q3_margin_normal(input,height,.045);
 // Desert's continuous authored sand height, transferred beneath coast water.
 float2 sand_uv=world*q3_source_repeat(.26)+float2(.31,.17);
 float sand_height=desert_height_texture.Sample(material_sampler,sand_uv).r;
 float shelf=coast_family*(1-smoothstep(.18,.55,input.hydrology_data.w));
 float2 broad_uv=float2(world.y,-world.x)*q3_source_repeat(.115)+float2(.43,.61);
 float broad_height=desert_hills_height_texture.Sample(material_sampler,broad_uv).r;
 float3 continuous_normal=q3_margin_normal(input,
  sand_height*.38+bed_alpha*.16+broad_height*.46,.55);
 float3 irregular_normal=normalize(lerp(decal_normal,continuous_normal,shelf));
 float control_alpha=shallow_bed_texture.Sample(material_sampler,bed_uv).a;
 float3 control_detail=q3_margin_normal(input,control_alpha,.025);
 float3 control_normal=normalize(lerp(decal_normal,control_detail,coast_family));
 return normalize(lerp(control_normal,irregular_normal,
  q3_coast_irregular_region(world)));
}
float3 q3_scene_bed(PixelInput input) {
 float sd=input.hydrology_data.x,rocky=saturate(input.hydrology_data.z);
 float2 uv=q3_source_world(input)*q3_source_repeat(.75);
 float4 beach=beach_base_texture.Sample(material_sampler,uv);
 float beach_grain=q3_surface_grain(beach.a,
  beach_base_texture.SampleBias(material_sampler,uv,3).a);
 float3 sand=beach.rgb*(1+beach_grain);
 float desert=saturate(input.material_weights.z);
 float3 desert_sand=desert_base_texture.Sample(material_sampler,input.uv).rgb;
 sand=lerp(sand,desert_sand,desert);
 float coast_family=1-smoothstep(.34,.63,saturate(input.surface_coordinate));
 float4 shallow=shallow_bed_texture.Sample(material_sampler,uv);
 float3 bed=lerp(shallow.rgb,shallow_bed_texture.SampleLevel(material_sampler,uv,10).rgb,coast_family);
 float2 world=q3_source_world(input);

 float4 authored=q3_authored_bed_detail(input);
 authored.a*=1-coast_family;
 float2 second_uv=float2(uv.y,-uv.x)*.47+float2(.19,.37);
 float second_alpha=shallow_bed_texture.Sample(material_sampler,second_uv).a;
 float shelf=coast_family*(1-smoothstep(.16,.58,input.hydrology_data.w));
 float structure=clamp((shallow.a-.34)*3.0+(second_alpha-.34)*1.6,-.42,.58);
 bed*=1+structure*.22*shelf;
 bed=lerp(bed,authored.rgb,authored.a);
 bed*=1+(sample_water_clutter_height(world)-.5)*authored.a*.30;
 float3 rock=cliff_base_texture.Sample(material_sampler,uv).rgb;
 float fine_height,fine_cavity;
 float4 fine=q3_margin_detail(world,fine_height,fine_cavity);
 float margin=(1-smoothstep(.08,.38,input.hydrology_data.w))*lerp(.25,1.0,q3_margin_patch(world))*(1-coast_family);
 float3 rim=lerp(sand,bed,authored.a*desert*.34);
 float3 color=lerp(rim,bed,smoothstep(0,.40,-sd));
#ifdef Q3_COAST_DETAIL
 color=lerp(rim,bed,smoothstep(0,lerp(.12,.34,desert),-sd));
#endif
 color=lerp(color,lerp(rock,bed,smoothstep(0,.70,-sd)),rocky);
 color*=lerp(.72,1.0,smoothstep(0,.32,-sd));
 color=lerp(color,fine.rgb,fine.a*margin*lerp(1,.35,desert));
 color*=lerp(1,fine_cavity,margin);
 float height=water_height_texture.Sample(material_sampler,uv).r;
 // Confirmed source height detail; no animated or inferred wave channels.
 // Spectral absorption tints the actual bed before coverage compositing.
 // This preserves authored contrast in shallows without a beige offshore plate.
 float coast_shelf=(1-smoothstep(.19,.65,input.hydrology_data.w))
  *(1-smoothstep(.34,.68,input.surface_coordinate));
 color*=1+beach_grain*.32*coast_shelf;
 float2 sand_uv=world*q3_source_repeat(.26)+float2(.31,.17);
 float sand_height=desert_height_texture.Sample(material_sampler,sand_uv).r;
 float sand_mean=desert_height_texture.SampleBias(material_sampler,sand_uv,3).r;
 // Source-height crest/cavity response, with no added stamps.
 float2 broad_uv=float2(world.y,-world.x)*q3_source_repeat(.115)+float2(.43,.61);
 float broad_height=desert_hills_height_texture.Sample(material_sampler,broad_uv).r;
 float broad_mean=desert_hills_height_texture.SampleBias(material_sampler,broad_uv,3).r;
 float surface=(sand_height-sand_mean)*.85+(broad_height-broad_mean)*2.7;
 float dune_fade=coast_shelf*(1-smoothstep(.27,.57,input.hydrology_data.w));
 color*=clamp(1+surface*dune_fade*q3_coast_irregular_region(world),.65,1.26);
 float3 absorption=exp(-input.hydrology_data.w*float3(9,5,3)
  *lerp(1,.42,coast_shelf));
 float tint_strength=coast_shelf*smoothstep(.05,.22,input.hydrology_data.w);
 float3 tint=lerp(1.0.xxx,float3(.48,1.12,1.80),tint_strength);
 return color*clamp(.86+(height-.426)*.55,.65,1.1)*absorption*tint;
}
float q3_beach_coverage(float distance,float width,float grain) {
 // Keep sand beneath the replacement ground's partial-coverage band. Ending
 // the beach at width+.06 exposed grass beneath fading grass, hiding its
 // texture breakup and leaving the beach's smooth contour as the visible edge.
 return 1-smoothstep(width+.10,width+.26,distance+(grain-.28)*.24);
}
void q3_shore_material(PixelInput input,float2 world_position,inout float3 albedo,inout float3 material_normal) {
 float sd=input.hydrology_data.x,width=input.hydrology_data.y;
 float rocky=saturate(input.hydrology_data.z);
 float desert=saturate(input.material_weights.z);
 float2 uv=q3_source_world(input)*q3_source_repeat(.75);
 float4 beach=beach_base_texture.Sample(material_sampler,uv);
 float beach_grain=q3_surface_grain(beach.a,
  beach_base_texture.SampleBias(material_sampler,uv,3).a);
 float3 sand=beach.rgb*(1+beach_grain);
 float grain=dot(sand,float3(.2126,.7152,.0722));
 float blend=q3_beach_coverage(sd,width,grain);
 // Desert already supplies a continuous sand surface. Replacing it with the
 // separate beach material created a flat, darker ribbon beside the desert's
 // fine ripples. Preserve that sand family through the wet edge; mixed biome
 // weights reduce the beach replacement continuously rather than branching by
 // tile ownership.
 float2 world=q3_source_world(input);
 float patch=q3_margin_patch(world);
 float beach_material=blend*(1-rocky)*(1-desert);
 albedo=lerp(albedo,sand*lerp(.82,1.0,patch),beach_material);
 float hx=beach_height_texture.Sample(material_sampler,uv+float2(.002,0)).r
  -beach_height_texture.Sample(material_sampler,uv-float2(.002,0)).r;
 float hy=beach_height_texture.Sample(material_sampler,uv+float2(0,.002)).r
  -beach_height_texture.Sample(material_sampler,uv-float2(0,.002)).r;
 float2 detail=clamp(float2(-hx-hy,-hx+hy)*12,-.18,.18)*beach_material;
 material_normal=normalize(float3(material_normal.xy+detail*material_normal.z,material_normal.z));
 material_normal=normalize(lerp(material_normal,
  q3_margin_normal(input,beach_grain,.022),beach_material));
 float3 rock=cliff_base_texture.Sample(material_sampler,q3_source_world(input)*q3_source_repeat(.75)).rgb;
 albedo=lerp(albedo,rock,rocky*(1-smoothstep(.03,.22,sd)));
 float fine_height,fine_cavity;
 float4 fine=q3_margin_detail(world,fine_height,fine_cavity);
 float margin=(1-smoothstep(width*.35,width+.10,sd))*lerp(.25,1.0,patch)*lerp(1,.35,desert);
 albedo=lerp(albedo,fine.rgb,fine.a*margin);
 albedo*=lerp(1,fine_cavity,margin);
 material_normal=normalize(lerp(material_normal,q3_margin_normal(input,fine_height,.065),margin));
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
  // About four pixels at gameplay zoom. Distance is interpolated linearly
  // across the ground grid, so an unbroken contour shows the mesh facets;
  // this field bends the waterline and bank edge between those samples.
  float shore_noise=river_bank_noise_texture.Sample(material_sampler,
   world*float2(q3_source_repeat(.67),q3_source_repeat(.59))+float2(.83,.21)).r-.5;
  float4 river_material=river_base_texture.Sample(material_sampler,uv);
  float grain=river_height_texture.Sample(material_sampler,uv).r;
  float material_grain=q3_surface_grain(river_material.a,
   river_base_texture.SampleBias(material_sampler,uv,3).a);
  // Keep each transition at least about one display pixel wide when zoomed out.
  float aa=max(fwidth(distance_pixels),.35);
  float water_width=5.8+noise*1.1;
  float shoreline=distance_pixels+shore_noise*1.5+(gravel_noise-.5)*.5;
  float water=1-smoothstep(water_width-1.0-aa,water_width+.6+aa,shoreline);
  // Keep the authored channel path, but let land reach close to the water.
  // The outer deposit is a broken, translucent margin rather than a solid
  // sand-colored ribbon on both sides of every bend.
  float bank_width=water_width+2.5+noise*1.4+(sediment_noise-.5)*1.3;
  float bank_feather=1.8+sediment_noise*.7;
  float bank_edge_distance=distance_pixels+(gravel_noise-.5)*.7+shore_noise*1.1;
  float bank=1-smoothstep(bank_width-bank_feather,bank_width+.6+aa,bank_edge_distance);
  bank=saturate(bank+material_grain*2.0*bank*(1-bank));
  // Fade the complete river surface into the sea across the optical shore.
  // The final alpha must keep this fade after the water/shore mix below.
  float land_bank=smoothstep(-.025,.065,sd);
  float outlet=smoothstep(-.28,.08,sd);
  bank*=outlet;
  water=lerp(1,water,land_bank);
  float3 bed=river_material.rgb;
  float3 sand=beach_base_texture.Sample(material_sampler,uv).rgb;
  float3 fine_sand=beach_base_texture.Sample(material_sampler,
   world*float2(q3_source_repeat(2.17),q3_source_repeat(2.63))+float2(.37,.59)).rgb;
  float gravel_height,gravel_cavity;
  float4 clutter=q3_margin_detail(world,gravel_height,gravel_cavity);
  float4 shore_cracks=sample_coast_clutter(world);
  // Isolated gravel and cracked-rock deposits can reach past the soil lip.
  // Their source alpha and broken noise field keep the surrounding terrain
  // visible, including at bends where a broad painted bank would look false.
  float grit_reach=smoothstep(water_width+.4,water_width+1.5,distance_pixels)
   *(1-smoothstep(bank_width+.5,bank_width+4.5,distance_pixels));
  float grit_seed=smoothstep(.37,.63,sediment_noise*.60
   +(noise+.5)*.25+gravel_noise*.15);
  float grit_spill=max(clutter.a*.62,shore_cracks.a*.72)
   *grit_reach*grit_seed*outlet*land_bank;
  clip(max(bank,grit_spill)-.001);
  float bank_position=saturate((distance_pixels-water_width)/max(1,bank_width-water_width));
  float sediment=smoothstep(.30,.70,sediment_noise*.52+(noise+.5)*.32+grain*.16);
  float3 textured_sand=lerp(sand,fine_sand,.46);
  float3 dry_sand=lerp(bed,textured_sand,.20);
  float3 bank_soil=lerp(bed,textured_sand,.22);
  float sand_patch=saturate((sediment-.28)*1.20)
   *smoothstep(.22,.62,bank_position);
  float3 dry=lerp(bank_soil,dry_sand,sand_patch*.32)*lerp(.96,1.04,sediment);
  dry*=1+material_grain;
  float gravel=smoothstep(.35,.68,gravel_noise)*clutter.a
   *smoothstep(.10,.58,bank_position);
  dry=lerp(dry,clutter.rgb*.82,gravel*.72);
  float pebble=smoothstep(.68,.84,gravel_noise)*smoothstep(.12,.68,bank_position);
  float3 pebble_color=lerp(float3(.16,.12,.07),textured_sand*1.18,sediment);
  dry=lerp(dry,pebble_color,pebble*.34);
  dry*=.91+gravel_noise*.16;
  // Reuse the authored shoreline crack cells as scattered exposed grit.
  float shore_grit=shore_cracks.a
   *grit_seed*smoothstep(water_width+.3,water_width+1.5,distance_pixels)
   *(1-smoothstep(bank_width,bank_width+1.0,distance_pixels));
  float bank_patch=smoothstep(.38,.66,sediment_noise*.57
   +(noise+.5)*.28+gravel_noise*.15);
  // The waterline: damp soil hugs the moving shoreline for a varying width,
  // and authored river gravel collects along it in broken runs. Both fade
  // into the land instead of ending on the old hard, faceted water edge.
  float wet_reach=1.1+sediment_noise*1.5+(shore_noise+.5)*.8;
  float wet=(1-smoothstep(water_width+.2,water_width+wet_reach+aa,shoreline))*land_bank;
  float waterline_gravel=clutter.a*land_bank
   *smoothstep(.40,.66,sediment_noise*.55+(shore_noise+.5)*.45)
   *smoothstep(water_width-1.2,water_width+.2,shoreline)
   *(1-smoothstep(water_width+1.4,water_width+2.8,shoreline));
  float deposit=(bank_patch*.22+gravel*.12+shore_grit*.14)
   *(1-smoothstep(water_width+.6,bank_width+.7,distance_pixels));
  float3 grit_color=lerp(clutter.rgb*.90,shore_cracks.rgb*.90,
   shore_cracks.a/max(.001,shore_cracks.a+clutter.a));
  float3 shore=lerp(dry,grit_color,max(shore_grit*.45,grit_spill*.55));
  // Damp ground is the land beneath, darkened: a translucent dark layer
  // keeps the grass or plains texture instead of painting a soil ribbon.
  float damp=wet*lerp(.28,.48,bank_patch);
  float stones=waterline_gravel*.72;
  float covered=1-(1-deposit)*(1-damp)*(1-stones);
  shore=(shore*deposit*(1-damp)*(1-stones)+float3(.030,.028,.018)*damp*(1-stones)
   +clutter.rgb*lerp(.80,.58,wet)*stones)/max(covered,.001);
  // Spilled grit beyond the bank edge keeps its own coverage and color.
  float spill=grit_spill*.43*(1-bank);
  shore=lerp(shore,grit_color,spill/max(spill+bank*covered,.001));
  bank=max(water,max(bank*covered,grit_spill*.43));
  // Shallows stay close to the channel color, so no pale rim outlines the
  // water; depth absorbs toward the middle.
  float channel=1-smoothstep(0,water_width,max(0,distance_pixels));
  float optical_depth=.16+.26*smoothstep(.10,.90,channel);
  float3 transmitted=bed*lerp(.62,1,channel)*exp(-optical_depth*float3(8,4,2));
  float3 river=lerp(transmitted,float3(.018,.074,.090),1-exp(-optical_depth*4));
  float submerged=smoothstep(water_width-3.0,water_width-1.0,distance_pixels)
   *(1-smoothstep(water_width-.15,water_width+.45,distance_pixels));
  river=lerp(river,clutter.rgb*float3(.30,.49,.52),submerged*clutter.a*.27);
  float bank_shadow=smoothstep(water_width-2.0,water_width-.65,distance_pixels)
   *(1-smoothstep(water_width-.05,water_width+.4,distance_pixels));
  river*=1-bank_shadow*.18;
  float2 river_uv=world*float2(q3_source_repeat(.92),q3_source_repeat(1.27));
  float2 lean=river_lean0_texture.Sample(material_sampler,river_uv).rg*2-1;
  float water_time=0;
#if defined(Q3_WATER_TIME)
  water_time=Q3_WATER_TIME;
  // Bounded two-phase advection hides resets while following the retained
  // channel tangent. Spatial phase variation avoids synchronized pulsing.
  if(Q3_WATER_TIME>0 && dot(input.relief_material.xy,input.relief_material.xy)>.01){
   float phase=frac(Q3_WATER_TIME*.20+sediment_noise);
   float other=frac(phase+.5),weight=1-abs(phase*2-1);
   float2 flow=input.relief_material.xy*.27*float2(q3_source_repeat(.92),q3_source_repeat(1.27));
   float2 a=river_lean0_texture.Sample(material_sampler,river_uv-flow*phase).rg*2-1;
   float2 b=river_lean0_texture.Sample(material_sampler,river_uv+float2(.37,.61)-flow*other).rg*2-1;
   lean=lerp(b,a,weight);
  }
#endif
  // Stream lines, as Civ VI draws them: thin pale lines parallel to the
  // banks, broken into dashes that drift downstream along the drainage
  // tangent. A broad world field picks occasional livelier reaches where the
  // lines gather and brighten. Unknown drainage has no direction and is calm.
  float2 downstream=input.relief_material.xy;
  float flowing=saturate(dot(downstream,downstream)*4)*land_bank;
  float reach=river_bank_noise_texture.Sample(material_sampler,
   world*float2(q3_source_repeat(.025),q3_source_repeat(.021))+float2(.37,.83)).r;
  float rapids=smoothstep(.62,.76,reach)*flowing;
  float stream_lines=0;
  if(flowing>.001){
   const float cycle=.32;
   float frequency=q3_source_repeat(1.10);
   float phase=frac(water_time*.30+sediment_noise);
   float other=frac(phase+.5),weight=1-abs(phase*2-1);
   float a=0,b=0;
   [unroll] for(int tap=0;tap<5;++tap){
    a+=river_bank_noise_texture.Sample(material_sampler,
     (world-downstream*(cycle*phase+tap*.012))*frequency).r;
    b+=river_bank_noise_texture.Sample(material_sampler,
     (world-downstream*(cycle*other+tap*.012))*frequency+float2(.29,.61)).r;
   }
   float dash=smoothstep(.50,.70,lerp(b,a,weight)*.2+rapids*.05);
   // Lines follow the channel distance, gently bent so they wander.
   float across=distance_pixels+noise*1.3+shore_noise*.9;
   float spacing=lerp(2.5,1.8,rapids);
   float offset=abs(frac(across/spacing)-.5)*spacing;
   float lane=smoothstep(spacing*.5-.30-aa*.5,spacing*.5-.05,offset);
   stream_lines=lane*dash*flowing*lerp(.22,.45,rapids)
    *(1-smoothstep(water_width-2.0,water_width-.4,distance_pixels));
  }
  float3 river_normal=normalize(float3(-lean*(.24+.16*rapids),1));
  float3 water_light=q6_receiver_illumination(input,river_normal,1,1);
  float river_visibility=q6_receiver_visibility(input,river_normal,1);
  // The same optics as shallow sea water: Fresnel-weighted sky and mirrored
  // scene over the transmitted body, plus a moving sun and moon glint.
  float3 eye=normalize(float3(.43,-.43,1));
#ifdef Q3_WATER_CAMERA
  {
   float2 delta=world-Q3_WATER_CAMERA.xy;
   float2 raw=float2(delta.x+delta.y,delta.x-delta.y);
   raw-=round(raw/max(Q3_WATER_CAMERA.zw,1))*Q3_WATER_CAMERA.zw;
   delta=float2(raw.x+raw.y,raw.x-raw.y)*.5;
   eye=normalize(float3(float2(.43,-.43)*2.5-delta,2.5));
  }
#endif
  float fresnel=clamp(pow(1.1-saturate(dot(river_normal,eye)),2)*1.5,.1,.75);
  float3 ray=reflect(-eye,river_normal);
  float3 sky_light=environment_ambient_color*.6+environment_sun_color*environment_sun_intensity*.6
   +environment_moon_color*environment_moon_intensity*.6;
  float3 reflected=sky_light*lerp(float3(.16,.25,.36),float3(.42,.55,.68),smoothstep(.30,.90,ray.y));
  float mirrored_alpha=0;
#ifdef Q3_OBJECT_REFLECTION
  // The mirror target reflects about the sea plane. Rivers keep the flat
  // datum, so the lookup moves by the river's own height above that plane;
  // a raised reach (unusual) would also mirror its own banks, so fade it.
  float lift=max(0,input.q6_world.z-NativeReflection.z);
  float2 mirror_uv=(input.position.xy+NativeReflectionTarget.zw+
   float2(0,2*lift*NativeReflection.x))/Q3_REFLECTION_SIZE;
  float2 mirror_sample=mirror_uv-river_normal.xy*float2(3,1.5)/
   (Q3_REFLECTION_SIZE*clamp(eye.z*2,.05,1));
  float4 mirrored=q3_object_reflection_texture.Sample(decal_sampler,mirror_sample);
  float inside=step(0,mirror_uv.x)*step(mirror_uv.x,1)*step(0,mirror_uv.y)*step(mirror_uv.y,1)
   *NativeReflection.w;
  mirrored_alpha=saturate(mirrored.a)*inside*(1-smoothstep(1.5,3.5,lift*112));
  reflected=lerp(reflected,mirrored.rgb,mirrored_alpha);
#endif
  float reflection=fresnel*max(mirrored_alpha,.4)*.68*land_bank;
  // Open-sea glare strength, so a river shares the sea's sun path. The calm
  // surface keeps the glare a smooth sheen; ripple normals only break it up.
  float3 glint_normal=normalize(float3(-lean*(.08+.10*rapids),1));
  float facing=max(dot(reflect(-environment_sun_direction,glint_normal),eye),0);
  // The sea's broad twilight sheen is open-ocean only; it would wash a
  // whole river pale at dusk.
  float glint=smoothstep(.82,.995,facing)*.9;
  float moon_glint=smoothstep(.82,.995,max(dot(reflect(-environment_moon_direction,glint_normal),eye),0))
   *smoothstep(.18,.28,environment_moon_intensity)*.65;
  float3 specular=(environment_sun_color*environment_sun_intensity*glint
   +environment_moon_color*environment_moon_intensity*moon_glint)*river_visibility;
  float3 surface=lerp(river*water_light,reflected,reflection)+specular;
  surface=lerp(surface,float3(.70,.78,.80)*water_light,stream_lines);
  // Shade the existing bank grain and authored gravel, not the river surface.
  float bank_height=grain*.25+material_grain*.30+(gravel_height-.5)*gravel*.30
   +shore_grit*.25+grit_spill*.28+(gravel_height-.5)*waterline_gravel*.30;
  float mean_grain=river_height_texture.SampleBias(material_sampler,uv,2).r;
  float cavity=q3_margin_visibility(grain,mean_grain)*
   lerp(1,gravel_cavity,max(gravel,waterline_gravel));
  shore*=lerp(1,cavity,.65);
  float3 bank_normal=q3_margin_normal(input,bank_height,.070);
  float3 bank_light=q6_receiver_illumination(input,bank_normal,1,1);
  return float4(lerp(shore*bank_light,surface,water),bank*outlet);
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

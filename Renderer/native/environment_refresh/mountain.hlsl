#define BEAUTY_COMPOSED_SHADOWS 1
#define BEAUTY_TERRAIN_TRANSITIONS 1
// One material and lighting implementation is compiled to both Metal and D3D11.
cbuffer Frame : register(b0) {
    float4 Sun;
    float4 SunColorExposure;
    float4 Ambient;
    float4 View;
    float4 Macro;
    float4 Quality;
};
Texture2D GrassColor : register(t0);
Texture2D GrassHeight : register(t1);
Texture2D GrassSpecular : register(t2);
Texture2D RockColor : register(t3);
Texture2D RockHeight : register(t4);
Texture2D RockSpecular : register(t5);
Texture2D TopColor : register(t6);
Texture2D TopHeight : register(t7);
Texture2D TopSpecular : register(t8);
Texture2D SnowColor : register(t9);
Texture2D SnowHeight : register(t10);
Texture2D SnowSpecular : register(t11);
Texture2D MacroHeight : register(t12);
#ifdef BEAUTY_TERRAIN_TRANSITIONS
Texture2D PlainsColor : register(t13);
Texture2D PlainsHeight : register(t14);
Texture2D PlainsSpecular : register(t15);
Texture2D TundraColor : register(t16);
Texture2D TundraHeight : register(t18);
Texture2D TundraSpecular : register(t19);
Texture2D DesertColor : register(t20);
Texture2D DesertHeight : register(t21);
Texture2D DesertSpecular : register(t22);
Texture2D AuthoredHillHeight : register(t23);
Texture2D GrassHillColor : register(t24);
Texture2D GrassHillHeight : register(t25);
Texture2D GrassHillSpecular : register(t26);
Texture2D PlainsHillColor : register(t27);
Texture2D PlainsHillHeight : register(t28);
Texture2D PlainsHillSpecular : register(t29);
Texture2D GroundSurfaceDetail : register(t30);
#endif
#ifdef BEAUTY_COMPOSED_SHADOWS
Texture2DArray ShadowField : register(t17);
cbuffer ShadowFrame : register(b2) {
    float4 ShadowU;
    float4 ShadowV;
    float4 ShadowL;
    float4 ShadowOrigin;
    float4 ShadowFlags;
};
// Binding adapter; the shared Lab provider owns filtering/contact/depth rules.
// World-aligned six-tile pages preserve the retained 6/1024 sampling density.
// Source depths use R32_FLOAT physical light distance, avoiding page-dependent
// normalization/quantization. Page identity never contains a screen anchor.


cbuffer C3XShadowPages : register(b4) { float4 pickup_pages[64]; };
int pickup_page(int2 page) {
 uint key=(uint(page.x)*73856093u ^ uint(page.y)*19349663u)&63u;
 [loop]for(int n=0;n<33;n++) {
  float4 entry=pickup_pages[(key+uint(n))&63u];
  if(entry.w<.5)return -1;
  if(all(page==int2(entry.xy)))return int(entry.z);
 }
 return -1;
}
float pickup_blocker(Texture2DArray field,int2 texel,int2 center_page,int center_slot) {
 int2 page=int2(floor(float2(texel)/1024.));
 int slot=all(page==center_page)?center_slot:pickup_page(page);
 return slot<0?-1e6:field.Load(int4(texel-page*1024,slot,0)).r;
}
float c3x_paged_visibility(Texture2DArray field,float4 world,float3 normal,bool water,
 float4 ShadowU,float4 ShadowV,float4 ShadowL,float4 ShadowFlags) {
 if(world.w<=.5 || ShadowFlags.x<=.5)return 1;
 const float texel=6./1024.;
 float3 offset=world.xyz+normal*texel;
 float2 uv=float2(dot(offset,ShadowU.xyz),dot(offset,ShadowV.xyz))/texel;
 float z=dot(offset,ShadowL.xyz);
 float2 plane=float2(dot(world.xyz,ShadowU.xyz),dot(world.xyz,ShadowV.xyz))/texel;
 float plane_z=dot(world.xyz,ShadowL.xyz);
 float2 ux=ddx(plane),uy=ddy(plane);float zx=ddx(plane_z),zy=ddy(plane_z);
 float determinant=ux.x*uy.y-ux.y*uy.x;
 float2 gradient=0;
 if(abs(determinant)>1e-12)gradient=float2(zx*uy.y-zy*ux.y,zy*ux.x-zx*uy.x)/determinant;
 int2 center=int2(floor(uv));int2 center_page=int2(floor(float2(center)/1024.));
 int center_slot=pickup_page(center_page);float sum=0,closest_delta=0;
 [unroll]for(int y=-1;y<=1;y++)[unroll]for(int x=-1;x<=1;x++) {
  int2 sample= center+int2(x,y);
  float blocker=pickup_blocker(field,sample,center_page,center_slot);
  float receiver=z+dot(gradient,float2(sample)+.5-uv);
  sum+=step(blocker,receiver+.00060);
  if(x==0 && y==0)closest_delta=blocker-receiver;
 }
 float soft=sum/9;
 if(!water && ShadowFlags.y>.5 && closest_delta>.0039 && closest_delta<.024)soft=min(soft,.15);
 return soft;
}

float q6_world_visibility(Texture2DArray field,float4 world,float3 normal,bool water) {
 return c3x_paged_visibility(field,world,normal,water,ShadowU,ShadowV,ShadowL,ShadowFlags);
}
float q6_shadow_visibility(Texture2DArray field,float3 world,float3 normal,float4 u,float4 v,float4 l,bool receive,bool contact) {
 return receive?q6_world_visibility(field,float4(world,1),normal,!contact):1;
}

#endif
SamplerState Wrap : register(s0);
SamplerState Clamp : register(s1);

struct V {
    float3 position : POSITION;
#ifdef BEAUTY_COMPOSED_SHADOWS
    float4 world : TEXCOORD0;
#else
    float3 world : TEXCOORD0;
#endif
    float3 normal : NORMAL;
    float2 uv : TEXCOORD1;
#ifdef BEAUTY_TERRAIN_TRANSITIONS
    float4 material : TEXCOORD2;
    float2 biome : TEXCOORD3;
    float coast_coverage : TEXCOORD4;
#else
    float3 material : TEXCOORD2;
#endif
};
struct P {
    float4 position : SV_POSITION;
    float3 world : TEXCOORD0;
    float3 normal : NORMAL;
    float2 uv : TEXCOORD1;
#ifdef BEAUTY_TERRAIN_TRANSITIONS
    float4 material : TEXCOORD2;
    float2 biome : TEXCOORD3;
    float coast_coverage : TEXCOORD4;
    float base_relief : TEXCOORD5;
#else
    float3 material : TEXCOORD2;
#endif
};
struct Output { float4 color : SV_Target0; float validity : SV_Target1; };

P VSMain(V input) {
    P output;
    output.position = float4(input.position, 1);
    output.world = input.world.xyz;
    output.normal = input.normal;
    output.uv = input.uv;
    output.material = input.material;
#ifdef BEAUTY_TERRAIN_TRANSITIONS
    output.biome = input.biome;
    output.coast_coverage = input.coast_coverage;
    output.base_relief = max(0, input.world.w - 1);
#endif
    return output;
}
P VSFeature(V input) { return VSMain(input); }

float3 triplanar(Texture2D texture_map, float3 p, float3 n) {
    float3 weight = pow(abs(n), 5);
    weight /= max(dot(weight, 1), 0.00001);
    p *= Quality.y;
    return texture_map.Sample(Wrap, p.yz).rgb * weight.x +
           texture_map.Sample(Wrap, p.xz).rgb * weight.y +
           texture_map.Sample(Wrap, p.xy).rgb * weight.z;
}

float triplanar_scalar(Texture2D texture_map, float3 p, float3 n) {
    return triplanar(texture_map, p, n).r;
}

float3 triplanar_neighborhood(Texture2D texture_map, float3 p, float3 n) {
    float3 weight = pow(abs(n), 5);
    weight /= max(dot(weight, 1), 0.00001);
    p *= Quality.y;
    // The wider mip footprint estimates the local material neighborhood at
    // the current viewing scale, rather than imposing a fixed texel size.
    return texture_map.SampleBias(Wrap, p.yz, 3).rgb * weight.x +
           texture_map.SampleBias(Wrap, p.xz, 3).rgb * weight.y +
           texture_map.SampleBias(Wrap, p.xy, 3).rgb * weight.z;
}

float rock_crevice_visibility(float height, float neighborhood) {
    // Valleys receive less fill; flat material and protrusions stay neutral.
    // No altitude, compass direction, or light vector enters this response.
    return 1 - 0.70 * saturate((neighborhood - height) * 12);
}

float macro_height_world(float2 world_xy) {
    float2 uv = world_xy / float2(3.20, 2.72) + 0.5;
    if (any(uv < 0) || any(uv > 1)) return 0;
    float raw = MacroHeight.SampleLevel(Clamp, uv, 0).r;
    return saturate((raw - Macro.x) / max(Macro.y - Macro.x, 0.0001)) * Macro.z;
}

float shadow_ray(float3 world, float lateral) {
    float2 toward_light = normalize(-Sun.xy);
    float2 perpendicular = float2(-toward_light.y, toward_light.x);
    world.xy += perpendicular * lateral;
    float visibility = 1;
    [unroll] for (int index = 1; index <= 20; ++index) {
        float travel = index * 0.075;
        float2 xy = world.xy + toward_light * travel;
        float ray = world.z + travel * Sun.z / max(length(Sun.xy), 0.05);
        float blocker = macro_height_world(xy);
        visibility = min(visibility, 1 - saturate((blocker - ray - 0.018) * 24));
    }
    return visibility;
}

float horizon_visibility(float3 world) {
    if (Quality.x < 0.5) return 1;
    float visibility = (shadow_ray(world, -0.045) + shadow_ray(world, 0) +
                        shadow_ray(world, 0.045)) / 3;
    return lerp(1, visibility, Quality.w);
}

float3 detail_normal(float3 geometric, float3 world, float detail, float strength) {
    float3 dx = ddx(world), dy = ddy(world);
    float3 r1 = cross(dy, geometric), r2 = cross(geometric, dx);
    float determinant = dot(dx, r1);
    float3 gradient = (ddx(detail) * r1 + ddy(detail) * r2) *
        sign(determinant) / max(abs(determinant), 0.000001);
    return normalize(geometric - gradient * strength);
}

// Differentiate each height projection before blending it.
// Differentiating the blended height also differentiates projection weights,
// so unrelated texture values can introduce false gradients as a face turns.
float2 triplanar_height_derivatives(Texture2D texture_map, float3 p, float3 n) {
    float3 weight = pow(abs(n), 5);
    weight /= max(dot(weight, 1), 0.00001);
    p *= Quality.y;
    float x = texture_map.Sample(Wrap, p.yz).r;
    float y = texture_map.Sample(Wrap, p.xz).r;
    float z = texture_map.Sample(Wrap, p.xy).r;
    return float2(ddx(x), ddy(x)) * weight.x +
           float2(ddx(y), ddy(y)) * weight.y +
           float2(ddx(z), ddy(z)) * weight.z;
}

float3 height_derivative_normal(float3 geometric, float3 world, float2 detail, float strength) {
    float3 dx = ddx(world), dy = ddy(world);
    float3 r1 = cross(dy, geometric), r2 = cross(geometric, dx);
    float determinant = dot(dx, r1);
    float3 gradient = (detail.x * r1 + detail.y * r2) *
        sign(determinant) / max(abs(determinant), 0.000001);
    return normalize(geometric - gradient * strength);
}

float ggx(float3 n, float3 light, float3 view, float roughness, float f0) {
    float3 halfway = normalize(light + view);
    float ndl = saturate(dot(n, light));
    float ndv = saturate(dot(n, view));
    float ndh = saturate(dot(n, halfway));
    float vdh = saturate(dot(view, halfway));
    float alpha = max(0.055, roughness * roughness);
    float a2 = alpha * alpha;
    float denominator = ndh * ndh * (a2 - 1) + 1;
    float distribution = a2 / max(3.14159265 * denominator * denominator, 0.0001);
    float k = (roughness + 1) * (roughness + 1) * 0.125;
    float geometry_v = ndv / max(ndv * (1 - k) + k, 0.0001);
    float geometry_l = ndl / max(ndl * (1 - k) + k, 0.0001);
    float fresnel = f0 + (1 - f0) * pow(1 - vdh, 5);
    return distribution * geometry_v * geometry_l * fresnel /
        max(4 * ndv * ndl, 0.0001) * ndl;
}

float3 atmosphere(float y) {
    float horizon = saturate(1 - abs(y - 0.56) * 2.25);
    float3 low = float3(0.040, 0.058, 0.074);
    float3 high = float3(0.155, 0.225, 0.300);
    return lerp(low, high, saturate(y * 0.78 + 0.18)) +
           float3(0.12, 0.085, 0.045) * horizon * 0.17;
}

void ground_material(P input, out float3 albedo, out float height_detail,
                     out float specular_map) {
#ifdef BEAUTY_TERRAIN_TRANSITIONS
    // Use the terrain provider's world-aligned sampling and normalized BIQ
    // family weights so the mountain collar is the terrain beneath it, not a
    // second grass-only surface. The broad authored modulation also matches.
    float2 uv0 = input.world.xy * 0.43 + float2(0.31, 0.17);
    float2 uv1 = float2(input.world.y, -input.world.x) * (0.43 * 0.91) +
                 float2(0.63, 0.29);
    float2 tundra_uv = input.world.xy * (0.43 * 0.84) + float2(0.19, 0.71);
    float desert_weight = saturate(input.biome.y);
    float tundra_weight = saturate(input.material.w /
        max(1 - desert_weight, 0.00001));
    float plains_weight = saturate(input.biome.x /
        max(1 - desert_weight - input.material.w, 0.00001));
    float3 grass = GrassColor.Sample(Wrap, uv0).rgb;
    float3 plains = PlainsColor.Sample(Wrap, uv1).rgb;
    float3 tundra = TundraColor.Sample(Wrap, tundra_uv).rgb;
    albedo = lerp(lerp(grass, plains, plains_weight), tundra, tundra_weight);
    height_detail = lerp(lerp(GrassHeight.Sample(Wrap, uv0).r,
                                      PlainsHeight.Sample(Wrap, uv1).r,
                                      plains_weight),
                               TundraHeight.Sample(Wrap, tundra_uv).r,
                               tundra_weight);
    specular_map = lerp(lerp(GrassSpecular.Sample(Wrap, uv0).r,
                                    PlainsSpecular.Sample(Wrap, uv1).r,
                                    plains_weight),
                             TundraSpecular.Sample(Wrap, tundra_uv).r,
                             tundra_weight);
    albedo = lerp(albedo, DesertColor.Sample(Wrap, uv0).rgb, desert_weight);
    height_detail = lerp(height_detail, DesertHeight.Sample(Wrap, uv0).r,
                         desert_weight);
    specular_map = lerp(specular_map, DesertSpecular.Sample(Wrap, uv0).r,
                        desert_weight);
    // The fourth world component carries only the underlying authored relief.
    // Reconstruct the same hill-top family response at the mountain collar so
    // an adjacent hill cannot look like a separate mesh pushed through it.
    float slope = 1 - saturate(normalize(input.normal).z);
    float hill_weight = smoothstep(0.008, 0.18, input.base_relief) *
                        saturate(0.42 + slope * 2.2) * (1 - tundra_weight);
    float3 hill = lerp(GrassHillColor.Sample(Wrap, uv0 * 1.08).rgb,
                       PlainsHillColor.Sample(Wrap, uv1 * 1.08).rgb,
                       plains_weight);
    hill = lerp(hill, tundra, tundra_weight);
    float hill_h = lerp(GrassHillHeight.Sample(Wrap, uv0 * 1.08).r,
                        PlainsHillHeight.Sample(Wrap, uv1 * 1.08).r,
                        plains_weight);
    hill_h = lerp(hill_h, TundraHeight.Sample(Wrap, tundra_uv).r,
                  tundra_weight);
    float hill_s = lerp(GrassHillSpecular.Sample(Wrap, uv0 * 1.08).r,
                        PlainsHillSpecular.Sample(Wrap, uv1 * 1.08).r,
                        plains_weight);
    hill_s = lerp(hill_s, TundraSpecular.Sample(Wrap, tundra_uv).r,
                  tundra_weight);
    albedo = lerp(albedo, hill, hill_weight * 0.90);
    height_detail = lerp(height_detail, hill_h, hill_weight);
    specular_map = lerp(specular_map, hill_s, hill_weight);
    // Match the ordinary terrain provider's world-space macro modulation so
    // the unified relief patch is indistinguishable at its outer boundary.
    float broad = GroundSurfaceDetail.Sample(Wrap,
        input.world.xy * 0.071 + float2(0.13, 0.37)).r * 0.72;
    broad += GroundSurfaceDetail.Sample(Wrap,
        float2(input.world.y, -input.world.x) * 0.183 + float2(0.61, 0.29)).r * 0.28;
    albedo *= lerp(float3(0.88, 0.94, 0.97),
                   float3(1.09, 1.045, 0.91), saturate(broad));
#else
    float2 uv = input.world.xy * 0.27 + 0.5;
    albedo = GrassColor.Sample(Wrap, uv).rgb;
    height_detail = GrassHeight.Sample(Wrap, uv).r;
    specular_map = GrassSpecular.Sample(Wrap, uv).r;
#endif
}

Output shade(P input) {
    Output output;
    if (input.material.y < 0.5) {
        output.color = float4(atmosphere(input.uv.y), 1);
        output.validity = 1;
        return output;
    }

    float3 geometric = normalize(input.normal);
    float3 albedo;
    float height_detail;
    float rock_crevice = 1;
    float2 rock_derivatives = 0;
    float specular_map;
#ifdef BEAUTY_TERRAIN_TRANSITIONS
    float mountain_rise = max(0, input.world.z - input.base_relief - 2.5 / 112.0);
#else
    float mountain_rise = max(0, input.material.x * Macro.z);
#endif
    // One terrain-to-rock rule for the whole composed surface. The source
    // footprint shapes geometry, but must not bypass the material transition;
    // doing so forced some low faces to pure rock while others retained grass.
    // Neither source blend nor face steepness changes coverage at equal rise.
    float mountain_coverage = smoothstep(0.02, 0.48, mountain_rise);
    float rock_albedo_coverage = mountain_coverage;
    float rock_detail_coverage = mountain_coverage;
#ifdef BEAUTY_TERRAIN_TRANSITIONS
    // 42 remains the source-shadow mountain discriminator. Its fractional
    // range carries the authoritative coast/source-family coverage.
    float coast_alpha = saturate(input.coast_coverage - 42);
    clip(coast_alpha - 0.001);
#else
    float coast_alpha = 1;
#endif
    float3 ground_albedo;
    float ground_height;
    float ground_specular;
    ground_material(input, ground_albedo, ground_height, ground_specular);
    if (input.material.y < 1.5) {
        albedo = ground_albedo;
        float ground_luma = dot(albedo, float3(0.2126, 0.7152, 0.0722));
        albedo = lerp(albedo, ground_luma.xxx, 0.15) * float3(1.03, 1.0, 0.92);
        height_detail = ground_height;
        specular_map = ground_specular;
    } else if (Quality.x < 0.5) {
        // Control: the old failure mode stretches one planar albedo lookup over
        // steep faces and omits the source material response.
        albedo = lerp(ground_albedo, RockColor.Sample(Wrap, input.uv).rgb,
                      rock_albedo_coverage);
        height_detail = ground_height;
        specular_map = lerp(ground_specular, 0.0, rock_detail_coverage);
    } else {
        float height = input.material.x;
        float snow = smoothstep(0.62, 0.78, height) * smoothstep(0.02, 0.25, geometric.z);
        // The top material contains patchy snow, not plain upper rock. Keep
        // both authored snow layers near the summit; ground coverage remains
        // tied to final rise independently of these source-height masks.
        float top = smoothstep(0.52, 0.68, height) * (1 - snow);
        float base = 1 - top - snow;
        float3 rock_albedo = triplanar(RockColor, input.world, geometric);
        float3 mountain_albedo = rock_albedo * base +
                                 triplanar(TopColor, input.world, geometric) * top +
                                 triplanar(SnowColor, input.world, geometric) * snow;
        float rock_detail = triplanar_scalar(RockHeight, input.world, geometric);
        float layered_detail = rock_detail * base +
                               triplanar_scalar(TopHeight, input.world, geometric) * top +
                               triplanar_scalar(SnowHeight, input.world, geometric) * snow;
        // Snow softens the rock relief. Preserve other packs' authored upper
        // height channel; the local base/upper height pair happens to be equal.
        float mountain_detail = lerp(layered_detail, rock_detail, snow * 0.60);
        // Retain the fine source-height contribution without amplifying the
        // broad material plateaus into horizontal ledges. Crevice fill stays
        // independent of light direction and separate from normal strength.
        float3 fine_world = input.world * 3.7 + float3(0.31, 0.17, 0.43);
        float fine_rock = triplanar_scalar(RockHeight,
            fine_world, geometric);
        float2 rock_gradient = triplanar_height_derivatives(RockHeight, input.world, geometric);
        float2 layered_gradient = rock_gradient * base +
            triplanar_height_derivatives(TopHeight, input.world, geometric) * top +
            triplanar_height_derivatives(SnowHeight, input.world, geometric) * snow;
        rock_derivatives = lerp(layered_gradient, rock_gradient, snow * 0.60) * 0.04;
        rock_derivatives += triplanar_height_derivatives(RockHeight, fine_world, geometric) * 0.12 * (1-snow);

        mountain_detail += (fine_rock - 0.5) * 0.12 * (1 - snow);
        float neighborhood = triplanar_neighborhood(RockHeight,
            input.world, geometric).r;
        float fine_neighborhood = triplanar_neighborhood(RockHeight,
            fine_world, geometric).r;
        rock_crevice = lerp(rock_crevice_visibility(rock_detail, neighborhood) *
            rock_crevice_visibility(fine_rock, fine_neighborhood), 1.0, snow);
        // Normalize fine authored color against its local mean: retain the
        // stone grain without painting a second broad color field over it.
        float3 fine_color = triplanar(RockColor, fine_world, geometric);
        float3 mean_color = triplanar_neighborhood(RockColor, fine_world, geometric);
        float fine_luma = dot(fine_color, float3(0.2126, 0.7152, 0.0722));
        float mean_luma = dot(mean_color, float3(0.2126, 0.7152, 0.0722));
        float grain = clamp(1 + 2.0 * (fine_luma - mean_luma) / max(mean_luma, 0.02), 0.50, 1.18);
        float rock_micro_relief = smoothstep(0.16, 0.84, rock_detail);
        mountain_albedo *= lerp(0.82, 1.06, rock_micro_relief);
        mountain_albedo *= lerp(1.0, grain, 1 - snow);
        albedo = lerp(ground_albedo, mountain_albedo, rock_albedo_coverage);
        height_detail = lerp(ground_height, mountain_detail, rock_detail_coverage);
        float mountain_specular = triplanar_scalar(RockSpecular, input.world, geometric) * base +
                                  triplanar_scalar(TopSpecular, input.world, geometric) * top +
                                  triplanar_scalar(SnowSpecular, input.world, geometric) * snow;
        specular_map = lerp(ground_specular, mountain_specular, rock_detail_coverage);
    }
    if (input.material.y > 1.5 && Quality.x > 0.5) {
        // Civ VI's broad cool skylight is a major part of its gray-rock read;
        // preserve authored luminance/detail while avoiding raw brown albedo.
        // Apply it only to the rock footprint; grading the inherited ground
        // was the visible gray halo at plains and shoreline transitions.
        float rock_luma = dot(albedo, float3(0.2126, 0.7152, 0.0722));
        float3 graded = lerp(albedo, rock_luma.xxx, 0.28) * float3(0.96, 1.0, 1.07);
        albedo = lerp(albedo, graded, rock_albedo_coverage);
    }
    // Apply the retained normal gain to separately calibrated broad/fine
    // gradients. This changes shading only, never the geometric surface.
    float rock_normal_strength = Quality.z * lerp(1.0, 1.60, rock_detail_coverage);
    float3 normal = Quality.x > 0.5 ? height_derivative_normal(geometric, input.world, lerp(float2(ddx(ground_height), ddy(ground_height)), rock_derivatives, rock_detail_coverage), rock_normal_strength) : geometric;
#ifdef BEAUTY_COMPOSED_SHADOWS
    // The shared shadow-frame light is authoritative for both the BRDF and
    // projection, so every mountain face and cast shadow agrees in direction.
    float3 light_direction = ShadowL.xyz;
    // The shared field still supplies the directional cast/self shadow, but a
    // mountain must not apply the generic near-contact clamp back onto its own
    // continuous shell; that clamp traces a dark ring around low rock slopes.
    float received_shadow = q6_shadow_visibility(ShadowField, input.world, normal,
        ShadowU, ShadowV, ShadowL, ShadowFlags.x > 0.5, false);
    // This is now one terrain-relief surface, so its ground portion receives
    // the same directional field as the raised rock. Coastal sky fill follows
    // the terrain receiver's continuous shoreline response.
    float coast_inland = smoothstep(0.18, 0.86, coast_alpha);
    float shadow = lerp(lerp(1.0, received_shadow, 0.48),
                        received_shadow, coast_inland);
#else
    float3 light_direction = Sun.xyz;
    float shadow = horizon_visibility(input.world);
#endif
    float ndl = saturate(dot(normal, light_direction));
    float wrap_bias = lerp(0.20, 0.18, rock_albedo_coverage);
    float wrap = saturate((dot(normal, light_direction) + wrap_bias) /
                          (1.0 + wrap_bias));
    float altitude = input.material.y > 1.5 ? input.material.x : 0;
    float mountain_cavity = lerp(0.76, 1.0, smoothstep(0.03, 0.48, altitude));
    #ifdef BEAUTY_TERRAIN_TRANSITIONS
    float ground_cavity = lerp(0.79, 1.0,
        smoothstep(0.02, 0.30, input.base_relief));
    #else
    float ground_cavity = 1.0;
    #endif
    float cavity = lerp(ground_cavity, mountain_cavity, rock_albedo_coverage);
    float crevice_visibility = lerp(1.0, rock_crevice, rock_albedo_coverage);
    float sky = saturate(normal.z * 0.5 + 0.5);
    float ambient_floor = lerp(0.56, 0.52, rock_albedo_coverage);
    float3 ambient = Ambient.rgb * Ambient.a * lerp(ambient_floor, 1.0, sky) * cavity * crevice_visibility;
    float diffuse_floor = lerp(0.07, 0.055, rock_albedo_coverage);
    float3 diffuse = albedo * (ambient + SunColorExposure.rgb * Sun.w *
                               (diffuse_floor + (1.0 - diffuse_floor) * wrap) * shadow *
                               lerp(1.0, crevice_visibility, 0.70));
    float ground_roughness = lerp(0.92, 0.48, saturate(specular_map));
    float rock_roughness = lerp(0.88, 0.38, saturate(specular_map));
    float roughness = lerp(ground_roughness, rock_roughness, rock_detail_coverage);
    float specular = Quality.x > 0.5 ? ggx(normal, light_direction,
        normalize(View.xyz), roughness, lerp(0.035, 0.045, rock_detail_coverage)) : 0;
    float rim = Quality.x > 0.5 ? pow(1 - saturate(dot(normal, normalize(View.xyz))), 3) *
        saturate(dot(normal, -light_direction) * 0.5 + 0.5) : 0;
    float3 radiance = diffuse + SunColorExposure.rgb * Sun.w * specular * shadow +
                      Ambient.rgb * rim * 0.13;
    // The relief patch owns its terrain as well as its rock. Only the
    // authoritative coast mask may make it transparent; there is no second
    // ground surface beneath it and therefore no collar to cross-fade.
    output.color = float4(max(radiance, 0) * coast_alpha, coast_alpha);
    output.validity = coast_alpha;
    return output;
}

Output PSMain(P input) { return shade(input); }
Output PSFeature(P input) { return shade(input); }

cbuffer NativeViewport : register(b1) {
 float2 translation; float depth_translation; float padding;
 float2 inverse_size; float2 reserved;
 float4 natural_projection; // owner column/row, tile width, target height; zero disables
};
float3 native_project_position(float3 position, float3 world) {
 if(natural_projection.z<=0) return position;
 float dx=world.x-natural_projection.x,dy=world.y-natural_projection.y;
 float h=world.z*112-2.5;
 float base=(dx-dy+1)*natural_projection.z*.25;
 return float3((dx+dy)*natural_projection.z*.5,
     base-h*(natural_projection.z/224*.82),
     base+h*.0016*natural_projection.w);
}
P VSNative(V input) {
 input.position=native_project_position(input.position,input.world.xyz);
 P o=VSMain(input);
 o.position.xy=(floor(input.position.xy*256+0.5)/256+translation)*inverse_size*float2(2,-2)+float2(-1,1);
 o.position.z=clamp(0.5-(floor(input.position.z*256+0.5)/256+translation.y)/16384.0,0.001,0.999);
 return o;
}

cbuffer NativeReflectionFrame : register(b5) {
 float4 NativeReflection; // world-height to native pixels, depth metric, plane Z, enabled
 float4 NativeReflectionTarget; // internal extent XY, sampling guard XY
};

P VSReflection(V input) {
 P o=VSNative(input);
 float h=max(0,input.world.z-NativeReflection.z);
 o.position.y-=h*NativeReflection.x*4*inverse_size.y;
 float base=native_project_position(input.position,input.world.xyz).y+h*NativeReflection.x;
 o.position.z=clamp(.5-(floor((base-h*NativeReflection.y)*256+.5)/256+translation.y)/16384,.001,.999);
 return o;
}
float4 PSReflection(P input):SV_Target {
 clip(input.world.z-NativeReflection.z-.0001);
 return PSMain(input).color;
}

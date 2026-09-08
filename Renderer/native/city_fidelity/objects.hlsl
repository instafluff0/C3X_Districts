#define BEAUTY_COMPOSED_SHADOWS 1
cbuffer Frame : register(b0) {
    float4 Sun;
    float4 SunColorExposure;
    float4 Ambient;
    float4 View;
    float4 Quality;
};
Texture2D GroundColor : register(t0);
Texture2D GroundHeight : register(t1);
Texture2D GroundSpecular : register(t2);
Texture2D BaseColor : register(t3);
Texture2D Normal0 : register(t4);
Texture2D Normal1 : register(t5);
Texture2D AmbientOcclusion : register(t6);
Texture2D Gloss : register(t7);
Texture2D Emissive : register(t8);
Texture2D Opacity : register(t9);
#ifdef BEAUTY_COMPOSED_SHADOWS
Texture2DArray ShadowField : register(t17);
cbuffer ShadowFrame : register(b2) {
    float4 ShadowU;
    float4 ShadowV;
    float4 ShadowL;
    float4 ShadowOrigin;
    float4 ShadowFlags;
};
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
float q6_world_visibility(Texture2DArray field,float4 world,float3 normal,bool water) {
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

float q6_shadow_visibility(Texture2DArray field,float3 world,float3 normal,float4 u,float4 v,float4 l,bool receive,bool contact) {
 return receive?q6_world_visibility(field,float4(world,1),normal,!contact):1;
}

#endif
SamplerState Wrap : register(s0);
SamplerState Clamp : register(s1);

struct V {
    float3 position : POSITION;
    float4 world : TEXCOORD0;
    float3 normal : NORMAL;
    float2 uv : TEXCOORD1;
    float4 material : TEXCOORD2;
    float2 secondary : TEXCOORD3;
};
struct P {
    float4 position : SV_POSITION;
    float3 world : TEXCOORD0;
    float3 normal : NORMAL;
    float2 uv : TEXCOORD1;
    float4 material : TEXCOORD2;
    float2 secondary : TEXCOORD3;
};
struct Output { float4 color : SV_Target0; float validity : SV_Target1; };

P VSMain(V input) {
    P output;
    output.position = float4(input.position, 1);
    output.world = input.world.xyz;
    output.normal = input.normal;
    output.uv = input.uv;
    output.material = input.material;
    output.secondary = input.secondary;
    return output;
}
P VSFeature(V input) { return VSMain(input); }

float3 height_normal(float3 geometric, float3 world, float height_value, float strength) {
    float3 dx = ddx(world), dy = ddy(world);
    float3 r1 = cross(dy, geometric), r2 = cross(geometric, dx);
    float determinant = dot(dx, r1);
    float3 gradient = (ddx(height_value) * r1 + ddy(height_value) * r2) *
        sign(determinant) / max(abs(determinant), 0.000001);
    return normalize(geometric - gradient * strength);
}

float3 mapped_normal(float3 geometric, float3 world, float2 uv, bool wrapping) {
    float3 dp1 = ddx(world), dp2 = ddy(world);
    float2 duv1 = ddx(uv), duv2 = ddy(uv);
    float3 tangent = normalize(dp1 * duv2.y - dp2 * duv1.y);
    tangent = normalize(tangent - geometric * dot(geometric, tangent));
    float3 bitangent = normalize(cross(geometric, tangent));
    float2 encoded = (wrapping ? Normal0.Sample(Wrap, uv) :
                                 Normal0.Sample(Clamp, uv)).rg * 2 - 1;
    float lean = wrapping ? Normal1.Sample(Wrap, uv).r :
                            Normal1.Sample(Clamp, uv).r;
    encoded *= lerp(0.68, 1.0, lean);
    float z = sqrt(saturate(1 - dot(encoded, encoded)));
    return normalize(tangent * encoded.x + bitangent * encoded.y + geometric * z);
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


// Eight independently bounded city fields fit within one D3D11 constant
// buffer. The CPU selects intersecting cities per guarded block; overflow
// fails the candidate draw instead of truncating a city's emitting facades.
cbuffer NativeCityLights : register(b6) {
 float4 CityLightCounts;
 float4 Q8LocalEnvelopeLow4;float4 Q8LocalEnvelopeHigh4;
 float4 Q8LocalPositionRange[1024];float4 Q8LocalColorIntensity[1024];
 float4 Q8LocalDirectionOwner[1024];float4 Q8LocalBoxLow[256];float4 Q8LocalBoxHigh[256];
};
#define Q8_LOCAL_LIGHT_COUNT int(CityLightCounts.x)
#define Q8_LOCAL_BLOCKER_COUNT int(CityLightCounts.y)
#define Q8LocalEnvelopeLow Q8LocalEnvelopeLow4.xyz
#define Q8LocalEnvelopeHigh Q8LocalEnvelopeHigh4.xyz
#define Q8_LOCAL_Z_METRIC 0.648266978876
#define Q8_LOCAL_LIGHT_GAIN 4
// Generic, bounded emissive-facade light proxies. Positions/colors are derived
// offline from normalized source materials, not recovered source light bindings.
#ifndef Q8_LOCAL_LIGHT_GAIN
#define Q8_LOCAL_LIGHT_GAIN 1
#endif
#ifndef Q8_LOCAL_OCCLUSION
#define Q8_LOCAL_OCCLUSION 1
#endif
bool q8_local_box_blocks(float3 start,float3 finish,float3 low,float3 high) {
 float3 ray=finish-start;
 float3 safe_ray=float3(ray.x<0?-max(abs(ray.x),1e-6):max(abs(ray.x),1e-6),
                        ray.y<0?-max(abs(ray.y),1e-6):max(abs(ray.y),1e-6),
                        ray.z<0?-max(abs(ray.z),1e-6):max(abs(ray.z),1e-6));
 float3 a=(low-start)/safe_ray,b=(high-start)/safe_ray;
 float3 entry=min(a,b),leave=max(a,b);
 float near_t=max(entry.x,max(entry.y,entry.z));
 float far_t=min(leave.x,min(leave.y,leave.z));
 return far_t>=max(near_t,.001) && near_t<.995;
}
float3 q8_local_irradiance(float4 world,float3 normal,float ambient_visibility) {
 if(world.w<.5 || CityLightCounts.z<=0 || Q8_LOCAL_LIGHT_GAIN<=0)return 0;
 float3 receiver_position=float3(world.x,-world.y,world.z*Q8_LOCAL_Z_METRIC);
 if(any(receiver_position<Q8LocalEnvelopeLow) || any(receiver_position>Q8LocalEnvelopeHigh))return 0;
 float3 light_sum=0;
 [loop]for(int i=0;i<Q8_LOCAL_LIGHT_COUNT;i++) {
  float3 to_light=Q8LocalPositionRange[i].xyz-receiver_position;
  float distance2=dot(to_light,to_light);
  float range=Q8LocalPositionRange[i].w;
  if(distance2>=range*range)continue;
  float3 direction=to_light*rsqrt(max(distance2,1e-8));
  float face=saturate(dot(Q8LocalDirectionOwner[i].xyz,-direction));
  float diffuse=saturate(dot(normal,direction));
  if(face*diffuse<=0)continue;
  bool blocked=false;
#if Q8_LOCAL_OCCLUSION
  [loop]for(int j=0;j<Q8_LOCAL_BLOCKER_COUNT;j++) {
   if(j==int(Q8LocalDirectionOwner[i].w))continue;
   if(q8_local_box_blocks(Q8LocalPositionRange[i].xyz,receiver_position,Q8LocalBoxLow[j].xyz,Q8LocalBoxHigh[j].xyz)) {blocked=true;break;}
  }
#endif
  if(blocked)continue;
  float normalized_distance=distance2/(range*range);
  float attenuation=pow(1-normalized_distance,2)/(1+8*normalized_distance);
  light_sum+=Q8LocalColorIntensity[i].rgb*Q8LocalColorIntensity[i].w*attenuation*face*diffuse;
 }
 return light_sum*(Q8_LOCAL_LIGHT_GAIN*CityLightCounts.z*CityLightCounts.w*ambient_visibility);
}

Output shade(P input) {
    Output output;
    float kind = input.material.x;
    if (kind > 3.5 && kind < 4.5) {
        if (input.secondary.y > 0.5)
            clip(Opacity.Sample(Wrap, input.uv).r - 0.5);
        float alpha = input.secondary.x;
        output.color = float4(float3(0.018, 0.026, 0.034) * alpha, alpha);
        output.validity = 1;
        return output;
    }
    if (kind > 4.5 && kind < 5.5) {
        float wave = sin(input.world.x * 18.0 + input.world.y * 11.0) * 0.5 +
                     sin(input.world.x * 7.0 - input.world.y * 23.0) * 0.5;
        float sparkle = pow(saturate(0.5 + wave * 0.5), 8.0);
        float3 water = lerp(float3(0.035, 0.115, 0.17),
                            float3(0.18, 0.34, 0.39), 0.46 + wave * 0.06);
        water += SunColorExposure.rgb * sparkle * 0.055;
        output.color = float4(water, 1);
        output.validity = 1;
        return output;
    }
    if (kind > 5.5) {
        float noise = GroundHeight.Sample(Wrap, input.world.xy * 0.42).r;
        float3 bank = lerp(float3(0.22, 0.17, 0.09),
                           float3(0.42, 0.33, 0.18), noise);
        output.color = float4(bank * (Ambient.rgb * 0.34 +
                              SunColorExposure.rgb * Sun.w * 0.48), 1);
        output.validity = 1;
        return output;
    }

    float3 geometric = normalize(input.normal);
    float3 albedo;
    float ao = 1;
    float gloss = 0.08;
    float3 normal = geometric;
    if (kind < 0.5) {
        float2 uv = input.world.xy * 0.27 + 0.5;
        albedo = GroundColor.Sample(Wrap, uv).rgb;
        float luma = dot(albedo, float3(0.2126, 0.7152, 0.0722));
        albedo = lerp(albedo, luma.xxx, 0.14) * float3(1.03, 1.0, 0.92);
        normal = height_normal(geometric, input.world, GroundHeight.Sample(Wrap, uv).r,
                               Quality.x);
        gloss = GroundSpecular.Sample(Wrap, uv).r;
    } else {
        bool foliage = kind > 0.5 && kind < 1.5;
        bool repeat_address = input.material.y > 1.5;
        bool has_normal = fmod(input.material.y, 2.0) > 0.5;
        // Civ V foliage UVs intentionally cross the 0..1 boundary. Clamping
        // them was the deeper cause of the solid green/yellow crowns: those
        // triangles repeatedly sampled an atlas edge instead of wrapping.
        float4 base = (foliage || repeat_address) ? BaseColor.Sample(Wrap, input.uv) :
                                                   BaseColor.Sample(Clamp, input.uv);
        if (foliage && input.secondary.y > 0.5)
            clip(Opacity.Sample(Wrap, input.uv).r - 0.5);
        albedo = base.rgb;
        if (input.secondary.x > 0.5) {
            float owner = (1 - base.a) * 0.82;
            float3 blue = float3(0.12, 0.35, 0.82);
            albedo = lerp(albedo, albedo * (0.45 + blue * 1.10), owner);
        }
        if (has_normal)
            normal = mapped_normal(geometric, input.world, input.uv,
                                   foliage || repeat_address);
        if (input.material.z > 0.5)
            ao = lerp(0.48, 1.0, (repeat_address ?
                AmbientOcclusion.Sample(Wrap, input.uv) :
                AmbientOcclusion.Sample(Clamp, input.uv)).r);
        if (input.material.w > 0.5)
            gloss = foliage ? Gloss.Sample(Wrap, input.uv).r :
                    (repeat_address ? Gloss.Sample(Wrap, input.uv).r :
                                      Gloss.Sample(Clamp, input.uv).r);
    }

#ifdef BEAUTY_COMPOSED_SHADOWS
    // Use the exact vector that projects the Q6 field. This makes canopy face
    // shading and every tree/mountain cast shadow share one sun direction.
    float3 light_direction = ShadowL.xyz;
#else
    float3 light_direction = Sun.xyz;
#endif
    float ndl = saturate(dot(normal, light_direction));
    float diffuse = ndl;
#ifdef BEAUTY_COMPOSED_SHADOWS
    float shadow = q6_shadow_visibility(ShadowField, input.world, normal,
        ShadowU, ShadowV, ShadowL, ShadowFlags.x > 0.5, true);
#else
    float shadow = 1.0;
#endif
    float sky = saturate(normal.z * 0.5 + 0.5);
    // Foliage needs readable key/fill separation at map scale; preserve a soft
    // skylight, but do not let it erase the canopy's shaded faces.
    float foliage_fill = kind > 0.5 && kind < 1.5 ? 0.72 : 1.0;
    float3 ambient = Ambient.rgb * Ambient.a * lerp(0.48, 1.0, sky) * ao * foliage_fill;
    float3 radiance = albedo * (ambient + SunColorExposure.rgb * Sun.w *
                                (0.035 + 0.965 * diffuse) * shadow);
    float roughness = lerp(0.91, 0.31, saturate(gloss));
    if (kind > 0.5 && kind < 1.5) roughness = max(roughness, 0.72);
    float specular_scale = kind > 0.5 && kind < 1.5 ? 0.10 : 0.52;
    radiance += SunColorExposure.rgb * Sun.w * specular_scale *
                ggx(normal, light_direction, normalize(View.xyz), roughness,
                    kind > 0.5 && kind < 1.5 ? 0.020 : 0.045) * shadow;
    if (!(kind > 0.5 && kind < 1.5) && input.secondary.y > 0.5)
        radiance += Emissive.Sample(Clamp, input.uv).rgb * 0.035;
    float rim = pow(1 - saturate(dot(normal, normalize(View.xyz))), 3);
    radiance += Ambient.rgb * rim * 0.08;
    radiance+=albedo*q8_local_irradiance(float4(input.world,1),normalize(normal*float3(1,-1,1/Q8_LOCAL_Z_METRIC)),1);
    output.color = float4(max(radiance, 0), 1);
    output.validity = 1;
    return output;
}

Output PSMain(P input) { return shade(input); }
Output PSFeature(P input) { return shade(input); }

cbuffer NativeViewport : register(b1) {
 float2 translation; float depth_translation; float padding;
 float2 inverse_size; float2 reserved;
};
P VSNative(V input) {
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
 float base=input.position.y+h*NativeReflection.x;
 o.position.z=clamp(.5-(floor((base-h*NativeReflection.y)*256+.5)/256+translation.y)/16384,.001,.999);
 return o;
}
float4 PSReflection(P input):SV_Target {
 clip(input.world.z-NativeReflection.z-.0001);
 return PSMain(input).color;
}

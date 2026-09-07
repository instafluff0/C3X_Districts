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
SamplerState Wrap : register(s0);
SamplerState Clamp : register(s1);

struct V {
    float3 position : POSITION;
    float3 world : TEXCOORD0;
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
    output.world = input.world;
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

    float ndl = saturate(dot(normal, Sun.xyz));
    float diffuse = ndl;
    float sky = saturate(normal.z * 0.5 + 0.5);
    float3 ambient = Ambient.rgb * Ambient.a * lerp(0.48, 1.0, sky) * ao;
    float3 radiance = albedo * (ambient + SunColorExposure.rgb * Sun.w *
                                (0.035 + 0.965 * diffuse));
    float roughness = lerp(0.91, 0.31, saturate(gloss));
    if (kind > 0.5 && kind < 1.5) roughness = max(roughness, 0.72);
    float specular_scale = kind > 0.5 && kind < 1.5 ? 0.10 : 0.52;
    radiance += SunColorExposure.rgb * Sun.w * specular_scale *
                ggx(normal, Sun.xyz, normalize(View.xyz), roughness,
                    kind > 0.5 && kind < 1.5 ? 0.020 : 0.045);
    if (!(kind > 0.5 && kind < 1.5) && input.secondary.y > 0.5)
        radiance += Emissive.Sample(Clamp, input.uv).rgb * 0.035;
    float rim = pow(1 - saturate(dot(normal, normalize(View.xyz))), 3);
    radiance += Ambient.rgb * rim * 0.08;
    output.color = float4(max(radiance, 0), 1);
    output.validity = 1;
    return output;
}

Output PSMain(P input) { return shade(input); }
Output PSFeature(P input) { return shade(input); }

// Appended to production's BIQ hydrology declarations. This is a D3D11
// adaptation of 0 A.D.'s water_high.fs getNormal/getSpecular/getReflection
// and main Fresnel blend. Prepared coast geometry and source textures remain.
float4 ShadeWaterSurface(PixelInput input) {
    clip(-input.hydrology_data.x - 0.0001);
    float depth = max(0, input.hydrology_data.w);
    float coastal_detail = 1 - smoothstep(.30, .39, depth);
    float2 world = q3_source_world(input);

    // 0 A.D. blends animated normals before Fresnel and reflected light.
    // The available BIQ pack has broad and fine source normal bands rather
    // than 60 frames, so drift and blend those two resident textures.
    float2 broad = water_large_lean0_texture.Sample(material_sampler,
        world * float2(q3_source_repeat(.40), q3_source_repeat(.54)) +
        Q3_WATER_DRIFT.xy).rg * 2 - 1;
    float2 fine = water_small_lean0_texture.Sample(material_sampler,
        world * float2(q3_source_repeat(.95), q3_source_repeat(1.24)) +
        float2(.27, .61) + float2(-Q3_WATER_DRIFT.y, Q3_WATER_DRIFT.z)).rg * 2 - 1;
    float2 slope = broad * .45 + fine * .55;
    float flatten = .5 + smoothstep(.015, .32, depth) * .5;
    float3 normal = normalize(float3(-slope * flatten, 1));

    // Civ III's pixel projection supplies a finite reflected view direction.
    float2 delta = world - Q3_WATER_CAMERA.xy;
    float2 wrapped = float2(delta.x + delta.y, delta.x - delta.y);
    wrapped -= round(wrapped / max(Q3_WATER_CAMERA.zw, 1)) * Q3_WATER_CAMERA.zw;
    delta = float2(wrapped.x + wrapped.y, wrapped.x - wrapped.y) * .5;
    float3 eye = normalize(float3(float2(.43, -.43) * 2.5 - delta, 2.5));
    float shadow = 1;
    if (coastal_detail > 0)
        shadow = q6_receiver_visibility(input, float3(0, 0, 1), 1);
    shadow = lerp(1, shadow, coastal_detail);
    float3 light = frame_illumination(float3(0, 0, 1), shadow, 1);

    // 0 A.D. separates depth-tinted refraction from the reflected sky and
    // objects, then mixes them with a bounded view/normal Fresnel term.
    float3 refracted = lerp(float3(.023, .074, .096),
        float3(.003, .015, .040), smoothstep(.18, .43, depth)) * light;
    refracted *= 1 + dot(normal.xy, float2(.85, -.65));
    float3 reflected_ray = reflect(-eye, normal);
    float3 sky_light = environment_ambient_color * .6 +
        environment_sun_color * environment_sun_intensity * .6 +
        environment_moon_color * environment_moon_intensity * .6;
    float3 reflected = sky_light * lerp(float3(.16, .25, .36),
        float3(.42, .55, .68), smoothstep(.30, .90, reflected_ray.y));
    float2 reflected_uv = (input.position.xy + NativeReflectionTarget.zw) /
        NativeReflectionTarget.xy;
    float ref_vy = clamp(eye.z * 2, .05, 1);
    float2 distorted_uv = reflected_uv -
        normal.xy * float2(3, 1.5) / (NativeReflectionTarget.xy * ref_vy);
    float object_alpha = 0;
    if (coastal_detail > 0) {
        float4 object = q3_object_reflection_texture.Sample(decal_sampler, distorted_uv);
        float inside = step(0, reflected_uv.x) * step(reflected_uv.x, 1) *
            step(0, reflected_uv.y) * step(reflected_uv.y, 1) * NativeReflection.w;
        object_alpha = saturate(object.a) * inside * coastal_detail;
        reflected = lerp(reflected, object.rgb, object_alpha);
    }
    float reflection_strength = max(object_alpha, .4);
    float ndotv = saturate(dot(normal, eye));
    float fresnel = clamp(pow(1.1 - ndotv, 2) * 1.5, .1, .75);
    fresnel = lerp(fresnel, fresnel * shadow, .09);

    float3 specular_direction = reflect(-environment_sun_direction, normal);
    float specular = smoothstep(.82, .995,
        max(dot(specular_direction, eye), 0)) * .9;
    float3 sun = environment_sun_color * environment_sun_intensity;
    float blend = fresnel * reflection_strength * .68;
    float3 color = lerp(refracted, reflected, blend) +
        shadow * saturate(specular * sun);
    // 0 A.D.'s water_high getFoam uses animated normal detail plus shoreline
    // coverage. The prepared BIQ coastal depth supplies that coverage here,
    // so the far-zoom scene needs no separate short-strip wave meshes.
    float foam_band = smoothstep(.055, .085, depth) *
        (1 - smoothstep(.105, .155, depth));
    float foam_pattern = smoothstep(.27, .55,
        broad.x * .42 + fine.y * .58 + .12);
    float foam = foam_band * foam_pattern * coastal_detail * .34;
    color = lerp(color, light * float3(.46, .60, .63), foam);
    float coverage = 1 - exp(-depth * lerp(2.3, 3.2,
        smoothstep(.10, .32, depth)));
    float alpha = coverage + (1 - coverage) * blend;
    alpha = lerp(alpha, 1, foam * .65);
    return q6_scene_output(float4(color, alpha)).color;
}
float4 PSWaterSurface(PixelInput input) : SV_Target {
    return ShadeWaterSurface(input);
}

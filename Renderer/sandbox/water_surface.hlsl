// Appended to production's BIQ hydrology declarations. This is a D3D11
// adaptation of 0 A.D.'s water_high.fs getNormal/getSpecular/getReflection
// and main Fresnel blend. Prepared coast geometry and source textures remain.
cbuffer SandboxAquaticBounds : register(b11) { float4 aquatic_bounds; };
float4 ShadeWaterSurface(PixelInput input) {
    clip(-input.hydrology_data.x - 0.0001);
    float depth = max(0, input.hydrology_data.w);
    float coastal_detail = 1 - smoothstep(.30, .39, depth);
    // Keep long ocean glints out of the narrow coastal wave zone. Distance is
    // the prepared signed shoreline field, so deep inlets still count as coast.
    float offshore = max(0, -input.hydrology_data.x);
    float open_ocean = smoothstep(.25, 1.65, offshore);
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
    // The prepared surface coordinate blends the authoritative Civ III water
    // families: coast .34, sea .68, ocean 1.0. Keep the ocean endpoint intact.
    float family = saturate(input.surface_coordinate);
    float coast_family = 1 - smoothstep(.34, .63, family);
    float sea_family = (1 - coast_family) * (1 - smoothstep(.65, .99, family));
    refracted += light * (float3(.012, .058, .046) * coast_family +
                          float3(.006, .026, .023) * sea_family);
    refracted *= 1 + dot(normal.xy, float2(.85, -.65));
    // t123 is the water variant's marine-color layer. Sample it through the
    // moving normal so fish and whales inherit surface refraction, reflection
    // and foam rather than being composited over the finished ocean.
    float4 aquatic = 0;
    if (all(input.position.xy >= aquatic_bounds.xy) &&
        all(input.position.xy < aquatic_bounds.zw)) {
        uint aquatic_width, aquatic_height;
        resource_base_texture_7.GetDimensions(aquatic_width, aquatic_height);
        float2 aquatic_uv = (input.position.xy - normal.xy * 5) /
            max(float2(aquatic_width, aquatic_height), 1);
        aquatic = resource_base_texture_7.SampleLevel(decal_sampler, aquatic_uv, 0);
    }
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
    float3 color = lerp(refracted, reflected, blend);
    float marine_coverage = saturate(aquatic.a * 1.6) * (1 - blend);
    float3 marine_body = aquatic.rgb / max(aquatic.a, .01);
    color = lerp(color, color * float3(.12, .30, .42) +
        marine_body * .035, marine_coverage);
    color += shadow * saturate(specular * sun) * lerp(.06, 1, open_ocean);
    // The broad low-sun reflection shares the forest-ray time window. Keep
    // the wave normals and angular specular term so this remains moving water,
    // while the noon and night ocean retain their established appearance.
    float low_sun = smoothstep(.10, .25, environment_sun_intensity) *
        (1 - smoothstep(.65, .82, environment_sun_intensity));
    float broad_glint = smoothstep(.10, .78,
        saturate(dot(specular_direction, eye)));
    float wave_glimmer = saturate(.42 + broad.x * .32 + fine.y * .30);
    float twilight_glint = shadow * low_sun * open_ocean *
        broad_glint * wave_glimmer;
    color += sun * twilight_glint * .35;
    // Keep a hint of the earlier purple-red twilight shimmer in the moving
    // specular path, rather than tinting the whole daytime ocean.
    color += float3(.045, .010, .032) * twilight_glint *
        environment_sun_intensity;
    // Moonlight follows the same moving normals and open-water selection as
    // the sun, with its own reflected direction and a neutral white glint.
    // Q6ShadowL is the active shadow-map light vector. Once night takes over,
    // the highlight therefore points back toward the shadow-casting moon.
    float3 moon_specular_direction = reflect(-normalize(Q6ShadowL.xyz), normal);
    float moon_facing = saturate(dot(moon_specular_direction, eye));
    float moon_broad = smoothstep(.10, .78, moon_facing);
    float moon_narrow = smoothstep(.82, .995, moon_facing);
    float moon_glint = shadow * environment_moon_intensity *
        smoothstep(.18, .28, environment_moon_intensity) *
        open_ocean * wave_glimmer;
    color += float3(1, 1, 1) * moon_glint *
        (moon_broad * .55 + moon_narrow * .65);
    // 0 A.D.'s water_high getFoam uses animated normal detail plus shoreline
    // coverage. The prepared BIQ coastal depth supplies that coverage here,
    // so the far-zoom scene needs no separate short-strip wave meshes.
    float foam_band = smoothstep(.055, .085, depth) *
        (1 - smoothstep(.105, .155, depth));
    float foam_pattern = smoothstep(.27, .55,
        broad.x * .42 + fine.y * .58 + .12);
    // The explicit breaker pass owns nearshore foam; the surface pattern
    // becomes visible only as that breaker zone gives way to open water.
    float foam = foam_band * foam_pattern * coastal_detail * open_ocean * .34;
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
float4 PSRiverSurface(PixelInput input) : SV_Target {
    // Production preparation supplies the river, wet bank and shoreline
    // materials. The sandbox's common scene shadow field also darkens their
    // ambient response where nearby terrain or vegetation blocks moon/sun.
    float4 river = q6_raw_main(input);
    float visibility = q6_receiver_visibility(input, float3(0, 0, 1), 1);
    river.rgb *= lerp(.25, 1, visibility);
    // River banks already use the prepared beach sand. At low sun their
    // separate river lighting ran hotter than the adjacent coastal shore.
    float low_sun = smoothstep(.10, .25, environment_sun_intensity) *
        (1 - smoothstep(.65, .82, environment_sun_intensity));
    float dry_bank = smoothstep(5.5, 9.5, input.river_data.x);
    river.rgb *= 1 - .20 * low_sun * dry_bank;
    return q6_scene_output(river).color;
}

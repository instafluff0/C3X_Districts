// Appended to the prepared city hydrology shader. The source declarations,
// material textures, world coordinates, reflection camera and light rig remain
// authoritative; this pass uses a bounded two-normal water variant.
float4 PSWaterSurface(PixelInput input) : SV_Target {
    clip(-input.hydrology_data.x - 0.0001);
    float depth = max(0, input.hydrology_data.w);
    float2 world = q3_source_world(input);
    float2 broad_uv = world * float2(q3_source_repeat(.40), q3_source_repeat(.54)) +
        Q3_WATER_DRIFT.xy;
    float2 fine_uv = world * float2(q3_source_repeat(.95), q3_source_repeat(1.24)) +
        float2(.27, .61) + float2(-Q3_WATER_DRIFT.y, Q3_WATER_DRIFT.z);
    float2 broad = water_large_lean0_texture.Sample(material_sampler, broad_uv).rg * 2 - 1;
    float2 fine = water_small_lean0_texture.Sample(material_sampler, fine_uv).rg * 2 - 1;
    float3 normal = normalize(float3(-(broad * .48 + fine * .30) *
        smoothstep(.015, .32, depth), 1));
    float2 delta = world - Q3_WATER_CAMERA.xy;
    float2 wrapped = float2(delta.x + delta.y, delta.x - delta.y);
    wrapped -= round(wrapped / max(Q3_WATER_CAMERA.zw, 1)) * Q3_WATER_CAMERA.zw;
    delta = float2(wrapped.x + wrapped.y, wrapped.x - wrapped.y) * .5;
    float3 view = normalize(float3(float2(.43, -.43) * 2.5 - delta, 2.5));
    float visibility = q6_receiver_visibility(input, float3(0, 0, 1), 1);
    float3 illumination = frame_illumination(float3(0, 0, 1), visibility, 1);
    float3 body = lerp(float3(.023, .074, .096), float3(.003, .015, .040),
        smoothstep(.18, .43, depth)) * illumination;
    float fresnel = saturate(environment_water_fresnel) +
        (1 - saturate(environment_water_fresnel)) *
        pow(1 - saturate(dot(normal, view)), 5);
    float3 ray = reflect(-view, normal);
    float3 sky_light = environment_ambient_color * .6 +
        environment_sun_color * environment_sun_intensity * .6 +
        environment_moon_color * environment_moon_intensity * .6;
    float3 sky = sky_light * lerp(float3(.16, .25, .36), float3(.42, .55, .68),
        smoothstep(.30, .90, ray.y));
    float2 reflected_uv = (input.position.xy + NativeReflectionTarget.zw) /
        NativeReflectionTarget.xy;
    float2 distortion = normal.xy * float2(3, 1.5) / NativeReflectionTarget.xy;
    float4 object = q3_object_reflection_texture.Sample(decal_sampler,
        reflected_uv + distortion);
    float inside = step(0, reflected_uv.x) * step(reflected_uv.x, 1) *
        step(0, reflected_uv.y) * step(reflected_uv.y, 1) * NativeReflection.w;
    sky = sky * (1 - saturate(object.a) * inside) + object.rgb * inside;
    float2 micro = water_small_lean0_texture.Sample(material_sampler,
        world * float2(q3_source_repeat(3.4), q3_source_repeat(4.12)) +
        float2(.71, .29) + float2(-Q3_WATER_DRIFT.z, Q3_WATER_DRIFT.y)).rg * 2 - 1;
    float3 glint_normal = normalize(normal + float3(-micro * .03, 0));
    float3 glint = (environment_sun_color * environment_sun_intensity *
        q3_water_glint(glint_normal, view, environment_sun_direction) +
        environment_moon_color * environment_moon_intensity *
        q3_water_glint(glint_normal, view, environment_moon_direction)) *
        3 * environment_water_specular * visibility;
    float coverage = 1 - exp(-depth * lerp(2.3, 3.2, smoothstep(.10, .32, depth)));
    float alpha = coverage + (1 - coverage) * fresnel;
    float3 premult = body * coverage * (1 - fresnel) + sky * fresnel + glint;
    return q6_scene_output(float4(premult / max(alpha, .0001), alpha)).color;
}

// Translate the authoritative Civ III viewport vertex contract into the
// approved terrain shader contract. Production keeps a frozen copy of the
// approved Lab implementation beside this thin Civ III input adapter, so
// in-progress Lab edits cannot change the game before their handoff.
#define C3X_GAME_RENDERER 1
#include "terrain_rendering.hlsl"

cbuffer C3XViewportSettings : register(b1)
{
    float2 c3x_viewport_translation;
    float c3x_viewport_depth_translation;
    float c3x_viewport_translation_padding;
    float2 c3x_inverse_viewport_size;
    float2 c3x_viewport_reserved;
};

struct IntegratedVertexInput
{
    float3 position : POSITION;
    float2 uv : TEXCOORD0;
    float panel : TEXCOORD1;
    float3 geometry_normal : NORMAL0;
    float2 shape_visibility : TEXCOORD2;
    float2 macro_uv : TEXCOORD3;
    float surface_kind : TEXCOORD4;
    float surface_coordinate : TEXCOORD5;
    float base_terrain : TEXCOORD6;
    float real_terrain : TEXCOORD7;
    float4 material_weights : TEXCOORD8;
    float2 authored_relief : TEXCOORD9;
    float shore_distance : TEXCOORD10;
    float4 river_data : TEXCOORD11;
    float active_effect : TEXCOORD12;
    float material_tundra : TEXCOORD13;
};

// Use a power-of-two pixel depth range. An integer camera movement now adds
// an exactly representable depth offset; coplanar neighbors cannot exchange
// their depth-test order because of viewport-dependent division roundoff.
float translated_depth(IntegratedVertexInput input, bool feature)
{
    float pixel_depth = floor(input.position.z * 256.0 + 0.5) / 256.0 + c3x_viewport_translation.y;
    float depth = clamp(0.5 - pixel_depth / 16384.0, 0.001, 0.999);
    if (feature) return depth;
    // Preserve the existing layer separation in physical pixels at each size.
    float bias_scale = c3x_viewport_reserved.x / 16384.0;
    float kind = input.surface_kind;
    if (kind > 10.5) return max(0.003, depth - 0.010 * bias_scale);
    if (kind > 9.5) return max(0.001, depth - 0.003 * bias_scale);
    if (kind > 8.5) return max(0.001, depth - 0.025 * bias_scale);
    if (kind > 6.5 && kind < 7.5) return max(0.001, depth - 0.004 * bias_scale);
    if (kind < 0.75) return min(0.999, depth + 0.000006 * bias_scale);
    if (kind < 1.25) return min(0.999, depth + 0.000004 * bias_scale);
    if (kind > 3.5 && kind < 4.5) return min(0.999, depth + 0.000002 * bias_scale);
    return depth;
}

// Match the rasterizer's subpixel grid before adding native integer anchors.
// Converting arbitrary local floats through NDC first can round a shared edge
// differently after scrolling, making an otherwise identical tile flicker.
float2 translated_position(IntegratedVertexInput input)
{
    float2 pixels = floor(input.position.xy * 256.0 + 0.5) / 256.0;
    return (pixels + c3x_viewport_translation) *
        c3x_inverse_viewport_size * float2(2.0, -2.0) + float2(-1.0, 1.0);
}

PixelInput VSIntegrated(IntegratedVertexInput input)
{
    PixelInput output;
    output.position = float4(translated_position(input),
                             translated_depth(input, false), 1.0);
    output.uv = input.uv;
    output.panel = input.panel;
    output.geometry_normal = input.geometry_normal;
    output.shape_visibility = input.shape_visibility;
    output.macro_uv = input.macro_uv;
    output.surface_kind = input.surface_kind;
    output.surface_coordinate = input.surface_coordinate;
    output.base_terrain = input.base_terrain;
    output.real_terrain = input.real_terrain;
    output.material_weights = input.material_weights;
    output.authored_relief = input.authored_relief;
    output.shore_distance = input.shore_distance;
    output.river_data = input.river_data;
    output.active_effect = input.active_effect;
    output.material_tundra = input.material_tundra;
    return output;
}

float4 PSIntegrated(PixelInput input) : SV_TARGET
{
    return PSMain(input);
}

FeaturePixelInput VSIntegratedFeature(IntegratedVertexInput input)
{
    FeaturePixelInput output;
    output.position = float4(translated_position(input),
                             translated_depth(input, true), 1.0);
    output.uv = input.uv;
    output.geometry_normal = input.geometry_normal;
    output.material_index = input.base_terrain;
    return output;
}

float4 PSIntegratedFeature(FeaturePixelInput input) : SV_TARGET
{
    return PSFeature(input);
}

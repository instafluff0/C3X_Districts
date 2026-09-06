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
};

PixelInput VSIntegrated(IntegratedVertexInput input)
{
    PixelInput output;
    output.position = float4(input.position.xy + c3x_viewport_translation,
                             input.position.z + c3x_viewport_depth_translation, 1.0);
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
    return output;
}

float4 PSIntegrated(PixelInput input) : SV_TARGET
{
    return PSMain(input);
}

FeaturePixelInput VSIntegratedFeature(IntegratedVertexInput input)
{
    FeaturePixelInput output;
    output.position = float4(input.position.xy + c3x_viewport_translation,
                             input.position.z + c3x_viewport_depth_translation, 1.0);
    output.uv = input.uv;
    output.geometry_normal = input.geometry_normal;
    output.material_index = input.base_terrain;
    return output;
}

float4 PSIntegratedFeature(FeaturePixelInput input) : SV_TARGET
{
    return PSFeature(input);
}

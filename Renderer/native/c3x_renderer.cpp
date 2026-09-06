#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <d3d11.h>
#include <d3dcompiler.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "c3x_renderer_api.h"
#include "terrain_scene_runtime.h"
#include "environment_runtime.h"
#include "terrain_definition_runtime.h"

namespace {

constexpr c3x_renderer_u32 viewport_cache_capacity = 8u;

struct Vertex {
    float x, y, z;
    float u, v;
    float panel;
    float normal_x, normal_y, normal_z;
    float shadow_visibility, ambient_visibility;
    float macro_u, macro_v;
    float surface_kind;
    float surface_coordinate;
    float base_terrain;
    float real_terrain;
    float material_grass, material_plains, material_desert, material_marsh;
    float authored_relief_height, authored_relief_blend;
    float shore_distance;
    float river_distance, river_branch_count, river_mouth_distance, river_padding;
    float active_effect;
};

// The production shader exposes only settings that are meaningful to the game.
// Standalone fixture and promotion switches are compile-time concerns outside
// this runtime contract.
struct TerrainShaderSettings {
    float height_texel[2];
    float normal_strength;
    float exposure;
    float light_direction[3];
    float sun_intensity;
    float sun_color[3];
    float shadow_strength;
    float moon_direction[3];
    float moon_intensity;
    float moon_color[3];
    float night_activation;
    float ambient_color[3];
    float environment_exposure;
    float water_fresnel;
    float water_specular;
    float emissive_scale;
    float hour;
};

struct TerrainTexture {
    ID3D11ShaderResourceView * view = nullptr;
    ID3D11ShaderResourceView * material_height_view = nullptr;
    ID3D11ShaderResourceView * specular_view = nullptr;
    ID3D11ShaderResourceView * elevated_view = nullptr;
    ID3D11ShaderResourceView * elevated_height_view = nullptr;
    ID3D11ShaderResourceView * elevated_specular_view = nullptr;
    std::array<ID3D11ShaderResourceView *, 5> relief_layer_views = {};
    std::array<ID3D11ShaderResourceView *, 5> water_surface_views = {};
    std::vector<std::uint8_t> dds;
    std::vector<std::uint8_t> material_height_dds;
    std::vector<std::uint8_t> specular_dds;
    std::vector<std::uint8_t> elevated_dds;
    std::vector<std::uint8_t> elevated_height_dds;
    std::vector<std::uint8_t> elevated_specular_dds;
    std::array<std::vector<std::uint8_t>, 5> relief_layer_dds;
    std::array<std::vector<std::uint8_t>, 5> water_surface_dds;
    std::vector<std::uint8_t> height_pixels;
    std::vector<std::uint8_t> blend_pixels;
    std::array<std::vector<std::uint8_t>, 5> relief_height_variants;
    std::array<std::vector<std::uint8_t>, 5> relief_blend_variants;
    std::array<std::uint32_t, 5> relief_variant_widths = {};
    std::array<std::uint32_t, 5> relief_variant_heights = {};
    std::array<float, 5> relief_height_minimum = {};
    std::array<float, 5> relief_height_maximum = {};
    std::array<float, 5> relief_blend_minimum = {};
    std::array<float, 5> relief_blend_maximum = {};
    std::uint32_t height_width = 0;
    std::uint32_t height_height = 0;
    float height_minimum = 0.0f;
    float height_maximum = 1.0f;
    float blend_minimum = 0.0f;
    float blend_maximum = 1.0f;
    float height_scale_px = 0.0f;
    int relief_profile = 0;
    bool configured = false;
};

struct CachedViewport {
    c3x_renderer::TerrainFrameSignature signature;
    std::vector<std::uint32_t> pixels;
    std::vector<c3x_renderer_tile_v1> tiles;
    std::vector<c3x_renderer_u32> replacement_flags;
    c3x_renderer_u32 rendered_tile_count = 0;
    c3x_renderer_u32 fallback_tile_count = 0;
    c3x_renderer_u32 textured_tile_count = 0;
};

class RendererState {
public:
    ID3D11Device * device = nullptr;
    ID3D11DeviceContext * context = nullptr;
    ID3D11VertexShader * vertex_shader = nullptr;
    ID3D11PixelShader * pixel_shader = nullptr;
    ID3D11VertexShader * feature_vertex_shader = nullptr;
    ID3D11PixelShader * feature_pixel_shader = nullptr;
    ID3D11InputLayout * input_layout = nullptr;
    ID3D11Buffer * terrain_settings_buffer = nullptr;
    ID3D11BlendState * blend_state = nullptr;
    ID3D11DepthStencilState * depth_state = nullptr;
    ID3D11RasterizerState * rasterizer_state = nullptr;
    ID3D11SamplerState * terrain_sampler = nullptr;
    ID3D11SamplerState * decal_sampler = nullptr;
    ID3D11Texture2D * render_texture = nullptr;
    ID3D11RenderTargetView * render_target = nullptr;
    ID3D11Texture2D * depth_texture = nullptr;
    ID3D11DepthStencilView * depth_target = nullptr;
    ID3D11Texture2D * readback_texture = nullptr;
    int width = 0;
    int height = 0;
    std::vector<std::uint32_t> pixels;
    std::vector<c3x_renderer_u32> fallback_tile_indices;
    std::vector<c3x_renderer_u32> replacement_tile_flags;
    std::array<TerrainTexture, c3x_renderer::terrain_type_count> terrain_textures;
    TerrainTexture dune_surface;
    c3x_renderer::FeatureBundle feature_bundle;
    std::array<std::vector<std::uint8_t>, 8> feature_texture_dds;
    std::array<ID3D11ShaderResourceView *, 8> feature_texture_views = {};
    // Extra terrain channels not already represented by TerrainTexture are
    // held here and bound to the shared shader's stable register contract.
    std::array<std::vector<std::uint8_t>, 19> terrain_extra_dds;
    std::array<ID3D11ShaderResourceView *, 19> terrain_extra_views = {};
    bool terrain_extra_assets_ready = false;
    bool authored_relief_assets_ready = false;
    std::string integrated_shader_path = "integrated_terrain.hlsl";
    std::vector<std::uint8_t> dune_decal_base_dds;
    std::vector<std::uint8_t> dune_decal_height_dds;
    ID3D11ShaderResourceView * dune_decal_base_view = nullptr;
    ID3D11ShaderResourceView * dune_decal_height_view = nullptr;
    std::vector<std::uint8_t> marsh_decal_base_dds;
    std::vector<std::uint8_t> marsh_decal_height_dds;
    std::vector<std::uint8_t> marsh_decal_specular_dds;
    ID3D11ShaderResourceView * marsh_decal_base_view = nullptr;
    ID3D11ShaderResourceView * marsh_decal_height_view = nullptr;
    ID3D11ShaderResourceView * marsh_decal_specular_view = nullptr;
    std::vector<std::uint8_t> volcano_base_dds;
    std::vector<std::uint8_t> volcano_height_dds;
    std::vector<std::uint8_t> volcano_active_base_dds;
    std::vector<std::uint8_t> volcano_active_specular_dds;
    ID3D11ShaderResourceView * volcano_base_view = nullptr;
    ID3D11ShaderResourceView * volcano_height_view = nullptr;
    ID3D11ShaderResourceView * volcano_active_base_view = nullptr;
    ID3D11ShaderResourceView * volcano_active_specular_view = nullptr;
    std::vector<std::uint8_t> water_clutter_base_dds;
    std::vector<std::uint8_t> water_clutter_height_dds;
    std::vector<std::uint8_t> grass_clutter_base_dds;
    std::vector<std::uint8_t> grass_clutter_height_dds;
    std::vector<std::uint8_t> plains_clutter_base_dds;
    std::vector<std::uint8_t> plains_clutter_height_dds;
    ID3D11ShaderResourceView * water_clutter_base_view = nullptr;
    ID3D11ShaderResourceView * water_clutter_height_view = nullptr;
    ID3D11ShaderResourceView * grass_clutter_base_view = nullptr;
    ID3D11ShaderResourceView * grass_clutter_height_view = nullptr;
    ID3D11ShaderResourceView * plains_clutter_base_view = nullptr;
    ID3D11ShaderResourceView * plains_clutter_height_view = nullptr;
    std::array<std::vector<std::uint8_t>, 10> river_surface_dds;
    std::array<ID3D11ShaderResourceView *, 10> river_surface_views = {};
    c3x_renderer::FeatureBundle river_rock_bundle;
    std::array<std::vector<std::uint8_t>, 5> river_rock_texture_dds;
    std::array<ID3D11ShaderResourceView *, 5> river_rock_texture_views = {};
    std::array<std::vector<std::uint8_t>, 10> route_texture_dds;
    std::array<ID3D11ShaderResourceView *, 10> route_texture_views = {};
    c3x_renderer::FeatureBundle bridge_bundle;
    std::array<std::vector<std::uint8_t>, 8> bridge_texture_dds;
    std::array<ID3D11ShaderResourceView *, 8> bridge_texture_views = {};
    c3x_renderer::FeatureBundle resource_bundle;
    std::array<std::vector<std::uint8_t>, 8> resource_texture_dds;
    std::array<ID3D11ShaderResourceView *, 8> resource_texture_views = {};
    c3x_renderer::FeatureBundle city_bundle;
    std::array<std::vector<std::uint8_t>, 4> city_base_dds;
    std::array<ID3D11ShaderResourceView *, 4> city_base_views = {};
    std::array<std::vector<std::uint8_t>, 4> city_emissive_dds;
    std::array<ID3D11ShaderResourceView *, 4> city_emissive_views = {};
    c3x_renderer::FeatureBundle wall_bundle;
    std::vector<std::uint8_t> wall_texture_dds;
    ID3D11ShaderResourceView * wall_texture_view = nullptr;
    c3x_renderer::FeatureBundle mine_bundle;
    std::array<std::vector<std::uint8_t>, 6> mine_base_dds;
    std::array<ID3D11ShaderResourceView *, 6> mine_base_views = {};
    std::array<std::vector<std::uint8_t>, 2> mine_emissive_dds;
    std::array<ID3D11ShaderResourceView *, 2> mine_emissive_views = {};
    c3x_renderer::TerrainFrameSignature cached_signature;
    c3x_renderer::TerrainFrameSignature previous_signature;
    std::uint64_t content_revision = 0;
    std::uint64_t previous_content_revision = 0;
    c3x_renderer_u32 cached_rendered_tile_count = 0;
    c3x_renderer_u32 cached_fallback_tile_count = 0;
    c3x_renderer_u32 cached_textured_tile_count = 0;
    c3x_renderer_u32 cached_visible_animation_count = 0;
    c3x_renderer_u32 cached_request_continuous_redraw = 0;
    c3x_renderer_u32 cache_hits = 0;
    c3x_renderer_u32 cache_misses = 0;
    c3x_renderer_u32 cache_evictions = 0;
    c3x_renderer_u32 cache_stale_rejections = 0;
    c3x_renderer_u32 device_generation = 1;
    c3x_renderer_u32 device_recoveries = 0;
    std::uint64_t shadow_field_signature = 0;
    std::vector<float> shadow_visibility_cache;
    std::vector<c3x_renderer_tile_v1> cached_tiles;
    std::vector<c3x_renderer_u32> cached_replacement_tile_flags;
    std::vector<CachedViewport> viewport_cache;
    bool cache_valid = false;
    bool feature_assets_ready = false;
    bool dune_assets_ready = false;
    bool marsh_assets_ready = false;
    bool volcano_assets_ready = false;
    bool clutter_assets_ready = false;
    bool river_assets_ready = false;
    bool route_assets_ready = false;
    bool resource_assets_ready = false;
    bool city_assets_ready = false;
    bool mine_assets_ready = false;

    ~RendererState() { reset(); }

    template <typename T>
    void release(T *& value) {
        if (value != nullptr) {
            value->Release();
            value = nullptr;
        }
    }

    void reset_targets() {
        release(readback_texture);
        release(depth_target);
        release(depth_texture);
        release(render_target);
        release(render_texture);
        width = 0;
        height = 0;
        pixels.clear();
        cache_valid = false;
    }

    void reset() {
        if (context != nullptr)
            context->ClearState();
        reset_targets();
        for (TerrainTexture & texture : terrain_textures)
        {
            release(texture.view);
            release(texture.material_height_view);
            release(texture.specular_view);
            release(texture.elevated_view);
            release(texture.elevated_height_view);
            release(texture.elevated_specular_view);
            for (ID3D11ShaderResourceView *& view : texture.relief_layer_views)
                release(view);
            for (ID3D11ShaderResourceView *& view : texture.water_surface_views)
                release(view);
        }
        release(dune_surface.view);
        release(dune_surface.material_height_view);
        release(dune_surface.specular_view);
        release(dune_decal_base_view);
        release(dune_decal_height_view);
        release(marsh_decal_base_view);
        release(marsh_decal_height_view);
        release(marsh_decal_specular_view);
        release(volcano_base_view);
        release(volcano_height_view);
        release(volcano_active_base_view);
        release(volcano_active_specular_view);
        release(water_clutter_base_view);
        release(water_clutter_height_view);
        release(grass_clutter_base_view);
        release(grass_clutter_height_view);
        release(plains_clutter_base_view);
        release(plains_clutter_height_view);
        for (ID3D11ShaderResourceView *& view : feature_texture_views)
            release(view);
        for (ID3D11ShaderResourceView *& view : terrain_extra_views)
            release(view);
        for (ID3D11ShaderResourceView *& view : river_surface_views)
            release(view);
        for (ID3D11ShaderResourceView *& view : river_rock_texture_views)
            release(view);
        for (ID3D11ShaderResourceView *& view : route_texture_views)
            release(view);
        for (ID3D11ShaderResourceView *& view : bridge_texture_views)
            release(view);
        for (ID3D11ShaderResourceView *& view : resource_texture_views)
            release(view);
        for (ID3D11ShaderResourceView *& view : city_base_views)
            release(view);
        for (ID3D11ShaderResourceView *& view : city_emissive_views)
            release(view);
        release(wall_texture_view);
        for (ID3D11ShaderResourceView *& view : mine_base_views)
            release(view);
        for (ID3D11ShaderResourceView *& view : mine_emissive_views)
            release(view);
        release(decal_sampler);
        release(terrain_sampler);
        release(rasterizer_state);
        release(depth_state);
        release(blend_state);
        release(input_layout);
        release(terrain_settings_buffer);
        release(feature_pixel_shader);
        release(feature_vertex_shader);
        release(pixel_shader);
        release(vertex_shader);
        release(context);
        release(device);
        shadow_field_signature = 0;
        shadow_visibility_cache.clear();
        cached_tiles.clear();
        cached_replacement_tile_flags.clear();
        viewport_cache.clear();
        cache_valid = false;
        if (device_generation != 0xffffffffu)
            ++device_generation;
    }

    bool initialize() {
        if (device != nullptr)
            return true;

        UINT flags = D3D11_CREATE_DEVICE_BGRA_SUPPORT;
        D3D_FEATURE_LEVEL levels[] = {
            D3D_FEATURE_LEVEL_11_0,
            D3D_FEATURE_LEVEL_10_1,
            D3D_FEATURE_LEVEL_10_0
        };
        D3D_FEATURE_LEVEL selected = D3D_FEATURE_LEVEL_10_0;
        HRESULT hr = D3D11CreateDevice(
            nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, flags,
            levels, static_cast<UINT>(std::size(levels)), D3D11_SDK_VERSION,
            &device, &selected, &context);
        if (FAILED(hr)) {
            hr = D3D11CreateDevice(
                nullptr, D3D_DRIVER_TYPE_WARP, nullptr, flags,
                levels, static_cast<UINT>(std::size(levels)), D3D11_SDK_VERSION,
                &device, &selected, &context);
        }
        if (FAILED(hr)) {
            reset();
            return false;
        }

        // Production executes the material and feature entry points copied
        // from the approved Lab handoff, isolated from in-progress Lab edits.
        auto compile_terrain_shader = [this](char const * entry, char const * target,
                                             ID3DBlob ** blob) {
            int count = MultiByteToWideChar(CP_UTF8, 0, integrated_shader_path.c_str(),
                                            -1, nullptr, 0);
            if (count <= 0)
                return false;
            std::wstring wide_path(static_cast<std::size_t>(count), L'\0');
            MultiByteToWideChar(CP_UTF8, 0, integrated_shader_path.c_str(), -1,
                                wide_path.data(), count);
            ID3DBlob * errors = nullptr;
            HRESULT result = D3DCompileFromFile(
                wide_path.c_str(), nullptr, D3D_COMPILE_STANDARD_FILE_INCLUDE,
                entry, target, D3DCOMPILE_OPTIMIZATION_LEVEL3, 0, blob, &errors);
            if (errors != nullptr) {
                OutputDebugStringA(static_cast<char const *>(errors->GetBufferPointer()));
                errors->Release();
            }
            return SUCCEEDED(result);
        };
        ID3DBlob * vertex_blob = nullptr;
        ID3DBlob * pixel_blob = nullptr;
        ID3DBlob * feature_vertex_blob = nullptr;
        ID3DBlob * feature_pixel_blob = nullptr;
        if (!compile_terrain_shader("VSIntegrated", "vs_4_0", &vertex_blob) ||
            !compile_terrain_shader("PSIntegrated", "ps_4_0", &pixel_blob) ||
            !compile_terrain_shader("VSIntegratedFeature", "vs_4_0", &feature_vertex_blob) ||
            !compile_terrain_shader("PSIntegratedFeature", "ps_4_0", &feature_pixel_blob)) {
            release(feature_pixel_blob);
            release(feature_vertex_blob);
            release(pixel_blob);
            release(vertex_blob);
            reset();
            return false;
        }

        hr = device->CreateVertexShader(vertex_blob->GetBufferPointer(), vertex_blob->GetBufferSize(),
                                        nullptr, &vertex_shader);
        if (SUCCEEDED(hr))
            hr = device->CreatePixelShader(pixel_blob->GetBufferPointer(), pixel_blob->GetBufferSize(),
                                           nullptr, &pixel_shader);
        if (SUCCEEDED(hr))
            hr = device->CreateVertexShader(feature_vertex_blob->GetBufferPointer(),
                                            feature_vertex_blob->GetBufferSize(), nullptr,
                                            &feature_vertex_shader);
        if (SUCCEEDED(hr))
            hr = device->CreatePixelShader(feature_pixel_blob->GetBufferPointer(),
                                           feature_pixel_blob->GetBufferSize(), nullptr,
                                           &feature_pixel_shader);
        D3D11_INPUT_ELEMENT_DESC elements[] = {
            {"POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D11_INPUT_PER_VERTEX_DATA, 0},
            {"TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 12, D3D11_INPUT_PER_VERTEX_DATA, 0},
            {"TEXCOORD", 1, DXGI_FORMAT_R32_FLOAT, 0, 20, D3D11_INPUT_PER_VERTEX_DATA, 0},
            {"NORMAL", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 24, D3D11_INPUT_PER_VERTEX_DATA, 0},
            {"TEXCOORD", 2, DXGI_FORMAT_R32G32_FLOAT, 0, 36, D3D11_INPUT_PER_VERTEX_DATA, 0},
            {"TEXCOORD", 3, DXGI_FORMAT_R32G32_FLOAT, 0, 44, D3D11_INPUT_PER_VERTEX_DATA, 0},
            {"TEXCOORD", 4, DXGI_FORMAT_R32_FLOAT, 0, 52, D3D11_INPUT_PER_VERTEX_DATA, 0},
            {"TEXCOORD", 5, DXGI_FORMAT_R32_FLOAT, 0, 56, D3D11_INPUT_PER_VERTEX_DATA, 0},
            {"TEXCOORD", 6, DXGI_FORMAT_R32_FLOAT, 0, 60, D3D11_INPUT_PER_VERTEX_DATA, 0},
            {"TEXCOORD", 7, DXGI_FORMAT_R32_FLOAT, 0, 64, D3D11_INPUT_PER_VERTEX_DATA, 0},
            {"TEXCOORD", 8, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, 68, D3D11_INPUT_PER_VERTEX_DATA, 0},
            {"TEXCOORD", 9, DXGI_FORMAT_R32G32_FLOAT, 0, 84, D3D11_INPUT_PER_VERTEX_DATA, 0},
            {"TEXCOORD", 10, DXGI_FORMAT_R32_FLOAT, 0, 92, D3D11_INPUT_PER_VERTEX_DATA, 0},
            {"TEXCOORD", 11, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, 96, D3D11_INPUT_PER_VERTEX_DATA, 0},
            {"TEXCOORD", 12, DXGI_FORMAT_R32_FLOAT, 0, 112, D3D11_INPUT_PER_VERTEX_DATA, 0}
        };
        if (SUCCEEDED(hr)) {
            hr = device->CreateInputLayout(elements, static_cast<UINT>(std::size(elements)),
                                            vertex_blob->GetBufferPointer(), vertex_blob->GetBufferSize(),
                                            &input_layout);
        }
        release(feature_pixel_blob);
        release(feature_vertex_blob);
        release(pixel_blob);
        release(vertex_blob);
        if (FAILED(hr)) {
            reset();
            return false;
        }

        TerrainShaderSettings settings = {};
        settings.height_texel[0] = 1.0f / 2048.0f;
        settings.height_texel[1] = 1.0f / 2048.0f;
        settings.normal_strength = 4.0f;
        settings.exposure = 1.0f;
        settings.light_direction[0] = -0.55f;
        settings.light_direction[1] = -0.35f;
        settings.light_direction[2] = 0.22f;
        settings.sun_intensity = 1.0f;
        settings.sun_color[0] = settings.sun_color[1] = settings.sun_color[2] = 1.0f;
        settings.shadow_strength = 0.84f;
        settings.moon_direction[2] = 1.0f;
        settings.ambient_color[0] = settings.ambient_color[1] = settings.ambient_color[2] = 0.78f;
        settings.environment_exposure = 1.0f;
        settings.water_fresnel = 0.04f;
        settings.water_specular = 0.62f;
        settings.emissive_scale = 0.25f;
        settings.hour = 12.0f;
        D3D11_BUFFER_DESC settings_desc = {};
        settings_desc.ByteWidth = sizeof(settings);
        settings_desc.Usage = D3D11_USAGE_DEFAULT;
        settings_desc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
        D3D11_SUBRESOURCE_DATA settings_data = {&settings, 0, 0};
        hr = device->CreateBuffer(&settings_desc, &settings_data,
                                  &terrain_settings_buffer);

        D3D11_BLEND_DESC blend = {};
        blend.RenderTarget[0].BlendEnable = TRUE;
        blend.RenderTarget[0].SrcBlend = D3D11_BLEND_SRC_ALPHA;
        blend.RenderTarget[0].DestBlend = D3D11_BLEND_INV_SRC_ALPHA;
        blend.RenderTarget[0].BlendOp = D3D11_BLEND_OP_ADD;
        blend.RenderTarget[0].SrcBlendAlpha = D3D11_BLEND_ONE;
        blend.RenderTarget[0].DestBlendAlpha = D3D11_BLEND_INV_SRC_ALPHA;
        blend.RenderTarget[0].BlendOpAlpha = D3D11_BLEND_OP_ADD;
        blend.RenderTarget[0].RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_ALL;
        if (SUCCEEDED(hr))
            hr = device->CreateBlendState(&blend, &blend_state);

        D3D11_DEPTH_STENCIL_DESC depth = {};
        depth.DepthEnable = TRUE;
        depth.DepthWriteMask = D3D11_DEPTH_WRITE_MASK_ALL;
        depth.DepthFunc = D3D11_COMPARISON_LESS_EQUAL;
        if (SUCCEEDED(hr))
            hr = device->CreateDepthStencilState(&depth, &depth_state);

        D3D11_RASTERIZER_DESC raster = {};
        raster.FillMode = D3D11_FILL_SOLID;
        raster.CullMode = D3D11_CULL_NONE;
        raster.DepthClipEnable = TRUE;
        raster.ScissorEnable = TRUE;
        if (SUCCEEDED(hr))
            hr = device->CreateRasterizerState(&raster, &rasterizer_state);
        D3D11_SAMPLER_DESC sampler = {};
        sampler.Filter = D3D11_FILTER_ANISOTROPIC;
        sampler.MaxAnisotropy = 8;
        sampler.MipLODBias = 0.0f;
        sampler.AddressU = D3D11_TEXTURE_ADDRESS_WRAP;
        sampler.AddressV = D3D11_TEXTURE_ADDRESS_WRAP;
        sampler.AddressW = D3D11_TEXTURE_ADDRESS_WRAP;
        sampler.MaxLOD = D3D11_FLOAT32_MAX;
        if (SUCCEEDED(hr))
            hr = device->CreateSamplerState(&sampler, &terrain_sampler);
        sampler.AddressU = D3D11_TEXTURE_ADDRESS_CLAMP;
        sampler.AddressV = D3D11_TEXTURE_ADDRESS_CLAMP;
        sampler.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
        if (SUCCEEDED(hr))
            hr = device->CreateSamplerState(&sampler, &decal_sampler);
        if (FAILED(hr)) {
            reset();
            return false;
        }
        return true;
    }

    static bool read_file(char const * path, std::vector<std::uint8_t> & output) {
        HANDLE file = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                                  FILE_ATTRIBUTE_NORMAL, nullptr);
        if (file == INVALID_HANDLE_VALUE)
            return false;
        LARGE_INTEGER size = {};
        bool ok = GetFileSizeEx(file, &size) != 0 && size.QuadPart > 0 && size.QuadPart <= 256ll * 1024ll * 1024ll;
        if (ok) {
            output.resize(static_cast<std::size_t>(size.QuadPart));
            DWORD bytes_read = 0;
            ok = ReadFile(file, output.data(), static_cast<DWORD>(output.size()), &bytes_read, nullptr) != 0 &&
                 bytes_read == static_cast<DWORD>(output.size());
        }
        CloseHandle(file);
        if (!ok)
            output.clear();
        return ok;
    }

    static bool pack_path(char const * root, char const * relative, char * output, std::size_t capacity) {
        std::size_t root_length = std::strlen(root);
        std::size_t relative_length = std::strlen(relative);
        if (root_length == 0 || root_length + relative_length + 2 > capacity)
            return false;
        std::memcpy(output, root, root_length);
        output[root_length] = '\\';
        std::memcpy(output + root_length + 1, relative, relative_length + 1);
        return true;
    }

    static bool safe_relative(std::string const & path) {
        if (path.empty() || path.front() == '/' || path.front() == '\\' || path.find(':') != std::string::npos)
            return false;
        std::string normalized = path;
        std::replace(normalized.begin(), normalized.end(), '\\', '/');
        std::size_t cursor = 0;
        while (cursor <= normalized.size()) {
            std::size_t end = normalized.find('/', cursor);
            std::string part = normalized.substr(cursor, end == std::string::npos ? std::string::npos : end - cursor);
            if (part == "..")
                return false;
            if (end == std::string::npos)
                break;
            cursor = end + 1;
        }
        return true;
    }

    static bool contains(std::vector<std::uint8_t> const & data, char const * text) {
        auto const * begin = reinterpret_cast<char const *>(data.data());
        auto const * end = begin + data.size();
        std::size_t length = std::strlen(text);
        return std::search(begin, end, text, text + length) != end;
    }

    static std::size_t find_text(std::vector<std::uint8_t> const & data, std::string const & text,
                                 std::size_t start = 0) {
        if (start > data.size())
            return std::string::npos;
        auto begin = data.begin() + static_cast<std::ptrdiff_t>(start);
        auto found = std::search(begin, data.end(), text.begin(), text.end());
        return found == data.end() ? std::string::npos : static_cast<std::size_t>(found - data.begin());
    }

    static bool json_string_after(std::vector<std::uint8_t> const & data, char const * key,
                                  std::size_t start, std::string & value) {
        std::string marker = std::string("\"") + key + "\"";
        std::size_t position = find_text(data, marker, start);
        if (position == std::string::npos)
            return false;
        position = find_text(data, ":", position + marker.size());
        if (position == std::string::npos)
            return false;
        auto const * bytes = reinterpret_cast<char const *>(data.data());
        while (++position < data.size() && std::isspace(static_cast<unsigned char>(bytes[position])) != 0) {}
        if (position >= data.size() || bytes[position] != '\"')
            return false;
        std::size_t end = position + 1;
        while (end < data.size() && bytes[end] != '\"')
            ++end;
        if (end >= data.size())
            return false;
        value.assign(bytes + position + 1, bytes + end);
        return safe_relative(value);
    }

    static bool json_number_after(std::vector<std::uint8_t> const & data, char const * key,
                                  std::size_t start, float & value) {
        std::string marker = std::string("\"") + key + "\"";
        std::size_t position = find_text(data, marker, start);
        if (position == std::string::npos)
            return false;
        position = find_text(data, ":", position + marker.size());
        if (position == std::string::npos)
            return false;
        auto const * bytes = reinterpret_cast<char const *>(data.data());
        while (++position < data.size() && std::isspace(static_cast<unsigned char>(bytes[position])) != 0) {}
        char number[48] = {};
        std::size_t count = 0;
        while (position < data.size() && count + 1 < std::size(number) &&
               (std::isdigit(static_cast<unsigned char>(bytes[position])) != 0 ||
                bytes[position] == '-' || bytes[position] == '+' || bytes[position] == '.'))
            number[count++] = bytes[position++];
        if (count == 0)
            return false;
        char * end = nullptr;
        value = std::strtof(number, &end);
        return end != number && *end == '\0' && std::isfinite(value);
    }

    static std::uint32_t read_u32(std::vector<std::uint8_t> const & data, std::size_t offset) {
        return static_cast<std::uint32_t>(data[offset]) |
               (static_cast<std::uint32_t>(data[offset + 1]) << 8) |
               (static_cast<std::uint32_t>(data[offset + 2]) << 16) |
               (static_cast<std::uint32_t>(data[offset + 3]) << 24);
    }

    void clear_terrain_assets() {
        for (TerrainTexture & texture : terrain_textures) {
            release(texture.view);
            release(texture.material_height_view);
            release(texture.specular_view);
            release(texture.elevated_view);
            release(texture.elevated_height_view);
            release(texture.elevated_specular_view);
            for (ID3D11ShaderResourceView *& view : texture.relief_layer_views)
                release(view);
            for (ID3D11ShaderResourceView *& view : texture.water_surface_views)
                release(view);
            texture.dds.clear();
            texture.material_height_dds.clear();
            texture.specular_dds.clear();
            texture.elevated_dds.clear();
            texture.elevated_height_dds.clear();
            texture.elevated_specular_dds.clear();
            for (std::vector<std::uint8_t> & layer : texture.relief_layer_dds)
                layer.clear();
            for (std::vector<std::uint8_t> & channel : texture.water_surface_dds)
                channel.clear();
            texture.height_pixels.clear();
            texture.blend_pixels.clear();
            for (std::vector<std::uint8_t> & field : texture.relief_height_variants)
                field.clear();
            for (std::vector<std::uint8_t> & field : texture.relief_blend_variants)
                field.clear();
            texture.relief_variant_widths.fill(0);
            texture.relief_variant_heights.fill(0);
            texture.relief_height_minimum.fill(0.0f);
            texture.relief_height_maximum.fill(0.0f);
            texture.relief_blend_minimum.fill(0.0f);
            texture.relief_blend_maximum.fill(0.0f);
            texture.height_width = 0;
            texture.height_height = 0;
            texture.height_minimum = 0.0f;
            texture.height_maximum = 1.0f;
            texture.blend_minimum = 0.0f;
            texture.blend_maximum = 1.0f;
            texture.height_scale_px = 0.0f;
            texture.relief_profile = 0;
            texture.configured = false;
        }
        release(dune_surface.view);
        release(dune_surface.material_height_view);
        release(dune_surface.specular_view);
        dune_surface.dds.clear();
        dune_surface.material_height_dds.clear();
        dune_surface.specular_dds.clear();
        dune_surface.configured = false;
        release(dune_decal_base_view);
        release(dune_decal_height_view);
        dune_decal_base_dds.clear();
        dune_decal_height_dds.clear();
        release(marsh_decal_base_view);
        release(marsh_decal_height_view);
        release(marsh_decal_specular_view);
        marsh_decal_base_dds.clear();
        marsh_decal_height_dds.clear();
        marsh_decal_specular_dds.clear();
        release(volcano_base_view);
        release(volcano_height_view);
        release(volcano_active_base_view);
        release(volcano_active_specular_view);
        volcano_base_dds.clear();
        volcano_height_dds.clear();
        volcano_active_base_dds.clear();
        volcano_active_specular_dds.clear();
        release(water_clutter_base_view);
        release(water_clutter_height_view);
        release(grass_clutter_base_view);
        release(grass_clutter_height_view);
        release(plains_clutter_base_view);
        release(plains_clutter_height_view);
        water_clutter_base_dds.clear();
        water_clutter_height_dds.clear();
        grass_clutter_base_dds.clear();
        grass_clutter_height_dds.clear();
        plains_clutter_base_dds.clear();
        plains_clutter_height_dds.clear();
        for (std::size_t index = 0; index < river_surface_views.size(); ++index) {
            release(river_surface_views[index]);
            river_surface_dds[index].clear();
        }
        for (std::size_t index = 0; index < river_rock_texture_views.size(); ++index) {
            release(river_rock_texture_views[index]);
            river_rock_texture_dds[index].clear();
        }
        river_rock_bundle = {};
        for (std::size_t index = 0; index < route_texture_views.size(); ++index) {
            release(route_texture_views[index]);
            route_texture_dds[index].clear();
        }
        for (std::size_t index = 0; index < bridge_texture_views.size(); ++index) {
            release(bridge_texture_views[index]);
            bridge_texture_dds[index].clear();
        }
        bridge_bundle = {};
        for (std::size_t index = 0; index < resource_texture_views.size(); ++index) {
            release(resource_texture_views[index]);
            resource_texture_dds[index].clear();
        }
        resource_bundle = {};
        for (std::size_t index = 0; index < city_base_views.size(); ++index) {
            release(city_base_views[index]);
            release(city_emissive_views[index]);
            city_base_dds[index].clear();
            city_emissive_dds[index].clear();
        }
        city_bundle = {};
        release(wall_texture_view);
        wall_texture_dds.clear();
        wall_bundle = {};
        for (std::size_t index = 0; index < mine_base_views.size(); ++index) {
            release(mine_base_views[index]);
            mine_base_dds[index].clear();
        }
        for (std::size_t index = 0; index < mine_emissive_views.size(); ++index) {
            release(mine_emissive_views[index]);
            mine_emissive_dds[index].clear();
        }
        mine_bundle = {};
        for (std::size_t index = 0; index < feature_texture_views.size(); ++index) {
            release(feature_texture_views[index]);
            feature_texture_dds[index].clear();
        }
        for (std::size_t index = 0; index < terrain_extra_views.size(); ++index) {
            release(terrain_extra_views[index]);
            terrain_extra_dds[index].clear();
        }
        feature_bundle = {};
        feature_assets_ready = false;
        dune_assets_ready = false;
        marsh_assets_ready = false;
        volcano_assets_ready = false;
        clutter_assets_ready = false;
        river_assets_ready = false;
        route_assets_ready = false;
        resource_assets_ready = false;
        city_assets_ready = false;
        mine_assets_ready = false;
        terrain_extra_assets_ready = false;
        authored_relief_assets_ready = false;
        cache_valid = false;
        viewport_cache.clear();
    }

    void mix_content_revision(std::vector<std::uint8_t> const & data) {
        if (content_revision == 0)
            content_revision = 1469598103934665603ull;
        for (std::uint8_t value : data) {
            content_revision ^= value;
            content_revision *= 1099511628211ull;
        }
    }

    bool load_material_only(char const * root, char const * material_relative,
                            TerrainTexture & output) {
        char path[4 * MAX_PATH];
        std::vector<std::uint8_t> material;
        if (!pack_path(root, material_relative, path, std::size(path)) ||
            !read_file(path, material) || !contains(material, "c3x.material.v0"))
            return false;
        std::array<std::pair<char const *, std::vector<std::uint8_t> *>, 3> channels = {
            std::make_pair("base_color", &output.dds),
            std::make_pair("height", &output.material_height_dds),
            std::make_pair("specular", &output.specular_dds)
        };
        for (std::size_t index = 0; index < channels.size(); ++index) {
            std::size_t position = find_text(material, std::string("\"") + channels[index].first + "\"");
            std::string relative;
            std::vector<std::uint8_t> dds;
            if (position == std::string::npos ||
                !json_string_after(material, "texture", position, relative) ||
                !pack_path(root, relative.c_str(), path, std::size(path)) ||
                !read_file(path, dds) || dds.size() < 156 ||
                std::memcmp(dds.data(), "DDS ", 4) != 0 ||
                std::memcmp(dds.data() + 84, "DX10", 4) != 0)
                return false;
            std::uint32_t format = read_u32(dds, 128);
            bool valid = index == 0 ?
                (format == DXGI_FORMAT_BC3_UNORM || format == DXGI_FORMAT_BC3_UNORM_SRGB) :
                (format == DXGI_FORMAT_BC4_UNORM || format == DXGI_FORMAT_BC4_SNORM);
            if (!valid)
                return false;
            mix_content_revision(dds);
            channels[index].second->swap(dds);
        }
        mix_content_revision(material);
        output.configured = true;
        return true;
    }

    bool load_decal_channel(char const * root, std::vector<std::uint8_t> const & record,
                            char const * channel, std::vector<std::uint8_t> & output,
                            std::uint32_t expected_format) {
        std::size_t position = find_text(record, std::string("\"") + channel + "\"");
        std::string relative;
        char path[4 * MAX_PATH];
        if (position == std::string::npos ||
            !json_string_after(record, "texture", position, relative) ||
            !pack_path(root, relative.c_str(), path, std::size(path)) ||
            !read_file(path, output) || output.size() < 156 ||
            std::memcmp(output.data(), "DDS ", 4) != 0 ||
            std::memcmp(output.data() + 84, "DX10", 4) != 0 ||
            read_u32(output, 128) != expected_format)
            return false;
        mix_content_revision(output);
        return true;
    }

    bool load_dds_bytes(char const * root, char const * relative,
                        std::vector<std::uint8_t> & output,
                        std::uint32_t expected_format,
                        std::uint32_t alternate_format = 0) {
        char path[4 * MAX_PATH];
        if (!pack_path(root, relative, path, std::size(path)) ||
            !read_file(path, output) || output.size() < 156 ||
            std::memcmp(output.data(), "DDS ", 4) != 0 ||
            std::memcmp(output.data() + 84, "DX10", 4) != 0 ||
            (read_u32(output, 128) != expected_format &&
             read_u32(output, 128) != alternate_format))
            return false;
        mix_content_revision(output);
        return true;
    }

    bool load_r8_field(char const * root, char const * relative,
                       std::vector<std::uint8_t> & field_pixels,
                       std::uint32_t & field_width, std::uint32_t & field_height) {
        std::vector<std::uint8_t> dds;
        char path[4 * MAX_PATH];
        if (!pack_path(root, relative, path, std::size(path)) ||
            !read_file(path, dds) || dds.size() < 149 ||
            std::memcmp(dds.data(), "DDS ", 4) != 0 ||
            std::memcmp(dds.data() + 84, "DX10", 4) != 0 ||
            read_u32(dds, 128) != DXGI_FORMAT_R8_UNORM)
            return false;
        field_width = read_u32(dds, 16);
        field_height = read_u32(dds, 12);
        if (field_width == 0 || field_height == 0 ||
            field_width > 4096 || field_height > 4096 ||
            148ull + static_cast<std::uint64_t>(field_width) * field_height > dds.size())
            return false;
        field_pixels.assign(dds.begin() + 148,
            dds.begin() + 148 + static_cast<std::ptrdiff_t>(field_width * field_height));
        mix_content_revision(dds);
        return true;
    }

    void configure_integrated_assets(char const * terrain_root,
                                     c3x_renderer::RendererPackRoots const & companion_packs) {
        std::string packs_root = terrain_root == nullptr ? "" : terrain_root;
        std::size_t slash = packs_root.find_last_of("\\/");
        if (slash == std::string::npos)
            return;
        packs_root.resize(slash);
        std::string vegetation_root = companion_packs.vegetation.empty() ?
            packs_root + "\\VegetationNormalized" : companion_packs.vegetation;
        std::string decal_root = companion_packs.decals.empty() ?
            packs_root + "\\DecalsNormalized" : companion_packs.decals;
        std::string terrain_elements_root = companion_packs.terrain_elements.empty() ?
            packs_root + "\\TerrainElementsNormalized" : companion_packs.terrain_elements;
        std::string shore_root = companion_packs.shore.empty() ?
            packs_root + "\\ShoreNormalized" : companion_packs.shore;
        bool dune_material = load_material_only(
            terrain_root, "materials\\library\\desert_hills.json", dune_surface);
        std::vector<std::uint8_t> decal_record;
        char decal_path[4 * MAX_PATH];
        bool decal = pack_path(decal_root.c_str(),
            "decals\\terrain_desert_dune_decal_01.json", decal_path, std::size(decal_path)) &&
            read_file(decal_path, decal_record) && contains(decal_record, "c3x.decal.v0") &&
            load_decal_channel(decal_root.c_str(), decal_record, "base_color",
                               dune_decal_base_dds, DXGI_FORMAT_BC3_UNORM_SRGB) &&
            load_decal_channel(decal_root.c_str(), decal_record, "height",
                               dune_decal_height_dds, DXGI_FORMAT_BC5_UNORM);
        if (decal)
            mix_content_revision(decal_record);
        dune_assets_ready = dune_material && decal;

        TerrainTexture & hills = terrain_textures[5];
        std::uint32_t hill_width = 0, hill_height = 0;
        bool hill_geometry = load_r8_field(
            terrain_root, "textures\\relief\\hills\\standard\\height_lod0.dds",
            hills.height_pixels, hill_width, hill_height);
        if (hill_geometry) {
            hills.height_width = hill_width;
            hills.height_height = hill_height;
            measure_field_limits(hills.height_pixels,
                                 hills.height_minimum, hills.height_maximum);
            hills.height_scale_px = 52.0f;
            hills.relief_profile = 2;
        }

        TerrainTexture & mountains = terrain_textures[6];
        bool mountain_geometry = true;
        for (int variant = 0; variant < 5; ++variant) {
            char height_path[256], blend_path[256];
            sprintf_s(height_path,
                "textures\\relief\\mountains\\standard\\variant_%02d\\height_lod0.dds",
                variant + 1);
            sprintf_s(blend_path,
                "textures\\relief\\mountains\\standard\\variant_%02d\\blend_lod0.dds",
                variant + 1);
            std::uint32_t height_width = 0, height_height = 0;
            std::uint32_t blend_width = 0, blend_height = 0;
            mountain_geometry = mountain_geometry &&
                load_r8_field(terrain_root, height_path,
                    mountains.relief_height_variants[variant],
                    height_width, height_height) &&
                load_r8_field(terrain_root, blend_path,
                    mountains.relief_blend_variants[variant],
                    blend_width, blend_height) &&
                height_width == blend_width && height_height == blend_height;
            mountains.relief_variant_widths[variant] = height_width;
            mountains.relief_variant_heights[variant] = height_height;
            measure_field_limits(mountains.relief_height_variants[variant],
                mountains.relief_height_minimum[variant],
                mountains.relief_height_maximum[variant]);
            measure_field_limits(mountains.relief_blend_variants[variant],
                mountains.relief_blend_minimum[variant],
                mountains.relief_blend_maximum[variant]);
        }
        if (mountain_geometry) {
            mountains.height_scale_px = 104.0f;
            mountains.relief_profile = 4;
        }
        authored_relief_assets_ready = hills.configured && hill_geometry &&
            mountains.configured && mountain_geometry;

        std::vector<std::uint8_t> marsh_record;
        char marsh_path[4 * MAX_PATH];
        bool marsh_decal = pack_path(decal_root.c_str(),
            "decals\\terrain_marsh_decal_01.json", marsh_path, std::size(marsh_path)) &&
            read_file(marsh_path, marsh_record) && contains(marsh_record, "c3x.decal.v0") &&
            load_decal_channel(decal_root.c_str(), marsh_record, "base_color",
                               marsh_decal_base_dds, DXGI_FORMAT_BC3_UNORM_SRGB) &&
            load_decal_channel(decal_root.c_str(), marsh_record, "height",
                               marsh_decal_height_dds, DXGI_FORMAT_BC5_UNORM) &&
            load_decal_channel(decal_root.c_str(), marsh_record, "specular",
                               marsh_decal_specular_dds, DXGI_FORMAT_BC4_UNORM);
        if (marsh_decal)
            mix_content_revision(marsh_record);
        marsh_assets_ready = terrain_textures[9].configured && marsh_decal;

        TerrainTexture & volcano = terrain_textures[10];
        std::uint32_t volcano_width = 0, volcano_height = 0;
        std::uint32_t blend_width = 0, blend_height = 0;
        bool volcano_geometry = load_r8_field(
            terrain_elements_root.c_str(),
            "textures\\terrain_elements\\terrain_feature_volcano\\height_lod0.dds",
            volcano.height_pixels, volcano_width, volcano_height) &&
            load_r8_field(
                terrain_elements_root.c_str(),
                "textures\\terrain_elements\\terrain_feature_volcano\\blend_lod0.dds",
                volcano.blend_pixels, blend_width, blend_height) &&
            volcano_width == blend_width && volcano_height == blend_height;
        if (volcano_geometry) {
            volcano.height_width = volcano_width;
            volcano.height_height = volcano_height;
            measure_field_limits(volcano.height_pixels,
                                 volcano.height_minimum, volcano.height_maximum);
            measure_field_limits(volcano.blend_pixels,
                                 volcano.blend_minimum, volcano.blend_maximum);
            volcano.height_scale_px = 104.0f;
            volcano.relief_profile = 5;
        }
        volcano_assets_ready = volcano.configured && volcano_geometry &&
            load_dds_bytes(terrain_root, "textures\\water\\volcano\\base.dds",
                           volcano_base_dds, DXGI_FORMAT_BC3_UNORM_SRGB,
                           DXGI_FORMAT_BC3_UNORM) &&
            load_dds_bytes(terrain_root, "textures\\water\\volcano\\height.dds",
                           volcano_height_dds, DXGI_FORMAT_BC5_UNORM) &&
            load_dds_bytes(terrain_root, "textures\\water\\volcano\\active_base.dds",
                           volcano_active_base_dds, DXGI_FORMAT_BC3_UNORM_SRGB,
                           DXGI_FORMAT_BC3_UNORM) &&
            load_dds_bytes(terrain_root, "textures\\water\\volcano\\active_specular.dds",
                           volcano_active_specular_dds, DXGI_FORMAT_BC4_UNORM);

        std::vector<std::uint8_t> water_clutter_record, grass_clutter_record,
            plains_clutter_record;
        char clutter_path[4 * MAX_PATH];
        bool water_clutter = pack_path(decal_root.c_str(),
            "decals\\terrain_water_ocean_decal_01.json", clutter_path,
            std::size(clutter_path)) && read_file(clutter_path, water_clutter_record) &&
            load_decal_channel(decal_root.c_str(), water_clutter_record, "base_color",
                               water_clutter_base_dds, DXGI_FORMAT_BC3_UNORM_SRGB) &&
            load_decal_channel(decal_root.c_str(), water_clutter_record, "height",
                               water_clutter_height_dds, DXGI_FORMAT_BC5_UNORM);
        bool grass_clutter = pack_path(decal_root.c_str(),
            "decals\\terrain_grassland_decal_02.json", clutter_path,
            std::size(clutter_path)) && read_file(clutter_path, grass_clutter_record) &&
            load_decal_channel(decal_root.c_str(), grass_clutter_record, "base_color",
                               grass_clutter_base_dds, DXGI_FORMAT_BC3_UNORM_SRGB) &&
            load_decal_channel(decal_root.c_str(), grass_clutter_record, "height",
                               grass_clutter_height_dds, DXGI_FORMAT_BC5_UNORM);
        bool plains_clutter = pack_path(decal_root.c_str(),
            "decals\\terrain_plains_decal_01.json", clutter_path,
            std::size(clutter_path)) && read_file(clutter_path, plains_clutter_record) &&
            load_decal_channel(decal_root.c_str(), plains_clutter_record, "base_color",
                               plains_clutter_base_dds, DXGI_FORMAT_BC3_UNORM_SRGB) &&
            load_decal_channel(decal_root.c_str(), plains_clutter_record, "height",
                               plains_clutter_height_dds, DXGI_FORMAT_BC5_UNORM);
        clutter_assets_ready = water_clutter && grass_clutter && plains_clutter;
        if (clutter_assets_ready) {
            mix_content_revision(water_clutter_record);
            mix_content_revision(grass_clutter_record);
            mix_content_revision(plains_clutter_record);
        }

        std::string bundle_path = vegetation_root + "\\vegetation_runtime.bin";
        feature_assets_ready = load_feature_bundle(bundle_path, feature_bundle) &&
            feature_bundle.texture_paths.size() <= feature_texture_dds.size();
        if (feature_assets_ready) {
            for (std::size_t index = 0; index < feature_bundle.texture_paths.size(); ++index) {
                char texture_path[4 * MAX_PATH];
                if (!pack_path(vegetation_root.c_str(),
                               feature_bundle.texture_paths[index].c_str(), texture_path,
                               std::size(texture_path)) ||
                    !read_file(texture_path, feature_texture_dds[index]) ||
                    feature_texture_dds[index].size() < 156 ||
                    std::memcmp(feature_texture_dds[index].data(), "DDS ", 4) != 0 ||
                    std::memcmp(feature_texture_dds[index].data() + 84, "DX10", 4) != 0 ||
                    (read_u32(feature_texture_dds[index], 128) != DXGI_FORMAT_BC1_UNORM_SRGB &&
                     read_u32(feature_texture_dds[index], 128) != DXGI_FORMAT_BC1_UNORM &&
                     read_u32(feature_texture_dds[index], 128) != DXGI_FORMAT_BC3_UNORM_SRGB &&
                     read_u32(feature_texture_dds[index], 128) != DXGI_FORMAT_BC3_UNORM)) {
                    feature_assets_ready = false;
                    break;
                }
                mix_content_revision(feature_texture_dds[index]);
            }
        }

        struct ExtraTexture {
            char const * path;
            std::uint32_t format;
            std::uint32_t alternate;
        };
        std::array<ExtraTexture, 19> const extras = {{
            {"textures\\beach_height.dds", DXGI_FORMAT_BC4_UNORM, 0},
            {"textures\\beach_specular.dds", DXGI_FORMAT_BC4_UNORM, 0},
            {"textures\\cliff_height.dds", DXGI_FORMAT_BC4_UNORM, 0},
            {"textures\\cliff_specular.dds", DXGI_FORMAT_BC4_UNORM, 0},
            {"textures\\water\\surface\\gloss.dds", DXGI_FORMAT_BC1_UNORM_SRGB,
             DXGI_FORMAT_BC1_UNORM},
            {"textures\\water\\surface\\tiling_mask.dds", DXGI_FORMAT_BC4_UNORM, 0},
            {"textures\\water\\surface\\non_tiling_mask.dds", DXGI_FORMAT_BC4_UNORM, 0},
            {"textures\\water\\surface\\small_secondary_lean0.dds",
             DXGI_FORMAT_R16G16B16A16_UNORM, 0},
            {"textures\\water\\surface\\small_secondary_lean1.dds",
             DXGI_FORMAT_R16G16_UNORM, 0},
            {"textures\\water\\effects\\ripples_primary.dds",
             DXGI_FORMAT_BC3_UNORM_SRGB, DXGI_FORMAT_BC3_UNORM},
            {"textures\\water\\effects\\turbulence.dds",
             DXGI_FORMAT_BC3_UNORM_SRGB, DXGI_FORMAT_BC3_UNORM},
            {"textures\\water\\profiles\\coast\\dark.dds",
             DXGI_FORMAT_R16G16B16A16_FLOAT, 0},
            {"textures\\water\\profiles\\coast\\scatter.dds",
             DXGI_FORMAT_R16G16B16A16_FLOAT, 0},
            {"textures\\water\\surface\\tiling_normal0.dds", DXGI_FORMAT_BC5_UNORM, 0},
            {"textures\\water\\surface\\tiling_normal1.dds", DXGI_FORMAT_BC4_UNORM, 0},
            {"textures\\water\\surface\\non_tiling_normal0.dds", DXGI_FORMAT_BC5_UNORM, 0},
            {"textures\\water\\surface\\non_tiling_normal1.dds", DXGI_FORMAT_BC4_UNORM, 0},
            {"textures\\mtn_desert_base_height.dds", DXGI_FORMAT_BC4_UNORM, 0},
            {"textures\\mtn_desert_base_specular.dds", DXGI_FORMAT_BC4_UNORM, 0},
        }};
        terrain_extra_assets_ready = true;
        for (std::size_t index = 0; index < extras.size(); ++index)
            terrain_extra_assets_ready = terrain_extra_assets_ready && load_dds_bytes(
                terrain_root, extras[index].path, terrain_extra_dds[index],
                extras[index].format, extras[index].alternate);

        std::array<ExtraTexture, 10> const river_channels = {{
            {"textures\\river_base_color.dds", DXGI_FORMAT_BC3_UNORM_SRGB,
             DXGI_FORMAT_BC3_UNORM},
            {"textures\\river_height.dds", DXGI_FORMAT_BC4_UNORM, 0},
            {"textures\\river_specular.dds", DXGI_FORMAT_BC4_UNORM, 0},
            {"textures\\water\\surface\\river_lean0.dds",
             DXGI_FORMAT_R16G16B16A16_UNORM, 0},
            {"textures\\water\\surface\\river_lean1.dds",
             DXGI_FORMAT_R16G16_UNORM, 0},
            {"textures\\water\\river\\source_decal_base.dds",
             DXGI_FORMAT_BC3_UNORM_SRGB, DXGI_FORMAT_BC3_UNORM},
            {"textures\\water\\river\\source_decal_height.dds",
             DXGI_FORMAT_BC5_UNORM, 0},
            {"textures\\water\\river\\clutter_decal_base.dds",
             DXGI_FORMAT_BC3_UNORM_SRGB, DXGI_FORMAT_BC3_UNORM},
            {"textures\\water\\river\\clutter_decal_height.dds",
             DXGI_FORMAT_BC5_UNORM, 0},
            {"textures\\water\\relief\\river_bank_noise\\height_lod0.dds",
             DXGI_FORMAT_R8_UNORM, 0},
        }};
        bool river_surface_ready = true;
        for (std::size_t index = 0; index < river_channels.size(); ++index)
            river_surface_ready = river_surface_ready && load_dds_bytes(
                terrain_root, river_channels[index].path, river_surface_dds[index],
                river_channels[index].format, river_channels[index].alternate);
        std::string river_bundle_path = shore_root + "\\shore_runtime.bin";
        std::vector<std::uint8_t> river_bundle_bytes;
        bool river_rocks_ready = read_file(river_bundle_path.c_str(), river_bundle_bytes) &&
            load_feature_bundle(river_bundle_path, river_rock_bundle) &&
            river_rock_bundle.texture_paths.size() == river_rock_texture_dds.size() &&
            find_feature_group(river_rock_bundle, "river_rock") != nullptr;
        if (river_rocks_ready) {
            mix_content_revision(river_bundle_bytes);
            for (std::size_t index = 0; index < river_rock_texture_dds.size(); ++index) {
                char texture_path[4 * MAX_PATH];
                river_rocks_ready = pack_path(
                    shore_root.c_str(), river_rock_bundle.texture_paths[index].c_str(),
                    texture_path, std::size(texture_path)) &&
                    read_file(texture_path, river_rock_texture_dds[index]) &&
                    river_rock_texture_dds[index].size() >= 156 &&
                    std::memcmp(river_rock_texture_dds[index].data(), "DDS ", 4) == 0 &&
                    std::memcmp(river_rock_texture_dds[index].data() + 84, "DX10", 4) == 0 &&
                    (read_u32(river_rock_texture_dds[index], 128) == DXGI_FORMAT_BC1_UNORM_SRGB ||
                     read_u32(river_rock_texture_dds[index], 128) == DXGI_FORMAT_BC1_UNORM);
                if (!river_rocks_ready)
                    break;
                mix_content_revision(river_rock_texture_dds[index]);
            }
        }
        river_assets_ready = river_surface_ready && river_rocks_ready;

        std::string route_root = packs_root + "\\RouteStylesNormalized";
        char const * route_textures[] = {
            "textures\\routes\\base_color_f5c58170975c3c8b.dds",
            "textures\\routes\\base_color_6c2f0cc0e38fff7b.dds",
            "textures\\routes\\base_color_5ae696c133345a83.dds",
            "textures\\routes\\base_color_ba7297551e6ece35.dds",
            "textures\\routes\\base_color_3631908100843d0d.dds",
            "textures\\routes\\base_color_678d822a9338aab5.dds",
            "textures\\routes\\base_color_fc46370d90163f16.dds",
            "textures\\routes\\base_color_a832a6518fdab3b1.dds",
            "textures\\routes\\base_color_c3ae5ee0b5879164.dds",
            "textures\\routes\\base_color_39a5855bea35f6d1.dds",
        };
        route_assets_ready = true;
        for (std::size_t index = 0; index < route_texture_dds.size(); ++index)
            route_assets_ready = route_assets_ready && load_dds_bytes(
                route_root.c_str(), route_textures[index], route_texture_dds[index],
                DXGI_FORMAT_BC3_UNORM_SRGB, DXGI_FORMAT_BC3_UNORM);

        auto load_runtime_bundle = [&](std::string const & root, char const * filename,
                                       c3x_renderer::FeatureBundle & bundle,
                                       auto & texture_bytes) {
            std::string runtime_path = root + "\\" + filename;
            std::vector<std::uint8_t> runtime_bytes;
            if (!read_file(runtime_path.c_str(), runtime_bytes) ||
                !load_feature_bundle(runtime_path, bundle) ||
                bundle.texture_paths.size() != texture_bytes.size())
                return false;
            mix_content_revision(runtime_bytes);
            for (std::size_t index = 0; index < texture_bytes.size(); ++index)
                if (!load_dds_bytes(root.c_str(), bundle.texture_paths[index].c_str(),
                                    texture_bytes[index], DXGI_FORMAT_BC1_UNORM_SRGB,
                                    DXGI_FORMAT_BC1_UNORM))
                    return false;
            return true;
        };
        std::string bridge_root = packs_root + "\\RouteDoodadsNormalized";
        route_assets_ready = route_assets_ready && load_runtime_bundle(
            bridge_root, "bridge_runtime.bin", bridge_bundle, bridge_texture_dds);

        std::string resource_root = packs_root + "\\ResourceNormalized";
        resource_assets_ready = load_runtime_bundle(
            resource_root, "resource_runtime.bin", resource_bundle, resource_texture_dds);

        std::string city_root = packs_root + "\\CityComponentsNormalized";
        std::string wall_root = packs_root + "\\CityAdjunctsNormalized";
        std::string city_runtime_path = city_root + "\\city_runtime.bin";
        std::string wall_runtime_path = wall_root + "\\wall_runtime.bin";
        std::vector<std::uint8_t> city_runtime_bytes, wall_runtime_bytes;
        city_assets_ready = read_file(city_runtime_path.c_str(), city_runtime_bytes) &&
            load_feature_bundle(city_runtime_path, city_bundle) &&
            city_bundle.texture_paths.size() == 8u &&
            read_file(wall_runtime_path.c_str(), wall_runtime_bytes) &&
            load_feature_bundle(wall_runtime_path, wall_bundle) &&
            wall_bundle.texture_paths.size() == 1u;
        if (city_assets_ready) {
            mix_content_revision(city_runtime_bytes);
            mix_content_revision(wall_runtime_bytes);
            for (std::size_t index = 0; index < city_base_dds.size(); ++index) {
                city_assets_ready = city_assets_ready && load_dds_bytes(
                    city_root.c_str(), city_bundle.texture_paths[index].c_str(),
                    city_base_dds[index], DXGI_FORMAT_BC1_UNORM_SRGB, DXGI_FORMAT_BC1_UNORM) &&
                    load_dds_bytes(city_root.c_str(), city_bundle.texture_paths[index + 4u].c_str(),
                                   city_emissive_dds[index], DXGI_FORMAT_BC1_UNORM_SRGB,
                                   DXGI_FORMAT_BC1_UNORM);
            }
            city_assets_ready = city_assets_ready && load_dds_bytes(
                wall_root.c_str(), wall_bundle.texture_paths[0].c_str(),
                wall_texture_dds, DXGI_FORMAT_BC1_UNORM_SRGB, DXGI_FORMAT_BC1_UNORM);
        }

        std::string mine_root = packs_root + "\\ImprovementsNormalized";
        std::string mine_runtime_path = mine_root + "\\mine_runtime.bin";
        std::vector<std::uint8_t> mine_runtime_bytes;
        mine_assets_ready = read_file(mine_runtime_path.c_str(), mine_runtime_bytes);
        if (!mine_assets_ready)
            SetLastError(1801u);
        if (mine_assets_ready) {
            mine_assets_ready = load_feature_bundle(mine_runtime_path, mine_bundle);
            if (!mine_assets_ready)
                SetLastError(1802u);
        }
        if (mine_assets_ready) {
            mine_assets_ready = mine_bundle.texture_paths.size() == 8u;
            if (!mine_assets_ready)
                SetLastError(1803u);
        }
        auto load_mine_texture = [&](std::size_t source_index,
                                     std::vector<std::uint8_t> & output) {
            char path[4 * MAX_PATH];
            if (!pack_path(mine_root.c_str(), mine_bundle.texture_paths[source_index].c_str(),
                           path, std::size(path)) ||
                !read_file(path, output) || output.size() < 156 ||
                std::memcmp(output.data(), "DDS ", 4) != 0 ||
                std::memcmp(output.data() + 84, "DX10", 4) != 0)
                return false;
            std::uint32_t format = read_u32(output, 128);
            if (format != DXGI_FORMAT_BC1_UNORM &&
                format != DXGI_FORMAT_BC1_UNORM_SRGB &&
                format != DXGI_FORMAT_BC3_UNORM &&
                format != DXGI_FORMAT_BC3_UNORM_SRGB)
                return false;
            mix_content_revision(output);
            return true;
        };
        if (mine_assets_ready) {
            mix_content_revision(mine_runtime_bytes);
            for (std::size_t index = 0; index < mine_base_dds.size(); ++index) {
                mine_assets_ready = mine_assets_ready &&
                    load_mine_texture(index, mine_base_dds[index]);
                if (!mine_assets_ready)
                    SetLastError(static_cast<DWORD>(1810u + index));
            }
            for (std::size_t index = 0; index < mine_emissive_dds.size(); ++index) {
                mine_assets_ready = mine_assets_ready &&
                    load_mine_texture(index + mine_base_dds.size(), mine_emissive_dds[index]);
                if (!mine_assets_ready)
                    SetLastError(static_cast<DWORD>(1820u + index));
            }
        }
    }

    bool configure_asset(int terrain_type, char const * root, char const * logical_asset_id) {
        if (terrain_type < 0 || terrain_type >= c3x_renderer::terrain_type_count || root == nullptr ||
            root[0] == '\0' || logical_asset_id == nullptr || logical_asset_id[0] == '\0')
            return false;
        char path[4 * MAX_PATH];
        std::vector<std::uint8_t> manifest, mesh, material, dds, height_dds;
        if (!pack_path(root, "manifest.json", path, std::size(path)) || !read_file(path, manifest) ||
            !contains(manifest, "c3x.asset_pack.v0"))
            return false;
        std::string logical_marker = std::string("\"") + logical_asset_id + "\"";
        std::size_t asset_position = find_text(manifest, logical_marker);
        std::string mesh_relative, material_relative, texture_relative;
        if (asset_position == std::string::npos ||
            !json_string_after(manifest, "mesh", asset_position, mesh_relative) ||
            !json_string_after(manifest, "material", asset_position, material_relative))
            return false;
        if (!pack_path(root, mesh_relative.c_str(), path, std::size(path)) || !read_file(path, mesh) ||
            !contains(mesh, "c3x.normalized_mesh.v0") || !contains(mesh, "\"primitive\": \"triangles\""))
            return false;
        if (!pack_path(root, material_relative.c_str(), path, std::size(path)) || !read_file(path, material) ||
            !contains(material, "c3x.material.v0") || !json_string_after(material, "texture", 0, texture_relative))
            return false;
        if (!pack_path(root, texture_relative.c_str(), path, std::size(path)) || !read_file(path, dds) ||
            dds.size() < 164 || std::memcmp(dds.data(), "DDS ", 4) != 0 || read_u32(dds, 4) != 124 ||
            std::memcmp(dds.data() + 84, "DX10", 4) != 0 ||
            (read_u32(dds, 128) != DXGI_FORMAT_BC3_UNORM && read_u32(dds, 128) != DXGI_FORMAT_BC3_UNORM_SRGB))
            return false;
        TerrainTexture & texture = terrain_textures[terrain_type];
        release(texture.view);
        release(texture.material_height_view);
        release(texture.specular_view);
        release(texture.elevated_view);
        release(texture.elevated_height_view);
        release(texture.elevated_specular_view);
        for (ID3D11ShaderResourceView *& view : texture.relief_layer_views)
            release(view);
        for (ID3D11ShaderResourceView *& view : texture.water_surface_views)
            release(view);
        texture.dds.swap(dds);
        texture.material_height_dds.clear();
        texture.specular_dds.clear();
        texture.elevated_dds.clear();
        texture.elevated_height_dds.clear();
        texture.elevated_specular_dds.clear();
        for (std::vector<std::uint8_t> & layer : texture.relief_layer_dds)
            layer.clear();
        for (std::vector<std::uint8_t> & channel : texture.water_surface_dds)
            channel.clear();
        texture.height_pixels.clear();
        texture.height_width = 0;
        texture.height_height = 0;
        texture.height_scale_px = 0.0f;
        texture.relief_profile = 0;
        for (auto const & channel : std::array<std::pair<char const *, std::vector<std::uint8_t> *>, 2>{
                 std::make_pair("height", &texture.material_height_dds),
                 std::make_pair("specular", &texture.specular_dds)}) {
            std::size_t channel_position = find_text(material, std::string("\"") + channel.first + "\"");
            std::string channel_relative;
            std::vector<std::uint8_t> channel_dds;
            if (channel_position == std::string::npos ||
                !json_string_after(material, "texture", channel_position, channel_relative) ||
                !pack_path(root, channel_relative.c_str(), path, std::size(path)) ||
                !read_file(path, channel_dds) || channel_dds.size() < 156 ||
                std::memcmp(channel_dds.data(), "DDS ", 4) != 0 || read_u32(channel_dds, 4) != 124 ||
                std::memcmp(channel_dds.data() + 84, "DX10", 4) != 0 ||
                (read_u32(channel_dds, 128) != DXGI_FORMAT_BC4_UNORM &&
                 read_u32(channel_dds, 128) != DXGI_FORMAT_BC4_SNORM))
                continue;
            channel.second->swap(channel_dds);
        }
        std::size_t elevated_position = find_text(material, "\"elevated\"");
        if (elevated_position != std::string::npos) {
            for (auto const & channel : std::array<std::pair<char const *, std::vector<std::uint8_t> *>, 3>{
                     std::make_pair("base_color", &texture.elevated_dds),
                     std::make_pair("height", &texture.elevated_height_dds),
                     std::make_pair("specular", &texture.elevated_specular_dds)}) {
                std::size_t channel_position = find_text(
                    material, std::string("\"") + channel.first + "\"", elevated_position);
                std::string channel_relative;
                std::vector<std::uint8_t> channel_dds;
                if (channel_position == std::string::npos ||
                    !json_string_after(material, "texture", channel_position, channel_relative) ||
                    !pack_path(root, channel_relative.c_str(), path, std::size(path)) ||
                    !read_file(path, channel_dds) || channel_dds.size() < 156 ||
                    std::memcmp(channel_dds.data(), "DDS ", 4) != 0 || read_u32(channel_dds, 4) != 124 ||
                    std::memcmp(channel_dds.data() + 84, "DX10", 4) != 0)
                    return false;
                std::uint32_t format = read_u32(channel_dds, 128);
                bool format_valid = channel.first == std::string("base_color") ?
                    (format == DXGI_FORMAT_BC3_UNORM || format == DXGI_FORMAT_BC3_UNORM_SRGB) :
                    (format == DXGI_FORMAT_BC4_UNORM || format == DXGI_FORMAT_BC4_SNORM);
                if (!format_valid)
                    return false;
                channel.second->swap(channel_dds);
            }
        }
        std::size_t authored_layers_position = find_text(material, "\"authored_layers\"");
        if (authored_layers_position != std::string::npos) {
            std::array<char const *, 5> layer_names = {};
            std::size_t layer_count = 0;
            if (terrain_type == 6) {
                layer_names = {"snow", "desert_base", "desert_stripe_1", "desert_stripe_2", "desert_stripe_3"};
                layer_count = 5;
            } else if (terrain_type == 11) {
                layer_names = {"beach", "cliff", "cliff_white", nullptr, nullptr};
                layer_count = 3;
            } else {
                return false;
            }
            for (std::size_t layer_index = 0; layer_index < layer_count; ++layer_index) {
                std::size_t layer_position = find_text(
                    material, std::string("\"") + layer_names[layer_index] + "\"", authored_layers_position);
                std::size_t channel_position = find_text(material, "\"base_color\"", layer_position);
                std::string channel_relative;
                std::vector<std::uint8_t> channel_dds;
                if (layer_position == std::string::npos || channel_position == std::string::npos ||
                    !json_string_after(material, "texture", channel_position, channel_relative) ||
                    !pack_path(root, channel_relative.c_str(), path, std::size(path)) ||
                    !read_file(path, channel_dds) || channel_dds.size() < 164 ||
                    std::memcmp(channel_dds.data(), "DDS ", 4) != 0 || read_u32(channel_dds, 4) != 124 ||
                    std::memcmp(channel_dds.data() + 84, "DX10", 4) != 0 ||
                    (read_u32(channel_dds, 128) != DXGI_FORMAT_BC3_UNORM &&
                     read_u32(channel_dds, 128) != DXGI_FORMAT_BC3_UNORM_SRGB))
                    return false;
                texture.relief_layer_dds[layer_index].swap(channel_dds);
            }
        }
        std::size_t water_surface_position = find_text(material, "\"water_surface\"");
        if (water_surface_position != std::string::npos) {
            if (terrain_type != 11)
                return false;
            std::array<char const *, 5> channel_names = {
                "large_lean0", "large_lean1", "small_lean0", "small_lean1", "foam"
            };
            std::array<std::uint32_t, 5> expected_formats = {
                DXGI_FORMAT_R16G16B16A16_UNORM, DXGI_FORMAT_R16G16_UNORM,
                DXGI_FORMAT_R16G16B16A16_UNORM, DXGI_FORMAT_R16G16_UNORM,
                DXGI_FORMAT_BC3_UNORM_SRGB,
            };
            for (std::size_t channel_index = 0; channel_index < channel_names.size(); ++channel_index) {
                std::size_t channel_position = find_text(
                    material, std::string("\"") + channel_names[channel_index] + "\"",
                    water_surface_position);
                std::string channel_relative;
                std::vector<std::uint8_t> channel_dds;
                if (channel_position == std::string::npos ||
                    !json_string_after(material, "texture", channel_position, channel_relative) ||
                    !pack_path(root, channel_relative.c_str(), path, std::size(path)) ||
                    !read_file(path, channel_dds) || channel_dds.size() < 156 ||
                    std::memcmp(channel_dds.data(), "DDS ", 4) != 0 || read_u32(channel_dds, 4) != 124 ||
                    std::memcmp(channel_dds.data() + 84, "DX10", 4) != 0)
                    return false;
                std::uint32_t format = read_u32(channel_dds, 128);
                bool format_matches = format == expected_formats[channel_index] ||
                    (channel_index == 4 && format == DXGI_FORMAT_BC3_UNORM);
                if (!format_matches)
                    return false;
                texture.water_surface_dds[channel_index].swap(channel_dds);
            }
        }
        std::size_t relief_position = find_text(material, "\"relief\"");
        if (relief_position != std::string::npos) {
            std::string height_relative, profile;
            float scale = 0.0f;
            if (!json_string_after(material, "texture", relief_position, height_relative) ||
                !json_string_after(material, "profile", relief_position, profile) ||
                !json_number_after(material, "height_scale_px", relief_position, scale) ||
                scale < 0.0f || scale > 256.0f ||
                !pack_path(root, height_relative.c_str(), path, std::size(path)) ||
                !read_file(path, height_dds) || height_dds.size() < 149 ||
                std::memcmp(height_dds.data(), "DDS ", 4) != 0 || read_u32(height_dds, 4) != 124 ||
                std::memcmp(height_dds.data() + 84, "DX10", 4) != 0 ||
                read_u32(height_dds, 128) != DXGI_FORMAT_R8_UNORM)
                return false;
            std::uint32_t height_width = read_u32(height_dds, 16);
            std::uint32_t height_height = read_u32(height_dds, 12);
            if (height_width == 0 || height_height == 0 || height_width > 4096 || height_height > 4096 ||
                148ull + static_cast<std::uint64_t>(height_width) * height_height > height_dds.size())
                return false;
            if (profile == "continuous")
                texture.relief_profile = 1;
            else if (profile == "connected_hills")
                texture.relief_profile = 2;
            else if (profile == "mountain_massif")
                texture.relief_profile = 3;
            else if (profile == "mountain_atlas")
                texture.relief_profile = 4;
            else
                return false;
            texture.height_width = height_width;
            texture.height_height = height_height;
            texture.height_scale_px = scale;
            texture.height_pixels.assign(height_dds.begin() + 148,
                height_dds.begin() + 148 + static_cast<std::ptrdiff_t>(height_width * height_height));
        }
        texture.configured = true;
        mix_content_revision(manifest);
        mix_content_revision(mesh);
        mix_content_revision(material);
        mix_content_revision(texture.dds);
        return true;
    }

    bool configure_pack(char const * root) {
        clear_terrain_assets();
        previous_content_revision = content_revision;
        content_revision = 0;
        return configure_asset(2, root, "terrain/grassland/base");
    }

    bool configure_definitions(char const * mod_root, char const * default_path,
                               char const * scenario_path, char const * custom_path) {
        char shader_path[4 * MAX_PATH];
        if (mod_root != nullptr &&
            pack_path(mod_root, "Renderer\\native\\integrated_terrain.hlsl",
                      shader_path, std::size(shader_path)) &&
            GetFileAttributesA(shader_path) != INVALID_FILE_ATTRIBUTES)
            integrated_shader_path = shader_path;
        std::array<c3x_renderer::TerrainAssetBinding, c3x_renderer::terrain_type_count> bindings;
        c3x_renderer::RendererPackRoots companion_packs;
        std::string diagnostic;
        if (!c3x_renderer::load_terrain_definition_layers(
                mod_root, default_path, scenario_path, custom_path, bindings,
                companion_packs, diagnostic))
            return false;
        std::string default_name = default_path == nullptr ? "" : default_path;
        std::replace(default_name.begin(), default_name.end(), '/', '\\');
        std::size_t default_separator = default_name.find_last_of('\\');
        std::string default_filename = default_separator == std::string::npos
            ? default_name : default_name.substr(default_separator + 1);
        bool production = default_filename == "default.custom_rendering.txt";
        clear_terrain_assets();
        previous_content_revision = content_revision;
        content_revision = 0;
        std::vector<std::uint8_t> shader_source;
        if (!read_file(integrated_shader_path.c_str(), shader_source) && production)
            return false;
        if (!shader_source.empty())
            mix_content_revision(shader_source);
        char production_shader_path[4 * MAX_PATH];
        shader_source.clear();
        bool has_production_shader = mod_root != nullptr &&
            pack_path(mod_root, "Renderer\\native\\terrain_rendering.hlsl",
                      production_shader_path, std::size(production_shader_path)) &&
            read_file(production_shader_path, shader_source);
        if (!has_production_shader && production)
            return false;
        if (has_production_shader)
            mix_content_revision(shader_source);
        for (char const * path : {default_path, scenario_path, custom_path}) {
            std::vector<std::uint8_t> definition;
            if (path != nullptr && path[0] != '\0' && read_file(path, definition))
                mix_content_revision(definition);
        }
        char const * terrain_root = nullptr;
        for (int index = 0; index < c3x_renderer::terrain_type_count; ++index) {
            c3x_renderer::TerrainAssetBinding const & binding = bindings[index];
            if (binding.configured) {
                if (!configure_asset(index, binding.pack_root.c_str(),
                                     binding.logical_asset_id.c_str())) {
                    if (production)
                        return false;
                    continue;
                }
                if (terrain_root == nullptr)
                    terrain_root = binding.pack_root.c_str();
            }
        }
        if (production) {
            if (terrain_root == nullptr)
                return false;
            for (TerrainTexture const & texture : terrain_textures)
                if (!texture.configured)
                    return false;
            configure_integrated_assets(terrain_root, companion_packs);
            if (!terrain_extra_assets_ready || !authored_relief_assets_ready ||
                !dune_assets_ready ||
                !marsh_assets_ready || !volcano_assets_ready ||
                !clutter_assets_ready || !feature_assets_ready || !river_assets_ready ||
                !route_assets_ready || !resource_assets_ready || !city_assets_ready ||
                !mine_assets_ready)
                return false;
        }
        return true;
    }

    bool ensure_dds_texture(std::vector<std::uint8_t> const & dds,
                            ID3D11ShaderResourceView *& view, bool required) {
        if (view != nullptr)
            return true;
        if (dds.empty())
            return !required;
        if (device == nullptr || dds.size() < 156)
            return false;
        std::uint32_t width_px = read_u32(dds, 16);
        std::uint32_t height_px = read_u32(dds, 12);
        std::uint32_t mip_count = std::max(1u, read_u32(dds, 28));
        DXGI_FORMAT format = static_cast<DXGI_FORMAT>(read_u32(dds, 128));
        if (width_px == 0 || height_px == 0 || width_px > 16384 || height_px > 16384 || mip_count > 15)
            return false;
        std::uint32_t block_bytes = 0;
        std::uint32_t bytes_per_pixel = 0;
        if (format == DXGI_FORMAT_BC1_UNORM || format == DXGI_FORMAT_BC1_UNORM_SRGB ||
            format == DXGI_FORMAT_BC4_UNORM || format == DXGI_FORMAT_BC4_SNORM)
            block_bytes = 8u;
        else if (format == DXGI_FORMAT_BC3_UNORM || format == DXGI_FORMAT_BC3_UNORM_SRGB ||
                 format == DXGI_FORMAT_BC5_UNORM)
            block_bytes = 16u;
        else if (format == DXGI_FORMAT_R16G16B16A16_UNORM ||
                 format == DXGI_FORMAT_R16G16B16A16_FLOAT)
            bytes_per_pixel = 8u;
        else if (format == DXGI_FORMAT_R16G16_UNORM)
            bytes_per_pixel = 4u;
        else if (format == DXGI_FORMAT_R8_UNORM)
            bytes_per_pixel = 1u;
        if (format != DXGI_FORMAT_BC1_UNORM && format != DXGI_FORMAT_BC1_UNORM_SRGB &&
            format != DXGI_FORMAT_BC3_UNORM && format != DXGI_FORMAT_BC3_UNORM_SRGB &&
            format != DXGI_FORMAT_BC4_UNORM && format != DXGI_FORMAT_BC4_SNORM &&
            format != DXGI_FORMAT_BC5_UNORM &&
            format != DXGI_FORMAT_R16G16B16A16_UNORM &&
            format != DXGI_FORMAT_R16G16B16A16_FLOAT &&
            format != DXGI_FORMAT_R16G16_UNORM &&
            format != DXGI_FORMAT_R8_UNORM)
            return false;
        std::vector<D3D11_SUBRESOURCE_DATA> subresources(mip_count);
        std::size_t offset = 148;
        std::uint32_t mip_width = width_px, mip_height = height_px;
        for (std::uint32_t mip = 0; mip < mip_count; ++mip) {
            std::uint32_t row_pitch = bytes_per_pixel != 0
                ? mip_width * bytes_per_pixel
                : std::max(1u, (mip_width + 3) / 4) * block_bytes;
            std::uint32_t rows = bytes_per_pixel != 0
                ? mip_height : std::max(1u, (mip_height + 3) / 4);
            std::size_t byte_count = static_cast<std::size_t>(row_pitch) * rows;
            if (offset + byte_count > dds.size())
                return false;
            subresources[mip].pSysMem = dds.data() + offset;
            subresources[mip].SysMemPitch = row_pitch;
            subresources[mip].SysMemSlicePitch = static_cast<UINT>(byte_count);
            offset += byte_count;
            mip_width = std::max(1u, mip_width / 2);
            mip_height = std::max(1u, mip_height / 2);
        }
        // Civ III is a 32-bit process and only displays 128x64 map cells. Keep
        // the authored mip chain, but do not allocate source-detail levels that
        // cannot contribute at the in-game projection scale.
        std::uint32_t first_mip = 0;
        std::uint32_t runtime_width = width_px, runtime_height = height_px;
        while (first_mip + 1 < mip_count &&
               (runtime_width > 2048 || runtime_height > 2048)) {
            ++first_mip;
            runtime_width = std::max(1u, runtime_width / 2);
            runtime_height = std::max(1u, runtime_height / 2);
        }
        D3D11_TEXTURE2D_DESC desc = {};
        desc.Width = runtime_width;
        desc.Height = runtime_height;
        desc.MipLevels = mip_count - first_mip;
        desc.ArraySize = 1;
        desc.Format = format;
        desc.SampleDesc.Count = 1;
        desc.Usage = D3D11_USAGE_IMMUTABLE;
        desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
        ID3D11Texture2D * texture = nullptr;
        HRESULT hr = device->CreateTexture2D(&desc, subresources.data() + first_mip, &texture);
        if (SUCCEEDED(hr))
            hr = device->CreateShaderResourceView(texture, nullptr, &view);
        release(texture);
        return SUCCEEDED(hr);
    }

    bool ensure_pack_texture(int terrain_type) {
        TerrainTexture & asset = terrain_textures[terrain_type];
        if (!asset.configured)
            return false;
        bool layers_ready = true;
        for (std::size_t index = 0; index < asset.relief_layer_dds.size(); ++index)
            layers_ready = layers_ready && ensure_dds_texture(
                asset.relief_layer_dds[index], asset.relief_layer_views[index], false);
        for (std::size_t index = 0; index < asset.water_surface_dds.size(); ++index)
            layers_ready = layers_ready && ensure_dds_texture(
                asset.water_surface_dds[index], asset.water_surface_views[index], false);
        return layers_ready && ensure_dds_texture(asset.dds, asset.view, true) &&
            ensure_dds_texture(asset.material_height_dds, asset.material_height_view, false) &&
            ensure_dds_texture(asset.specular_dds, asset.specular_view, false) &&
            ensure_dds_texture(asset.elevated_dds, asset.elevated_view, false) &&
            ensure_dds_texture(asset.elevated_height_dds, asset.elevated_height_view, false) &&
            ensure_dds_texture(asset.elevated_specular_dds, asset.elevated_specular_view, false);
    }

    bool ensure_terrain_textures() {
        if (dune_assets_ready) {
            dune_assets_ready = ensure_dds_texture(dune_surface.dds, dune_surface.view, true) &&
                ensure_dds_texture(dune_surface.material_height_dds,
                                   dune_surface.material_height_view, true) &&
                ensure_dds_texture(dune_surface.specular_dds, dune_surface.specular_view, true) &&
                ensure_dds_texture(dune_decal_base_dds, dune_decal_base_view, true) &&
                ensure_dds_texture(dune_decal_height_dds, dune_decal_height_view, true);
        }
        if (feature_assets_ready) {
            for (std::size_t index = 0; index < feature_bundle.texture_paths.size(); ++index)
                feature_assets_ready = feature_assets_ready && ensure_dds_texture(
                    feature_texture_dds[index], feature_texture_views[index], true);
        }
        if (marsh_assets_ready) {
            marsh_assets_ready = ensure_dds_texture(marsh_decal_base_dds, marsh_decal_base_view, true) &&
                ensure_dds_texture(marsh_decal_height_dds, marsh_decal_height_view, true) &&
                ensure_dds_texture(marsh_decal_specular_dds, marsh_decal_specular_view, true);
        }
        if (volcano_assets_ready) {
            volcano_assets_ready = ensure_dds_texture(volcano_base_dds, volcano_base_view, true) &&
                ensure_dds_texture(volcano_height_dds, volcano_height_view, true) &&
                ensure_dds_texture(volcano_active_base_dds, volcano_active_base_view, true) &&
                ensure_dds_texture(volcano_active_specular_dds,
                                   volcano_active_specular_view, true);
        }
        if (clutter_assets_ready) {
            clutter_assets_ready = ensure_dds_texture(water_clutter_base_dds,
                                                      water_clutter_base_view, true) &&
                ensure_dds_texture(water_clutter_height_dds,
                                   water_clutter_height_view, true) &&
                ensure_dds_texture(grass_clutter_base_dds,
                                   grass_clutter_base_view, true) &&
                ensure_dds_texture(grass_clutter_height_dds,
                                   grass_clutter_height_view, true) &&
                ensure_dds_texture(plains_clutter_base_dds,
                                   plains_clutter_base_view, true) &&
                ensure_dds_texture(plains_clutter_height_dds,
                                   plains_clutter_height_view, true);
        }
        if (terrain_extra_assets_ready) {
            for (std::size_t index = 0; index < terrain_extra_dds.size(); ++index)
                if (!ensure_dds_texture(terrain_extra_dds[index], terrain_extra_views[index], true))
                    return false;
        }
        if (river_assets_ready) {
            for (std::size_t index = 0; index < river_surface_dds.size(); ++index)
                river_assets_ready = river_assets_ready && ensure_dds_texture(
                    river_surface_dds[index], river_surface_views[index], true);
            for (std::size_t index = 0; index < river_rock_texture_dds.size(); ++index)
                river_assets_ready = river_assets_ready && ensure_dds_texture(
                    river_rock_texture_dds[index], river_rock_texture_views[index], true);
        }
        if (route_assets_ready) {
            for (std::size_t index = 0; index < route_texture_views.size(); ++index)
                route_assets_ready = route_assets_ready && ensure_dds_texture(
                    route_texture_dds[index], route_texture_views[index], true);
            for (std::size_t index = 0; index < bridge_texture_views.size(); ++index)
                route_assets_ready = route_assets_ready && ensure_dds_texture(
                    bridge_texture_dds[index], bridge_texture_views[index], true);
        }
        if (resource_assets_ready) {
            for (std::size_t index = 0; index < resource_texture_views.size(); ++index)
                resource_assets_ready = resource_assets_ready && ensure_dds_texture(
                    resource_texture_dds[index], resource_texture_views[index], true);
        }
        if (city_assets_ready) {
            for (std::size_t index = 0; index < city_base_views.size(); ++index) {
                city_assets_ready = city_assets_ready && ensure_dds_texture(
                    city_base_dds[index], city_base_views[index], true) &&
                    ensure_dds_texture(city_emissive_dds[index], city_emissive_views[index], true);
            }
            city_assets_ready = city_assets_ready && ensure_dds_texture(
                wall_texture_dds, wall_texture_view, true);
        }
        if (mine_assets_ready) {
            for (std::size_t index = 0; index < mine_base_views.size(); ++index)
                mine_assets_ready = mine_assets_ready && ensure_dds_texture(
                    mine_base_dds[index], mine_base_views[index], true);
            for (std::size_t index = 0; index < mine_emissive_views.size(); ++index)
                mine_assets_ready = mine_assets_ready && ensure_dds_texture(
                    mine_emissive_dds[index], mine_emissive_views[index], true);
        }
        return true;
    }

    bool ensure_targets(int requested_width, int requested_height) {
        if (requested_width == width && requested_height == height && render_texture != nullptr)
            return true;
        reset_targets();

        D3D11_TEXTURE2D_DESC desc = {};
        desc.Width = static_cast<UINT>(requested_width);
        desc.Height = static_cast<UINT>(requested_height);
        desc.MipLevels = 1;
        desc.ArraySize = 1;
        desc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
        desc.SampleDesc.Count = 1;
        desc.Usage = D3D11_USAGE_DEFAULT;
        desc.BindFlags = D3D11_BIND_RENDER_TARGET;
        HRESULT hr = device->CreateTexture2D(&desc, nullptr, &render_texture);
        if (SUCCEEDED(hr))
            hr = device->CreateRenderTargetView(render_texture, nullptr, &render_target);

        D3D11_TEXTURE2D_DESC depth_desc = {};
        depth_desc.Width = static_cast<UINT>(requested_width);
        depth_desc.Height = static_cast<UINT>(requested_height);
        depth_desc.MipLevels = 1;
        depth_desc.ArraySize = 1;
        depth_desc.Format = DXGI_FORMAT_D24_UNORM_S8_UINT;
        depth_desc.SampleDesc.Count = 1;
        depth_desc.Usage = D3D11_USAGE_DEFAULT;
        depth_desc.BindFlags = D3D11_BIND_DEPTH_STENCIL;
        if (SUCCEEDED(hr))
            hr = device->CreateTexture2D(&depth_desc, nullptr, &depth_texture);
        if (SUCCEEDED(hr))
            hr = device->CreateDepthStencilView(depth_texture, nullptr, &depth_target);

        desc.Usage = D3D11_USAGE_STAGING;
        desc.BindFlags = 0;
        desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
        if (SUCCEEDED(hr))
            hr = device->CreateTexture2D(&desc, nullptr, &readback_texture);
        if (FAILED(hr)) {
            reset_targets();
            return false;
        }
        width = requested_width;
        height = requested_height;
        pixels.resize(static_cast<std::size_t>(width) * static_cast<std::size_t>(height));
        return true;
    }

    static int ground_type(c3x_renderer_tile_v1 const & tile) {
        // Real terrain types 5-10 are relief or vegetation over the base
        // biome.  They are composed separately and must not become a
        // tile-sized ground decal.  Flood plain (4) and water remain direct
        // surface families.
        if (tile.real_terrain_type == 4 || tile.real_terrain_type >= 11)
            return tile.real_terrain_type;
        return tile.terrain_type;
    }

    static int relief_type(c3x_renderer_tile_v1 const & tile) {
        // Volcanoes, hills, and mountains are geometry categories over their
        // underlying ground terrain.
        return tile.real_terrain_type == 5 || tile.real_terrain_type == 6 ||
               tile.real_terrain_type == 10
            ? tile.real_terrain_type : -1;
    }

    static void measure_field_limits(std::vector<std::uint8_t> const & pixels,
                                     float & minimum, float & maximum) {
        if (pixels.empty()) {
            minimum = 0.0f;
            maximum = 1.0f;
            return;
        }
        auto limits = std::minmax_element(pixels.begin(), pixels.end());
        minimum = static_cast<float>(*limits.first) / 255.0f;
        maximum = static_cast<float>(*limits.second) / 255.0f;
    }

    static float sample_byte_field(std::vector<std::uint8_t> const & pixels,
                                   std::uint32_t width, std::uint32_t height,
                                   float u, float v, bool wrap) {
        if (pixels.empty() || width == 0 || height == 0)
            return 0.0f;
        if (wrap) {
            u -= std::floor(u);
            v -= std::floor(v);
        } else {
            u = std::clamp(u, 0.0f, 1.0f);
            v = std::clamp(v, 0.0f, 1.0f);
        }
        float x = u * static_cast<float>(wrap ? width : width - 1);
        float y = v * static_cast<float>(wrap ? height : height - 1);
        std::uint32_t x0 = static_cast<std::uint32_t>(x);
        std::uint32_t y0 = static_cast<std::uint32_t>(y);
        if (wrap) { x0 %= width; y0 %= height; }
        std::uint32_t x1 = wrap ? (x0 + 1) % width : std::min(x0 + 1, width - 1);
        std::uint32_t y1 = wrap ? (y0 + 1) % height : std::min(y0 + 1, height - 1);
        float fx = x - static_cast<float>(x0);
        float fy = y - static_cast<float>(y0);
        auto value = [&pixels, width](std::uint32_t px, std::uint32_t py) {
            return static_cast<float>(pixels[static_cast<std::size_t>(py) * width + px]) / 255.0f;
        };
        float top = value(x0, y0) * (1.0f - fx) + value(x1, y0) * fx;
        float bottom = value(x0, y1) * (1.0f - fx) + value(x1, y1) * fx;
        return top * (1.0f - fy) + bottom * fy;
    }

    static float sample_height_field(TerrainTexture const & asset, float u, float v, bool wrap) {
        return sample_byte_field(asset.height_pixels, asset.height_width,
                                 asset.height_height, u, v, wrap);
    }

    static float sample_normalized_field(std::vector<std::uint8_t> const & pixels,
                                         std::uint32_t width, std::uint32_t height,
                                         float minimum, float maximum,
                                         float u, float v) {
        // This is the Lab HeightField::sample implementation. Authored relief
        // fields are normalized to their own observed range and sampled with
        // wrapped bilinear coordinates, including at the source border.
        if (pixels.empty() || width == 0 || height == 0)
            return 0.0f;
        u -= std::floor(u);
        v -= std::floor(v);
        float px = u * static_cast<float>(width);
        float py = v * static_cast<float>(height);
        std::uint32_t x0 = static_cast<std::uint32_t>(std::floor(px)) % width;
        std::uint32_t y0 = static_cast<std::uint32_t>(std::floor(py)) % height;
        std::uint32_t x1 = (x0 + 1) % width;
        std::uint32_t y1 = (y0 + 1) % height;
        float tx = px - std::floor(px);
        float ty = py - std::floor(py);
        auto value = [&pixels, width, minimum, maximum](std::uint32_t x,
                                                        std::uint32_t y) {
            float raw = static_cast<float>(
                pixels[static_cast<std::size_t>(y) * width + x]) / 255.0f;
            return (raw - minimum) / std::max(0.0001f, maximum - minimum);
        };
        float top = value(x0, y0) * (1.0f - tx) + value(x1, y0) * tx;
        float bottom = value(x0, y1) * (1.0f - tx) + value(x1, y1) * tx;
        return top * (1.0f - ty) + bottom * ty;
    }

    static float smooth_edge(float distance) {
        float amount = std::clamp(distance / 0.42f, 0.0f, 1.0f);
        return amount * amount * (3.0f - 2.0f * amount);
    }

    static float smoothstep01(float value) {
        value = std::clamp(value, 0.0f, 1.0f);
        return value * value * (3.0f - 2.0f * value);
    }

    static std::array<float, 3> relief_height(
            TerrainTexture const & asset, float u, float v,
            float world_u, float world_v, int base_ground,
            c3x_renderer_u32 seed, bool const connected_edges[4]) {
        if (asset.relief_profile == 1)
            return {(sample_height_field(asset, world_u, world_v, true) - 0.5f) *
                asset.height_scale_px, 0.0f, 0.0f};
        if (asset.relief_profile == 2) {
            // Match the accepted Lab hill body: a source-space nine-tap
            // low-pass extracts the authored macro landform, and a calibrated
            // remap removes the source field's positive floor. Neighbor
            // topology weights outside this function own its continuous skirt.
            (void)u;
            (void)v;
            (void)base_ground;
            (void)seed;
            (void)connected_edges;
            constexpr float radius = 0.018f;
            float source_u = 0.11f + world_u * 0.035f;
            float source_v = 0.17f + world_v * 0.035f;
            auto sample = [&asset](float x, float y) {
                return sample_normalized_field(asset.height_pixels,
                    asset.height_width, asset.height_height,
                    asset.height_minimum, asset.height_maximum, x, y);
            };
            float center = sample(source_u, source_v) * 4.0f;
            float cardinal = sample(source_u - radius, source_v) +
                sample(source_u + radius, source_v) +
                sample(source_u, source_v - radius) +
                sample(source_u, source_v + radius);
            float diagonal = sample(source_u - radius, source_v - radius) +
                sample(source_u + radius, source_v - radius) +
                sample(source_u - radius, source_v + radius) +
                sample(source_u + radius, source_v + radius);
            float authored_macro = (center + cardinal * 2.0f + diagonal) / 16.0f;
            float shape = smoothstep01((authored_macro - 0.22f) / 0.38f);
            return {shape * asset.height_scale_px, 0.0f, 0.0f};
        }
        if (asset.relief_profile == 4) {
            // Mountain bodies are composed across neighboring cells below,
            // using the Lab's chain sampler verbatim. A tile-local fallback
            // here would reintroduce the divergent diamond-shaped mountain.
            return {0.0f, 0.0f, 0.0f};
        }
        if (asset.relief_profile == 3) {
            float sample_u = u, sample_v = v;
            switch (seed & 3u) {
            case 1: sample_u = v; sample_v = 1.0f - u; break;
            case 2: sample_u = 1.0f - u; sample_v = 1.0f - v; break;
            case 3: sample_u = 1.0f - v; sample_v = u; break;
            default: break;
            }
            return {sample_height_field(asset, sample_u, sample_v, false) *
                asset.height_scale_px, 0.0f, 0.0f};
        }
        if (asset.relief_profile == 5) {
            // Use the normalized ordinary-volcano height and blend fields
            // exactly as authored. Only deterministic rigid orientation,
            // bounded aspect, Civ III footprint fit, and vertical calibration
            // are applied. Adjacent mountain/volcano cells enlarge the same
            // source footprint so shoulders overlap without connector geometry.
            bool connected = connected_edges[0] || connected_edges[1] ||
                connected_edges[2] || connected_edges[3];
            float footprint = connected ? 0.60f : 1.0f;
            float aspect = (seed & 4u) != 0 ? 0.88f : 1.12f;
            float source_u = 0.5f + (u - 0.5f) * footprint * aspect;
            float source_v = 0.5f + (v - 0.5f) * footprint / aspect;
            switch (seed & 3u) {
            case 1: std::swap(source_u, source_v); break;
            case 2: source_u = 1.0f - source_u; break;
            case 3: std::swap(source_u, source_v); source_u = 1.0f - source_u; break;
            default: break;
            }
            if (source_u < 0.0f || source_u > 1.0f ||
                source_v < 0.0f || source_v > 1.0f)
                return {0.0f, 0.0f, 0.0f};
            float height = sample_height_field(asset, source_u, source_v, false);
            float blend = sample_byte_field(asset.blend_pixels,
                asset.height_width, asset.height_height, source_u, source_v, false);
            float blend_weight = std::clamp(blend / 0.34f, 0.0f, 1.0f);
            blend_weight = blend_weight * blend_weight * (3.0f - 2.0f * blend_weight);
            float vertical = ((seed >> 3) & 1u) != 0 ? 104.0f : 88.0f;
            return {height * blend_weight * vertical, height, blend};
        }
        return {0.0f, 0.0f, 0.0f};
    }

    void fill_output(c3x_renderer_frame_v1 const & frame,
                     c3x_renderer_output_v1 & output, c3x_renderer_u32 invalidations,
                     c3x_renderer_i64 renderer_ticks) {
        output.api_version = C3X_RENDERER_API_VERSION;
        output.struct_size = sizeof(output);
        output.width = width;
        output.height = height;
        output.stride_bytes = width * static_cast<int>(sizeof(std::uint32_t));
        output.clip_left = frame.clip_left;
        output.clip_top = frame.clip_top;
        output.clip_right = frame.clip_right;
        output.clip_bottom = frame.clip_bottom;
        output.rendered_tile_count = cached_rendered_tile_count;
        output.fallback_tile_count = cached_fallback_tile_count;
        output.bgra_pixels = pixels.data();
        // Terrain is independent of retained native unit/effect animation.  A
        // cache hit must still report the current frame's animation demand so
        // Civ III keeps driving those overlay planes without rerendering the
        // static map underneath them.
        output.visible_animation_count = frame.visible_animation_count;
        output.request_continuous_redraw = frame.visible_animation_count != 0;
        output.renderer_cpu_ticks = renderer_ticks;
        output.textured_tile_count = cached_textured_tile_count;
        output.fallback_tile_indices = fallback_tile_indices.empty() ? nullptr : fallback_tile_indices.data();
        output.replacement_tile_flags = replacement_tile_flags.empty() ? nullptr : replacement_tile_flags.data();
        output.replacement_tile_count = static_cast<c3x_renderer_u32>(replacement_tile_flags.size());
        output.frame_invalidation_flags = invalidations;
        output.cache_hits = cache_hits;
        output.cache_misses = cache_misses;
        output.cache_evictions = cache_evictions;
        output.cache_stale_rejections = cache_stale_rejections;
        output.cache_entries = static_cast<c3x_renderer_u32>(viewport_cache.size());
        output.cache_capacity = viewport_cache_capacity;
        output.device_generation = device_generation;
        output.device_recoveries = device_recoveries;
        output.content_revision = static_cast<c3x_renderer_i64>(content_revision);
    }

    c3x_renderer_u32 invalidations_for(c3x_renderer::TerrainFrameSignature const & signature) const {
        if (!cache_valid)
            return C3X_RENDERER_INVALIDATE_ALL;
        c3x_renderer_u32 flags = 0;
        if (signature.camera != cached_signature.camera)
            flags |= C3X_RENDERER_INVALIDATE_CAMERA;
        if (signature.scene != cached_signature.scene)
            flags |= C3X_RENDERER_INVALIDATE_SCENE;
        if (signature.environment != cached_signature.environment)
            flags |= C3X_RENDERER_INVALIDATE_ENVIRONMENT;
        if (signature.wrap != cached_signature.wrap)
            flags |= C3X_RENDERER_INVALIDATE_WRAP;
        if (signature.ownership != cached_signature.ownership)
            flags |= C3X_RENDERER_INVALIDATE_OWNERSHIP;
        if (content_revision != previous_content_revision)
            flags |= C3X_RENDERER_INVALIDATE_PACK_DEFINITION;
        return flags;
    }

    bool same_terrain_record(c3x_renderer_tile_v1 const & left,
                             c3x_renderer_tile_v1 const & right) const {
        return left.tile_x == right.tile_x && left.tile_y == right.tile_y &&
            left.anchor_x == right.anchor_x && left.anchor_y == right.anchor_y &&
            left.terrain_type == right.terrain_type &&
            left.real_terrain_type == right.real_terrain_type &&
            left.variant_seed == right.variant_seed &&
            left.tile_flags == right.tile_flags &&
            left.feature_flags == right.feature_flags &&
            left.improvement_flags == right.improvement_flags &&
            left.has_effect == right.has_effect &&
            left.river_code == right.river_code &&
            left.road_mask == right.road_mask &&
            left.railroad_mask == right.railroad_mask &&
            left.route_style == right.route_style &&
            left.resource_id == right.resource_id &&
            left.resource_class == right.resource_class &&
            std::memcmp(left.resource_name, right.resource_name,
                        sizeof(left.resource_name)) == 0 &&
            left.city_id == right.city_id &&
            left.city_owner_id == right.city_owner_id &&
            left.city_size == right.city_size &&
            left.city_culture_group == right.city_culture_group &&
            left.city_era == right.city_era &&
            left.city_flags == right.city_flags;
    }

    bool reuse_cached_subset(c3x_renderer_frame_v1 const & frame,
                             c3x_renderer::TerrainFrameSignature const & signature) {
        if (!cache_valid || cached_tiles.empty() ||
            frame.tile_count > cached_tiles.size() ||
            signature.camera != cached_signature.camera ||
            signature.environment != cached_signature.environment ||
            signature.wrap != cached_signature.wrap ||
            content_revision != previous_content_revision ||
            cached_replacement_tile_flags.size() != cached_tiles.size())
            return false;
        std::vector<c3x_renderer_u32> subset_flags;
        subset_flags.reserve(frame.tile_count);
        for (c3x_renderer_u32 index = 0; index < frame.tile_count; ++index) {
            bool found = false;
            for (std::size_t cached_index = 0; cached_index < cached_tiles.size(); ++cached_index) {
                if (same_terrain_record(frame.tiles[index], cached_tiles[cached_index])) {
                    subset_flags.push_back(cached_replacement_tile_flags[cached_index]);
                    found = true;
                    break;
                }
            }
            if (!found)
                return false;
        }
        replacement_tile_flags = std::move(subset_flags);
        fallback_tile_indices.clear();
        return true;
    }

    bool render(c3x_renderer_frame_v1 const & frame, c3x_renderer_output_v1 & output) {
        LARGE_INTEGER started = {}, finished = {};
        QueryPerformanceCounter(&started);
        if (!initialize()) {
            OutputDebugStringA("[C3X renderer] native-failure=initialize\n");
            return false;
        }
        if (!ensure_targets(frame.target_width, frame.target_height)) {
            OutputDebugStringA("[C3X renderer] native-failure=targets\n");
            return false;
        }
        for (int index = 0; index < c3x_renderer::terrain_type_count; ++index)
            if (terrain_textures[index].configured && !ensure_pack_texture(index))
                terrain_textures[index].configured = false;
        if (!ensure_terrain_textures()) {
            OutputDebugStringA("[C3X renderer] native-failure=terrain-textures\n");
            return false;
        }
        c3x_renderer::TerrainFrameSignature signature = c3x_renderer::terrain_frame_signature(
            frame, content_revision, device_generation);
        if (cache_valid &&
            signature.complete == cached_signature.complete) {
            if (cache_hits != 0xffffffffu)
                ++cache_hits;
            replacement_tile_flags = cached_replacement_tile_flags;
            fallback_tile_indices.clear();
            fill_output(frame, output, 0, 0);
            return true;
        }
        {
            for (std::size_t cache_index = 0; cache_index < viewport_cache.size(); ++cache_index) {
                if (viewport_cache[cache_index].signature.complete != signature.complete)
                    continue;
                CachedViewport hit = std::move(viewport_cache[cache_index]);
                viewport_cache.erase(viewport_cache.begin() + static_cast<std::ptrdiff_t>(cache_index));
                pixels = hit.pixels;
                cached_tiles = hit.tiles;
                cached_replacement_tile_flags = hit.replacement_flags;
                replacement_tile_flags = hit.replacement_flags;
                cached_rendered_tile_count = hit.rendered_tile_count;
                cached_fallback_tile_count = hit.fallback_tile_count;
                cached_textured_tile_count = hit.textured_tile_count;
                cached_visible_animation_count = 0;
                cached_request_continuous_redraw = 0;
                cached_signature = hit.signature;
                cache_valid = true;
                viewport_cache.push_back(std::move(hit));
                fallback_tile_indices.clear();
                if (cache_hits != 0xffffffffu)
                    ++cache_hits;
                fill_output(frame, output, 0, 0);
                return true;
            }
        }
        if (reuse_cached_subset(frame, signature)) {
            if (cache_hits != 0xffffffffu)
                ++cache_hits;
            fill_output(frame, output, 0, 0);
            return true;
        }
        c3x_renderer_u32 invalidations = invalidations_for(signature);
        if (cache_valid) {
            if (cache_stale_rejections != 0xffffffffu)
                ++cache_stale_rejections;
        }
        if (cache_misses != 0xffffffffu)
            ++cache_misses;

        int const base_ground_grid = frame.tile_width >= 96 ?
            (frame.tile_count <= 768 ? 16 : 12) : 8;
        std::vector<Vertex> underlay_vertices;
        std::vector<Vertex> land_vertices;
        std::vector<Vertex> bed_vertices;
        std::vector<Vertex> water_vertices;
        std::vector<Vertex> river_vertices;
        std::vector<Vertex> route_vertices;
        std::vector<Vertex> shadow_vertices;
        std::vector<Vertex> feature_vertices;
        std::vector<Vertex> city_vertices;
        std::vector<Vertex> wall_vertices;
        std::vector<Vertex> mine_vertices;
        std::size_t rendered_reserve = 0, land_reserve = 0, river_reserve = 0;
        for (c3x_renderer_u32 index = 0; index < frame.tile_count; ++index) {
            c3x_renderer_tile_v1 const & tile = frame.tiles[index];
            if ((tile.tile_flags & C3X_RENDERER_TILE_RENDER) == 0)
                continue;
            ++rendered_reserve;
            if (ground_type(tile) < 11)
                ++land_reserve;
            if ((tile.river_code & 170u) != 0)
                ++river_reserve;
        }
        std::size_t const ground_vertices_per_tile =
            static_cast<std::size_t>(base_ground_grid) * base_ground_grid * 6u;
        underlay_vertices.reserve(rendered_reserve * ground_vertices_per_tile);
        land_vertices.reserve(land_reserve * ground_vertices_per_tile);
        bed_vertices.reserve(rendered_reserve * ground_vertices_per_tile);
        water_vertices.reserve(rendered_reserve * ground_vertices_per_tile);
        river_vertices.reserve(river_reserve * ground_vertices_per_tile);
        shadow_vertices.reserve(land_reserve * ground_vertices_per_tile);
        float half_w = static_cast<float>(frame.tile_width) * 0.5f;
        float half_h = static_cast<float>(frame.tile_height) * 0.5f;
        auto ndc_x = [&frame](float x) { return 2.0f * x / frame.target_width - 1.0f; };
        auto ndc_y = [&frame](float y) { return 1.0f - 2.0f * y / frame.target_height; };
        c3x_renderer_u32 textured_tile_count = 0;
        c3x_renderer_u32 fallback_tile_count = 0;
        fallback_tile_indices.clear();
        replacement_tile_flags.assign(frame.tile_count, 0u);
        // Terrain cast visibility is independent of screen anchors.  Keep one
        // bounded field for the current authoritative tile snapshot so pixel
        // scrolling can move the already-sampled field instead of repeating
        // millions of relief ray samples.  Every terrain/environment/wrap/
        // zoom/content input is fingerprinted; retained overlays are not.
        std::uint64_t current_shadow_field_signature = 1469598103934665603ull;
        auto hash_shadow_value = [&current_shadow_field_signature](auto value) {
            auto const * bytes = reinterpret_cast<std::uint8_t const *>(&value);
            for (std::size_t byte = 0; byte < sizeof(value); ++byte) {
                current_shadow_field_signature ^= bytes[byte];
                current_shadow_field_signature *= 1099511628211ull;
            }
        };
        hash_shadow_value(frame.tile_count);
        hash_shadow_value(frame.tile_width);
        hash_shadow_value(frame.tile_height);
        hash_shadow_value(frame.hour);
        hash_shadow_value(frame.season);
        hash_shadow_value(frame.world_width_tiles);
        hash_shadow_value(frame.world_height_tiles);
        hash_shadow_value(frame.world_wrap_x);
        hash_shadow_value(frame.world_wrap_y);
        hash_shadow_value(content_revision);
        hash_shadow_value(device_generation);
        for (c3x_renderer_u32 index = 0; index < frame.tile_count; ++index) {
            c3x_renderer_tile_v1 const & tile = frame.tiles[index];
            for (auto value : {tile.tile_x, tile.tile_y, tile.terrain_type,
                               tile.real_terrain_type})
                hash_shadow_value(value);
            for (auto value : {tile.tile_flags, tile.river_code})
                hash_shadow_value(value);
        }
        bool reuse_shadow_field = !shadow_visibility_cache.empty() &&
            current_shadow_field_signature == shadow_field_signature;
        std::size_t shadow_visibility_index = 0;
        std::vector<float> next_shadow_visibility;
        if (!reuse_shadow_field)
            next_shadow_visibility.reserve(65536);
        c3x_renderer::EnvironmentState environment = c3x_renderer::evaluate_environment(
            static_cast<float>(frame.hour), frame.season);
        TerrainShaderSettings frame_settings = {};
        frame_settings.height_texel[0] = 1.0f / 2048.0f;
        frame_settings.height_texel[1] = 1.0f / 2048.0f;
        frame_settings.normal_strength = 4.0f;
        frame_settings.exposure = 1.0f;
        float const * key_light = environment.sun_intensity >= environment.moon_intensity
            ? environment.sun_direction : environment.moon_direction;
        std::copy(key_light, key_light + 3,
                  std::begin(frame_settings.light_direction));
        frame_settings.sun_intensity = environment.sun_intensity;
        std::copy(std::begin(environment.sun_color), std::end(environment.sun_color),
                  std::begin(frame_settings.sun_color));
        frame_settings.shadow_strength = environment.shadow_strength;
        std::copy(std::begin(environment.moon_direction), std::end(environment.moon_direction),
                  std::begin(frame_settings.moon_direction));
        frame_settings.moon_intensity = environment.moon_intensity;
        std::copy(std::begin(environment.moon_color), std::end(environment.moon_color),
                  std::begin(frame_settings.moon_color));
        frame_settings.night_activation = environment.night_activation;
        std::copy(std::begin(environment.ambient_color), std::end(environment.ambient_color),
                  std::begin(frame_settings.ambient_color));
        frame_settings.environment_exposure = environment.exposure;
        frame_settings.water_fresnel = environment.water_fresnel;
        frame_settings.water_specular = environment.water_specular;
        frame_settings.emissive_scale = environment.emissive_scale;
        frame_settings.hour = static_cast<float>(frame.hour);
        context->UpdateSubresource(terrain_settings_buffer, 0, nullptr,
                                   &frame_settings, 0, 0);
        auto canonical_component = [](int value, int size, c3x_renderer_u32 wraps) {
            if (wraps == 0 || size <= 0)
                return value;
            int result = value % size;
            return result < 0 ? result + size : result;
        };
        auto coordinate_key = [&frame, &canonical_component](int x, int y) {
            x = canonical_component(x, frame.world_width_tiles, frame.world_wrap_x);
            y = canonical_component(y, frame.world_height_tiles, frame.world_wrap_y);
            return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(x)) << 32) |
                static_cast<std::uint32_t>(y);
        };
        std::unordered_map<std::uint64_t, int> ground_by_coordinate;
        std::unordered_map<std::uint64_t, int> real_by_coordinate;
        std::unordered_map<std::uint64_t, int> relief_by_coordinate;
        std::unordered_map<std::uint64_t, int> surface_by_coordinate;
        std::unordered_map<std::uint64_t, c3x_renderer_tile_v1 const *> tile_by_coordinate;
        ground_by_coordinate.reserve(frame.tile_count);
        real_by_coordinate.reserve(frame.tile_count);
        relief_by_coordinate.reserve(frame.tile_count);
        surface_by_coordinate.reserve(frame.tile_count);
        tile_by_coordinate.reserve(frame.tile_count);
        for (c3x_renderer_u32 index = 0; index < frame.tile_count; ++index) {
            c3x_renderer_tile_v1 const & tile = frame.tiles[index];
            if ((tile.tile_flags & (C3X_RENDERER_TILE_RENDER |
                                    C3X_RENDERER_TILE_TOPOLOGY_HALO)) == 0)
                continue;
            int ground = ground_type(tile);
            int relief = relief_type(tile);
            tile_by_coordinate[coordinate_key(tile.tile_x, tile.tile_y)] = &tile;
            if (relief == 10 && !volcano_assets_ready)
                relief = -1;
            ground_by_coordinate[coordinate_key(tile.tile_x, tile.tile_y)] =
                ground >= 0 && ground < c3x_renderer::terrain_type_count &&
                terrain_textures[ground].view != nullptr ? ground : -1;
            real_by_coordinate[coordinate_key(tile.tile_x, tile.tile_y)] =
                tile.real_terrain_type;
            relief_by_coordinate[coordinate_key(tile.tile_x, tile.tile_y)] =
                relief >= 0 && relief < c3x_renderer::terrain_type_count &&
                terrain_textures[relief].view != nullptr ? relief : -1;
            int integrated_marsh = marsh_assets_ready &&
                tile.real_terrain_type == 9 && terrain_textures[9].view != nullptr ? 9 : -1;
            surface_by_coordinate[coordinate_key(tile.tile_x, tile.tile_y)] =
                integrated_marsh >= 0 ? integrated_marsh :
                (relief >= 0 && relief < c3x_renderer::terrain_type_count &&
                 terrain_textures[relief].view != nullptr ? relief :
                 (ground >= 0 && ground < c3x_renderer::terrain_type_count &&
                  terrain_textures[ground].view != nullptr ? ground : -1));
        }
        struct RiverNode {
            int lattice_x;
            int lattice_y;
            unsigned degree;
            bool touches_water;
        };
        std::vector<RiverNode> river_nodes;
        auto river_node_at = [&river_nodes](int x, int y) -> RiverNode & {
            for (RiverNode & node : river_nodes)
                if (node.lattice_x == x && node.lattice_y == y)
                    return node;
            river_nodes.push_back(RiverNode{x, y, 0u, false});
            return river_nodes.back();
        };
        auto add_river_edge = [&river_node_at](int start_x, int start_y,
                                               int endpoint_x, int endpoint_y) {
            river_node_at(start_x, start_y).degree += 1u;
            river_node_at(endpoint_x, endpoint_y).degree += 1u;
        };
        for (c3x_renderer_u32 index = 0; index < frame.tile_count; ++index) {
            c3x_renderer_tile_v1 const & tile = frame.tiles[index];
            if ((tile.tile_flags & (C3X_RENDERER_TILE_RENDER |
                                    C3X_RENDERER_TILE_TOPOLOGY_HALO)) == 0)
                continue;
            unsigned mask = tile.river_code & 170u;
            if ((mask & 2u) != 0)
                add_river_edge(tile.tile_x, tile.tile_y - 1,
                               tile.tile_x + 1, tile.tile_y);
            if ((mask & 8u) != 0)
                add_river_edge(tile.tile_x + 1, tile.tile_y,
                               tile.tile_x, tile.tile_y + 1);
        }
        constexpr int river_corner_offsets[4][2] = {
            {0, -1}, {1, 0}, {0, 1}, {-1, 0}
        };
        for (c3x_renderer_u32 index = 0; index < frame.tile_count; ++index) {
            c3x_renderer_tile_v1 const & tile = frame.tiles[index];
            if ((tile.tile_flags & (C3X_RENDERER_TILE_RENDER |
                                    C3X_RENDERER_TILE_TOPOLOGY_HALO)) == 0 ||
                ground_type(tile) < 11)
                continue;
            for (auto const & offset : river_corner_offsets)
                for (RiverNode & node : river_nodes)
                    if (node.lattice_x == tile.tile_x + offset[0] &&
                        node.lattice_y == tile.tile_y + offset[1])
                        node.touches_water = true;
        }
        c3x_renderer::FeatureGroup broadleaf_forest;
        c3x_renderer::FeatureGroup const * forest_group =
            c3x_renderer::find_feature_group(feature_bundle, "forest");
        if (forest_group != nullptr) {
            broadleaf_forest.name = "forest";
            for (c3x_renderer::FeaturePlacement const & placement : forest_group->placements)
                if (placement.asset_index < feature_bundle.assets.size() &&
                    feature_bundle.assets[placement.asset_index].id.find(
                        "feature/forest/leafy") != std::string::npos)
                    broadleaf_forest.placements.push_back(placement);
            if (!broadleaf_forest.placements.empty())
                forest_group = &broadleaf_forest;
        }
        c3x_renderer::FeatureGroup const * river_rock_group = river_assets_ready
            ? c3x_renderer::find_feature_group(river_rock_bundle, "river_rock") : nullptr;
        auto stable_feature_hash = [](std::uint32_t value) {
            value ^= value >> 16;
            value *= 0x7feb352du;
            value ^= value >> 15;
            value *= 0x846ca68bu;
            return value ^ (value >> 16);
        };
        auto append_object_shadow = [&](c3x_renderer::FeatureAsset const & asset,
                                        float scale, float center_x, float center_y,
                                        float ground_height_screen) {
            float radius = 0.0f;
            float feature_height = 0.0f;
            for (c3x_renderer::FeatureSourceVertex const & vertex : asset.vertices) {
                radius = std::max(radius, std::sqrt(
                    vertex.position[0] * vertex.position[0] +
                    vertex.position[1] * vertex.position[1]) * scale);
                feature_height = std::max(feature_height, vertex.position[2] * scale);
            }
            float shadow_width = std::max(4.0f, radius * half_w * 0.65f);
            float horizontal = std::sqrt(key_light[0] * key_light[0] +
                                         key_light[1] * key_light[1]);
            float cast_world_x = horizontal > 0.001f ? -key_light[0] / horizontal : 0.0f;
            float cast_world_y = horizontal > 0.001f ? -key_light[1] / horizontal : 1.0f;
            float cast_screen_x = cast_world_x - cast_world_y;
            float cast_screen_y = (cast_world_x + cast_world_y) * half_h / half_w;
            float cast_length = std::sqrt(cast_screen_x * cast_screen_x +
                                          cast_screen_y * cast_screen_y);
            if (cast_length > 0.001f) {
                cast_screen_x /= cast_length;
                cast_screen_y /= cast_length;
            }
            float height_shadow_length = feature_height * 150.0f *
                (static_cast<float>(frame.tile_width) / 224.0f) * 0.72f;
            float shadow_length = std::clamp(
                std::max(shadow_width * 2.40f, height_shadow_length),
                shadow_width * 2.55f, std::min(180.0f, shadow_width * 10.0f));
            float perpendicular_x = -cast_screen_y;
            float perpendicular_y = cast_screen_x;
            float ground_base_screen_y = center_y + ground_height_screen;
            auto make_shadow_vertex = [&](float screen_x, float screen_y, float u, float v) {
                float projected_base_y = ground_base_screen_y + (screen_y - center_y);
                float depth = std::max(0.001f, std::clamp(
                    1.0f - projected_base_y / static_cast<float>(frame.target_height) -
                    ground_height_screen * 0.75f / static_cast<float>(frame.target_height),
                    0.001f, 0.999f) - 0.004f);
                return Vertex{
                    ndc_x(screen_x), ndc_y(screen_y), depth, u, v, 1.0f,
                    0.0f, 0.0f, 1.0f, 1.0f, 1.0f, 0.0f, 0.0f,
                    7.0f, 0.0f, 0.0f, 0.0f,
                    0.0f, 0.0f, 0.0f, 0.0f,
                    0.0f, 0.0f, 1.0f,
                    1000.0f, 0.0f, 1000.0f, 0.0f, -1.0f};
            };
            float near_left_x = center_x - perpendicular_x * shadow_width * 0.42f;
            float near_left_y = center_y - perpendicular_y * shadow_width * 0.42f;
            float near_right_x = center_x + perpendicular_x * shadow_width * 0.42f;
            float near_right_y = center_y + perpendicular_y * shadow_width * 0.42f;
            float far_right_x = center_x + cast_screen_x * shadow_length +
                                perpendicular_x * shadow_width * 0.72f;
            float far_right_y = center_y + cast_screen_y * shadow_length +
                                perpendicular_y * shadow_width * 0.72f;
            float far_left_x = center_x + cast_screen_x * shadow_length -
                               perpendicular_x * shadow_width * 0.72f;
            float far_left_y = center_y + cast_screen_y * shadow_length -
                               perpendicular_y * shadow_width * 0.72f;
            Vertex near_left = make_shadow_vertex(near_left_x, near_left_y, 0.0f, 0.0f);
            Vertex near_right = make_shadow_vertex(near_right_x, near_right_y, 1.0f, 0.0f);
            Vertex far_right = make_shadow_vertex(far_right_x, far_right_y, 1.0f, 1.0f);
            Vertex far_left = make_shadow_vertex(far_left_x, far_left_y, 0.0f, 1.0f);
            Vertex triangles[] = {near_left, near_right, far_right,
                                  near_left, far_right, far_left};
            shadow_vertices.insert(shadow_vertices.end(),
                                   std::begin(triangles), std::end(triangles));
        };
        for (c3x_renderer_u32 index = 0; index < frame.tile_count; ++index) {
            c3x_renderer_tile_v1 const & tile = frame.tiles[index];
            if ((tile.tile_flags & C3X_RENDERER_TILE_RENDER) == 0)
                continue;
            int ground = ground_type(tile);
            int relief = relief_type(tile);
            if (relief == 10 && !volcano_assets_ready)
                relief = -1;
            if (ground < 0 || ground >= c3x_renderer::terrain_type_count ||
                terrain_textures[ground].view == nullptr)
                continue;
            if (relief < 0 || relief >= c3x_renderer::terrain_type_count ||
                terrain_textures[relief].view == nullptr)
                relief = -1;
            bool draw_feature = feature_assets_ready &&
                (tile.real_terrain_type == 7 || tile.real_terrain_type == 8);
            bool draw_marsh = marsh_assets_ready && tile.real_terrain_type == 9;
            bool draw_volcano = volcano_assets_ready && tile.real_terrain_type == 10;
            bool draw_dunes = dune_assets_ready &&
                tile.real_terrain_type == 0 && tile.terrain_type == 0;
            ++textured_tile_count;
            replacement_tile_flags[index] = C3X_RENDERER_TILE_CUSTOM_TERRAIN_REPLACED;
            if (draw_feature || draw_marsh || draw_volcano)
                replacement_tile_flags[index] |= C3X_RENDERER_TILE_CUSTOM_FEATURE_REPLACED;
            if (draw_dunes)
                replacement_tile_flags[index] |= C3X_RENDERER_TILE_CUSTOM_DUNES_REPLACED;
            if (river_assets_ready && (tile.river_code & 170u) != 0)
                replacement_tile_flags[index] |= C3X_RENDERER_TILE_CUSTOM_RIVER_REPLACED;
            if (route_assets_ready && tile.road_mask != 0)
                replacement_tile_flags[index] |= C3X_RENDERER_TILE_CUSTOM_ROAD_REPLACED;
            if (route_assets_ready && tile.railroad_mask != 0)
                replacement_tile_flags[index] |= C3X_RENDERER_TILE_CUSTOM_RAILROAD_REPLACED;
            if (city_assets_ready && tile.city_id >= 0)
                replacement_tile_flags[index] |= C3X_RENDERER_TILE_CUSTOM_CITY_REPLACED;
            if (mine_assets_ready &&
                (tile.improvement_flags & C3X_RENDERER_IMPROVEMENT_MINE) != 0)
                replacement_tile_flags[index] |= C3X_RENDERER_TILE_CUSTOM_MINE_REPLACED;
            float left = static_cast<float>(tile.anchor_x);
            float top = static_cast<float>(tile.anchor_y);
            // The source terrain materials are detail textures, not one enormous
            // decal per viewport.  This scale puts a repeat across roughly six
            // Civ III tiles and is close to the cell-relative density used by
            // modern connected-grid terrain renderers.
            float const uv_scale = 0.26f;
            // The approved BIQ Lab projection uses a 224 px tile and applies
            // 0.82 vertical scale to terrain relief. Preserve that exact ratio
            // at both Civ III zoom levels. Feature bodies use the Lab's full
            // 150-pixel vertical basis rather than terrain's 0.82 multiplier.
            float const relief_projection_scale =
                static_cast<float>(frame.tile_width) / 224.0f * 0.82f;
            float const feature_projection_scale =
                static_cast<float>(frame.tile_width) / 224.0f;
            float ground_slot = static_cast<float>(ground);
            float surface_slot = static_cast<float>(draw_marsh ? 9 :
                (relief >= 0 ? relief : ground));
            int neighbor_coordinates[4][2] = {
                {tile.tile_x - 1, tile.tile_y - 1}, {tile.tile_x + 1, tile.tile_y - 1},
                {tile.tile_x + 1, tile.tile_y + 1}, {tile.tile_x - 1, tile.tile_y + 1}
            };
            bool relief_neighborhood = relief == 5 || relief == 6 || relief == 10 || draw_dunes;
            for (int edge = 0; edge < 4; ++edge) {
                auto found = relief_by_coordinate.find(coordinate_key(
                    neighbor_coordinates[edge][0], neighbor_coordinates[edge][1]));
                relief_neighborhood = relief_neighborhood || (found != relief_by_coordinate.end() &&
                    (found->second == 5 || found->second == 6 || found->second == 10));
            }
            // Keep close authored relief dense when the viewport contains only
            // a few hundred tiles, then spend the same geometry budget across
            // wider views.  Unbounded 24x24 patches can exhaust the 32-bit
            // preview/game process and exceed D3D11's per-buffer size limit.
            int const relief_grid = frame.tile_width >= 96 ?
                (frame.tile_count <= 512 ? 24 : (frame.tile_count <= 768 ? 16 : 12)) :
                (frame.tile_count <= 2048 ? 12 : 8);
            int const tile_ground_grid = relief_neighborhood ? relief_grid : base_ground_grid;
            auto ground_at = [&](int x, int y) {
                auto found = ground_by_coordinate.find(coordinate_key(x, y));
                return found == ground_by_coordinate.end() ? ground_slot : static_cast<float>(found->second);
            };
            auto surface_at = [&](int x, int y) {
                auto found = surface_by_coordinate.find(coordinate_key(x, y));
                return found == surface_by_coordinate.end() ? surface_slot : static_cast<float>(found->second);
            };
            auto terrain_at_lattice = [&](int lattice_u, int lattice_v) {
                int x = lattice_u + lattice_v;
                int y = lattice_u - lattice_v;
                return surface_at(x, y);
            };
            auto ground_at_lattice = [&](int lattice_u, int lattice_v) {
                int x = lattice_u + lattice_v;
                int y = lattice_u - lattice_v;
                return ground_at(x, y);
            };
            auto relief_at_lattice = [&](int lattice_u, int lattice_v) {
                int x = lattice_u + lattice_v;
                int y = lattice_u - lattice_v;
                auto found = relief_by_coordinate.find(coordinate_key(x, y));
                return found == relief_by_coordinate.end() ? -1 : found->second;
            };
            auto periodic_surface_uv = [&](float world_u, float world_v, float frequency) {
                float map_x = world_u + world_v - 1.0f;
                float map_y = world_u - world_v;
                float half_frequency = frequency * 0.5f;
                float x_component = map_x * half_frequency;
                float y_component = map_y * half_frequency;
                if (frame.world_wrap_x != 0 && frame.world_width_tiles > 0) {
                    float cycles = std::max(1.0f, std::round(frame.world_width_tiles * half_frequency));
                    float canonical_map_x = static_cast<float>(canonical_component(
                        tile.tile_x, frame.world_width_tiles, frame.world_wrap_x)) + map_x - tile.tile_x;
                    x_component = cycles * canonical_map_x / static_cast<float>(frame.world_width_tiles);
                }
                if (frame.world_wrap_y != 0 && frame.world_height_tiles > 0) {
                    float cycles = std::max(1.0f, std::round(frame.world_height_tiles * half_frequency));
                    float canonical_map_y = static_cast<float>(canonical_component(
                        tile.tile_y, frame.world_height_tiles, frame.world_wrap_y)) + map_y - tile.tile_y;
                    y_component = cycles * canonical_map_y / static_cast<float>(frame.world_height_tiles);
                }
                return std::array<float, 2>{x_component + y_component, y_component - x_component};
            };
            auto river_segment_distance = [](float point_x, float point_y,
                                              float start_x, float start_y,
                                              float endpoint_x, float endpoint_y) {
                float segment_x = endpoint_x - start_x;
                float segment_y = endpoint_y - start_y;
                float point_offset_x = point_x - start_x;
                float point_offset_y = point_y - start_y;
                float denominator = segment_x * segment_x + segment_y * segment_y;
                float t = denominator > 0.0f
                    ? std::clamp((point_offset_x * segment_x + point_offset_y * segment_y) /
                                     denominator, 0.0f, 1.0f)
                    : 0.0f;
                float delta_x = point_offset_x - segment_x * t;
                float delta_y = point_offset_y - segment_y * t;
                return std::sqrt(delta_x * delta_x + delta_y * delta_y);
            };
            auto river_edge_distance = [&](c3x_renderer_tile_v1 const & river_tile,
                                           float u, float v,
                                           float start_u, float start_v,
                                           float endpoint_u, float endpoint_v,
                                           unsigned direction_bit) {
                auto screen_point = [&](float local_u, float local_v) {
                    return std::array<float, 2>{
                        (local_u - local_v) * half_w,
                        (local_u + local_v - 1.0f) * half_h};
                };
                std::array<float, 2> point = screen_point(u, v);
                std::array<float, 2> start = screen_point(start_u, start_v);
                std::array<float, 2> endpoint = screen_point(endpoint_u, endpoint_v);
                int canonical_river_x = canonical_component(
                    river_tile.tile_x, frame.world_width_tiles, frame.world_wrap_x);
                int canonical_river_y = canonical_component(
                    river_tile.tile_y, frame.world_height_tiles, frame.world_wrap_y);
                int canonical_column = (canonical_river_x + canonical_river_y) / 2;
                int canonical_row = (canonical_river_x - canonical_river_y) / 2;
                unsigned edge_family = 0u;
                if (direction_bit == 32u)
                    canonical_row -= 1;
                else if (direction_bit == 128u) {
                    canonical_column -= 1;
                    edge_family = 1u;
                } else if (direction_bit == 8u)
                    edge_family = 1u;
                unsigned seed = static_cast<unsigned>(canonical_column + 4096) * 73856093u ^
                                static_cast<unsigned>(canonical_row + 4096) * 19349663u ^
                                edge_family * 83492791u;
                seed ^= seed >> 13;
                float direction_x = endpoint[0] - start[0];
                float direction_y = endpoint[1] - start[1];
                float direction_length = std::sqrt(direction_x * direction_x +
                                                   direction_y * direction_y);
                float normal_x = -direction_y / std::max(direction_length, 0.001f);
                float normal_y = direction_x / std::max(direction_length, 0.001f);
                float primary_bend = 5.0f +
                    static_cast<float>((seed >> 5) & 15u) / 15.0f * 8.0f;
                float secondary_bend = 1.5f +
                    static_cast<float>((seed >> 11) & 7u) / 7.0f * 4.0f;
                if ((seed & 1u) != 0)
                    primary_bend = -primary_bend;
                if ((seed & 2u) != 0)
                    secondary_bend = -secondary_bend;
                float distance = 1000.0f;
                float previous_x = start[0];
                float previous_y = start[1];
                for (int segment = 1; segment <= 16; ++segment) {
                    float t = static_cast<float>(segment) / 16.0f;
                    float offset = std::sin(t * 3.14159265f) * primary_bend +
                                   std::sin(t * 6.28318531f) * secondary_bend;
                    float curve_x = start[0] + direction_x * t + normal_x * offset;
                    float curve_y = start[1] + direction_y * t + normal_y * offset;
                    distance = std::min(distance, river_segment_distance(
                        point[0], point[1], previous_x, previous_y, curve_x, curve_y));
                    previous_x = curve_x;
                    previous_y = curve_y;
                }
                return distance;
            };
            auto river_distance = [&](c3x_renderer_tile_v1 const & river_tile,
                                      float u, float v) {
                float distance = 1000.0f;
                unsigned mask = river_tile.river_code & 170u;
                if ((mask & 2u) != 0)
                    distance = std::min(distance, river_edge_distance(
                        river_tile, u, v, 0.0f, 0.0f, 1.0f, 0.0f, 2u));
                if ((mask & 8u) != 0)
                    distance = std::min(distance, river_edge_distance(
                        river_tile, u, v, 1.0f, 0.0f, 1.0f, 1.0f, 8u));
                if ((mask & 32u) != 0)
                    distance = std::min(distance, river_edge_distance(
                        river_tile, u, v, 0.0f, 1.0f, 1.0f, 1.0f, 32u));
                if ((mask & 128u) != 0)
                    distance = std::min(distance, river_edge_distance(
                        river_tile, u, v, 0.0f, 0.0f, 0.0f, 1.0f, 128u));
                return distance;
            };
            auto river_node_distance = [&](float u, float v, unsigned node_kind) {
                float point_x = static_cast<float>(tile.tile_x) + u - v;
                float point_y = static_cast<float>(tile.tile_y) + u + v - 1.0f;
                float distance = 1000.0f;
                for (RiverNode const & node : river_nodes) {
                    bool selected = node_kind == 0u
                        ? node.degree == 1u && !node.touches_water
                        : (node_kind == 1u ? node.degree >= 3u
                                          : node.degree == 1u && node.touches_water);
                    if (!selected)
                        continue;
                    float delta_x = (point_x - static_cast<float>(node.lattice_x)) * half_w;
                    float delta_y = (point_y - static_cast<float>(node.lattice_y)) * half_h;
                    distance = std::min(distance,
                        std::sqrt(delta_x * delta_x + delta_y * delta_y));
                }
                return distance;
            };
            auto center_material_weights = [&](int lattice_u, int lattice_v) {
                std::array<float, 4> result = {};
                int base = static_cast<int>(ground_at_lattice(lattice_u, lattice_v));
                int surface = static_cast<int>(terrain_at_lattice(lattice_u, lattice_v));
                auto add_material = [&result](int sample_base, int sample_surface, float amount) {
                    if (sample_base >= 11)
                        return;
                    int material = sample_surface == 9 ? 3 :
                        (sample_base == 0 || sample_base == 4 ? 2 :
                         (sample_base == 1 ? 1 : 0));
                    result[material] += amount;
                };
                if (base < 11)
                    add_material(base, surface, 1.0f);
                else {
                    for (int y = -1; y <= 1; ++y) {
                        for (int x = -1; x <= 1; ++x) {
                            int sample_base = static_cast<int>(
                                ground_at_lattice(lattice_u + x, lattice_v + y));
                            int sample_surface = static_cast<int>(
                                terrain_at_lattice(lattice_u + x, lattice_v + y));
                            add_material(sample_base, sample_surface,
                                         (x == 0 || y == 0) ? 1.0f : 0.70f);
                        }
                    }
                }
                float total = result[0] + result[1] + result[2] + result[3];
                if (total <= 0.0f)
                    result[0] = 1.0f;
                else
                    for (float & value : result)
                        value /= total;
                return result;
            };
            auto material_weights_for = [&](float world_u, float world_v) {
                float source_x = world_u + world_v - 1.0f;
                float source_y = world_u - world_v;
                float warp_x = std::sin(source_x * 0.83f + source_y * 1.19f) * 0.10f +
                    std::sin(source_x * 2.31f - source_y * 0.67f) * 0.035f;
                float warp_y = std::sin(source_x * 1.07f - source_y * 0.91f) * 0.10f +
                    std::sin(source_x * 0.59f + source_y * 2.03f) * 0.035f;
                float grid_x = world_u - 0.5f + warp_x;
                float grid_y = world_v - 0.5f + warp_y;
                int x0 = static_cast<int>(std::floor(grid_x));
                int y0 = static_cast<int>(std::floor(grid_y));
                float tx = smoothstep01((grid_x - static_cast<float>(x0) - 0.20f) / 0.60f);
                float ty = smoothstep01((grid_y - static_cast<float>(y0) - 0.20f) / 0.60f);
                std::array<float, 4> c00 = center_material_weights(x0, y0);
                std::array<float, 4> c10 = center_material_weights(x0 + 1, y0);
                std::array<float, 4> c01 = center_material_weights(x0, y0 + 1);
                std::array<float, 4> c11 = center_material_weights(x0 + 1, y0 + 1);
                std::array<float, 4> result = {};
                for (int material = 0; material < 4; ++material) {
                    float top = c00[material] * (1.0f - tx) + c10[material] * tx;
                    float bottom = c01[material] * (1.0f - tx) + c11[material] * tx;
                    result[material] = top * (1.0f - ty) + bottom * ty;
                }
                return result;
            };
            auto water_family_depth = [&](float world_u, float world_v) {
                float grid_x = world_u - 0.5f;
                float grid_y = world_v - 0.5f;
                int x0 = static_cast<int>(std::floor(grid_x));
                int y0 = static_cast<int>(std::floor(grid_y));
                float tx = smoothstep01(grid_x - static_cast<float>(x0));
                float ty = smoothstep01(grid_y - static_cast<float>(y0));
                auto center_depth = [&](int x, int y) {
                    int base = static_cast<int>(ground_at_lattice(x, y));
                    return base >= 11 ? std::clamp((base - 10) * 0.34f, 0.18f, 1.0f) : 0.34f;
                };
                float top = center_depth(x0, y0) * (1.0f - tx) +
                    center_depth(x0 + 1, y0) * tx;
                float bottom = center_depth(x0, y0 + 1) * (1.0f - tx) +
                    center_depth(x0 + 1, y0 + 1) * tx;
                return top * (1.0f - ty) + bottom * ty;
            };
            auto signed_shore_distance = [&](float world_u, float world_v,
                                             float local_u, float local_v) {
                float grid_x = world_u - 0.5f;
                float grid_y = world_v - 0.5f;
                int x0 = static_cast<int>(std::floor(grid_x));
                int y0 = static_cast<int>(std::floor(grid_y));
                float tx = smoothstep01(grid_x - static_cast<float>(x0));
                float ty = smoothstep01(grid_y - static_cast<float>(y0));
                auto raw_sign = [&](int x, int y) {
                    return ground_at_lattice(x, y) >= 11.0f ? 1.0f : -1.0f;
                };
                auto center_sign = [&](int x, int y) {
                    return raw_sign(x, y) * 0.62f +
                        (raw_sign(x - 1, y) + raw_sign(x + 1, y) +
                         raw_sign(x, y - 1) + raw_sign(x, y + 1)) * 0.095f;
                };
                float top = center_sign(x0, y0) * (1.0f - tx) +
                    center_sign(x0 + 1, y0) * tx;
                float bottom = center_sign(x0, y0 + 1) * (1.0f - tx) +
                    center_sign(x0 + 1, y0 + 1) * tx;
                float field = top * (1.0f - ty) + bottom * ty;
                float center_dx = local_u - 0.5f;
                float center_dy = local_v - 0.5f;
                float center_anchor = 1.0f - smoothstep01(
                    std::sqrt(center_dx * center_dx + center_dy * center_dy) / 0.34f);
                float own_sign = ground >= 11 ? 1.0f : -1.0f;
                field = field * (1.0f - center_anchor * 0.88f) +
                    own_sign * center_anchor * 0.88f;
                float source_x = world_u + world_v - 1.0f;
                float source_y = world_u - world_v;
                float contour_noise = std::sin(source_x * 1.37f + source_y * 0.71f) * 0.38f +
                    std::sin(source_x * 0.79f - source_y * 1.11f) * 0.20f +
                    std::sin(source_x * 3.83f - source_y * 2.17f) * 0.13f +
                    std::sin(source_x * 0.53f + source_y * 2.91f) * 0.07f;
                float boundary_weight = 1.0f - smoothstep01(std::abs(field) / 0.92f);
                contour_noise *= 1.0f - center_anchor * 0.90f;
                float result = field + contour_noise * boundary_weight;
                for (int row = y0 - 1; row <= y0 + 1; ++row) {
                    for (int column = x0 - 1; column <= x0 + 1; ++column) {
                        if (raw_sign(column, row) < 0.0f)
                            continue;
                        bool connected = raw_sign(column - 1, row) > 0.0f ||
                            raw_sign(column + 1, row) > 0.0f ||
                            raw_sign(column, row - 1) > 0.0f ||
                            raw_sign(column, row + 1) > 0.0f;
                        if (connected)
                            continue;
                        float dx = world_u - (static_cast<float>(column) + 0.5f);
                        float dy = world_v - (static_cast<float>(row) + 0.5f);
                        float basin = 1.0f - smoothstep01(
                            std::sqrt(dx * dx + dy * dy) / 0.46f);
                        result = std::max(result, basin);
                    }
                }
                return std::clamp(result, -1.0f, 1.0f);
            };
            auto sample_mountain = [&](int column, int row,
                                       float local_x, float local_y,
                                       float & height, float & blend) {
                height = 0.0f;
                blend = 0.0f;
                if (relief_at_lattice(column, row) != 6)
                    return;
                int source_x = canonical_component(column + row,
                    frame.world_width_tiles, frame.world_wrap_x);
                int source_y = canonical_component(column - row,
                    frame.world_height_tiles, frame.world_wrap_y);
                unsigned seed = static_cast<unsigned>(source_x * 73 + source_y * 151);
                unsigned variant = (seed >> 3) % 5u;
                TerrainTexture const & mountains = terrain_textures[6];
                if (mountains.relief_height_variants[variant].empty() ||
                    mountains.relief_blend_variants[variant].empty())
                    return;

                // Copied from the canonical Lab mountain sampler. Preserve the
                // five authored silhouettes and only apply its deterministic
                // rigid orientation and connected-massif footprint fit.
                unsigned transform = seed & 7u;
                if ((transform & 1u) != 0)
                    std::swap(local_x, local_y);
                if ((transform & 2u) != 0)
                    local_x = 1.0f - local_x;
                if ((transform & 4u) != 0)
                    local_y = 1.0f - local_y;
                constexpr int offsets[4][2] = {
                    {-1, 0}, {1, 0}, {0, -1}, {0, 1}};
                bool has_relief_neighbor = false;
                for (auto const & offset : offsets) {
                    int neighbor = relief_at_lattice(
                        column + offset[0], row + offset[1]);
                    has_relief_neighbor = has_relief_neighbor ||
                        neighbor == 6 || neighbor == 10;
                }
                float footprint_scale = has_relief_neighbor ? 0.50f : 0.68f;
                float source_u = 0.5f + (local_x - 0.5f) * footprint_scale;
                float source_v = 0.5f + (local_y - 0.5f) * footprint_scale;
                if (source_u < 0.0f || source_u > 1.0f ||
                    source_v < 0.0f || source_v > 1.0f)
                    return;
                float source_edge = std::min(std::min(source_u, 1.0f - source_u),
                    std::min(source_v, 1.0f - source_v));
                float edge = smoothstep01(source_edge / 0.055f);
                height = sample_normalized_field(
                    mountains.relief_height_variants[variant],
                    mountains.relief_variant_widths[variant],
                    mountains.relief_variant_heights[variant],
                    mountains.relief_height_minimum[variant],
                    mountains.relief_height_maximum[variant], source_u, source_v);
                blend = sample_normalized_field(
                    mountains.relief_blend_variants[variant],
                    mountains.relief_variant_widths[variant],
                    mountains.relief_variant_heights[variant],
                    mountains.relief_blend_minimum[variant],
                    mountains.relief_blend_maximum[variant], source_u, source_v) * edge;
            };
            auto sample_volcano = [&](int column, int row,
                                      float local_x, float local_y,
                                      float & height, float & blend) {
                height = 0.0f;
                blend = 0.0f;
                if (!volcano_assets_ready || relief_at_lattice(column, row) != 10)
                    return;
                int source_x = canonical_component(column + row,
                    frame.world_width_tiles, frame.world_wrap_x);
                int source_y = canonical_component(column - row,
                    frame.world_height_tiles, frame.world_wrap_y);
                unsigned seed = static_cast<unsigned>(source_x) * 73856093u ^
                    static_cast<unsigned>(source_y) * 19349663u;
                seed ^= seed >> 13;
                if ((seed & 1u) != 0)
                    std::swap(local_x, local_y);
                if ((seed & 2u) != 0)
                    local_x = 1.0f - local_x;
                constexpr int offsets[4][2] = {
                    {-1, 0}, {1, 0}, {0, -1}, {0, 1}};
                bool has_relief_neighbor = false;
                for (auto const & offset : offsets) {
                    int neighbor = relief_at_lattice(
                        column + offset[0], row + offset[1]);
                    has_relief_neighbor = has_relief_neighbor ||
                        neighbor == 6 || neighbor == 10;
                }
                float footprint_scale = has_relief_neighbor ? 0.60f : 1.0f;
                float aspect = (seed & 4u) != 0 ? 0.88f : 1.12f;
                float source_u = 0.5f + (local_x - 0.5f) * footprint_scale * aspect;
                float source_v = 0.5f + (local_y - 0.5f) * footprint_scale / aspect;
                if (source_u < 0.0f || source_u > 1.0f ||
                    source_v < 0.0f || source_v > 1.0f)
                    return;
                float source_edge = std::min(std::min(source_u, 1.0f - source_u),
                    std::min(source_v, 1.0f - source_v));
                float edge = smoothstep01(source_edge / 0.055f);
                TerrainTexture const & volcano = terrain_textures[10];
                height = sample_normalized_field(volcano.height_pixels,
                    volcano.height_width, volcano.height_height,
                    volcano.height_minimum, volcano.height_maximum,
                    source_u, source_v);
                blend = sample_normalized_field(volcano.blend_pixels,
                    volcano.height_width, volcano.height_height,
                    volcano.blend_minimum, volcano.blend_maximum,
                    source_u, source_v) * edge;
            };
            auto sample_relief_chain = [&](float world_u, float world_v,
                                           float & height, float & blend,
                                           float & displacement,
                                           bool include_volcano) {
                height = 0.0f;
                blend = 0.0f;
                displacement = 0.0f;
                int center_u = static_cast<int>(std::floor(world_u));
                int center_v = static_cast<int>(std::floor(world_v));
                constexpr int candidates[5][2] = {
                    {0, 0}, {-1, 0}, {1, 0}, {0, -1}, {0, 1}};
                for (auto const & offset : candidates) {
                    int candidate_u = center_u + offset[0];
                    int candidate_v = center_v + offset[1];
                    int candidate_relief = relief_at_lattice(candidate_u, candidate_v);
                    if ((candidate_relief != 6 && candidate_relief != 10) ||
                        (!include_volcano && candidate_relief == 10))
                        continue;
                    float local_x = world_u - static_cast<float>(candidate_u);
                    float local_y = 1.0f - (world_v - static_cast<float>(candidate_v));
                    float candidate_height = 0.0f;
                    float candidate_blend = 0.0f;
                    if (candidate_relief == 6)
                        sample_mountain(candidate_u, candidate_v, local_x, local_y,
                                        candidate_height, candidate_blend);
                    else
                        sample_volcano(candidate_u, candidate_v, local_x, local_y,
                                       candidate_height, candidate_blend);
                    int source_x = canonical_component(candidate_u + candidate_v,
                        frame.world_width_tiles, frame.world_wrap_x);
                    int source_y = canonical_component(candidate_u - candidate_v,
                        frame.world_height_tiles, frame.world_wrap_y);
                    unsigned vertical_seed = static_cast<unsigned>(source_x) * 73856093u ^
                        static_cast<unsigned>(source_y) * 19349663u;
                    float candidate_displacement = candidate_height *
                        smoothstep01(candidate_blend / 0.34f) *
                        (candidate_relief == 6 ? 104.0f :
                         (((vertical_seed >> 3) & 1u) != 0 ? 104.0f : 88.0f));
                    if (candidate_displacement > displacement) {
                        height = candidate_height;
                        blend = candidate_blend;
                        displacement = candidate_displacement;
                    }
                }
            };
            auto relief_at_world = [&](float world_u, float world_v) {
                // This is the approved Lab BIQ hill path, copied literally
                // into production-space accessors. Base-material height maps
                // are intentionally not evaluated here: the Lab BIQ renderer
                // keeps ordinary ground geometrically flat and only raises
                // authored hills, mountain/volcano chains, and dunes.
                int owner_u = static_cast<int>(std::floor(world_u));
                int owner_v = static_cast<int>(std::floor(world_v));
                auto owner_found = tile_by_coordinate.find(coordinate_key(
                    owner_u + owner_v, owner_u - owner_v));
                c3x_renderer_tile_v1 const & height_tile =
                    owner_found == tile_by_coordinate.end() ? tile : *owner_found->second;
                TerrainTexture const & hill_asset = terrain_textures[5];
                auto hill_value = [&](float sample_u, float sample_v) {
                    if (hill_asset.relief_profile != 2 || hill_asset.height_pixels.empty())
                        return 0.0f;
                    constexpr float radius = 0.018f;
                    float source_u = 0.11f + sample_u * 0.035f;
                    float source_v = 0.17f + sample_v * 0.035f;
                    auto sample = [&](float u, float v) {
                        return sample_normalized_field(hill_asset.height_pixels,
                            hill_asset.height_width, hill_asset.height_height,
                            hill_asset.height_minimum, hill_asset.height_maximum,
                            u, v);
                    };
                    float center = sample(source_u, source_v) * 4.0f;
                    float cardinal = sample(source_u - radius, source_v) +
                        sample(source_u + radius, source_v) +
                        sample(source_u, source_v - radius) +
                        sample(source_u, source_v + radius);
                    float diagonal = sample(source_u - radius, source_v - radius) +
                        sample(source_u + radius, source_v - radius) +
                        sample(source_u - radius, source_v + radius) +
                        sample(source_u + radius, source_v + radius);
                    float authored_macro =
                        (center + cardinal * 2.0f + diagonal) / 16.0f;
                    return smoothstep01((authored_macro - 0.22f) / 0.38f);
                };
                auto hill_support = [&]() {
                    int center_u = static_cast<int>(std::floor(world_u));
                    int center_v = static_cast<int>(std::floor(world_v));
                    float support = 0.0f;
                    for (int row = center_v - 1; row <= center_v + 1; ++row) {
                        for (int column = center_u - 1; column <= center_u + 1; ++column) {
                            if (relief_at_lattice(column, row) != 5)
                                continue;
                            float dx = (world_u - (static_cast<float>(column) + 0.5f)) / 0.92f;
                            float dy = (world_v - (static_cast<float>(row) + 0.5f)) / 0.78f;
                            float distance = std::sqrt(dx * dx + dy * dy);
                            support = std::max(support,
                                smoothstep01((1.0f - distance) / 0.42f));
                        }
                    }
                    return support;
                };
                float tile_world_u =
                    static_cast<float>(height_tile.tile_x + height_tile.tile_y) * 0.5f;
                float tile_world_v =
                    static_cast<float>(height_tile.tile_x - height_tile.tile_y) * 0.5f;
                float local_u = world_u - tile_world_u;
                float local_v = 1.0f - (world_v - tile_world_v);
                constexpr int source_offsets[4][2] = {
                    {-1, -1}, {1, -1}, {1, 1}, {-1, 1}};
                float distances[4] = {
                    local_u, local_v, 1.0f - local_u, 1.0f - local_v};
                float hill_compatibility = 1.0f;
                for (int edge = 0; edge < 4; ++edge) {
                    auto neighbor = ground_by_coordinate.find(coordinate_key(
                        height_tile.tile_x + source_offsets[edge][0],
                        height_tile.tile_y + source_offsets[edge][1]));
                    bool compatible = neighbor != ground_by_coordinate.end() &&
                        neighbor->second >= 0 && neighbor->second < 11;
                    if (!compatible)
                        hill_compatibility *= smoothstep01(distances[edge] / 0.22f);
                }
                float height = hill_value(world_u, world_v) * 52.0f *
                    hill_support() * hill_compatibility;
                float authored_height = 0.0f;
                float authored_blend = 0.0f;
                int current_real = height_tile.real_terrain_type;
                if (current_real == 5 || current_real == 6 || current_real == 10) {
                    float relief_envelope = 1.0f;
                    if (current_real == 6 || current_real == 10) {
                        for (int edge = 0; edge < 4; ++edge) {
                            auto neighbor = relief_by_coordinate.find(coordinate_key(
                                height_tile.tile_x + source_offsets[edge][0],
                                height_tile.tile_y + source_offsets[edge][1]));
                            int neighbor_real = neighbor == relief_by_coordinate.end()
                                ? -1 : neighbor->second;
                            bool continues = neighbor_real == 5 || neighbor_real == 6 ||
                                neighbor_real == 10;
                            if (!continues)
                                relief_envelope *= smoothstep01(distances[edge] / 0.28f);
                        }
                    }
                    float transition_envelope = 1.0f;
                    if (current_real == 5) {
                        transition_envelope = 0.0f;
                        for (int edge = 0; edge < 4; ++edge) {
                            auto neighbor = relief_by_coordinate.find(coordinate_key(
                                height_tile.tile_x + source_offsets[edge][0],
                                height_tile.tile_y + source_offsets[edge][1]));
                            if (neighbor != relief_by_coordinate.end() &&
                                neighbor->second == 6)
                                transition_envelope = std::max(transition_envelope,
                                    1.0f - smoothstep01(distances[edge] / 0.24f));
                        }
                    }
                    float chain_height = 0.0f;
                    float chain_blend = 0.0f;
                    float chain_displacement = 0.0f;
                    sample_relief_chain(world_u, world_v, chain_height, chain_blend,
                                        chain_displacement, current_real != 5);
                    float mountain_displacement = chain_displacement *
                        relief_envelope * transition_envelope;
                    if (height <= 0.001f || mountain_displacement <= 0.001f)
                        height = std::max(height, mountain_displacement);
                    else {
                        constexpr float blend_width = 12.0f;
                        float weight = std::clamp(0.5f + 0.5f *
                            (mountain_displacement - height) / blend_width, 0.0f, 1.0f);
                        height = height * (1.0f - weight) +
                            mountain_displacement * weight +
                            blend_width * weight * (1.0f - weight);
                    }
                    float material_envelope = 0.0f;
                    if (current_real == 5)
                        material_envelope = transition_envelope * 0.35f;
                    else {
                        material_envelope = 1.0f;
                        for (int edge = 0; edge < 4; ++edge) {
                            auto neighbor = relief_by_coordinate.find(coordinate_key(
                                height_tile.tile_x + source_offsets[edge][0],
                                height_tile.tile_y + source_offsets[edge][1]));
                            int neighbor_real = neighbor == relief_by_coordinate.end()
                                ? -1 : neighbor->second;
                            if (neighbor_real != 6 && neighbor_real != 10)
                                material_envelope *= smoothstep01(
                                    distances[edge] / 0.28f);
                        }
                    }
                    authored_height = chain_height;
                    authored_blend = chain_blend * material_envelope;
                }
                bool height_tile_has_dunes = dune_assets_ready &&
                    height_tile.real_terrain_type == 0 && height_tile.terrain_type == 0;
                if (height_tile_has_dunes) {
                    float dune_envelope = 1.0f;
                    for (int edge = 0; edge < 4; ++edge) {
                        std::uint64_t key = coordinate_key(
                            height_tile.tile_x + source_offsets[edge][0],
                            height_tile.tile_y + source_offsets[edge][1]);
                        auto neighbor_ground = ground_by_coordinate.find(key);
                        auto neighbor_real = real_by_coordinate.find(key);
                        bool continues = neighbor_ground != ground_by_coordinate.end() &&
                            neighbor_real != real_by_coordinate.end() &&
                            neighbor_ground->second == 0 && neighbor_real->second == 0;
                        if (!continues)
                            dune_envelope *= smoothstep01(distances[edge] / 0.16f);
                    }
                    height += c3x_renderer::dune_height(
                        world_u, world_v, 1.0f) * dune_envelope;
                }
                if (river_assets_ready && height_tile.terrain_type < 11 &&
                    (height_tile.river_code & 170u) != 0) {
                    float local_height_u = world_u - tile_world_u;
                    float local_height_v = 1.0f - (world_v - tile_world_v);
                    float distance = river_distance(
                        height_tile, local_height_u, local_height_v);
                    float valley = 1.0f - smoothstep01((distance - 4.0f) / 16.0f);
                    float valley_floor = height * 0.10f;
                    height = height * (1.0f - valley * 0.92f) +
                             valley_floor * valley * 0.92f;
                }
                return std::array<float, 3>{height, authored_height, authored_blend};
            };
            auto cast_shadow_visibility = [&](float world_u, float world_v,
                                              float origin_height) {
                if (reuse_shadow_field &&
                    shadow_visibility_index < shadow_visibility_cache.size())
                    return shadow_visibility_cache[shadow_visibility_index++];
                float horizontal = std::sqrt(key_light[0] * key_light[0] +
                                             key_light[1] * key_light[1]);
                if (horizontal < 0.001f) {
                    next_shadow_visibility.push_back(1.0f);
                    return 1.0f;
                }
                float direction_u = key_light[0] / horizontal;
                float direction_v = -key_light[1] / horizontal;
                float perpendicular_u = -direction_v;
                float perpendicular_v = direction_u;
                float occlusion = 0.0f;
                for (int lane = -1; lane <= 1; ++lane) {
                    float greatest_obstruction = 0.0f;
                    float lane_offset = static_cast<float>(lane) * 0.075f;
                    for (int step = 1; step <= 48; ++step) {
                        float distance = static_cast<float>(step) * 0.12f;
                        float sample_u = world_u + direction_u * distance +
                                         perpendicular_u * lane_offset;
                        float sample_v = world_v + direction_v * distance +
                                         perpendicular_v * lane_offset;
                        float sample_height = relief_at_world(sample_u, sample_v)[0];
                        float ray_height = origin_height + 96.0f * distance + 0.8f;
                        greatest_obstruction = std::max(
                            greatest_obstruction, sample_height - ray_height);
                    }
                    occlusion += std::clamp(
                        (greatest_obstruction - 0.5f) / 12.0f, 0.0f, 0.78f);
                }
                float visibility = 1.0f - occlusion / 3.0f;
                next_shadow_visibility.push_back(visibility);
                return visibility;
            };
            struct GroundPoint {
                float u, v;
                float world_u, world_v;
                float ground_x, ground_y;
                float material_u, material_v;
                float material_weights[4];
                float signed_shore;
                float surface_coordinate;
            };
            std::unordered_map<std::uint64_t, GroundPoint> ground_point_cache;
            ground_point_cache.reserve(2048);
            auto ground_point_at = [&](float u, float v) -> GroundPoint const & {
                std::uint32_t u_bits = 0, v_bits = 0;
                std::memcpy(&u_bits, &u, sizeof(u_bits));
                std::memcpy(&v_bits, &v, sizeof(v_bits));
                std::uint64_t key = (static_cast<std::uint64_t>(u_bits) << 32) | v_bits;
                auto found = ground_point_cache.find(key);
                if (found != ground_point_cache.end())
                    return found->second;
                GroundPoint point = {};
                point.u = u;
                point.v = v;
                point.world_u =
                    (static_cast<float>(tile.tile_x + tile.tile_y) * 0.5f) + u;
                point.world_v =
                    (static_cast<float>(tile.tile_x - tile.tile_y) * 0.5f) + (1.0f - v);
                point.ground_x = left + half_w + (u - v) * half_w;
                point.ground_y = top + (u + v) * half_h;
                std::array<float, 2> material_uv =
                    periodic_surface_uv(point.world_u, point.world_v, uv_scale);
                point.material_u = material_uv[0];
                point.material_v = material_uv[1];
                std::array<float, 4> weights =
                    material_weights_for(point.world_u, point.world_v);
                std::copy(weights.begin(), weights.end(), point.material_weights);
                point.signed_shore = signed_shore_distance(
                    point.world_u, point.world_v, u, v);
                point.surface_coordinate = point.signed_shore <= 0.0f
                    ? point.signed_shore
                    : std::sqrt(smoothstep01(point.signed_shore)) *
                        water_family_depth(point.world_u, point.world_v);
                return ground_point_cache.emplace(key, point).first->second;
            };
            auto make_ground_vertex = [&](float u, float v, float layer) {
                GroundPoint const & point = ground_point_at(u, v);
                float world_u = point.world_u;
                float world_v = point.world_v;
                bool land_surface = layer > 0.75f && layer < 1.25f;
                bool terrain_conforming_surface = land_surface ||
                    (layer > 8.5f && layer < 10.5f);
                std::array<float, 3> relief_sample = terrain_conforming_surface
                    ? relief_at_world(world_u, world_v) :
                      std::array<float, 3>{0.0f, 0.0f, 0.0f};
                float h = relief_sample[0] * relief_projection_scale;
                float signed_shore = point.signed_shore;
                if (land_surface && h > 0.0f) {
                    float shore_envelope = smoothstep01((-signed_shore - 0.02f) / 0.42f);
                    h *= shore_envelope;
                    relief_sample[1] *= shore_envelope;
                    relief_sample[2] *= shore_envelope;
                }
                float ground_x = point.ground_x;
                float ground_y = point.ground_y;
                // Elevation moves toward the isometric camera as well as up on
                // screen.  Keeping flat-ground depth made steep micro-quads
                // fold over one another and appear as bright contour seams.
                float depth = std::clamp(1.0f - ground_y / static_cast<float>(frame.target_height) -
                    h * 0.75f / static_cast<float>(frame.target_height), 0.001f, 0.999f);
                float normal_x = 0.0f, normal_y = 0.0f, normal_z = 1.0f;
                if (terrain_conforming_surface) {
                    constexpr float normal_step = 0.006f;
                    float left_height = relief_at_world(world_u - normal_step, world_v)[0];
                    float right_height = relief_at_world(world_u + normal_step, world_v)[0];
                    float down_height = relief_at_world(world_u, world_v - normal_step)[0];
                    float up_height = relief_at_world(world_u, world_v + normal_step)[0];
                    float slope_u = (right_height - left_height) * relief_projection_scale /
                        (2.0f * normal_step * static_cast<float>(frame.tile_width));
                    float slope_v = (up_height - down_height) * relief_projection_scale /
                        (2.0f * normal_step * static_cast<float>(frame.tile_width));
                    float length = std::sqrt(slope_u * slope_u + slope_v * slope_v + 1.0f);
                    normal_x = -slope_u / length;
                    normal_y = -slope_v / length;
                    normal_z = 1.0f / length;
                }
                float surface_coordinate = point.surface_coordinate;
                float shadow_visibility = layer > 9.5f
                    ? cast_shadow_visibility(world_u, world_v, relief_sample[0]) : 1.0f;
                // River topology is consumed only by the river surface pass.
                // Computing its curved-edge and global node distances for the
                // four terrain passes and the shadow pass was pure discarded
                // work, and scaled especially badly with a full Civ III view.
                bool river_surface = layer > 8.5f && layer < 9.5f;
                float river_surface_distance = river_surface
                    ? river_distance(tile, u, v) : 1000.0f;
                if (layer > 8.5f && layer < 9.5f)
                    depth = std::max(0.001f, depth - 0.025f);
                else if (layer > 9.5f)
                    depth = std::max(0.001f, depth - 0.003f);
                return Vertex{
                    ndc_x(ground_x), ndc_y(ground_y - h), depth,
                    point.material_u, point.material_v,
                    1.0f, normal_x, normal_y, normal_z,
                    shadow_visibility, 1.0f, world_u * 0.5f, world_v * 0.5f,
                    layer, surface_coordinate,
                    static_cast<float>(tile.terrain_type),
                    static_cast<float>(tile.real_terrain_type),
                    point.material_weights[0], point.material_weights[1],
                    point.material_weights[2], point.material_weights[3],
                    relief_sample[1], relief_sample[2], signed_shore,
                    river_surface_distance,
                    river_surface ? river_node_distance(u, v, 1u) : 1000.0f,
                    river_surface ? river_node_distance(u, v, 2u) : 1000.0f,
                    river_surface ? river_node_distance(u, v, 0u) : 1000.0f,
                    tile.has_effect != 0 ? 1.0f : 0.0f
                };
            };
            auto append_ground_layer = [&](std::vector<Vertex> & target, float layer,
                                           int subdivisions) {
                // Adjacent cells share grid corners. Build and upload each
                // corner once, then preserve the original triangle-list order.
                // This avoids repeated terrain/shadow evaluation for a point.
                int const row_width = subdivisions + 1;
                std::vector<Vertex> grid_vertices(
                    static_cast<std::size_t>(row_width) * row_width);
                for (int grid_v = 0; grid_v <= subdivisions; ++grid_v) {
                    for (int grid_u = 0; grid_u <= subdivisions; ++grid_u) {
                        float u = static_cast<float>(grid_u) / subdivisions;
                        float v = static_cast<float>(grid_v) / subdivisions;
                        grid_vertices[static_cast<std::size_t>(grid_v) * row_width + grid_u] =
                            make_ground_vertex(u, v, layer);
                    }
                }
                for (int grid_v = 0; grid_v < subdivisions; ++grid_v) {
                    for (int grid_u = 0; grid_u < subdivisions; ++grid_u) {
                        auto vertex_at = [&](int x, int y) -> Vertex const & {
                            return grid_vertices[
                                static_cast<std::size_t>(y) * row_width + x];
                        };
                        Vertex const & a = vertex_at(grid_u, grid_v);
                        Vertex const & b0 = vertex_at(grid_u + 1, grid_v);
                        Vertex const & c = vertex_at(grid_u + 1, grid_v + 1);
                        Vertex const & d = vertex_at(grid_u, grid_v + 1);
                        Vertex triangles[] = {a, b0, c, a, c, d};
                        target.insert(target.end(), std::begin(triangles), std::end(triangles));
                    }
                }
            };
            auto append_feature_instance = [&](c3x_renderer::FeatureBundle const & bundle,
                                               c3x_renderer::FeaturePlacement const & placement,
                                               float local_u, float local_v, float rotation,
                                               float scale, float material_offset,
                                               float owner_code, bool cast_shadow,
                                               std::vector<Vertex> & target) {
                if (placement.asset_index >= bundle.assets.size())
                    return;
                c3x_renderer::FeatureAsset const & asset = bundle.assets[placement.asset_index];
                float tile_world_u = static_cast<float>(tile.tile_x + tile.tile_y) * 0.5f;
                float tile_world_v = static_cast<float>(tile.tile_x - tile.tile_y) * 0.5f;
                std::array<float, 3> ground_sample = relief_at_world(
                    tile_world_u + local_u, tile_world_v + (1.0f - local_v));
                float center_x = left + half_w + (local_u - local_v) * half_w;
                float center_y = top + (local_u + local_v) * half_h -
                    ground_sample[0] * relief_projection_scale;
                if (cast_shadow)
                    append_object_shadow(asset, scale, center_x, center_y,
                                         ground_sample[0] * relief_projection_scale);
                float cosine = std::cos(rotation);
                float sine = std::sin(rotation);
                std::vector<Vertex> transformed(asset.vertices.size());
                for (std::size_t vertex_index = 0; vertex_index < asset.vertices.size(); ++vertex_index) {
                    c3x_renderer::FeatureSourceVertex const & source = asset.vertices[vertex_index];
                    float local_x = (source.position[0] * cosine - source.position[1] * sine) * scale;
                    float local_y = (source.position[0] * sine + source.position[1] * cosine) * scale;
                    float local_z = source.position[2] * scale;
                    float screen_x = center_x + (local_x - local_y) * half_w;
                    float screen_y = center_y + (local_x + local_y) * half_h -
                        local_z * 150.0f * feature_projection_scale;
                    float normal_x = source.normal[0] * cosine - source.normal[1] * sine;
                    float normal_y = source.normal[0] * sine + source.normal[1] * cosine;
                    float ground_height_pixels = ground_sample[0] * relief_projection_scale;
                    float base_ground_y = center_y + ground_height_pixels +
                        (local_x + local_y) * half_h;
                    float feature_height_tiles = local_z * 150.0f *
                        feature_projection_scale / relief_projection_scale;
                    float depth = std::clamp(
                        1.0f - base_ground_y / static_cast<float>(frame.target_height) -
                        ground_height_pixels * 0.75f / static_cast<float>(frame.target_height) -
                        feature_height_tiles * 0.0012f, 0.001f, 0.999f);
                    transformed[vertex_index] = Vertex{
                        ndc_x(screen_x), ndc_y(screen_y), depth,
                        source.uv[0], source.uv[1], 1.0f,
                        normal_x, normal_y, source.normal[2],
                        1.0f, 1.0f, 0.0f, 0.0f,
                        0.0f, 0.0f,
                        static_cast<float>(asset.texture_index) + material_offset + owner_code,
                        0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                        0.0f, 0.0f, 1.0f,
                        1000.0f, 0.0f, 1000.0f, 0.0f, -1.0f};
                }
                for (std::uint32_t source_index : asset.indices)
                    target.push_back(transformed[source_index]);
            };
            auto append_route_segment = [&](float u0, float v0, float u1, float v1,
                                            unsigned style, bool railroad) {
                constexpr int subdivisions = 16;
                float route_half_width = railroad ? 0.076f : 0.105f;
                float atlas_half_width = railroad ? 0.058f : 0.075f;
                float du = u1 - u0, dv = v1 - v0;
                float original_length = std::sqrt(du * du + dv * dv);
                if (original_length < 0.001f)
                    return;
                float direction_u = du / original_length;
                float direction_v = dv / original_length;
                float original_u0 = u0, original_v0 = v0;
                float original_u1 = u1, original_v1 = v1;
                u0 -= direction_u * 0.14f; v0 -= direction_v * 0.14f;
                u1 += direction_u * 0.14f; v1 += direction_v * 0.14f;
                du = u1 - u0; dv = v1 - v0;
                float length = std::sqrt(du * du + dv * dv);
                float perpendicular_u = -dv / length;
                float perpendicular_v = du / length;
                float atlas_dx = 1.0f;
                float atlas_dy = 0.99021526f - 0.90606654f;
                float atlas_length = std::sqrt(atlas_dx * atlas_dx + atlas_dy * atlas_dy);
                float atlas_perpendicular_u = -atlas_dy / atlas_length;
                float atlas_perpendicular_v = atlas_dx / atlas_length;
                float wave_seed = std::fmod(std::fabs(
                    original_u0 * 17.0f + original_v0 * 31.0f +
                    original_u1 * 47.0f + original_v1 * 61.0f), 19.0f) / 19.0f;
                float wave_phase = wave_seed * 6.28318530718f;
                auto route_vertex = [&](float along, float across) {
                    float source_along = (along * length - 0.14f) / original_length;
                    float curve_t = std::clamp(source_along, 0.0f, 1.0f);
                    float curve_envelope = std::sin(curve_t * 3.14159265359f);
                    float curve_amplitude = railroad ? 0.028f : 0.042f;
                    float road_wave = curve_envelope * curve_amplitude *
                        (0.62f * std::sin(wave_phase) +
                         0.38f * std::sin(curve_t * 6.28318530718f + wave_phase));
                    float route_u = u0 + du * along + perpendicular_u *
                        (route_half_width * across + road_wave);
                    float route_v = v0 + dv * along + perpendicular_v *
                        (route_half_width * across + road_wave);
                    float atlas_u = atlas_dx * source_along +
                        atlas_perpendicular_u * atlas_half_width * across;
                    float atlas_v = 0.90606654f + atlas_dy * source_along +
                        atlas_perpendicular_v * atlas_half_width * across;
                    float tile_world_u = static_cast<float>(tile.tile_x + tile.tile_y) * 0.5f;
                    float tile_world_v = static_cast<float>(tile.tile_x - tile.tile_y) * 0.5f;
                    std::array<float, 3> ground_sample = relief_at_world(
                        tile_world_u + route_u, tile_world_v + (1.0f - route_v));
                    float ground_x = left + half_w + (route_u - route_v) * half_w;
                    float ground_y = top + (route_u + route_v) * half_h;
                    float h = ground_sample[0] * relief_projection_scale;
                    float depth = std::max(0.003f, std::clamp(
                        1.0f - ground_y / static_cast<float>(frame.target_height) -
                        h * 0.75f / static_cast<float>(frame.target_height),
                        0.001f, 0.999f) - 0.010f);
                    return Vertex{
                        ndc_x(ground_x), ndc_y(ground_y - h), depth,
                        atlas_u, atlas_v, 1.0f, 0.0f, 0.0f, 1.0f,
                        across, curve_t, source_along, 0.90606654f + atlas_dy * source_along,
                        11.0f, 0.0f, static_cast<float>(style), 0.0f,
                        route_u, route_v, 0.0f, 0.0f,
                        0.0f, 0.0f, 1.0f,
                        1000.0f, 0.0f, 1000.0f, 0.0f, -1.0f};
                };
                for (int segment = 0; segment < subdivisions; ++segment) {
                    float a0 = static_cast<float>(segment) / subdivisions;
                    float a1 = static_cast<float>(segment + 1) / subdivisions;
                    Vertex left0 = route_vertex(a0, -1.0f);
                    Vertex right0 = route_vertex(a0, 1.0f);
                    Vertex right1 = route_vertex(a1, 1.0f);
                    Vertex left1 = route_vertex(a1, -1.0f);
                    Vertex triangles[] = {left0, right0, right1, left0, right1, left1};
                    route_vertices.insert(route_vertices.end(), std::begin(triangles), std::end(triangles));
                }
            };
            // Match the standalone terrain stack exactly: a flat material
            // underlay, raised land, submerged bed, then transparent water.
            // Keeping these pass-major vectors prevents a later land tile
            // from overwriting an earlier neighbor's continuous shoreline.
            append_ground_layer(underlay_vertices, 0.5f, tile_ground_grid);
            if (ground < 11)
                append_ground_layer(land_vertices, 1.0f, tile_ground_grid);
            append_ground_layer(bed_vertices, 4.0f, tile_ground_grid);
            append_ground_layer(water_vertices, 5.0f, tile_ground_grid);
            if (river_assets_ready && (tile.river_code & 170u) != 0)
                append_ground_layer(river_vertices, 9.0f,
                                    frame.tile_width >= 96 ? 32 : 16);
            if (ground < 11) {
                // Cast-shadow visibility ray-marches the authored relief field.
                // Retain the approved 16x16 near grid for canonical fixtures,
                // but use the already-approved reduced grid when a live m19
                // capture contains hundreds of companion records.  The shader
                // interpolates visibility across the unchanged terrain body.
                int const shadow_grid = frame.tile_width >= 96 && frame.tile_count <= 512
                    ? 16 : 8;
                append_ground_layer(shadow_vertices, 10.0f,
                                    shadow_grid);
            }
            if (route_assets_ready && (tile.road_mask != 0 || tile.railroad_mask != 0)) {
                constexpr int route_offsets[4][2] = {
                    {1, -1}, {2, 0}, {1, 1}, {0, 2}
                };
                constexpr unsigned river_edge_bits[4] = {2u, 0u, 8u, 0u};
                constexpr unsigned opposite_river_bits[4] = {32u, 0u, 128u, 0u};
                float base_world_u = static_cast<float>(tile.tile_x + tile.tile_y) * 0.5f;
                float base_world_v = static_cast<float>(tile.tile_x - tile.tile_y) * 0.5f;
                for (int direction = 0; direction < 4; ++direction) {
                    int neighbor_x = tile.tile_x + route_offsets[direction][0];
                    int neighbor_y = tile.tile_y + route_offsets[direction][1];
                    auto found = tile_by_coordinate.find(coordinate_key(neighbor_x, neighbor_y));
                    if (found == tile_by_coordinate.end())
                        continue;
                    c3x_renderer_tile_v1 const & neighbor = *found->second;
                    bool railroad = tile.railroad_mask != 0 && neighbor.railroad_mask != 0;
                    bool road = tile.road_mask != 0 && neighbor.road_mask != 0;
                    if (!railroad && !road)
                        continue;
                    float end_u = (static_cast<float>(neighbor_x + neighbor_y) * 0.5f + 0.5f) -
                        base_world_u;
                    float end_v = 1.0f - ((static_cast<float>(neighbor_x - neighbor_y) * 0.5f + 0.5f) -
                        base_world_v);
                    unsigned style = railroad ? 4u : static_cast<unsigned>(
                        std::clamp(tile.route_style, 0, 3));
                    append_route_segment(0.5f, 0.5f, end_u, end_v, style, railroad);
                    bool bridge = river_edge_bits[direction] != 0 &&
                        (((tile.river_code & river_edge_bits[direction]) != 0) ||
                         ((neighbor.river_code & opposite_river_bits[direction]) != 0));
                    if (bridge) {
                        char const * bridge_style = railroad ? "railroad" :
                            (style >= 3u ? "modern" : (style >= 2u ? "industrial" : "medieval"));
                        std::string group_name = std::string("bridge_") + bridge_style + "_normal";
                        c3x_renderer::FeatureGroup const * bridge_group =
                            c3x_renderer::find_feature_group(bridge_bundle, group_name.c_str());
                        if (bridge_group != nullptr && !bridge_group->placements.empty()) {
                            float rotation = std::atan2(end_v - 0.5f, end_u - 0.5f);
                            c3x_renderer::FeaturePlacement const & placement =
                                bridge_group->placements.front();
                            append_feature_instance(bridge_bundle, placement,
                                (0.5f + end_u) * 0.5f, (0.5f + end_v) * 0.5f,
                                rotation, placement.scale, 13.0f, 0.0f, true,
                                feature_vertices);
                        }
                    }
                }
            }
            if (feature_assets_ready &&
                (tile.real_terrain_type == 7 || tile.real_terrain_type == 8)) {
                char const * group_name = tile.real_terrain_type == 7 ? "forest" : "jungle";
                c3x_renderer::FeatureGroup const * group =
                    tile.real_terrain_type == 7 ? forest_group :
                    c3x_renderer::find_feature_group(feature_bundle, group_name);
                // Forest uses a 6x6 canopy body and jungle a 7x7 body. Stable
                // grid jitter prevents the random
                // interior holes visible in the earlier in-game port.
                unsigned instance_count = tile.real_terrain_type == 7 ? 36u : 49u;
                char const * forest_anchors[] = {"pine_01", "pine_clump_01", "shrub_01"};
                char const * jungle_anchors[] = {
                    "grass_04", "palm_01", "palm_02", "plant_01", "plant_02", "plant_03"};
                char const * const * anchors = tile.real_terrain_type == 7 ?
                    forest_anchors : jungle_anchors;
                unsigned anchor_count = tile.real_terrain_type == 7 ? 3u : 6u;
                float tile_world_u = static_cast<float>(tile.tile_x + tile.tile_y) * 0.5f;
                float tile_world_v = static_cast<float>(tile.tile_x - tile.tile_y) * 0.5f;
                int canonical_feature_x = canonical_component(
                    tile.tile_x, frame.world_width_tiles, frame.world_wrap_x);
                int canonical_feature_y = canonical_component(
                    tile.tile_y, frame.world_height_tiles, frame.world_wrap_y);
                std::uint32_t feature_seed =
                    static_cast<std::uint32_t>(canonical_feature_x * 0x193) ^
                    static_cast<std::uint32_t>(canonical_feature_y * 0x217);
                if (group != nullptr) {
                    for (unsigned instance = 0; instance < instance_count; ++instance) {
                        c3x_renderer::FeaturePlacement const * placement =
                            instance < anchor_count ?
                            c3x_renderer::find_feature_placement_by_suffix(
                                feature_bundle, *group, anchors[instance]) :
                            c3x_renderer::select_feature_placement(
                                *group, feature_seed + instance * 31u);
                        if (placement == nullptr || placement->asset_index >= feature_bundle.assets.size())
                            continue;
                        c3x_renderer::FeatureAsset const & asset =
                            feature_bundle.assets[placement->asset_index];
                        unsigned grid_side = tile.real_terrain_type == 7 ? 6u : 7u;
                        unsigned column = instance % grid_side;
                        unsigned row = instance / grid_side;
                        float jitter_u = c3x_renderer::stable_random(
                            feature_seed + instance * 103u + 59u) - 0.5f;
                        float jitter_v = c3x_renderer::stable_random(
                            feature_seed + instance * 107u + 61u) - 0.5f;
                        float u_t = (static_cast<float>(column) + 0.5f + jitter_u * 0.68f) /
                            static_cast<float>(grid_side);
                        float v_t = (static_cast<float>(row) + 0.5f + jitter_v * 0.68f) /
                            static_cast<float>(grid_side);
                        float u = 0.07f + 0.86f * u_t;
                        float v = 0.07f + 0.86f * v_t;
                        float scale_variation =
                            (c3x_renderer::stable_random(feature_seed + instance * 71u + 23u) * 2.0f - 1.0f) *
                            placement->scale_variation;
                        float scene_feature_scale = tile.real_terrain_type == 7 ? 0.42f : 0.40f;
                        float scale = placement->scale * (1.0f + scale_variation) *
                            scene_feature_scale;
                        float rotation = c3x_renderer::stable_random(
                            feature_seed + instance * 97u + 47u) * 6.28318530718f;
                        float cosine = std::cos(rotation);
                        float sine = std::sin(rotation);
                        std::array<float, 3> ground_sample = relief_at_world(
                            tile_world_u + u, tile_world_v + (1.0f - v));
                        float center_x = left + half_w + (u - v) * half_w;
                        float center_y = top + (u + v) * half_h -
                            ground_sample[0] * relief_projection_scale;
                        append_object_shadow(asset, scale, center_x, center_y,
                            ground_sample[0] * relief_projection_scale);
                        std::vector<Vertex> transformed(asset.vertices.size());
                        for (std::size_t vertex_index = 0; vertex_index < asset.vertices.size(); ++vertex_index) {
                            c3x_renderer::FeatureSourceVertex const & source = asset.vertices[vertex_index];
                            float local_x = (source.position[0] * cosine - source.position[1] * sine) * scale;
                            float local_y = (source.position[0] * sine + source.position[1] * cosine) * scale;
                            float local_z = source.position[2] * scale;
                            float screen_x = center_x + (local_x - local_y) * half_w;
                            float screen_y = center_y + (local_x + local_y) * half_h -
                                local_z * 150.0f * feature_projection_scale;
                            float normal_x = source.normal[0] * cosine - source.normal[1] * sine;
                            float normal_y = source.normal[0] * sine + source.normal[1] * cosine;
                            // Match Lab's BIQ feature depth: position the body
                            // from its ground-plane Y, then pull authored height
                            // toward the camera separately. Using lifted screen_y
                            // here pushed tall trees behind neighboring ground and
                            // clipped their meshes at exact tile diamonds.
                            float ground_height_pixels =
                                ground_sample[0] * relief_projection_scale;
                            float base_ground_y = center_y + ground_height_pixels +
                                (local_x + local_y) * half_h;
                            float feature_height_tiles = local_z * 150.0f *
                                feature_projection_scale / relief_projection_scale;
                            float depth = std::clamp(
                                1.0f - base_ground_y / static_cast<float>(frame.target_height) -
                                ground_height_pixels * 0.75f /
                                    static_cast<float>(frame.target_height) -
                                feature_height_tiles * 0.0012f,
                                0.001f, 0.999f);
                            transformed[vertex_index] = Vertex{
                                ndc_x(screen_x), ndc_y(screen_y), depth,
                                source.uv[0], source.uv[1], 1.0f,
                                normal_x, normal_y, source.normal[2],
                                1.0f, 1.0f, 0.0f, 0.0f,
                                0.0f, 0.0f,
                                static_cast<float>(asset.texture_index), 0.0f,
                                0.0f, 0.0f, 0.0f, 0.0f,
                                0.0f, 0.0f, 1.0f,
                                1000.0f, 0.0f, 1000.0f, 0.0f, -1.0f
                            };
                        }
                        for (std::uint32_t source_index : asset.indices)
                            feature_vertices.push_back(transformed[source_index]);
                    }
                }
            }
            if (river_rock_group != nullptr && !river_rock_group->placements.empty()) {
                struct RiverRockEdge {
                    unsigned bit;
                    int neighbor_x;
                    int neighbor_y;
                    bool north_edge;
                };
                RiverRockEdge edges[] = {
                    {2u, tile.tile_x + 1, tile.tile_y - 1, true},
                    {8u, tile.tile_x + 1, tile.tile_y + 1, false},
                };
                for (RiverRockEdge const & edge : edges) {
                    if ((tile.river_code & edge.bit) == 0)
                        continue;
                    std::uint32_t seed = stable_feature_hash(
                        static_cast<std::uint32_t>(canonical_component(
                            tile.tile_x, frame.world_width_tiles, frame.world_wrap_x) + 4096) * 0x193u ^
                        static_cast<std::uint32_t>(canonical_component(
                            tile.tile_y, frame.world_height_tiles, frame.world_wrap_y) + 4096) * 0x217u ^
                        edge.bit);
                    if ((seed % 3u) != 0u)
                        continue;
                    c3x_renderer::FeaturePlacement const & placement =
                        river_rock_group->placements[(seed >> 5) %
                            river_rock_group->placements.size()];
                    if (placement.asset_index >= river_rock_bundle.assets.size())
                        continue;
                    float along = 0.28f + c3x_renderer::stable_random(seed ^ 0x73a52u) * 0.44f;
                    c3x_renderer_tile_v1 const * owner = &tile;
                    float local_u = edge.north_edge ? along : 0.975f;
                    float local_v = edge.north_edge ? 0.025f : along;
                    auto neighbor = tile_by_coordinate.find(
                        coordinate_key(edge.neighbor_x, edge.neighbor_y));
                    if (ground_type(*owner) >= 11 && neighbor != tile_by_coordinate.end() &&
                        ground_type(*neighbor->second) < 11) {
                        owner = neighbor->second;
                        local_u = edge.north_edge ? along : 0.025f;
                        local_v = edge.north_edge ? 0.975f : along;
                    }
                    if (ground_type(*owner) >= 11)
                        continue;
                    c3x_renderer::FeatureAsset const & asset =
                        river_rock_bundle.assets[placement.asset_index];
                    float scale = 0.155f +
                        c3x_renderer::stable_random(seed ^ 0x91c37u) * 0.070f;
                    float rotation = c3x_renderer::stable_random(seed ^ 0x4ad91u) *
                        6.28318530718f;
                    float cosine = std::cos(rotation);
                    float sine = std::sin(rotation);
                    float owner_world_u =
                        static_cast<float>(owner->tile_x + owner->tile_y) * 0.5f;
                    float owner_world_v =
                        static_cast<float>(owner->tile_x - owner->tile_y) * 0.5f;
                    std::array<float, 3> ground_sample = relief_at_world(
                        owner_world_u + local_u, owner_world_v + (1.0f - local_v));
                    float center_x = static_cast<float>(owner->anchor_x) + half_w +
                        (local_u - local_v) * half_w;
                    float center_y = static_cast<float>(owner->anchor_y) +
                        (local_u + local_v) * half_h -
                        ground_sample[0] * relief_projection_scale;
                    append_object_shadow(asset, scale, center_x, center_y,
                        ground_sample[0] * relief_projection_scale);
                    std::vector<Vertex> transformed(asset.vertices.size());
                    for (std::size_t vertex_index = 0;
                         vertex_index < asset.vertices.size(); ++vertex_index) {
                        c3x_renderer::FeatureSourceVertex const & source =
                            asset.vertices[vertex_index];
                        float local_x = (source.position[0] * cosine -
                                         source.position[1] * sine) * scale;
                        float local_y = (source.position[0] * sine +
                                         source.position[1] * cosine) * scale;
                        float local_z = source.position[2] * scale;
                        float screen_x = center_x + (local_x - local_y) * half_w;
                        float screen_y = center_y + (local_x + local_y) * half_h -
                            local_z * 150.0f * feature_projection_scale;
                        float normal_x = source.normal[0] * cosine - source.normal[1] * sine;
                        float normal_y = source.normal[0] * sine + source.normal[1] * cosine;
                        float ground_height_pixels =
                            ground_sample[0] * relief_projection_scale;
                        float base_ground_y = center_y + ground_height_pixels +
                            (local_x + local_y) * half_h;
                        float feature_height_tiles = local_z * 150.0f *
                            feature_projection_scale / relief_projection_scale;
                        float depth = std::clamp(
                            1.0f - base_ground_y / static_cast<float>(frame.target_height) -
                            ground_height_pixels * 0.75f /
                                static_cast<float>(frame.target_height) -
                            feature_height_tiles * 0.0012f,
                            0.001f, 0.999f);
                        transformed[vertex_index] = Vertex{
                            ndc_x(screen_x), ndc_y(screen_y), depth,
                            source.uv[0], source.uv[1], 1.0f,
                            normal_x, normal_y, source.normal[2],
                            1.0f, 1.0f, 0.0f, 0.0f,
                            0.0f, 0.0f,
                            static_cast<float>(asset.texture_index + 8u), 0.0f,
                            0.0f, 0.0f, 0.0f, 0.0f,
                            0.0f, 0.0f, 1.0f,
                            1000.0f, 0.0f, 1000.0f, 0.0f, -1.0f};
                    }
                    for (std::uint32_t source_index : asset.indices)
                        feature_vertices.push_back(transformed[source_index]);
                }
            }
            if (resource_assets_ready && tile.resource_id >= 0) {
                std::string resource_name = tile.resource_name;
                std::transform(resource_name.begin(), resource_name.end(), resource_name.begin(),
                    [](unsigned char value) { return static_cast<char>(std::tolower(value)); });
                char const * group_name = nullptr;
                for (char const * candidate : {"horses", "iron", "uranium", "gold",
                                               "dye", "wheat", "cattle", "fish"}) {
                    if (resource_name.find(candidate) != std::string::npos) {
                        group_name = candidate;
                        break;
                    }
                }
                c3x_renderer::FeatureGroup const * group = group_name == nullptr ? nullptr :
                    c3x_renderer::find_feature_group(resource_bundle, group_name);
                if (group != nullptr && !group->placements.empty()) {
                    replacement_tile_flags[index] |= C3X_RENDERER_TILE_CUSTOM_RESOURCE_REPLACED;
                    c3x_renderer::FeaturePlacement const & placement = group->placements.front();
                    unsigned count = std::max(1u, placement.count);
                    for (unsigned body = 0; body < count; ++body) {
                        float angle = 6.28318530718f *
                            (static_cast<float>(body) / static_cast<float>(count) +
                             c3x_renderer::stable_random(tile.variant_seed * 101u + body * 37u) * 0.11f);
                        float ring = count == 1u ? 0.0f :
                            (body == 0u ? 0.045f : 0.10f + 0.055f * static_cast<float>((body - 1u) % 3u));
                        float variation =
                            (c3x_renderer::stable_random(tile.variant_seed * 59u + body * 71u + 13u) *
                             2.0f - 1.0f) * placement.scale_variation;
                        float scale = placement.scale * (1.0f + variation) * 0.78f;
                        float rotation = c3x_renderer::stable_random(
                            tile.variant_seed * 83u + body * 97u + 29u) * 6.28318530718f;
                        bool fish = std::strcmp(group_name, "fish") == 0;
                        append_feature_instance(resource_bundle, placement,
                            0.5f + std::cos(angle) * ring,
                            0.5f + std::sin(angle) * ring * 0.78f,
                            rotation, scale, 21.0f, 0.0f, !fish, feature_vertices);
                    }
                }
            }
            if (mine_assets_ready && ground < 11 &&
                (tile.improvement_flags & C3X_RENDERER_IMPROVEMENT_MINE) != 0) {
                unsigned era = static_cast<unsigned>(std::clamp(tile.route_style, 0, 3));
                unsigned family = era < 2u ? 0u : 1u;
                unsigned variant = tile.variant_seed % 3u;
                std::string group_name = "mine_" + std::to_string(family * 3u + variant);
                c3x_renderer::FeatureGroup const * group =
                    c3x_renderer::find_feature_group(mine_bundle, group_name.c_str());
                if (group != nullptr && !group->placements.empty()) {
                    float rotation = c3x_renderer::stable_random(
                        static_cast<std::uint32_t>(tile.tile_x * 71 + tile.tile_y * 113) +
                        era * 29u) * 0.48f - 0.24f;
                    for (std::size_t part = 0; part < group->placements.size(); ++part) {
                        c3x_renderer::FeaturePlacement const & placement =
                            group->placements[part];
                        if (placement.asset_index >= mine_bundle.assets.size())
                            continue;
                        c3x_renderer::FeatureAsset const & asset =
                            mine_bundle.assets[placement.asset_index];
                        unsigned emissive_code = 0u;
                        std::size_t marker = asset.id.rfind(":e");
                        if (marker != std::string::npos)
                            emissive_code = static_cast<unsigned>(std::strtoul(
                                asset.id.c_str() + marker + 2u, nullptr, 10));
                        append_feature_instance(mine_bundle, placement,
                            0.5f, 0.5f, rotation, placement.scale, 21.0f,
                            0.01f * static_cast<float>(emissive_code + 1u),
                            part == 0u, mine_vertices);
                    }
                }
            }
            if (city_assets_ready && tile.city_id >= 0 && ground < 11) {
                constexpr char const * era_names[] = {
                    "ancient", "medieval", "industrial", "modern"};
                constexpr char const * wall_names[] = {
                    "wall_ancient", "wall_medieval", "wall_industrial"};
                constexpr unsigned counts[] = {4u, 7u, 11u};
                constexpr float radii[] = {0.25f, 0.33f, 0.41f};
                constexpr float size_scales[] = {0.92f, 1.00f, 1.08f};
                constexpr float golden_angle = 2.39996322973f;
                unsigned era = static_cast<unsigned>(std::clamp(tile.city_era, 0, 3));
                unsigned size = static_cast<unsigned>(std::clamp(tile.city_size, 0, 2));
                unsigned culture = static_cast<unsigned>(std::max(0, tile.city_culture_group));
                unsigned owner = static_cast<unsigned>(std::max(0, tile.city_owner_id));
                c3x_renderer::FeatureGroup const * group =
                    c3x_renderer::find_feature_group(city_bundle, era_names[era]);
                if (group != nullptr && !group->placements.empty()) {
                    unsigned component_count = counts[size];
                    for (unsigned slot = 0; slot < component_count; ++slot) {
                        c3x_renderer::FeaturePlacement const & placement = group->placements[
                            (culture + tile.variant_seed + slot) % group->placements.size()];
                        float angle = static_cast<float>(slot) * golden_angle +
                            c3x_renderer::stable_random(tile.variant_seed * 53u + culture * 19u) * 0.72f;
                        float radius = slot == 0u ? 0.0f : radii[size] *
                            std::sqrt(static_cast<float>(slot) /
                                      static_cast<float>(component_count - 1u));
                        float scale = placement.scale * size_scales[size] *
                            (slot == 0u && (tile.city_flags & C3X_RENDERER_CITY_CAPITAL) != 0 ? 1.30f : 1.0f);
                        append_feature_instance(city_bundle, placement,
                            0.5f + std::cos(angle) * radius,
                            0.5f + std::sin(angle) * radius * 0.78f,
                            angle + 0.55f, scale, 29.0f,
                            0.08f * static_cast<float>(owner + 1u), true, city_vertices);
                    }
                }
                if ((tile.city_flags & C3X_RENDERER_CITY_WALLED) != 0) {
                    c3x_renderer::FeatureGroup const * walls = c3x_renderer::find_feature_group(
                        wall_bundle, wall_names[std::min(era, 2u)]);
                    if (walls != nullptr && !walls->placements.empty()) {
                        c3x_renderer::FeaturePlacement const & wall = walls->placements.front();
                        constexpr float offsets[4][3] = {
                            {-0.29f, 0.00f, 0.785398163f},
                            {0.29f, 0.00f, 0.785398163f},
                            {0.00f, -0.23f, -0.785398163f},
                            {0.00f, 0.23f, -0.785398163f},
                        };
                        for (auto const & offset : offsets)
                            append_feature_instance(wall_bundle, wall,
                                0.5f + offset[0], 0.5f + offset[1], offset[2],
                                wall.scale * (size == 0u ? 0.82f : 1.0f), 29.0f,
                                0.08f * static_cast<float>(owner + 1u), true, wall_vertices);
                    }
                }
            }
        }
        float clear[4] = {0, 0, 0, 0};
        context->ClearRenderTargetView(render_target, clear);
        context->ClearDepthStencilView(depth_target, D3D11_CLEAR_DEPTH | D3D11_CLEAR_STENCIL, 1.0f, 0);
        context->OMSetRenderTargets(1, &render_target, depth_target);
        context->OMSetDepthStencilState(depth_state, 0);
        float blend_factor[4] = {0, 0, 0, 0};
        context->OMSetBlendState(blend_state, blend_factor, 0xffffffffu);
        context->RSSetState(rasterizer_state);
        D3D11_VIEWPORT viewport = {0.0f, 0.0f, static_cast<float>(width), static_cast<float>(height), 0.0f, 1.0f};
        context->RSSetViewports(1, &viewport);
        // Always retain a complete terrain surface. The output clip is applied
        // only by c3x_renderer_blit when Civ III composites its dirty rectangle.
        D3D11_RECT scissor = {0, 0, width, height};
        context->RSSetScissorRects(1, &scissor);

        if (!underlay_vertices.empty() || !land_vertices.empty() ||
            !bed_vertices.empty() || !water_vertices.empty() ||
            !river_vertices.empty() || !route_vertices.empty() ||
            !shadow_vertices.empty() || !feature_vertices.empty() ||
            !city_vertices.empty() || !wall_vertices.empty() || !mine_vertices.empty()) {
            context->IASetInputLayout(input_layout);
            context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            context->VSSetShader(vertex_shader, nullptr, 0);
            context->PSSetShader(pixel_shader, nullptr, 0);
            // Register-for-register match with the frozen approved terrain
            // shader. No production-only palette remains in this binding path.
            std::array<ID3D11ShaderResourceView *, 128> views = {};
            TerrainTexture const & grass = terrain_textures[2];
            TerrainTexture const & plains = terrain_textures[1];
            TerrainTexture const & desert = terrain_textures[0];
            TerrainTexture const & hills = terrain_textures[5];
            TerrainTexture const & mountain = terrain_textures[6];
            TerrainTexture const & marsh = terrain_textures[9];
            TerrainTexture const & coast = terrain_textures[11];
            TerrainTexture const & ocean = terrain_textures[13];
            views[0] = grass.view; views[1] = grass.material_height_view; views[2] = grass.specular_view;
            views[3] = mountain.material_height_view; views[4] = mountain.material_height_view;
            views[5] = mountain.material_height_view;
            views[6] = mountain.view; views[7] = mountain.elevated_view;
            views[8] = mountain.relief_layer_views[0];
            views[9] = mountain.material_height_view; views[10] = mountain.specular_view;
            views[11] = coast.relief_layer_views[0]; views[12] = terrain_extra_views[0];
            views[13] = terrain_extra_views[1]; views[14] = coast.relief_layer_views[1];
            views[15] = terrain_extra_views[2]; views[16] = terrain_extra_views[3];
            views[17] = coast.view; views[18] = ocean.view;
            views[19] = coast.material_height_view;
            for (std::size_t index = 0; index < coast.water_surface_views.size(); ++index)
                views[20 + index] = coast.water_surface_views[index];
            for (std::size_t index = 0; index < 4; ++index)
                views[25 + index] = feature_texture_views[index];
            views[29] = coast.specular_view; views[30] = ocean.material_height_view;
            views[31] = ocean.specular_view;
            for (std::size_t index = 0; index < 13; ++index)
                views[32 + index] = terrain_extra_views[4 + index];
            views[45] = plains.view; views[46] = plains.material_height_view;
            views[47] = plains.specular_view;
            views[48] = desert.view; views[49] = desert.material_height_view;
            views[50] = desert.specular_view; views[51] = hills.material_height_view;
            views[52] = dune_surface.view; views[53] = dune_surface.material_height_view;
            views[54] = dune_surface.specular_view; views[55] = dune_decal_base_view;
            views[56] = dune_decal_height_view;
            for (std::size_t index = 1; index < mountain.relief_layer_views.size(); ++index)
                views[56 + index] = mountain.relief_layer_views[index];
            views[61] = terrain_extra_views[17]; views[62] = terrain_extra_views[18];
            views[63] = marsh.view; views[64] = marsh.material_height_view;
            views[65] = marsh.specular_view; views[66] = marsh_decal_base_view;
            views[67] = marsh_decal_height_view; views[68] = marsh_decal_specular_view;
            views[69] = volcano_base_view; views[70] = volcano_height_view;
            views[71] = volcano_active_base_view; views[72] = volcano_active_specular_view;
            views[73] = water_clutter_base_view; views[74] = water_clutter_height_view;
            views[75] = grass_clutter_base_view; views[76] = grass_clutter_height_view;
            views[77] = plains_clutter_base_view; views[78] = plains_clutter_height_view;
            for (std::size_t index = 0; index < river_surface_views.size(); ++index)
                views[79 + index] = river_surface_views[index];
            for (std::size_t index = 0; index < river_rock_texture_views.size(); ++index)
                views[89 + index] = river_rock_texture_views[index];
            for (std::size_t index = 4; index < feature_texture_views.size(); ++index)
                views[90 + index] = feature_texture_views[index];
            for (std::size_t index = 0; index < route_texture_views.size(); ++index)
                views[98 + index] = route_texture_views[index];
            for (std::size_t index = 0; index < bridge_texture_views.size(); ++index)
                views[108 + index] = bridge_texture_views[index];
            for (std::size_t index = 0; index < resource_texture_views.size(); ++index)
                views[116 + index] = resource_texture_views[index];
            for (std::size_t index = 0; index < city_base_views.size(); ++index)
                views[124 + index] = city_base_views[index];
            context->PSSetShaderResources(0, static_cast<UINT>(views.size()), views.data());
            ID3D11SamplerState * samplers[] = {terrain_sampler, decal_sampler};
            context->PSSetSamplers(0, 2, samplers);
            context->PSSetConstantBuffers(0, 1, &terrain_settings_buffer);
            // Upload triangle-aligned chunks instead of one viewport-sized
            // immutable buffer.  This keeps all zooms below D3D11's buffer
            // limits without changing draw order or shoreline continuity.
            std::size_t const chunk_capacity = 262143u;
            auto draw_batches = [&](std::vector<Vertex> const & batch_vertices) {
                for (std::size_t first = 0; first < batch_vertices.size(); first += chunk_capacity) {
                    std::size_t count = std::min(chunk_capacity, batch_vertices.size() - first);
                    D3D11_BUFFER_DESC buffer_desc = {};
                    buffer_desc.ByteWidth = static_cast<UINT>(count * sizeof(Vertex));
                    buffer_desc.Usage = D3D11_USAGE_IMMUTABLE;
                    buffer_desc.BindFlags = D3D11_BIND_VERTEX_BUFFER;
                    D3D11_SUBRESOURCE_DATA initial = {batch_vertices.data() + first, 0, 0};
                    ID3D11Buffer * vertex_buffer = nullptr;
                    HRESULT hr = device->CreateBuffer(&buffer_desc, &initial, &vertex_buffer);
                    if (FAILED(hr)) {
                        OutputDebugStringA("[C3X renderer] native-failure=vertex-buffer\n");
                        return false;
                    }
                    UINT stride = sizeof(Vertex);
                    UINT offset = 0;
                    context->IASetVertexBuffers(0, 1, &vertex_buffer, &stride, &offset);
                    context->Draw(static_cast<UINT>(count), 0);
                    vertex_buffer->Release();
                }
                return true;
            };
            if (!draw_batches(underlay_vertices) || !draw_batches(land_vertices) ||
                !draw_batches(bed_vertices) || !draw_batches(water_vertices) ||
                !draw_batches(river_vertices) || !draw_batches(shadow_vertices) ||
                !draw_batches(route_vertices)) {
                return false;
            }
            if (!feature_vertices.empty()) {
                context->VSSetShader(feature_vertex_shader, nullptr, 0);
                context->PSSetShader(feature_pixel_shader, nullptr, 0);
                if (!draw_batches(feature_vertices))
                    return false;
            }
            if (!mine_vertices.empty()) {
                context->VSSetShader(feature_vertex_shader, nullptr, 0);
                context->PSSetShader(feature_pixel_shader, nullptr, 0);
                context->PSSetShaderResources(116, 6, mine_base_views.data());
                context->PSSetShaderResources(124, 2, mine_emissive_views.data());
                if (!draw_batches(mine_vertices))
                    return false;
            }
            if (!city_vertices.empty()) {
                context->VSSetShader(feature_vertex_shader, nullptr, 0);
                context->PSSetShader(feature_pixel_shader, nullptr, 0);
                context->PSSetShaderResources(116, 4, city_emissive_views.data());
                context->PSSetShaderResources(124, 4, city_base_views.data());
                if (!draw_batches(city_vertices))
                    return false;
            }
            if (!wall_vertices.empty()) {
                std::array<ID3D11ShaderResourceView *, 4> no_emissive = {};
                std::array<ID3D11ShaderResourceView *, 4> wall_views = {
                    wall_texture_view, nullptr, nullptr, nullptr};
                context->VSSetShader(feature_vertex_shader, nullptr, 0);
                context->PSSetShader(feature_pixel_shader, nullptr, 0);
                context->PSSetShaderResources(116, 4, no_emissive.data());
                context->PSSetShaderResources(124, 4, wall_views.data());
                if (!draw_batches(wall_vertices))
                    return false;
            }
        }

        context->CopyResource(readback_texture, render_texture);
        D3D11_MAPPED_SUBRESOURCE mapped = {};
        HRESULT hr = context->Map(readback_texture, 0, D3D11_MAP_READ, 0, &mapped);
        if (FAILED(hr)) {
            OutputDebugStringA("[C3X renderer] native-failure=readback\n");
            return false;
        }
        for (int y = 0; y < height; ++y) {
            std::memcpy(pixels.data() + static_cast<std::size_t>(y) * width,
                        static_cast<std::uint8_t const *>(mapped.pData) + static_cast<std::size_t>(y) * mapped.RowPitch,
                        static_cast<std::size_t>(width) * sizeof(std::uint32_t));
        }
        context->Unmap(readback_texture, 0);

        cached_rendered_tile_count = textured_tile_count;
        cached_fallback_tile_count = fallback_tile_count;
        cached_textured_tile_count = textured_tile_count;
        cached_visible_animation_count = frame.visible_animation_count;
        cached_request_continuous_redraw = frame.visible_animation_count != 0;
        if (!reuse_shadow_field) {
            shadow_visibility_cache = std::move(next_shadow_visibility);
            shadow_field_signature = current_shadow_field_signature;
        }
        cached_signature = signature;
        previous_signature = signature;
        previous_content_revision = content_revision;
        cache_valid = true;
        if (frame.tile_count == 0)
            cached_tiles.clear();
        else
            cached_tiles.assign(frame.tiles, frame.tiles + frame.tile_count);
        cached_replacement_tile_flags = replacement_tile_flags;
        if (cache_valid) {
            for (auto existing = viewport_cache.begin(); existing != viewport_cache.end(); ++existing) {
                if (existing->signature.complete == signature.complete) {
                    viewport_cache.erase(existing);
                    break;
                }
            }
            if (viewport_cache.size() == viewport_cache_capacity) {
                viewport_cache.erase(viewport_cache.begin());
                if (cache_evictions != 0xffffffffu)
                    ++cache_evictions;
            }
            CachedViewport stored;
            stored.signature = signature;
            stored.pixels = pixels;
            stored.tiles = cached_tiles;
            stored.replacement_flags = cached_replacement_tile_flags;
            stored.rendered_tile_count = cached_rendered_tile_count;
            stored.fallback_tile_count = cached_fallback_tile_count;
            stored.textured_tile_count = cached_textured_tile_count;
            viewport_cache.push_back(std::move(stored));
        }
        QueryPerformanceCounter(&finished);
        fill_output(frame, output, invalidations, finished.QuadPart - started.QuadPart);
        return true;
    }
};

RendererState renderer;

bool valid_frame(c3x_renderer_frame_v1 const * frame, c3x_renderer_output_v1 const * output) {
    if (frame == nullptr || output == nullptr)
        return false;
    if (frame->api_version != C3X_RENDERER_API_VERSION || frame->struct_size != sizeof(*frame))
        return false;
    if (output->struct_size != sizeof(*output))
        return false;
    if (frame->target_width <= 0 || frame->target_height <= 0 ||
        frame->target_width > 8192 || frame->target_height > 8192)
        return false;
    if (frame->tile_width <= 0 || frame->tile_height <= 0 || frame->tile_count > 8192u)
        return false;
    if (frame->presentation_time_ticks < 0 || frame->presentation_frequency <= 0)
        return false;
    if (frame->world_width_tiles < 0 || frame->world_height_tiles < 0 ||
        frame->world_width_tiles > 100000 || frame->world_height_tiles > 100000 ||
        (frame->world_wrap_x != 0 && frame->world_width_tiles == 0) ||
        (frame->world_wrap_y != 0 && frame->world_height_tiles == 0))
        return false;
    if (frame->tile_count != 0 && frame->tiles == nullptr)
        return false;
    if (frame->clip_left < 0 || frame->clip_top < 0 ||
        frame->clip_right > frame->target_width || frame->clip_bottom > frame->target_height ||
        frame->clip_left >= frame->clip_right || frame->clip_top >= frame->clip_bottom)
        return false;
    return true;
}

} // namespace

extern "C" __declspec(dllexport) c3x_renderer_u32 c3x_renderer_get_api_version(void) {
    return C3X_RENDERER_API_VERSION;
}

extern "C" __declspec(dllexport) int c3x_renderer_set_pack_path(char const * pack_path) {
    return renderer.configure_pack(pack_path) ? C3X_RENDERER_RESULT_OK : C3X_RENDERER_RESULT_ERROR;
}

extern "C" __declspec(dllexport) int c3x_renderer_set_definition_paths(
    char const * mod_root, char const * default_path, char const * scenario_path, char const * custom_path) {
    return renderer.configure_definitions(mod_root, default_path, scenario_path, custom_path)
        ? C3X_RENDERER_RESULT_OK : C3X_RENDERER_RESULT_ERROR;
}

extern "C" __declspec(dllexport) int c3x_renderer_render(
    c3x_renderer_frame_v1 const * frame, c3x_renderer_output_v1 * output) {
    if (!valid_frame(frame, output))
        return C3X_RENDERER_RESULT_BAD_ARGUMENT;
    if (renderer.render(*frame, *output))
        return C3X_RENDERER_RESULT_OK;
    renderer.reset();
    if (renderer.device_recoveries != 0xffffffffu)
        ++renderer.device_recoveries;
    return renderer.render(*frame, *output)
        ? C3X_RENDERER_RESULT_OK : C3X_RENDERER_RESULT_DEVICE_ERROR;
}

extern "C" __declspec(dllexport) int c3x_renderer_blit(
    c3x_renderer_output_v1 const * output, void * destination_hdc) {
    if (output == nullptr || destination_hdc == nullptr ||
        output->api_version != C3X_RENDERER_API_VERSION || output->struct_size != sizeof(*output) ||
        output->width <= 0 || output->height <= 0 || output->stride_bytes != output->width * 4 ||
        output->clip_left < 0 || output->clip_top < 0 ||
        output->clip_right > output->width || output->clip_bottom > output->height ||
        output->clip_left >= output->clip_right || output->clip_top >= output->clip_bottom ||
        output->bgra_pixels == nullptr)
        return C3X_RENDERER_RESULT_BAD_ARGUMENT;

    BITMAPINFO info = {};
    info.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
    info.bmiHeader.biWidth = output->width;
    info.bmiHeader.biHeight = -output->height;
    info.bmiHeader.biPlanes = 1;
    info.bmiHeader.biBitCount = 32;
    info.bmiHeader.biCompression = BI_RGB;
    void * dib_bits = nullptr;
    HDC source_dc = CreateCompatibleDC(nullptr);
    HBITMAP bitmap = CreateDIBSection(source_dc, &info, DIB_RGB_COLORS, &dib_bits, nullptr, 0);
    if (source_dc == nullptr || bitmap == nullptr || dib_bits == nullptr) {
        if (bitmap != nullptr)
            DeleteObject(bitmap);
        if (source_dc != nullptr)
            DeleteDC(source_dc);
        return C3X_RENDERER_RESULT_ERROR;
    }
    std::memcpy(dib_bits, output->bgra_pixels,
                static_cast<std::size_t>(output->stride_bytes) * output->height);
    HGDIOBJ previous = SelectObject(source_dc, bitmap);
    int blit_width = output->clip_right - output->clip_left;
    int blit_height = output->clip_bottom - output->clip_top;
    BOOL ok = BitBlt(static_cast<HDC>(destination_hdc),
                     output->clip_left, output->clip_top, blit_width, blit_height,
                     source_dc, output->clip_left, output->clip_top, SRCCOPY);
    SelectObject(source_dc, previous);
    DeleteObject(bitmap);
    DeleteDC(source_dc);
    return ok ? C3X_RENDERER_RESULT_OK : C3X_RENDERER_RESULT_ERROR;
}

extern "C" __declspec(dllexport) void c3x_renderer_reset(void) {
    renderer.reset();
}

// Focused, source-faithful mountain study. The packet is replayed unchanged by
// Metal and D3D11; this file contains no graphics-backend implementation.
#define main frozen_scene_unused_main
#include "../../shared/frozen_scene.cpp"
#undef main
#include "../../shared/environment_runtime.cpp"

#include <fstream>
#include <sstream>

namespace {

struct BeautyVertex {
    float position[3];
    float world[3];
    float normal[3];
    float uv[2];
    float material[3]; // source height, surface kind, authored footprint
};

struct ComposedMountainVertex {
    float position[3];
    float world[4];
    float normal[3];
    float uv[2];
    float material[3];
};

struct BeautyFrame {
    float sun[4];
    float sun_color_exposure[4];
    float ambient[4];
    float view[4];
    float macro[4]; // source minimum, maximum, height scale, world span
    float quality[4]; // enabled, material scale, detail strength, shadow strength
};

static_assert(sizeof(BeautyVertex) == 56, "beauty vertex wire drift");
static_assert(sizeof(ComposedMountainVertex) == 60, "composed mountain vertex wire drift");
static_assert(sizeof(BeautyFrame) == 96, "beauty frame alignment drift");

float clamp01(float value) {
    return std::max(0.0f, std::min(1.0f, value));
}

void normalize3(float value[3]) {
    float length = std::sqrt(value[0] * value[0] + value[1] * value[1] + value[2] * value[2]);
    if (length <= 1.0e-8f) return;
    for (unsigned index = 0; index < 3; ++index) value[index] /= length;
}

BeautyVertex project_vertex(float x, float y, float z, float source_height,
                            float surface_kind) {
    // Orthographic presentation intentionally follows the source art rather
    // than the Civ III tile basis. Camera-to-surface view direction is fixed.
    const float right[3] = {0.8320503f, 0.5547002f, 0.0f};
    const float up[3] = {-0.258886f, 0.388329f, 0.884652f};
    const float forward[3] = {-0.490290f, 0.735435f, -0.469979f};
    float sx = x * right[0] + y * right[1];
    float sy = x * up[0] + y * up[1] + z * up[2];
    float distance = 8.0f + x * forward[0] + y * forward[1] + z * forward[2];
    BeautyVertex out = {};
    out.position[0] = sx / 3.25f;
    out.position[1] = (sy - 0.20f) / 2.30f;
    out.position[2] = clamp01((distance - 4.0f) / 8.0f);
    out.world[0] = x;
    out.world[1] = y;
    out.world[2] = z;
    out.uv[0] = x / 5.8f + 0.5f;
    out.uv[1] = y / 5.8f + 0.5f;
    out.material[0] = source_height;
    out.material[1] = surface_kind;
    return out;
}

void add_triangle(std::vector<BeautyVertex> &vertices, const BeautyVertex &a,
                  const BeautyVertex &b, const BeautyVertex &c) {
    vertices.push_back(a);
    vertices.push_back(b);
    vertices.push_back(c);
}

void add_draw(const std::vector<BeautyVertex> &vertices, unsigned constants,
              const std::array<unsigned, 13> &textures, bool depth) {
    recorded.buffers.emplace_back(
        reinterpret_cast<const std::uint8_t *>(vertices.data()),
        reinterpret_cast<const std::uint8_t *>(vertices.data() + vertices.size()));
    labv2::Draw draw;
    draw.vertex_buffer = unsigned(recorded.buffers.size() - 1);
    draw.constant_buffer = constants;
    draw.count = unsigned(vertices.size());
    draw.stride = sizeof(BeautyVertex);
    draw.feature = 1;
    draw.depth_mode = depth ? 2 : 0;
    draw.blend_mode = 0;
    draw.attributes = {{3, 0}, {3, 12}, {3, 24}, {2, 36}, {3, 44}};
    for (unsigned index = 0; index < textures.size(); ++index)
        draw.textures[index] = textures[index];
    recorded.draws.push_back(draw);
}

void add_composed_draw(const std::vector<ComposedMountainVertex> &vertices,
                       unsigned constants,
                       const std::array<unsigned, 13> &textures) {
    if (vertices.empty()) return;
    recorded.buffers.emplace_back(
        reinterpret_cast<const std::uint8_t *>(vertices.data()),
        reinterpret_cast<const std::uint8_t *>(vertices.data() + vertices.size()));
    labv2::Draw draw;
    draw.vertex_buffer = unsigned(recorded.buffers.size() - 1);
    draw.constant_buffer = constants;
    draw.count = unsigned(vertices.size());
    draw.stride = sizeof(ComposedMountainVertex);
    draw.feature = 1;
    draw.depth_mode = 2;
    draw.blend_mode = 0;
    draw.attributes = {{3, 0}, {4, 12}, {3, 28}, {2, 40}, {3, 48}};
    for (unsigned index = 0; index < textures.size(); ++index)
        draw.textures[index] = textures[index];
    draw.world_attribute = 1;
    draw.normal_attribute = 2;
    draw.uv_attribute = 3;
    draw.geometry_flags = 3u;
    recorded.draws.push_back(draw);
}

std::uint32_t mountain_seed(BiqWindowTile const &tile) {
    return std::uint32_t(tile.source_x * 0x193u) ^
           std::uint32_t(tile.source_y * 0x217u) ^ 0x6b91u;
}

ComposedMountainVertex project_composed_mountain(BiqWindowTile const &tile,
        float u, float v, float authoring_height, float source_height,
        float footprint, unsigned width, unsigned height) {
    constexpr float half_width = 64.0f, half_height = 32.0f;
    constexpr float vertical_scale = 0.82f * half_width / 112.0f;
    float world_x = float(tile.column) + 0.5f + (u - 0.5f) * 1.85f;
    float world_y = float(tile.row) + 0.5f + (0.5f - v) * 1.55f;
    // The composed terrain and forest providers use the same authoritative
    // world-to-screen basis: screen_x = 40 + (world_x + world_y) * 64.
    // Keeping the mountain visible projection on that basis is required for
    // its world-space silhouette and Q6 cast shadow to meet on the ground.
    float center_x = 40.0f + (world_x + world_y) * half_width;
    float base_y = 380.0f + (world_x - world_y) * half_height;
    float screen_y = base_y - authoring_height * vertical_scale;
    ComposedMountainVertex out = {};
    out.position[0] = center_x / float(width) * 2.0f - 1.0f;
    out.position[1] = 1.0f - screen_y / float(height) * 2.0f;
    out.position[2] = clamp01(0.94f - base_y / float(height) * 0.75f -
                              authoring_height * 0.0012f - 0.00035f);
    out.world[0] = world_x;
    out.world[1] = world_y;
    out.world[2] = authoring_height / 112.0f;
    out.world[3] = 1.0f;
    out.uv[0] = u;
    out.uv[1] = v;
    out.material[0] = source_height;
    out.material[1] = 2.0f;
    out.material[2] = footprint;
    return out;
}

void add_composed_triangle(std::vector<ComposedMountainVertex> &vertices,
                           ComposedMountainVertex const &a,
                           ComposedMountainVertex const &b,
                           ComposedMountainVertex const &c) {
    vertices.push_back(a); vertices.push_back(b); vertices.push_back(c);
}

} // namespace

int main(int argc, char **argv) {
    if (argc != 7) return 2;
    try {
        recorded = {};
        const unsigned width = unsigned(std::stoul(argv[2]));
        const unsigned height = unsigned(std::stoul(argv[3]));
        std::ifstream descriptor(argv[6]);
        std::string fixture((std::istreambuf_iterator<char>(descriptor)), {});
        const bool quality = fixture.find("beauty-mountain-baseline") == std::string::npos;
        const bool combined_scene = fixture.find("beauty-scene") != std::string::npos;
        const bool composed = fixture.find("source-fidelity-r2-inland") != std::string::npos;

        ID3D11Device device;
        HeightField macro_height, macro_blend;
        ID3D11ShaderResourceView *macro_height_view = nullptr;
        ID3D11ShaderResourceView *macro_blend_view = nullptr;
        const std::string pack = "Renderer/packs/Civ5EnvironmentSkin/";
        const std::string relief = pack +
            "textures/relief/mountains/standard/variant_02/";
        if (!load_r8_height(&device, relief + "height_lod0.dds", DXGI_FORMAT_R8_UNORM,
                            macro_height, &macro_height_view) ||
            !load_r8_height(&device, relief + "blend_lod0.dds", DXGI_FORMAT_R8_UNORM,
                            macro_blend, &macro_blend_view))
            throw std::runtime_error("mountain macro source load failed");

        std::array<unsigned, 13> texture_ids = {};
        auto texture = [&](unsigned slot, const char *path, DXGI_FORMAT format) {
            ID3D11ShaderResourceView *view = nullptr;
            unsigned texture_width = 0, texture_height = 0;
            if (!load_dds(&device, pack + path, format, &view, texture_width, texture_height))
                throw std::runtime_error(std::string("material source load failed: ") + path);
            texture_ids[slot] = view->id;
            release(view);
        };
        texture(0, "textures/grassland_base_color.dds", DXGI_FORMAT_BC3_UNORM_SRGB);
        texture(1, "textures/grassland_height.dds", DXGI_FORMAT_BC4_UNORM);
        texture(2, "textures/grassland_specular.dds", DXGI_FORMAT_BC4_UNORM);
        texture(3, "textures/mtn_base_base_color.dds", DXGI_FORMAT_BC3_UNORM_SRGB);
        texture(4, "textures/mtn_base_height.dds", DXGI_FORMAT_BC4_UNORM);
        texture(5, "textures/mtn_base_specular.dds", DXGI_FORMAT_BC4_UNORM);
        texture(6, "textures/mtn_top_base_color.dds", DXGI_FORMAT_BC3_UNORM_SRGB);
        texture(7, "textures/mtn_top_height.dds", DXGI_FORMAT_BC4_UNORM);
        texture(8, "textures/mtn_top_specular.dds", DXGI_FORMAT_BC4_UNORM);
        texture(9, "textures/mtn_snow_base_color.dds", DXGI_FORMAT_BC3_UNORM_SRGB);
        texture(10, "textures/mtn_snow_height.dds", DXGI_FORMAT_BC4_UNORM);
        texture(11, "textures/mtn_snow_specular.dds", DXGI_FORMAT_BC4_UNORM);
        texture_ids[12] = macro_height_view->id;

        BeautyFrame frame = {};
        frame.sun[0] = -0.62f; frame.sun[1] = -0.42f; frame.sun[2] = 0.66f;
        normalize3(frame.sun);
        frame.sun[3] = 2.18f;
        frame.sun_color_exposure[0] = 1.00f;
        frame.sun_color_exposure[1] = 0.91f;
        frame.sun_color_exposure[2] = 0.76f;
        frame.sun_color_exposure[3] = 1.18f;
        frame.ambient[0] = 0.35f; frame.ambient[1] = 0.46f; frame.ambient[2] = 0.61f;
        frame.ambient[3] = 0.58f;
        frame.view[0] = 0.490290f; frame.view[1] = -0.735435f; frame.view[2] = 0.469979f;
        frame.macro[0] = macro_height.minimum;
        frame.macro[1] = macro_height.maximum;
        frame.macro[2] = 1.72f;
        frame.macro[3] = 3.25f;
        frame.quality[0] = quality ? 1.0f : 0.0f;
        frame.quality[1] = 0.72f;
        frame.quality[2] = 0.105f;
        frame.quality[3] = 0.72f;
        recorded.buffers.emplace_back(reinterpret_cast<const std::uint8_t *>(&frame),
                                      reinterpret_cast<const std::uint8_t *>(&frame + 1));
        const unsigned constants = unsigned(recorded.buffers.size() - 1);

        if (composed) {
            if (!load_biq_window(
                    "Renderer/terrain_lab/v2/fixtures/beauty/gameplay-100-v1/inland/terrain.csv",
                    biq_window))
                throw std::runtime_error("composed mountain window load failed");

            std::array<HeightField, 5> heights;
            std::array<HeightField, 5> blends;
            std::array<ID3D11ShaderResourceView *, 5> height_views{};
            std::array<ID3D11ShaderResourceView *, 5> blend_views{};
            heights[1] = macro_height;
            blends[1] = macro_blend;
            height_views[1] = macro_height_view;
            blend_views[1] = macro_blend_view;
            for (unsigned variant = 0; variant < 5; ++variant) {
                if (variant == 1) continue;
                char name[16];
                std::snprintf(name, sizeof(name), "variant_%02u/", variant + 1);
                std::string source = pack + "textures/relief/mountains/standard/" + name;
                if (!load_r8_height(&device, source + "height_lod0.dds", DXGI_FORMAT_R8_UNORM,
                                    heights[variant], &height_views[variant]) ||
                    !load_r8_height(&device, source + "blend_lod0.dds", DXGI_FORMAT_R8_UNORM,
                                    blends[variant], &blend_views[variant]))
                    throw std::runtime_error("composed mountain variant load failed");
            }

            constexpr unsigned grid = 128;
            constexpr float height_scale = 165.0f;
            std::vector<ComposedMountainVertex> mountains;
            unsigned mountain_count = 0;
            for (BiqWindowTile const &tile : biq_window.tiles) {
                if (tile.real != 6) continue;
                ++mountain_count;
                unsigned variant = mountain_seed(tile) % 5u;
                HeightField const &source_height = heights[variant];
                HeightField const &source_blend = blends[variant];
                auto vertex = [&](unsigned x, unsigned y) {
                    float u = float(x) / float(grid - 1);
                    float v = float(y) / float(grid - 1);
                    float h = source_height.sample(u, v);
                    float footprint = source_blend.sample(u, v);
                    ComposedMountainVertex out = project_composed_mountain(
                        tile, u, v, 2.5f + h * height_scale, h, footprint, width, height);
                    float du = 1.0f / float(grid - 1);
                    float hx = (source_height.sample(u + du, v) -
                                source_height.sample(u - du, v)) * (height_scale / 112.0f);
                    float hy = (source_height.sample(u, v + du) -
                                source_height.sample(u, v - du)) * (height_scale / 112.0f);
                    out.normal[0] = -hx / (2.0f * du * 1.85f);
                    out.normal[1] = hy / (2.0f * du * 1.55f);
                    out.normal[2] = 1.0f;
                    normalize3(out.normal);
                    return out;
                };
                for (unsigned y = 0; y + 1 < grid; ++y)
                    for (unsigned x = 0; x + 1 < grid; ++x) {
                        ComposedMountainVertex a = vertex(x, y);
                        ComposedMountainVertex b = vertex(x + 1, y);
                        ComposedMountainVertex c = vertex(x + 1, y + 1);
                        ComposedMountainVertex d = vertex(x, y + 1);
                        if (std::max(std::max(a.material[2], b.material[2]),
                                     std::max(c.material[2], d.material[2])) < 0.015f)
                            continue;
                        add_composed_triangle(mountains, a, b, c);
                        add_composed_triangle(mountains, a, c, d);
                    }
            }
            if (!mountain_count) throw std::runtime_error("composed mountain set is empty");
            add_composed_draw(mountains, constants, texture_ids);
            for (unsigned variant = 0; variant < 5; ++variant) {
                if (variant != 1) {
                    release(height_views[variant]);
                    release(blend_views[variant]);
                }
            }
            release(macro_height_view);
            release(macro_blend_view);
            recorded.width = width;
            recorded.height = height;
            recorded.downsample = std::atoi(argv[5]) == 2 ? 2u : 1u;
            recorded.color_branch = 1;
            recorded.geometry_contract = 1;
            recorded.valid_rect = {0, 0, width / recorded.downsample,
                                   height / recorded.downsample};
            recorded.exposure = 1.0f;
            return labv2::write_packet(argv[1], recorded) ? 0 : 1;
        }

        // A full-screen atmospheric background prevents transparent-edge
        // treatment from influencing perceived edge quality.
        std::vector<BeautyVertex> background;
        for (auto p : std::array<std::array<float, 2>, 6>{{
                 {{-1, -1}}, {{1, -1}}, {{1, 1}}, {{-1, -1}}, {{1, 1}}, {{-1, 1}}}}) {
            BeautyVertex v = {};
            v.position[0] = p[0]; v.position[1] = p[1]; v.position[2] = 0.999f;
            v.uv[0] = (p[0] + 1) * 0.5f; v.uv[1] = (1 - p[1]) * 0.5f;
            v.material[1] = 0;
            background.push_back(v);
        }
        add_draw(background, constants, texture_ids, false);

        std::vector<BeautyVertex> ground;
        // Keep every viewport corner on terrain. A smaller quad exposed the
        // atmospheric clear color in the far-right corner and made the study
        // read like a cut-out instead of a continuous Civ-style landscape.
        const float plane = 10.0f;
        BeautyVertex a = project_vertex(-plane, -plane, 0, 0, 1);
        BeautyVertex b = project_vertex( plane, -plane, 0, 0, 1);
        BeautyVertex c = project_vertex( plane,  plane, 0, 0, 1);
        BeautyVertex d = project_vertex(-plane,  plane, 0, 0, 1);
        for (BeautyVertex *v : {&a, &b, &c, &d}) v->normal[2] = 1;
        add_triangle(ground, a, b, c); add_triangle(ground, a, c, d);
        add_draw(ground, constants, texture_ids, true);

        // One source texel becomes one mesh sample. The previous whole-map path
        // used a much coarser grid and made the authored ridge network faceted.
        const unsigned grid = 256;
        const float span_x = 3.20f, span_y = 2.72f, z_scale = frame.macro[2];
        auto sample = [&](float u, float v) { return macro_height.sample(u, v); };
        auto footprint = [&](float u, float v) { return macro_blend.sample(u, v); };
        auto vertex = [&](unsigned x, unsigned y) {
            float u = float(x) / float(grid - 1), v = float(y) / float(grid - 1);
            float h = sample(u, v);
            float wx = (u - 0.5f) * span_x - (combined_scene ? 1.20f : 0.0f);
            float wy = (v - 0.5f) * span_y + (combined_scene ? 0.82f : 0.0f);
            BeautyVertex out = project_vertex(wx, wy, h * z_scale, h, 2);
            // Composition replays the relief and object packets together. A
            // small scene-only bias keeps the feathered, near-coplanar skirts
            // from losing their depth tie to the shared terrain plane.
            if (combined_scene) out.position[2] = clamp01(out.position[2] - 0.055f);
            out.uv[0] = u; out.uv[1] = v;
            out.material[2] = footprint(u, v);
            float du = 1.0f / float(grid - 1), dv = du;
            float hx = (sample(u + du, v) - sample(u - du, v)) * z_scale;
            float hy = (sample(u, v + dv) - sample(u, v - dv)) * z_scale;
            out.normal[0] = -hx / (2 * du * span_x);
            out.normal[1] = -hy / (2 * dv * span_y);
            out.normal[2] = 1;
            normalize3(out.normal);
            return out;
        };
        std::vector<BeautyVertex> mountain;
        mountain.reserve((grid - 1) * (grid - 1) * 6);
        for (unsigned y = 0; y + 1 < grid; ++y)
            for (unsigned x = 0; x + 1 < grid; ++x) {
                BeautyVertex v00 = vertex(x, y), v10 = vertex(x + 1, y);
                BeautyVertex v11 = vertex(x + 1, y + 1), v01 = vertex(x, y + 1);
                add_triangle(mountain, v00, v10, v11);
                add_triangle(mountain, v00, v11, v01);
            }
        add_draw(mountain, constants, texture_ids, true);

        release(macro_height_view);
        release(macro_blend_view);
        recorded.width = width;
        recorded.height = height;
        recorded.downsample = 1;
        recorded.color_branch = 1;
        recorded.valid_rect = {0, 0, width, height};
        recorded.exposure = 1.0f;
        return labv2::write_packet(argv[1], recorded) ? 0 : 1;
    } catch (const std::exception &error) {
        std::fprintf(stderr, "beauty_mountain: %s\n", error.what());
        return 1;
    }
}

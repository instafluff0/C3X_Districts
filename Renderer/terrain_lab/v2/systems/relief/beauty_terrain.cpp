// Focused Mac Metal terrain study using normalized Civ V environment-skin
// materials, authored hill relief, and the source grass-hill decal atlas.
#define main frozen_scene_unused_main
#include "../../shared/frozen_scene.cpp"
#undef main
#include "../../shared/environment_runtime.cpp"

#include <array>
#include <cstdint>

namespace {

struct TerrainVertex {
    float position[3];
    float world[3];
    float normal[3];
    float uv[2];
    float material[4]; // elevation, surface kind, hill support, plains weight
};

struct ComposedTerrainVertex {
    float position[3];
    float world[4];
    float normal[3];
    float uv[2];
    float material[4];
};

struct TerrainFrame {
    float sun[4];
    float sun_color_exposure[4];
    float ambient[4];
    float view[4];
    float detail[4]; // material repeat, detail strength, shadow strength, unused
};

struct Hill {
    float x, y, radius_x, radius_y, height, angle, source_u, source_v;
    float rockiness;
    std::uint32_t seed;
};

constexpr std::array<Hill, 6> hills = {{
    {-2.85f,  0.68f, 1.54f, 1.04f, 0.50f, -0.34f, 0.11f, 0.17f, 0.88f, 0x15a3u},
    {-1.12f, -0.76f, 1.38f, 0.96f, 0.38f,  0.53f, 0.47f, 0.08f, 0.40f, 0x31c7u},
    { 0.23f,  0.58f, 1.72f, 1.19f, 0.57f, -0.61f, 0.69f, 0.54f, 1.00f, 0x70bdu},
    { 2.16f, -0.53f, 1.46f, 1.02f, 0.44f,  0.26f, 0.32f, 0.79f, 0.66f, 0xa211u},
    { 3.25f,  1.12f, 1.28f, 0.90f, 0.35f, -0.82f, 0.87f, 0.31f, 0.27f, 0xc4d9u},
    {-0.46f,  1.78f, 1.14f, 0.81f, 0.32f,  0.14f, 0.23f, 0.91f, 0.54f, 0xe625u},
}};

float clamp01(float value) { return std::max(0.0f, std::min(1.0f, value)); }

float smooth01(float value) {
    value = clamp01(value);
    return value * value * (3.0f - 2.0f * value);
}

void normalize3(float value[3]) {
    float length = std::sqrt(value[0] * value[0] + value[1] * value[1] + value[2] * value[2]);
    if (length <= 1.0e-8f) return;
    for (unsigned index = 0; index < 3; ++index) value[index] /= length;
}

std::uint32_t random_u32(std::uint32_t &state) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

float random01(std::uint32_t &state) {
    return float(random_u32(state) & 0x00ffffffu) / float(0x01000000u);
}

float hill_support(Hill const &hill, float x, float y) {
    float c = std::cos(hill.angle), s = std::sin(hill.angle);
    float dx = x - hill.x, dy = y - hill.y;
    float rx = (c * dx + s * dy) / hill.radius_x;
    float ry = (-s * dx + c * dy) / hill.radius_y;
    float radius = std::sqrt(rx * rx + ry * ry);
    return 1.0f - smooth01((radius - 0.34f) / 0.66f);
}

float source_macro(HeightField const &field, Hill const &hill, float x, float y) {
    float c = std::cos(hill.angle), s = std::sin(hill.angle);
    float dx = x - hill.x, dy = y - hill.y;
    float u = hill.source_u + (c * dx + s * dy) * 0.092f;
    float v = hill.source_v + (-s * dx + c * dy) * 0.092f;
    constexpr float radius = 0.010f;
    return (field.sample(u, v) * 4.0f +
            (field.sample(u - radius, v) + field.sample(u + radius, v) +
             field.sample(u, v - radius) + field.sample(u, v + radius)) * 2.0f +
            field.sample(u - radius, v - radius) + field.sample(u + radius, v - radius) +
            field.sample(u - radius, v + radius) + field.sample(u + radius, v + radius)) / 16.0f;
}

float terrain_height(HeightField const &field, float x, float y, float *support_out = nullptr) {
    float broad = (field.sample(0.061f * x + 0.13f, 0.061f * y + 0.41f) - 0.5f) * 0.055f;
    float elevation = broad;
    float support = 0.0f;
    for (Hill const &hill : hills) {
        float authored = source_macro(field, hill, x, y);
        float raw_mask = hill_support(hill, x, y);
        // Source contours perturb the ownership edge as well as the crown.
        // This removes the repeated ellipse silhouette without inventing a
        // second procedural noise source.
        float mask = smooth01((raw_mask + (authored - 0.5f) * 0.42f - 0.05f) / 0.90f);
        // The support controls only placement/topology. Authored relief breaks
        // up the crown and shoulders so the hills are not smooth domes.
        float shaped = hill.height * mask * (0.34f + authored * 0.86f);
        elevation = std::max(elevation, broad + shaped);
        support = std::max(support, mask);
    }
    if (support_out) *support_out = support;
    return elevation;
}

TerrainVertex project_vertex(float x, float y, float z) {
    const float right[3] = {0.8320503f, 0.5547002f, 0.0f};
    const float up[3] = {-0.258886f, 0.388329f, 0.884652f};
    const float forward[3] = {-0.490290f, 0.735435f, -0.469979f};
    float sx = x * right[0] + y * right[1];
    float sy = x * up[0] + y * up[1] + z * up[2];
    float distance = 10.0f + x * forward[0] + y * forward[1] + z * forward[2];
    TerrainVertex out = {};
    out.position[0] = sx / 4.32f;
    out.position[1] = (sy - 0.10f) / 2.76f;
    out.position[2] = clamp01((distance - 4.0f) / 12.0f);
    out.world[0] = x; out.world[1] = y; out.world[2] = z;
    return out;
}

void add_triangle(std::vector<TerrainVertex> &vertices, TerrainVertex const &a,
                  TerrainVertex const &b, TerrainVertex const &c) {
    vertices.push_back(a); vertices.push_back(b); vertices.push_back(c);
}

void add_draw(std::vector<TerrainVertex> const &vertices, unsigned constants,
              std::array<unsigned, 19> const &textures, unsigned depth_mode,
              unsigned blend_mode) {
    recorded.buffers.emplace_back(
        reinterpret_cast<std::uint8_t const *>(vertices.data()),
        reinterpret_cast<std::uint8_t const *>(vertices.data() + vertices.size()));
    labv2::Draw draw;
    draw.vertex_buffer = unsigned(recorded.buffers.size() - 1);
    draw.constant_buffer = constants;
    draw.count = unsigned(vertices.size());
    draw.stride = sizeof(TerrainVertex);
    draw.feature = 1;
    draw.depth_mode = depth_mode;
    draw.blend_mode = blend_mode;
    draw.attributes = {{3, 0}, {3, 12}, {3, 24}, {2, 36}, {4, 44}};
    for (unsigned index = 0; index < textures.size(); ++index)
        draw.textures[index] = textures[index];
    recorded.draws.push_back(draw);
}

void add_composed_draw(std::vector<ComposedTerrainVertex> const &vertices,
                       unsigned constants,
                       std::array<unsigned, 19> const &textures,
                       unsigned depth_mode, unsigned blend_mode,
                       unsigned geometry_flags) {
    if (vertices.empty()) return;
    recorded.buffers.emplace_back(
        reinterpret_cast<std::uint8_t const *>(vertices.data()),
        reinterpret_cast<std::uint8_t const *>(vertices.data() + vertices.size()));
    labv2::Draw draw;
    draw.vertex_buffer = unsigned(recorded.buffers.size() - 1);
    draw.constant_buffer = constants;
    draw.count = unsigned(vertices.size());
    draw.stride = sizeof(ComposedTerrainVertex);
    draw.feature = 1;
    draw.depth_mode = depth_mode;
    draw.blend_mode = blend_mode;
    draw.attributes = {{3, 0}, {4, 12}, {3, 28}, {2, 40}, {4, 48}};
    for (unsigned index = 0; index < textures.size(); ++index)
        draw.textures[index] = textures[index];
    draw.world_attribute = 1;
    draw.normal_attribute = 2;
    draw.uv_attribute = 3;
    draw.geometry_flags = geometry_flags;
    recorded.draws.push_back(draw);
}

std::uint32_t composed_seed(BiqWindowTile const &tile) {
    return std::uint32_t(tile.source_x * 0x193u) ^
           std::uint32_t(tile.source_y * 0x217u) ^ 0x8d31u;
}

Hill composed_hill(BiqWindowTile const &tile) {
    std::uint32_t state = composed_seed(tile);
    float angle = random01(state) * 6.283185307f;
    float radius_x = 0.43f + random01(state) * 0.05f;
    float radius_y = 0.37f + random01(state) * 0.05f;
    float height = 29.0f + random01(state) * 8.0f;
    float source_u = random01(state);
    float source_v = random01(state);
    float rockiness = 0.62f + random01(state) * 0.34f;
    return {float(tile.column) + 0.5f, float(tile.row) + 0.5f,
            radius_x, radius_y, height, angle, source_u, source_v,
            rockiness, state};
}

float composed_source_macro(HeightField const &field, Hill const &hill,
                            float world_x, float world_y) {
    float c = std::cos(hill.angle), s = std::sin(hill.angle);
    float dx = world_x - hill.x, dy = world_y - hill.y;
    float u = hill.source_u + (c * dx + s * dy) * 0.30f;
    float v = hill.source_v + (-s * dx + c * dy) * 0.30f;
    constexpr float radius = 0.012f;
    return (field.sample(u, v) * 4.0f +
            (field.sample(u - radius, v) + field.sample(u + radius, v) +
             field.sample(u, v - radius) + field.sample(u, v + radius)) * 2.0f +
            field.sample(u - radius, v - radius) + field.sample(u + radius, v - radius) +
            field.sample(u - radius, v + radius) + field.sample(u + radius, v + radius)) / 16.0f;
}

float composed_height(HeightField const &field, float world_x, float world_y,
                      float *support_out = nullptr) {
    float support = 0.0f;
    float height = 2.5f;
    for (BiqWindowTile const &tile : biq_window.tiles) {
        if (tile.real != 5) continue;
        Hill hill = composed_hill(tile);
        if (std::abs(world_x - hill.x) > hill.radius_x ||
            std::abs(world_y - hill.y) > hill.radius_y)
            continue;
        float authored = composed_source_macro(field, hill, world_x, world_y);
        float raw_mask = hill_support(hill, world_x, world_y);
        float local_support = smooth01(
            (raw_mask + (authored - 0.5f) * 0.56f - 0.08f) / 0.88f);
        float local_height = 2.5f + hill.height * local_support *
            (0.20f + authored * 0.94f);
        height = std::max(height, local_height);
        support = std::max(support, local_support);
    }
    if (support_out) *support_out = support;
    return height;
}

ComposedTerrainVertex project_composed_terrain(BiqWindowTile const &tile,
        float u, float v, float height, float support, float material_kind,
        float biome, unsigned output_width, unsigned output_height) {
    float half_width = 64.0f, half_height = 32.0f;
    float vertical_scale = 0.82f * half_width / 112.0f;
    float center_x = 104.0f + float(tile.column + tile.row) * half_width;
    float center_y = 380.0f + float(tile.column - tile.row) * half_height;
    float screen_x = center_x + (u - v) * half_width;
    float base_y = center_y + (u + v - 1.0f) * half_height;
    float screen_y = base_y - height * vertical_scale;
    ComposedTerrainVertex out = {};
    out.position[0] = screen_x / float(output_width) * 2.0f - 1.0f;
    out.position[1] = 1.0f - screen_y / float(output_height) * 2.0f;
    out.position[2] = clamp01(0.94f - base_y / float(output_height) * 0.75f -
                              height * 0.0012f - 0.00015f);
    out.world[0] = float(tile.column) + u;
    out.world[1] = float(tile.row) + (1.0f - v);
    out.world[2] = height / 112.0f;
    out.world[3] = 1.0f;
    out.uv[0] = out.world[0]; out.uv[1] = out.world[1];
    out.material[0] = std::max(0.0f, (height - 2.5f) / 112.0f);
    out.material[1] = material_kind;
    out.material[2] = support;
    out.material[3] = biome;
    return out;
}

void add_composed_triangle(std::vector<ComposedTerrainVertex> &vertices,
        ComposedTerrainVertex const &a, ComposedTerrainVertex const &b,
        ComposedTerrainVertex const &c) {
    vertices.push_back(a); vertices.push_back(b); vertices.push_back(c);
}

float biome_coordinate(HeightField const &field, float x, float y) {
    float screen_x = x * 0.8320503f + y * 0.5547002f;
    float material_noise = field.sample(x * 0.035f + 0.73f, y * 0.035f + 0.22f);
    return std::max(0.0f, std::min(2.0f,
        (screen_x + 3.45f) / 3.45f + (material_noise - 0.5f) * 0.24f));
}

TerrainVertex surface_vertex(HeightField const &field, float x, float y, float lift = 0.0f) {
    float support = 0.0f;
    float z = terrain_height(field, x, y, &support);
    float epsilon = 0.018f;
    float hx = terrain_height(field, x + epsilon, y) - terrain_height(field, x - epsilon, y);
    float hy = terrain_height(field, x, y + epsilon) - terrain_height(field, x, y - epsilon);
    TerrainVertex out = project_vertex(x, y, z + lift);
    out.normal[0] = -hx / (2.0f * epsilon);
    out.normal[1] = -hy / (2.0f * epsilon);
    out.normal[2] = 1.0f;
    normalize3(out.normal);
    out.uv[0] = x; out.uv[1] = y;
    out.material[0] = z;
    out.material[1] = 1.0f;
    out.material[2] = support;
    out.material[3] = biome_coordinate(field, x, y);
    return out;
}

void add_decal(std::vector<TerrainVertex> &vertices, HeightField const &field,
               float center_x, float center_y, float size_x, float size_y,
               float angle, unsigned atlas_cell) {
    constexpr unsigned grid = 12;
    float c = std::cos(angle), s = std::sin(angle);
    auto vertex = [&](unsigned ix, unsigned iy) {
        float u = float(ix) / float(grid), v = float(iy) / float(grid);
        float lx = (u - 0.5f) * size_x, ly = (v - 0.5f) * size_y;
        float x = center_x + c * lx - s * ly;
        float y = center_y + s * lx + c * ly;
        TerrainVertex out = surface_vertex(field, x, y, 0.030f);
        float inset = 0.0012f;
        out.uv[0] = (float(atlas_cell) + inset + u * (1.0f - 2.0f * inset)) * 0.25f;
        out.uv[1] = (inset + v * (1.0f - 2.0f * inset)) * 0.25f;
        out.material[1] = 2.0f;
        out.material[2] = angle;
        return out;
    };
    for (unsigned y = 0; y < grid; ++y)
        for (unsigned x = 0; x < grid; ++x) {
            TerrainVertex a = vertex(x, y), b = vertex(x + 1, y);
            TerrainVertex c0 = vertex(x + 1, y + 1), d = vertex(x, y + 1);
            add_triangle(vertices, a, b, c0); add_triangle(vertices, a, c0, d);
        }
}

} // namespace

int main(int argc, char **argv) {
    if (argc != 7) return 2;
    try {
        recorded = {};
        unsigned width = unsigned(std::stoul(argv[2]));
        unsigned height = unsigned(std::stoul(argv[3]));
        std::ifstream descriptor(argv[6]);
        std::string fixture((std::istreambuf_iterator<char>(descriptor)), {});
        bool composed = fixture.find("source-fidelity-r2-inland") != std::string::npos;
        ID3D11Device device;
        const std::string terrain_pack = "Renderer/packs/Civ5EnvironmentSkin/";
        const std::string decal_pack = "Renderer/packs/DecalsNormalized/";

        HeightField hill_field;
        ID3D11ShaderResourceView *hill_view = nullptr;
        if (!load_r8_height(&device,
                terrain_pack + "textures/relief/hills/standard/height_lod0.dds",
                DXGI_FORMAT_R8_UNORM, hill_field, &hill_view))
            throw std::runtime_error("authored hill relief load failed");

        std::array<unsigned, 19> texture_ids = {};
        auto texture = [&](unsigned slot, std::string const &root, char const *path,
                           DXGI_FORMAT format) {
            ID3D11ShaderResourceView *view = nullptr;
            unsigned texture_width = 0, texture_height = 0;
            if (!load_dds(&device, root + path, format, &view, texture_width, texture_height))
                throw std::runtime_error(std::string("terrain source load failed: ") + path);
            texture_ids[slot] = view->id;
            release(view);
        };
        texture(0, terrain_pack, "textures/grassland_base_color.dds", DXGI_FORMAT_BC3_UNORM_SRGB);
        texture(1, terrain_pack, "textures/grassland_height.dds", DXGI_FORMAT_BC4_UNORM);
        texture(2, terrain_pack, "textures/grassland_specular.dds", DXGI_FORMAT_BC4_UNORM);
        texture(3, terrain_pack, "textures/grasshill_top_base_color.dds", DXGI_FORMAT_BC3_UNORM_SRGB);
        texture(4, terrain_pack, "textures/grasshill_top_height.dds", DXGI_FORMAT_BC4_UNORM);
        texture(5, terrain_pack, "textures/grasshill_top_specular.dds", DXGI_FORMAT_BC4_UNORM);
        texture(6, terrain_pack, "textures/plains_base_color.dds", DXGI_FORMAT_BC3_UNORM_SRGB);
        texture(7, terrain_pack, "textures/plains_height.dds", DXGI_FORMAT_BC4_UNORM);
        texture(8, terrain_pack, "textures/plains_specular.dds", DXGI_FORMAT_BC4_UNORM);
        texture(9, terrain_pack, "textures/plainshill_top_base_color.dds", DXGI_FORMAT_BC3_UNORM_SRGB);
        texture(10, terrain_pack, "textures/plainshill_top_height.dds", DXGI_FORMAT_BC4_UNORM);
        texture(11, terrain_pack, "textures/plainshill_top_specular.dds", DXGI_FORMAT_BC4_UNORM);
        texture(12, decal_pack, "textures/decals/base_color_c996c6a9d015eebe.dds", DXGI_FORMAT_BC3_UNORM_SRGB);
        texture(13, decal_pack, "textures/decals/height_31eb0f0117ea3beb.dds", DXGI_FORMAT_BC5_UNORM);
        texture_ids[14] = hill_view->id;
        texture(15, terrain_pack, "textures/tundra_blend_base_color.dds", DXGI_FORMAT_BC3_UNORM_SRGB);
        texture(16, terrain_pack, "textures/tundra_blend_height.dds", DXGI_FORMAT_BC4_UNORM);
        texture(17, terrain_pack, "textures/tundra_blend_specular.dds", DXGI_FORMAT_BC4_UNORM);
        texture_ids[18] = texture_ids[17];

        TerrainFrame frame = {};
        frame.sun[0] = -0.62f; frame.sun[1] = -0.42f; frame.sun[2] = 0.66f;
        normalize3(frame.sun); frame.sun[3] = 2.08f;
        frame.sun_color_exposure[0] = 1.00f; frame.sun_color_exposure[1] = 0.91f;
        frame.sun_color_exposure[2] = 0.76f; frame.sun_color_exposure[3] = 1.16f;
        frame.ambient[0] = 0.35f; frame.ambient[1] = 0.46f; frame.ambient[2] = 0.61f;
        frame.ambient[3] = 0.61f;
        frame.view[0] = 0.490290f; frame.view[1] = -0.735435f; frame.view[2] = 0.469979f;
        frame.detail[0] = 0.43f; frame.detail[1] = 0.075f; frame.detail[2] = 0.70f;
        recorded.buffers.emplace_back(reinterpret_cast<std::uint8_t *>(&frame),
                                      reinterpret_cast<std::uint8_t *>(&frame + 1));
        unsigned constants = unsigned(recorded.buffers.size() - 1);

        if (composed) {
            if (!load_biq_window(
                    "Renderer/terrain_lab/v2/fixtures/beauty/gameplay-100-v1/inland/terrain.csv",
                    biq_window))
                throw std::runtime_error("composed terrain window load failed");
            build_river_graph();
            std::vector<ComposedTerrainVertex> surface;
            std::vector<ComposedTerrainVertex> decals;
            constexpr unsigned grid = 16;
            auto vertex = [&](BiqWindowTile const &tile, unsigned ix, unsigned iy) {
                float u = float(ix) / float(grid), v = float(iy) / float(grid);
                float world_x = float(tile.column) + u;
                float world_y = float(tile.row) + (1.0f - v);
                float support = 0.0f;
                float z = composed_height(hill_field, world_x, world_y, &support);
                ComposedTerrainVertex out = project_composed_terrain(
                    tile, u, v, z, support, 1.0f, 0.0f, width, height);
                float epsilon = 0.006f;
                float hx = composed_height(hill_field, world_x + epsilon, world_y) -
                           composed_height(hill_field, world_x - epsilon, world_y);
                float hy = composed_height(hill_field, world_x, world_y + epsilon) -
                           composed_height(hill_field, world_x, world_y - epsilon);
                out.normal[0] = -hx / (2.0f * epsilon * 128.0f);
                out.normal[1] = -hy / (2.0f * epsilon * 128.0f);
                out.normal[2] = 1.0f;
                normalize3(out.normal);
                return out;
            };
            for (BiqWindowTile const &tile : biq_window.tiles) {
                if (is_water_terrain(tile.base)) continue;
                for (unsigned y = 0; y < grid; ++y)
                    for (unsigned x = 0; x < grid; ++x) {
                        auto a = vertex(tile, x, y), b = vertex(tile, x + 1, y);
                        auto c = vertex(tile, x + 1, y + 1), d = vertex(tile, x, y + 1);
                        add_composed_triangle(surface, a, b, c);
                        add_composed_triangle(surface, a, c, d);
                    }
                if (tile.real != 5) continue;
                Hill hill = composed_hill(tile);
                std::uint32_t state = hill.seed;
                constexpr unsigned weighted_cells[10] = {0, 0, 0, 0, 1, 1, 1, 2, 2, 2};
                for (unsigned ordinal = 0; ordinal < 10; ++ordinal) {
                    float keep = random01(state);
                    float angle = random01(state) * 6.283185307f;
                    float radius = std::sqrt(random01(state)) * 0.22f;
                    float phase = random01(state) * 6.283185307f;
                    float scale = 0.90f + 0.20f * random01(state);
                    if (keep > hill.rockiness) continue;
                    float center_u = 0.5f + std::cos(phase) * radius;
                    float center_v = 0.5f + std::sin(phase) * radius;
                    constexpr unsigned decal_grid = 8;
                    for (unsigned y = 0; y < decal_grid; ++y)
                        for (unsigned x = 0; x < decal_grid; ++x) {
                            auto point = [&](unsigned px, unsigned py) {
                                float du = (float(px) / float(decal_grid) - 0.5f) * 0.42f * scale;
                                float dv = (float(py) / float(decal_grid) - 0.5f) * 0.37f * scale;
                                float cu = std::cos(angle), su = std::sin(angle);
                                float u = center_u + cu * du - su * dv;
                                float v = center_v + su * du + cu * dv;
                                float world_x = float(tile.column) + u;
                                float world_y = float(tile.row) + (1.0f - v);
                                float support = 0.0f;
                                float z = composed_height(hill_field, world_x, world_y,
                                                          &support) + 0.45f;
                                auto out = project_composed_terrain(tile, u, v, z, support,
                                    2.0f, 0.0f, width, height);
                                float inset = 0.0012f;
                                out.uv[0] = (float(weighted_cells[ordinal]) + inset +
                                    float(px) / float(decal_grid) * (1.0f - 2.0f * inset)) * 0.25f;
                                out.uv[1] = (inset + float(py) / float(decal_grid) *
                                    (1.0f - 2.0f * inset)) * 0.25f;
                                out.material[2] = angle;
                                out.normal[2] = 1.0f;
                                return out;
                            };
                            auto a = point(x, y), b = point(x + 1, y);
                            auto c = point(x + 1, y + 1), d = point(x, y + 1);
                            add_composed_triangle(decals, a, b, c);
                            add_composed_triangle(decals, a, c, d);
                        }
                }
            }
            add_composed_draw(surface, constants, texture_ids, 2, 0, 3u);
            add_composed_draw(decals, constants, texture_ids, 1, 1, 2u);
            release(hill_view);
            recorded.width = width; recorded.height = height;
            recorded.downsample = std::atoi(argv[5]) == 2 ? 2u : 1u;
            recorded.color_branch = 1;
            recorded.geometry_contract = 1;
            recorded.valid_rect = {0, 0, width / recorded.downsample,
                                   height / recorded.downsample};
            recorded.exposure = 1.0f;
            return labv2::write_packet(argv[1], recorded) ? 0 : 1;
        }

        std::vector<TerrainVertex> background;
        for (auto p : std::array<std::array<float, 2>, 6>{{
                 {{-1, -1}}, {{1, -1}}, {{1, 1}}, {{-1, -1}}, {{1, 1}}, {{-1, 1}}}}) {
            TerrainVertex v = {};
            v.position[0] = p[0]; v.position[1] = p[1]; v.position[2] = 0.999f;
            v.uv[0] = (p[0] + 1) * 0.5f; v.uv[1] = (1 - p[1]) * 0.5f;
            background.push_back(v);
        }
        add_draw(background, constants, texture_ids, 0, 0);

        constexpr unsigned grid_x = 300, grid_y = 220;
        constexpr float span_x = 10.6f, span_y = 7.2f;
        auto vertex = [&](unsigned ix, unsigned iy) {
            float x = (float(ix) / float(grid_x) - 0.5f) * span_x;
            float y = (float(iy) / float(grid_y) - 0.5f) * span_y;
            return surface_vertex(hill_field, x, y);
        };
        std::vector<TerrainVertex> terrain;
        terrain.reserve(grid_x * grid_y * 6);
        for (unsigned y = 0; y < grid_y; ++y)
            for (unsigned x = 0; x < grid_x; ++x) {
                TerrainVertex a = vertex(x, y), b = vertex(x + 1, y);
                TerrainVertex c = vertex(x + 1, y + 1), d = vertex(x, y + 1);
                add_triangle(terrain, a, b, c); add_triangle(terrain, a, c, d);
            }
        add_draw(terrain, constants, texture_ids, 2, 0);

        std::vector<TerrainVertex> decals;
        for (Hill const &hill : hills) {
            // The confirmed grass/plains HB set does not own tundra hills.
            // Their separate high-altitude snow/tundra decal set remains a
            // distinct future import instead of being approximated here.
            if (biome_coordinate(hill_field, hill.x, hill.y) > 1.48f) continue;
            std::uint32_t state = hill.seed;
            // Source recipe is HB01/HB02/HB03 at 3/2/2. Rockiness applies
            // deterministic thinning, while every retained placement receives
            // independent position, rotation, and +/-10% source scale variation.
            constexpr unsigned weighted_cells[7] = {0, 0, 0, 1, 1, 2, 2};
            for (unsigned ordinal = 0; ordinal < 7; ++ordinal) {
                float keep = random01(state);
                float angle = random01(state) * 6.283185307f;
                float radius = std::sqrt(random01(state)) * 0.76f;
                float phase = random01(state) * 6.283185307f;
                float scale = (0.90f + 0.20f * random01(state));
                if (keep > hill.rockiness) continue;
                float c = std::cos(hill.angle), s = std::sin(hill.angle);
                float lx = std::cos(phase) * radius * hill.radius_x;
                float ly = std::sin(phase) * radius * hill.radius_y;
                float x = hill.x + c * lx - s * ly;
                float y = hill.y + s * lx + c * ly;
                float patch = 0.92f * scale;
                add_decal(decals, hill_field, x, y, patch, patch * 0.88f,
                          angle, weighted_cells[ordinal]);
            }
        }
        add_draw(decals, constants, texture_ids, 1, 1);

        release(hill_view);
        recorded.width = width; recorded.height = height;
        recorded.downsample = 1;
        recorded.color_branch = 1;
        recorded.valid_rect = {0, 0, width, height};
        recorded.exposure = 1.0f;
        return labv2::write_packet(argv[1], recorded) ? 0 : 1;
    } catch (std::exception const &error) {
        std::fprintf(stderr, "beauty_terrain: %s\n", error.what());
        return 1;
    }
}

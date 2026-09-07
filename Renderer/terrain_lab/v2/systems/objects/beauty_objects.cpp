// Standalone source-faithful object studies. This provider emits a portable
// packet and contains no Metal, Direct3D, Civ III, or tile-layout behavior.
#define main frozen_scene_unused_main
#include "../../shared/frozen_scene.cpp"
#undef main
#include "../../shared/environment_runtime.cpp"

#include <fstream>
#include <map>

namespace {

struct BeautyVertex {
    float position[3];
    float world[3];
    float normal[3];
    float uv[2];
    float material[4]; // kind, has normal, has AO, has gloss
    float secondary[2]; // owner tint / shadow alpha, has emissive
};

struct BeautyFrame {
    float sun[4];
    float sun_color_exposure[4];
    float ambient[4];
    float view[4];
    float quality[4];
};

struct BeautyMaterial {
    std::array<std::string, 7> paths;
    unsigned owner_tint = 0;
    unsigned repeat = 0;
    std::array<unsigned, 7> textures{};
};

struct BeautyObject {
    std::string id;
    unsigned kind = 0;
    unsigned material = 0;
    std::vector<FeatureSourceVertex> vertices;
};

struct BeautyRecipe {
    unsigned object = 0;
    float scale = 1.0f;
    float scale_variation = 0.0f;
    unsigned count = 0;
    unsigned min_count = 0;
    unsigned priority = 0;
    unsigned flags = 0;
    float width = 0.0f;
    float low_end_reduction = 0.0f;
};

struct Placement {
    unsigned object;
    float x, y, scale, rotation;
};

static_assert(sizeof(BeautyVertex) == 68, "beauty object vertex wire drift");
static_assert(sizeof(BeautyFrame) == 80, "beauty object frame wire drift");

float clamp01(float value) {
    return std::max(0.0f, std::min(1.0f, value));
}

void normalize3(float value[3]) {
    float length = std::sqrt(value[0] * value[0] + value[1] * value[1] + value[2] * value[2]);
    if (length <= 1.0e-8f) return;
    for (unsigned index = 0; index < 3; ++index) value[index] /= length;
}

bool consume_optional_string(std::vector<std::uint8_t> const &data, std::size_t &cursor,
                             std::string &output) {
    std::uint32_t bytes = 0;
    if (!consume_u32(data, cursor, bytes) || bytes > 4096 || cursor + bytes > data.size())
        return false;
    output.assign(reinterpret_cast<char const *>(data.data() + cursor), bytes);
    cursor += bytes;
    return true;
}

bool load_beauty_objects(std::string const &path, std::vector<BeautyMaterial> &materials,
                         std::vector<BeautyObject> &objects,
                         std::vector<BeautyRecipe> &tree_recipes) {
    std::vector<std::uint8_t> data;
    if (!read_file(path, data) || data.size() < 24 ||
        std::memcmp(data.data(), "C3XBTO1\0", 8) != 0)
        return false;
    std::size_t cursor = 8;
    std::uint32_t version = 0, material_count = 0, object_count = 0, recipe_count = 0;
    if (!consume_u32(data, cursor, version) || !consume_u32(data, cursor, material_count) ||
        !consume_u32(data, cursor, object_count) || !consume_u32(data, cursor, recipe_count) ||
        version != 3 || material_count == 0 || material_count > 64 || object_count == 0 ||
        object_count > 256 || recipe_count == 0 || recipe_count > 128)
        return false;
    materials.resize(material_count);
    for (BeautyMaterial &material : materials) {
        for (std::string &path_value : material.paths)
            if (!consume_optional_string(data, cursor, path_value)) return false;
        if (!consume_u32(data, cursor, material.owner_tint) || material.owner_tint > 1 ||
            !consume_u32(data, cursor, material.repeat) || material.repeat > 1)
            return false;
    }
    objects.resize(object_count);
    for (BeautyObject &object : objects) {
        std::uint32_t count = 0;
        if (!consume_optional_string(data, cursor, object.id) || object.id.empty() ||
            !consume_u32(data, cursor, object.kind) ||
            !consume_u32(data, cursor, object.material) ||
            !consume_u32(data, cursor, count) || object.kind < 1 || object.kind > 3 ||
            object.material >= materials.size() || count < 3 || count > 300000 || count % 3)
            return false;
        object.vertices.resize(count);
        for (FeatureSourceVertex &vertex : object.vertices) {
            for (float &value : vertex.position)
                if (!consume_float(data, cursor, value)) return false;
            for (float &value : vertex.normal)
                if (!consume_float(data, cursor, value)) return false;
            for (float &value : vertex.uv)
                if (!consume_float(data, cursor, value)) return false;
        }
    }
    tree_recipes.resize(recipe_count);
    for (BeautyRecipe &recipe : tree_recipes) {
        if (!consume_u32(data, cursor, recipe.object) ||
            !consume_float(data, cursor, recipe.scale) ||
            !consume_float(data, cursor, recipe.scale_variation) ||
            !consume_u32(data, cursor, recipe.count) ||
            !consume_u32(data, cursor, recipe.min_count) ||
            !consume_u32(data, cursor, recipe.priority) ||
            !consume_u32(data, cursor, recipe.flags) ||
            !consume_float(data, cursor, recipe.width) ||
            !consume_float(data, cursor, recipe.low_end_reduction) ||
            recipe.object >= objects.size() || objects[recipe.object].kind != 1 ||
            recipe.scale <= 0 || recipe.scale_variation < 0 ||
            recipe.scale_variation > 2 || recipe.flags > 7)
            return false;
    }
    return cursor == data.size();
}

BeautyVertex project(float x, float y, float z, float view_x, float view_y,
                     float center_y, float kind) {
    const float right[3] = {0.8320503f, 0.5547002f, 0.0f};
    const float up[3] = {-0.258886f, 0.388329f, 0.884652f};
    const float forward[3] = {-0.490290f, 0.735435f, -0.469979f};
    BeautyVertex out = {};
    out.position[0] = (x * right[0] + y * right[1]) / view_x;
    out.position[1] = (x * up[0] + y * up[1] + z * up[2] - center_y) / view_y;
    float distance = 8.0f + x * forward[0] + y * forward[1] + z * forward[2];
    out.position[2] = clamp01((distance - 4.0f) / 8.0f);
    out.world[0] = x; out.world[1] = y; out.world[2] = z;
    out.material[0] = kind;
    return out;
}

void add_triangle(std::vector<BeautyVertex> &vertices, BeautyVertex const &a,
                  BeautyVertex const &b, BeautyVertex const &c) {
    vertices.push_back(a); vertices.push_back(b); vertices.push_back(c);
}

void add_draw(std::vector<BeautyVertex> const &vertices, unsigned constants,
              std::array<unsigned, 10> const &textures, unsigned depth_mode,
              unsigned blend_mode) {
    if (vertices.empty()) return;
    recorded.buffers.emplace_back(
        reinterpret_cast<std::uint8_t const *>(vertices.data()),
        reinterpret_cast<std::uint8_t const *>(vertices.data() + vertices.size()));
    labv2::Draw draw;
    draw.vertex_buffer = unsigned(recorded.buffers.size() - 1);
    draw.constant_buffer = constants;
    draw.count = unsigned(vertices.size());
    draw.stride = sizeof(BeautyVertex);
    draw.feature = 1;
    draw.depth_mode = depth_mode;
    draw.blend_mode = blend_mode;
    draw.attributes = {{3, 0}, {3, 12}, {3, 24}, {2, 36}, {4, 44}, {2, 60}};
    for (unsigned index = 0; index < textures.size(); ++index)
        draw.textures[index] = textures[index];
    recorded.draws.push_back(draw);
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
        bool trees_mode = fixture.find("beauty-trees") != std::string::npos;
        bool city_mode = fixture.find("beauty-city") != std::string::npos;
        bool warrior_mode = fixture.find("beauty-warrior") != std::string::npos;
        bool scene_mode = fixture.find("beauty-scene") != std::string::npos;
        if (unsigned(trees_mode) + unsigned(city_mode) + unsigned(warrior_mode) +
                unsigned(scene_mode) != 1)
            throw std::runtime_error("beauty object fixture mode missing or ambiguous");

        std::vector<BeautyMaterial> materials;
        std::vector<BeautyObject> objects;
        std::vector<BeautyRecipe> tree_recipes;
        if (!load_beauty_objects("Renderer/packs/BeautyStudies/beauty_objects.bin",
                                 materials, objects, tree_recipes))
            throw std::runtime_error("beauty object bundle load failed");

        ID3D11Device device;
        std::map<std::string, unsigned> loaded;
        auto texture = [&](std::string const &path) {
            if (path.empty()) return 0u;
            auto found = loaded.find(path);
            if (found != loaded.end()) return found->second;
            std::vector<std::uint8_t> dds;
            if (!read_file(path, dds) || dds.size() < 148)
                throw std::runtime_error("beauty texture missing: " + path);
            DXGI_FORMAT format = static_cast<DXGI_FORMAT>(read_u32(dds, 128));
            ID3D11ShaderResourceView *view = nullptr;
            unsigned texture_width = 0, texture_height = 0;
            if (!load_dds(&device, path, format, &view, texture_width, texture_height))
                throw std::runtime_error("beauty texture load failed: " + path);
            unsigned id = view->id;
            release(view);
            loaded[path] = id;
            return id;
        };

        std::array<unsigned, 10> common{};
        common[0] = texture("Renderer/packs/Civ5EnvironmentSkin/textures/grassland_base_color.dds");
        common[1] = texture("Renderer/packs/Civ5EnvironmentSkin/textures/grassland_height.dds");
        common[2] = texture("Renderer/packs/Civ5EnvironmentSkin/textures/grassland_specular.dds");
        for (BeautyMaterial &material : materials)
            for (unsigned channel = 0; channel < material.paths.size(); ++channel)
                material.textures[channel] = texture(material.paths[channel]);

        BeautyFrame frame = {};
        frame.sun[0] = -0.62f; frame.sun[1] = -0.42f; frame.sun[2] = 0.66f;
        normalize3(frame.sun); frame.sun[3] = 2.05f;
        // DEFAULT_LIGHTING noon is authored as RGB intensity 6.2/4.5/3.5.
        // Preserve that chromatic ratio while the Lab retains its bounded
        // source-independent radiance scale.
        frame.sun_color_exposure[0] = 1.0f;
        frame.sun_color_exposure[1] = 4.5f / 6.2f;
        frame.sun_color_exposure[2] = 3.5f / 6.2f;
        frame.sun_color_exposure[3] = 1.0f;
        frame.ambient[0] = 0.34f; frame.ambient[1] = 0.45f;
        frame.ambient[2] = 0.60f; frame.ambient[3] = 0.62f;
        frame.view[0] = 0.490290f; frame.view[1] = -0.735435f; frame.view[2] = 0.469979f;
        frame.quality[0] = 0.09f; frame.quality[1] = 0.82f;
        frame.quality[2] = 0.16f; frame.quality[3] = 1.0f;
        recorded.buffers.emplace_back(reinterpret_cast<std::uint8_t *>(&frame),
                                      reinterpret_cast<std::uint8_t *>(&frame + 1));
        unsigned constants = unsigned(recorded.buffers.size() - 1);

        float view_x = scene_mode ? 3.25f :
            (trees_mode ? 2.20f : (city_mode ? 2.20f : 0.78f));
        float view_y = scene_mode ? 2.30f :
            (trees_mode ? 1.65f : (city_mode ? 1.65f : 0.78f));
        float center_y = scene_mode ? 0.20f :
            (trees_mode ? 0.14f : (city_mode ? 0.10f : 0.30f));

        std::vector<BeautyVertex> ground;
        float plane = 10.0f;
        BeautyVertex a = project(-plane, -plane, 0, view_x, view_y, center_y, 0);
        BeautyVertex b = project( plane, -plane, 0, view_x, view_y, center_y, 0);
        BeautyVertex c = project( plane,  plane, 0, view_x, view_y, center_y, 0);
        BeautyVertex d = project(-plane,  plane, 0, view_x, view_y, center_y, 0);
        for (BeautyVertex *vertex : {&a, &b, &c, &d}) vertex->normal[2] = 1;
        add_triangle(ground, a, b, c); add_triangle(ground, a, c, d);
        // In the combined fixture the relief module already owns the identical
        // full-frame ground. Drawing it again would resolve equal-depth pixels
        // over the mountain's feathered foothills.
        if (!scene_mode) add_draw(ground, constants, common, 2, 0);

        if (scene_mode) {
            auto river_ribbon = [&](float half_width, float z, float kind) {
                std::vector<BeautyVertex> ribbon;
                std::array<std::array<float, 2>, 49> left{}, right{};
                for (unsigned index = 0; index < left.size(); ++index) {
                    float phase = float(index) / float(left.size() - 1);
                    float x = -4.4f + phase * 8.8f;
                    float y = -0.10f + 0.18f * std::sin(phase * 9.0f) +
                              0.08f * std::sin(phase * 19.0f);
                    float derivative = 1.62f * std::cos(phase * 9.0f) / 8.8f +
                                       1.52f * std::cos(phase * 19.0f) / 8.8f;
                    float inverse = 1.0f / std::sqrt(1.0f + derivative * derivative);
                    float nx = -derivative * inverse, ny = inverse;
                    left[index] = {x + nx * half_width, y + ny * half_width};
                    right[index] = {x - nx * half_width, y - ny * half_width};
                }
                for (unsigned index = 0; index + 1 < left.size(); ++index) {
                    BeautyVertex a = project(left[index][0], left[index][1], z,
                                             view_x, view_y, center_y, kind);
                    BeautyVertex b = project(right[index][0], right[index][1], z,
                                             view_x, view_y, center_y, kind);
                    BeautyVertex c = project(right[index + 1][0], right[index + 1][1], z,
                                             view_x, view_y, center_y, kind);
                    BeautyVertex d = project(left[index + 1][0], left[index + 1][1], z,
                                             view_x, view_y, center_y, kind);
                    for (BeautyVertex *vertex : {&a, &b, &c, &d}) vertex->normal[2] = 1;
                    add_triangle(ribbon, a, b, c); add_triangle(ribbon, a, c, d);
                }
                add_draw(ribbon, constants, common, 2, 0);
            };
            // Warm banks frame a narrow, sky-reflecting river and add the
            // large-scale color/value break missing from an all-grass tableau.
            river_ribbon(0.24f, 0.003f, 6.0f);
            river_ribbon(0.155f, 0.008f, 5.0f);
        }

        std::vector<Placement> placements;
        if (scene_mode) {
            unsigned city_indices[4] = {};
            unsigned warrior_indices[5] = {};
            unsigned city_count = 0, warrior_count = 0;
            for (unsigned index = 0; index < objects.size(); ++index) {
                if (objects[index].kind == 2 && city_count < 4)
                    city_indices[city_count++] = index;
                if (objects[index].kind == 3 && warrior_count < 5)
                    warrior_indices[warrior_count++] = index;
            }
            if (city_count != 4 || warrior_count != 5)
                throw std::runtime_error("combined scene source count drift");

            // Forest masses frame the built and military subjects without
            // falling back to a tile grid. They use only the two close-camera
            // vegetation sources proven by the isolated study.
            constexpr float golden = 2.39996323f;
            for (unsigned cluster = 0; cluster < 2; ++cluster) {
                float center_x = cluster ? 1.72f : -2.10f;
                float center_z = cluster ? 0.72f : -1.02f;
                for (unsigned index = 0; index < 11; ++index) {
                    float ring = std::sqrt(float(index) / 10.0f);
                    float angle = golden * index + cluster * 0.71f;
                    unsigned source = index % 7 == 0 ? 4u + cluster : 1u;
                    float scale = source >= 4 ? 0.42f :
                        0.72f + 0.055f * float((index * 5) % 5);
                    placements.push_back({source,
                        center_x + std::cos(angle) * ring * 0.68f,
                        center_z + std::sin(angle) * ring * 0.52f,
                        scale, angle * 0.37f});
                }
            }

            float city_placements[8][5] = {
                {0, 0.62f, 0.16f, 5.25f,  0.10f},
                {1, 1.04f, 0.12f, 5.05f, -0.12f},
                {2, 1.43f, 0.20f, 5.12f,  0.08f},
                {3, 0.68f, 0.58f, 5.02f, -0.08f},
                {2, 1.10f, 0.54f, 5.28f,  0.16f},
                {1, 1.49f, 0.64f, 5.05f, -0.14f},
                {3, 0.88f, 0.94f, 4.88f,  0.06f},
                {0, 1.30f, 0.98f, 5.12f, -0.04f},
            };
            for (unsigned index = 0; index < 8; ++index) {
                unsigned source = unsigned(city_placements[index][0]);
                placements.push_back({city_indices[source], city_placements[index][1],
                    city_placements[index][2], city_placements[index][3],
                    city_placements[index][4]});
            }

            // Three complete Warrior figures share the baked idle pose while
            // varying screen placement, scale, and heading like a Civ army.
            float warrior_placements[3][4] = {
                { 0.12f, -1.02f, 3.45f, -0.46f},
                { 0.51f, -0.86f, 3.30f, -0.31f},
                {-0.22f, -0.70f, 3.18f, -0.58f},
            };
            for (unsigned member = 0; member < 3; ++member)
                for (unsigned part = 0; part < warrior_count; ++part)
                    placements.push_back({warrior_indices[part],
                        warrior_placements[member][0], warrior_placements[member][1],
                        warrior_placements[member][2], warrior_placements[member][3]});
        } else if (trees_mode) {
            constexpr float golden = 2.39996323f;
            unsigned recipe_weight = 0;
            for (BeautyRecipe const &recipe : tree_recipes)
                recipe_weight += recipe.count;
            if (recipe_weight == 0)
                throw std::runtime_error("forest recipe has no authored weight");
            // ArtDef Count is the only authored variant-frequency signal. The
            // engine's exact scatter transforms are not serialized, so this
            // study applies those weights, scales, variations, and RotateZ
            // semantics to a deterministic low-discrepancy patch.
            for (unsigned index = 0; index < 36; ++index) {
                unsigned selected = (index * 73 + 19) % recipe_weight;
                BeautyRecipe const *recipe = nullptr;
                for (BeautyRecipe const &candidate : tree_recipes) {
                    if (selected < candidate.count) { recipe = &candidate; break; }
                    selected -= candidate.count;
                }
                if (!recipe) throw std::runtime_error("forest recipe selection failed");
                float ring = std::sqrt((float(index) + 0.5f) / 36.0f);
                float angle = golden * index;
                float signed_jitter = float((index * 37 + 11) % 101) / 50.0f - 1.0f;
                float scale = recipe->scale *
                    (1.0f + recipe->scale_variation * signed_jitter);
                placements.push_back({recipe->object,
                    std::cos(angle) * ring * 1.12f,
                    std::sin(angle) * ring * 0.80f,
                    scale,
                    angle * 0.73f});
            }
        } else if (city_mode) {
            unsigned city_indices[4] = {};
            unsigned count = 0;
            for (unsigned index = 0; index < objects.size() && count < 4; ++index)
                if (objects[index].kind == 2) city_indices[count++] = index;
            if (count != 4) throw std::runtime_error("city source count drift");
            float city_placements[8][5] = {
                {0, -0.48f, -0.30f, 6.25f,  0.10f},
                {1,  0.02f, -0.34f, 6.05f, -0.12f},
                {2,  0.48f, -0.20f, 6.15f,  0.08f},
                {3, -0.48f,  0.20f, 6.00f, -0.08f},
                {2,  0.00f,  0.18f, 6.30f,  0.16f},
                {1,  0.48f,  0.28f, 6.05f, -0.14f},
                {3, -0.22f,  0.62f, 5.85f,  0.06f},
                {0,  0.30f,  0.67f, 6.15f, -0.04f},
            };
            for (unsigned index = 0; index < 8; ++index) {
                unsigned source = unsigned(city_placements[index][0]);
                placements.push_back({city_indices[source], city_placements[index][1],
                    city_placements[index][2], city_placements[index][3],
                    city_placements[index][4]});
            }
        } else {
            for (unsigned index = 0; index < objects.size(); ++index)
                if (objects[index].kind == 3)
                    placements.push_back({index, 0.0f, 0.0f, 6.9f, -0.46f});
        }

        std::vector<std::vector<BeautyVertex>> shadow_batches(objects.size());
        std::vector<std::vector<BeautyVertex>> batches(objects.size());
        float sun_horizontal = std::max(0.05f, frame.sun[2]);
        float cast_x = -frame.sun[0] / sun_horizontal;
        float cast_y = -frame.sun[1] / sun_horizontal;
        for (Placement const &placement : placements) {
            BeautyObject const &object = objects[placement.object];
            BeautyMaterial const &material = materials[object.material];
            float cosine = std::cos(placement.rotation), sine = std::sin(placement.rotation);
            std::vector<BeautyVertex> transformed;
            transformed.reserve(object.vertices.size());
            for (FeatureSourceVertex const &source : object.vertices) {
                float local_x = (source.position[0] * cosine - source.position[1] * sine) * placement.scale;
                float local_y = (source.position[0] * sine + source.position[1] * cosine) * placement.scale;
                float local_z = source.position[2] * placement.scale;
                BeautyVertex vertex = project(placement.x + local_x, placement.y + local_y,
                                              local_z, view_x, view_y, center_y,
                                              float(object.kind));
                vertex.normal[0] = source.normal[0] * cosine - source.normal[1] * sine;
                vertex.normal[1] = source.normal[0] * sine + source.normal[1] * cosine;
                vertex.normal[2] = source.normal[2];
                vertex.uv[0] = source.uv[0]; vertex.uv[1] = source.uv[1];
                // Source-authored packed normals now drive foliage and units.
                // The paired LEAN moments remain bound for audit, but their
                // BRDF decoding is not yet proven and must not be treated as
                // an XYZ normal for those assets. Bit 1 carries the source
                // texture address mode without widening the vertex wire.
                vertex.material[1] = (object.kind == 1 || object.kind == 3 ||
                    material.paths[1].empty() ? 0.0f : 1.0f) +
                    (material.repeat ? 2.0f : 0.0f);
                vertex.material[2] = material.paths[3].empty() ? 0.0f : 1.0f;
                vertex.material[3] = material.paths[4].empty() ? 0.0f : 1.0f;
                vertex.secondary[0] = float(material.owner_tint);
                vertex.secondary[1] = object.kind == 1
                    ? (material.paths[6].empty() ? 0.0f : 1.0f)
                    : (material.paths[5].empty() ? 0.0f : 1.0f);
                transformed.push_back(vertex);
            }
            batches[placement.object].insert(batches[placement.object].end(),
                                             transformed.begin(), transformed.end());

            // Three lightly offset projections of light-facing source triangles
            // give each object a soft, shape-preserving ground shadow.
            for (std::size_t index = 0; index + 2 < transformed.size(); index += 3) {
                float facing = 0;
                for (unsigned corner = 0; corner < 3; ++corner)
                    facing += transformed[index + corner].normal[0] * frame.sun[0] +
                              transformed[index + corner].normal[1] * frame.sun[1] +
                              transformed[index + corner].normal[2] * frame.sun[2];
                if (facing <= 0) continue;
                for (int sample = -1; sample <= 1; ++sample) {
                    BeautyVertex projected[3];
                    for (unsigned corner = 0; corner < 3; ++corner) {
                        BeautyVertex const &source = transformed[index + corner];
                        float offset = float(sample) * 0.012f;
                        float x = source.world[0] + cast_x * source.world[2] - cast_y * offset;
                        float y = source.world[1] + cast_y * source.world[2] + cast_x * offset;
                        projected[corner] = project(x, y, 0.004f, view_x, view_y,
                                                    center_y, 4.0f);
                        projected[corner].normal[2] = 1;
                        projected[corner].uv[0] = source.uv[0];
                        projected[corner].uv[1] = source.uv[1];
                        projected[corner].secondary[0] = object.kind == 1 ? 0.075f :
                            (object.kind == 3 ? 0.075f : 0.052f);
                        projected[corner].secondary[1] = source.secondary[1];
                    }
                    add_triangle(shadow_batches[placement.object], projected[0],
                                 projected[1], projected[2]);
                }
            }
        }

        for (unsigned index = 0; index < objects.size(); ++index) {
            if (shadow_batches[index].empty()) continue;
            std::array<unsigned, 10> bindings = common;
            BeautyMaterial const &material = materials[objects[index].material];
            for (unsigned channel = 0; channel < material.textures.size(); ++channel)
                bindings[3 + channel] = material.textures[channel];
            add_draw(shadow_batches[index], constants, bindings, 1, 1);
        }

        for (unsigned index = 0; index < objects.size(); ++index) {
            if (batches[index].empty()) continue;
            std::array<unsigned, 10> bindings = common;
            BeautyMaterial const &material = materials[objects[index].material];
            for (unsigned channel = 0; channel < material.textures.size(); ++channel)
                bindings[3 + channel] = material.textures[channel];
            add_draw(batches[index], constants, bindings, 2, 0);
        }

        recorded.width = width;
        recorded.height = height;
        recorded.downsample = 1;
        recorded.color_branch = 1;
        recorded.valid_rect = {0, 0, width, height};
        recorded.exposure = 1.0f;
        return labv2::write_packet(argv[1], recorded) ? 0 : 1;
    } catch (std::exception const &error) {
        std::fprintf(stderr, "beauty_objects: %s\n", error.what());
        return 1;
    }
}

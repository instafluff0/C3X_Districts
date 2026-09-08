// Generated selected r13 numerical kernels; see prepare.py and provenance.json.
#pragma once
namespace c3x_renderer { namespace fidelity {
struct Tile { int source_x,source_y,column,row,real; };
using BiqWindowTile=Tile;
struct Hill {
    float x, y, radius_x, radius_y, height, angle, source_u, source_v;
    float rockiness;
    std::uint32_t seed;
};

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

std::uint32_t composed_seed(BiqWindowTile const &tile) {
    return std::uint32_t(tile.source_x * 0x193u) ^
           std::uint32_t(tile.source_y * 0x217u) ^ 0x8d31u;
}

Hill composed_hill(BiqWindowTile const &tile) {
    std::uint32_t state = composed_seed(tile);
    float angle = random01(state) * 6.283185307f;
    // Hills must read as terrain, not small decals: broad neighboring bodies
    // overlap into rolling chains while remaining far below the 165-unit
    // authored mountain silhouettes.
    float radius_x = 0.74f + random01(state) * 0.12f;
    float radius_y = 0.62f + random01(state) * 0.10f;
    float height = 48.0f + random01(state) * 12.0f;
    float source_u = random01(state);
    float source_v = random01(state);
    float rockiness = 0.62f + random01(state) * 0.34f;
    return {float(tile.column) + 0.5f, float(tile.row) + 0.5f,
            radius_x, radius_y, height, angle, source_u, source_v,
            rockiness, state};
}

template<class HeightField>
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

std::uint32_t mountain_seed(BiqWindowTile const &tile) {
    return std::uint32_t(tile.source_x * 0x193u) ^
           std::uint32_t(tile.source_y * 0x217u) ^ 0x6b91u;
}
} }

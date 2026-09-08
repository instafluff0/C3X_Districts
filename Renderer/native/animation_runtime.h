#ifndef C3X_ANIMATION_RUNTIME_H
#define C3X_ANIMATION_RUNTIME_H

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#include "terrain_scene_runtime.h"

namespace c3x_renderer {

// Generic, offline-bound skin palettes. No source-engine formats, game state,
// clocks, or terrain-cache keys belong here. Each payload is one material part.
struct AnimationVertex {
    FeatureSourceVertex source;
    std::array<std::uint32_t, 4> joints;
    std::array<float, 4> weights;
};

struct AnimationMesh {
    float duration = 0;
    std::uint32_t bones = 0, frames = 0;
    std::vector<AnimationVertex> vertices;
    std::vector<std::uint32_t> indices;
    std::vector<float> palettes;
};

inline bool decode_animation_mesh(std::vector<std::uint8_t> const & data,
                                  AnimationMesh & output) {
    // Validate the complete byte budget before allocating any count-sized array.
    if (data.size() < 32 || data.size() > 64u * 1024u * 1024u ||
        std::memcmp(data.data(), "C3XANM1\0", 8) != 0) return false;
    std::size_t cursor = 8;
    auto u32 = [&]() {
        std::uint32_t value = 0;
        for (unsigned i = 0; i < 4; ++i) value |= std::uint32_t(data[cursor++]) << (8 * i);
        return value;
    };
    auto f32 = [&]() { auto bits = u32(); float value; std::memcpy(&value, &bits, 4); return value; };
    if (u32() != 1) return false;
    auto vertex_count = u32(), index_count = u32(), bones = u32(), frames = u32();
    float duration = f32();
    if (!vertex_count || vertex_count > 65536 || !index_count || index_count > 393216 ||
        index_count % 3 || !bones || bones > 256 || frames < 2 || frames > 4096 ||
        !std::isfinite(duration) || duration <= 0 || duration > 3600) return false;
    std::uint64_t expected = 32ull + vertex_count * 64ull + index_count * 4ull +
        std::uint64_t(bones) * frames * 64ull;
    if (expected != data.size()) return false;
    AnimationMesh decoded;
    decoded.duration = duration; decoded.bones = bones; decoded.frames = frames;
    decoded.vertices.resize(vertex_count);
    decoded.indices.resize(index_count);
    decoded.palettes.resize(std::size_t(bones) * frames * 16);
    for (auto & vertex : decoded.vertices) {
        for (auto & v : vertex.source.position) { v = f32(); if (!std::isfinite(v)) return false; }
        for (auto & v : vertex.source.normal) { v = f32(); if (!std::isfinite(v)) return false; }
        for (auto & v : vertex.source.uv) { v = f32(); if (!std::isfinite(v)) return false; }
        for (auto & j : vertex.joints) { j = u32(); if (j >= bones) return false; }
        float sum = 0;
        for (auto & w : vertex.weights) {
            w = f32(); if (!std::isfinite(w) || w < 0 || w > 1) return false; sum += w;
        }
        if (std::abs(sum - 1.0f) > 0.00001f) return false;
    }
    for (auto & index : decoded.indices) { index = u32(); if (index >= vertex_count) return false; }
    for (auto & v : decoded.palettes) { v = f32(); if (!std::isfinite(v)) return false; }
    for (std::size_t i = 0; i < decoded.palettes.size(); i += 16) {
        auto p = decoded.palettes.data() + i;
        if (std::abs(p[3]) > 0.0001f || std::abs(p[7]) > 0.0001f ||
            std::abs(p[11]) > 0.0001f || std::abs(p[15] - 1) > 0.0001f) return false;
    }
    output = std::move(decoded); // A rejected payload never alters the live asset.
    return true;
}

// Returns absolute phase, not an accumulated delta. Camera changes, dropped
// redraws and repeated unit callbacks must not restart or slow a clip.
inline double ambient_animation_time(std::int64_t ticks, std::int64_t frequency,
                                     double duration, std::uint32_t seed) {
    if (ticks < 0 || frequency <= 0 || !std::isfinite(duration) || duration <= 0) return 0;
    double seconds = double(ticks / frequency) + double(ticks % frequency) / double(frequency);
    double offset = double(seed) / 4294967296.0 * duration;
    return std::fmod(std::fmod(seconds, duration) + offset, duration);
}

// Quantize source-time ambient playback to its authored pose-cache frames.
// The duration remains the source duration; Civ III cursor counts are irrelevant.
inline std::uint32_t ambient_animation_frame(std::int64_t ticks, std::int64_t frequency,
                                             double duration, std::uint32_t frames,
                                             std::uint32_t seed) {
    if (frames < 2) return 0;
    double time = ambient_animation_time(ticks, frequency, duration, seed);
    return std::min(frames - 1, static_cast<std::uint32_t>(
        time / duration * static_cast<double>(frames - 1)));
}

inline bool sample_animation_mesh(AnimationMesh const & mesh, double seconds, bool loop,
                                   std::vector<FeatureSourceVertex> & output) {
    if (!std::isfinite(seconds) || mesh.frames < 2 || !mesh.bones || mesh.bones > 256 ||
        mesh.duration <= 0 || mesh.palettes.size() != std::size_t(mesh.frames) * mesh.bones * 16)
        return false;
    double time = loop ? std::fmod(std::fmod(seconds, mesh.duration) + mesh.duration, mesh.duration) :
        std::clamp(seconds, 0.0, double(mesh.duration));
    double frame = time / mesh.duration * (mesh.frames - 1);
    auto first = std::min(mesh.frames - 1, static_cast<std::uint32_t>(frame));
    auto second = std::min(mesh.frames - 1, first + 1);
    float fraction = static_cast<float>(frame - first);
    std::array<std::array<float, 16>, 256> poses{};
    std::array<std::array<float, 9>, 256> normals{};
    for (std::uint32_t bone = 0; bone < mesh.bones; ++bone) {
        auto & p = poses[bone];
        auto a = mesh.palettes.data() + (std::size_t(first) * mesh.bones + bone) * 16;
        auto b = mesh.palettes.data() + (std::size_t(second) * mesh.bones + bone) * 16;
        for (unsigned j = 0; j < 16; ++j) p[j] = a[j] + (b[j] - a[j]) * fraction;
        // Inverse transpose preserves lighting under animated scale/shear.
        auto & n = normals[bone];
        n = {p[5]*p[10]-p[6]*p[9], p[6]*p[8]-p[4]*p[10], p[4]*p[9]-p[5]*p[8],
             p[2]*p[9]-p[1]*p[10], p[0]*p[10]-p[2]*p[8], p[1]*p[8]-p[0]*p[9],
             p[1]*p[6]-p[2]*p[5], p[2]*p[4]-p[0]*p[6], p[0]*p[5]-p[1]*p[4]};
        float determinant = p[0]*n[0] + p[1]*n[1] + p[2]*n[2];
        if (!std::isfinite(determinant)) return false;
        // Authored visibility tracks collapse some parts to zero scale. Their
        // positions must still collapse; a singular normal must not reject the
        // entire kit (including unused inventory helper bones).
        if (std::abs(determinant) >= 1e-12f)
            for (auto & v : n) v /= determinant;
    }
    output.resize(mesh.vertices.size());
    for (std::size_t i = 0; i < mesh.vertices.size(); ++i) {
        auto const & input = mesh.vertices[i];
        auto & vertex = output[i]; vertex = {};
        for (unsigned influence = 0; influence < 4; ++influence) {
            float weight = input.weights[influence]; if (weight == 0) continue;
            if (input.joints[influence] >= mesh.bones || !std::isfinite(weight)) return false;
            auto const & p = poses[input.joints[influence]];
            auto const & n = normals[input.joints[influence]];
            for (unsigned axis = 0; axis < 3; ++axis) {
                vertex.position[axis] += weight * (input.source.position[0]*p[axis] +
                    input.source.position[1]*p[4+axis] + input.source.position[2]*p[8+axis] + p[12+axis]);
                vertex.normal[axis] += weight * (input.source.normal[0]*n[axis] +
                    input.source.normal[1]*n[3+axis] + input.source.normal[2]*n[6+axis]);
            }
        }
        float length = std::sqrt(vertex.normal[0]*vertex.normal[0] +
            vertex.normal[1]*vertex.normal[1] + vertex.normal[2]*vertex.normal[2]);
        for (unsigned axis = 0; axis < 3; ++axis)
            vertex.normal[axis] = length > 1e-12f ? vertex.normal[axis]/length : input.source.normal[axis];
        for (unsigned axis = 0; axis < 3; ++axis)
            if (!std::isfinite(vertex.position[axis]) || !std::isfinite(vertex.normal[axis])) return false;
        std::copy(input.source.uv, input.source.uv + 2, vertex.uv);
    }
    return true;
}

} // namespace c3x_renderer
#endif

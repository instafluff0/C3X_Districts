#pragma once
// Runtime reader for flat effect packs (`effects.bin`), written by
// `tools/asset_compiler/effect_graph_compiler.write_runtime_pack`.
// Every count, index, string and curve range is bounds-checked; a malformed
// pack is rejected as a whole and the caller keeps native effects.
#include "effect_sampler.h"
#include <cstring>

namespace c3x_renderer::effects {

struct Texture {
    std::string path; // relative to the pack root
    unsigned width = 0, height = 0, dxgi_format = 0;
};

struct Pack {
    std::vector<Texture> textures;
    std::vector<Profile> profiles;
    Profile const* find(std::string_view id) const {
        for (auto const& profile : profiles) if (profile.id == id) return &profile;
        return nullptr;
    }
};

inline bool parse_pack(std::uint8_t const* data, std::size_t size, Pack& out) {
    constexpr std::size_t texture_bytes = 20, profile_bytes = 64, emitter_bytes = 152;
    if (!data || size < 32 || std::memcmp(data, "C3XFX1\0\0", 8) != 0) return false;
    std::uint32_t header[6];
    std::memcpy(header, data + 8, sizeof header);
    std::uint32_t const version = header[0], texture_count = header[1], profile_count = header[2],
                        emitter_count = header[3], point_count = header[4], string_bytes = header[5];
    if (version != 1 || texture_count > 4096 || profile_count > 4096 || emitter_count > 65536 ||
        point_count > 1u << 20 || string_bytes > 1u << 24) return false;
    std::size_t textures_at = 32, profiles_at = textures_at + texture_count * texture_bytes;
    std::size_t emitters_at = profiles_at + profile_count * profile_bytes;
    std::size_t points_at = emitters_at + std::size_t(emitter_count) * emitter_bytes;
    std::size_t strings_at = points_at + std::size_t(point_count) * 8;
    if (strings_at + string_bytes != size) return false;
    auto u32 = [&](std::size_t at) { std::uint32_t v; std::memcpy(&v, data + at, 4); return v; };
    auto f32 = [&](std::size_t at) { float v; std::memcpy(&v, data + at, 4); return v; };
    auto finite = [](double v) { return std::isfinite(v); };
    auto text = [&](std::size_t at, std::string& value) {
        std::uint32_t offset = u32(at), length = u32(at + 4);
        if (offset > string_bytes || length > string_bytes - offset || length == 0) return false;
        value.assign(reinterpret_cast<char const*>(data + strings_at + offset), length);
        return true;
    };
    auto curve = [&](std::size_t at, Curve& value, bool required) {
        std::uint32_t first = u32(at), count = u32(at + 4);
        if (count == 0) return !required;
        if (count < 2 || first > point_count || count > point_count - first) return false;
        value.points.resize(count);
        for (std::uint32_t i = 0; i < count; ++i) {
            value.points[i] = {f32(points_at + (first + i) * 8), f32(points_at + (first + i) * 8 + 4)};
            if (!finite(value.points[i][0]) || !finite(value.points[i][1]) || value.points[i][1] < 0 ||
                (i && value.points[i][0] < value.points[i - 1][0])) return false;
        }
        return value.points.front()[0] == 0 && value.points.back()[0] == 1;
    };
    Pack pack;
    pack.textures.resize(texture_count);
    for (std::uint32_t i = 0; i < texture_count; ++i) {
        std::size_t at = textures_at + i * texture_bytes;
        auto& texture = pack.textures[i];
        if (!text(at, texture.path) || texture.path.find("..") != std::string::npos) return false;
        texture.width = u32(at + 8); texture.height = u32(at + 12); texture.dxgi_format = u32(at + 16);
        if (!texture.width || !texture.height || texture.width > 16384 || texture.height > 16384) return false;
    }
    std::vector<Emitter> all(emitter_count);
    for (std::uint32_t i = 0; i < emitter_count; ++i) {
        std::size_t at = emitters_at + std::size_t(i) * emitter_bytes;
        auto& e = all[i];
        std::uint32_t blend = u32(at + 16), rotation = u32(at + 36);
        e.texture = u32(at + 8); e.alpha_texture = u32(at + 12);
        e.columns = u32(at + 20); e.rows = u32(at + 24); e.frames = u32(at + 28); e.variant_frames = u32(at + 32) != 0;
        e.rate_per_second = f32(at + 40); e.lifetime_ms = f32(at + 44); e.start_ms = f32(at + 48);
        e.burst_spread_ms = f32(at + 52);
        std::int32_t burst; std::memcpy(&burst, data + at + 56, 4); e.burst = burst;
        e.max_particles = u32(at + 60);
        for (int c = 0; c < 2; ++c) e.size[c] = f32(at + 64 + 4 * c);
        for (int c = 0; c < 3; ++c) e.velocity[c] = f32(at + 72 + 4 * c);
        for (int c = 0; c < 4; ++c) e.spread[c] = f32(at + 84 + 4 * c);
        for (int c = 0; c < 3; ++c) e.tint[c] = f32(at + 100 + 4 * c);
        e.spawn_radius = f32(at + 112); e.gravity = f32(at + 116); e.spin_per_second = f32(at + 120); e.intensity = f32(at + 124);
        e.pivot[0] = f32(at + 144); e.pivot[1] = f32(at + 148);
        if (!(e.pivot[0] >= 0 && e.pivot[0] <= 1 && e.pivot[1] >= 0 && e.pivot[1] <= 1)) return false;
        if (!text(at, e.id) || e.texture >= texture_count || (e.alpha_texture != ~0u && e.alpha_texture >= texture_count) ||
            blend > 2 || rotation > 2 || !e.columns || !e.rows || !e.frames || e.columns > 64 || e.rows > 64 ||
            e.frames > e.columns * e.rows || !(e.rate_per_second > 0) || !(e.lifetime_ms > 0) || e.start_ms < 0 ||
            e.burst_spread_ms < 0 || burst < 0 || !e.max_particles || e.max_particles > 4096 ||
            unsigned(burst) > e.max_particles || !(e.intensity > 0) || e.spawn_radius < 0 ||
            !curve(at + 128, e.opacity, true) || !curve(at + 136, e.size_curve, false)) return false;
        e.blend = Blend(blend); e.rotation = Rotation(rotation);
        for (double v : {e.rate_per_second, e.lifetime_ms, e.start_ms, e.burst_spread_ms, e.spawn_radius, e.gravity,
                         e.spin_per_second, e.intensity, double(e.size[0]), double(e.size[1]), double(e.velocity[0]),
                         double(e.velocity[1]), double(e.velocity[2]), double(e.spread[0]), double(e.spread[1]),
                         double(e.spread[2]), double(e.spread[3]), double(e.tint[0]), double(e.tint[1]), double(e.tint[2])})
            if (!finite(v)) return false;
    }
    pack.profiles.resize(profile_count);
    for (std::uint32_t i = 0; i < profile_count; ++i) {
        std::size_t at = profiles_at + i * profile_bytes;
        auto& profile = pack.profiles[i];
        std::uint32_t first = u32(at + 32), count = u32(at + 36);
        profile.loop = u32(at + 8) != 0; profile.duration_ms = f32(at + 12);
        for (int z = 0; z < 2; ++z) { profile.density[z] = f32(at + 16 + 4 * z); profile.size_scale[z] = f32(at + 24 + 4 * z); }
        profile.repeat_count = u32(at + 40); profile.repeat_interval_ms = f32(at + 44);
        for (int c = 0; c < 3; ++c) profile.repeat_spacing[c] = f32(at + 48 + 4 * c);
        profile.repeat_jitter = f32(at + 60);
        if (!text(at, profile.id) || !(profile.duration_ms > 0) || !finite(profile.duration_ms) || !count ||
            first > emitter_count || count > emitter_count - first || profile.repeat_count > 16 ||
            !(profile.repeat_interval_ms >= 0) || !(profile.repeat_jitter >= 0) || !finite(profile.repeat_interval_ms) ||
            !finite(profile.repeat_jitter) || !finite(profile.repeat_spacing[0]) || !finite(profile.repeat_spacing[1]) ||
            !finite(profile.repeat_spacing[2])) return false;
        for (int z = 0; z < 2; ++z)
            if (!(profile.density[z] > 0) || !(profile.size_scale[z] > 0) || !finite(profile.density[z]) || !finite(profile.size_scale[z]))
                return false;
        profile.emitters.assign(all.begin() + first, all.begin() + first + count);
    }
    out = std::move(pack);
    return true;
}

} // namespace c3x_renderer::effects

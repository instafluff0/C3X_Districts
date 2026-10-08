#pragma once
// Deterministic sprite-effect sampling (generic effect graphs).
//
// Mirrors `tools/asset_compiler/effect_graph_compiler.sample_effect`: every
// particle of an event is a pure function of the profile, the event's key and
// the event age, so skipped frames, replays and device resets need no
// simulation state. Positions are in the effect's own tile frame (x forward
// along the event direction, y lateral, z up); callers rotate and project.
#include <array>
#include <cmath>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace c3x_renderer::effects {

inline std::uint32_t fnv1a(std::string_view text, std::uint32_t value = 0x811C9DC5u) {
    for (unsigned char byte : text) value = (value ^ byte) * 0x01000193u;
    return value;
}
inline std::uint32_t mix32(std::uint32_t value) {
    value ^= value >> 16; value *= 0x7FEB352Du;
    value ^= value >> 15; value *= 0x846CA68Bu;
    return value ^ (value >> 16);
}
inline double random01(std::uint32_t key, std::uint32_t ordinal, std::uint32_t lane) {
    return mix32(key ^ mix32(ordinal * 0x9E3779B1u + lane * 0x85EBCA6Bu)) / 4294967296.0;
}

struct Curve {
    std::vector<std::array<float, 2>> points; // (normalized age, value), age 0..1 ascending
    double sample(double phase) const {
        for (std::size_t i = 1; i < points.size(); ++i) {
            auto const& left = points[i - 1];
            auto const& right = points[i];
            if (phase <= right[0]) {
                double span = right[0] - left[0];
                double mix = span <= 0 ? 0.0 : (phase - left[0]) / span;
                return left[1] + (right[1] - left[1]) * mix;
            }
        }
        return points.empty() ? 1.0 : points.back()[1];
    }
};

enum class Blend : std::uint8_t { alpha, additive, premultiplied };
enum class Rotation : std::uint8_t { none, random, emitter };

struct Emitter {
    std::string id;
    unsigned texture = 0, alpha_texture = ~0u; // indices into the pack's texture table
    Blend blend = Blend::alpha;
    unsigned columns = 1, rows = 1, frames = 1;
    bool variant_frames = false;               // frame_mode "variant": one random frame per particle
    Rotation rotation = Rotation::none;
    double rate_per_second = 1, lifetime_ms = 1, start_ms = 0, burst_spread_ms = 0;
    int burst = 0;                             // 0: continuous at rate_per_second
    unsigned max_particles = 1;
    float size[2] = {1, 1}, velocity[3] = {}, spread[4] = {}, tint[3] = {1, 1, 1};
    float pivot[2] = {.5f, .5f};               // sprite-space anchor (u, v) placed at the particle position
    double spawn_radius = 0, gravity = 0, spin_per_second = 0, intensity = 1;
    Curve opacity, size_curve;                 // empty size_curve: constant size
};

struct Profile {
    std::string id;
    bool loop = false;
    double duration_ms = 1;
    std::vector<Emitter> emitters;
    float density[2] = {1, 1}, size_scale[2] = {1, 1}; // normal, reduced zoom
    // A stick or salvo: `repeat_count` copies spaced along the event direction
    // (centered), scattered within `repeat_jitter`, `repeat_interval_ms` apart.
    unsigned repeat_count = 0;
    float repeat_interval_ms = 0, repeat_spacing[3] = {}, repeat_jitter = 0;
};

struct Particle {
    unsigned emitter = 0, ordinal = 0, frame = 0;
    float position[3] = {}, size[2] = {}, atlas[4] = {}, tint[3] = {1, 1, 1}, pivot[2] = {.5f, .5f};
    float opacity = 0, intensity = 1, rotation = 0;
    bool emitter_oriented = false;
};

// One copy of every emitter at `time_ms` after that copy's start.
inline void sample_once(Profile const& profile, std::string_view instance_id, long long time_ms, bool reduced,
                        std::vector<Particle>& out) {
    constexpr double tau = 6.283185307179586;
    double density = profile.density[reduced ? 1 : 0], size_scale = profile.size_scale[reduced ? 1 : 0];
    constexpr std::string_view separator("\0", 1);
    std::uint32_t prefix = fnv1a(separator, fnv1a(instance_id, fnv1a(separator, fnv1a(profile.id))));
    for (unsigned e = 0; e < profile.emitters.size(); ++e) {
        auto const& emitter = profile.emitters[e];
        double local = double(time_ms) - emitter.start_ms;
        if (local < 0) continue;
        std::uint32_t key = fnv1a(emitter.id, prefix);
        auto spawned = [&](unsigned ordinal, double spawn) {
            double age = local - spawn;
            if (age < 0 || age >= emitter.lifetime_ms) return;
            double phase = age / emitter.lifetime_ms, seconds = age / 1000.0;
            double angle = random01(key, ordinal, 0) * tau;
            double radius = std::sqrt(random01(key, ordinal, 1)) * emitter.spawn_radius;
            double outward = emitter.spread[0] + (emitter.spread[1] - emitter.spread[0]) * random01(key, ordinal, 2);
            double rise = emitter.spread[2] + (emitter.spread[3] - emitter.spread[2]) * random01(key, ordinal, 3);
            unsigned frame = emitter.variant_frames ? unsigned(random01(key, ordinal, 4) * emitter.frames)
                                                    : unsigned(phase * emitter.frames);
            if (frame > emitter.frames - 1) frame = emitter.frames - 1;
            Particle p;
            p.emitter = e; p.ordinal = ordinal; p.frame = frame;
            unsigned column = frame % emitter.columns, row = frame / emitter.columns;
            p.atlas[0] = float(column) / emitter.columns; p.atlas[1] = float(row) / emitter.rows;
            p.atlas[2] = float(column + 1) / emitter.columns; p.atlas[3] = float(row + 1) / emitter.rows;
            p.opacity = float(emitter.opacity.sample(phase));
            p.intensity = float(emitter.intensity);
            for (int c = 0; c < 3; ++c) p.tint[c] = emitter.tint[c];
            p.pivot[0] = emitter.pivot[0]; p.pivot[1] = emitter.pivot[1];
            p.rotation = float((emitter.rotation == Rotation::random ? random01(key, ordinal, 5) * tau : 0.0) +
                               emitter.spin_per_second * (random01(key, ordinal, 6) * 2 - 1) * seconds);
            p.emitter_oriented = emitter.rotation == Rotation::emitter;
            double reach = radius + outward * seconds;
            p.position[0] = float(std::cos(angle) * reach + emitter.velocity[0] * seconds);
            p.position[1] = float(std::sin(angle) * reach + emitter.velocity[1] * seconds);
            p.position[2] = float(std::max(0.0, (emitter.velocity[2] + rise) * seconds +
                                                   .5 * emitter.gravity * seconds * seconds));
            double size = emitter.size_curve.points.empty() ? 1.0 : emitter.size_curve.sample(phase);
            p.size[0] = float(emitter.size[0] * size_scale * size);
            p.size[1] = float(emitter.size[1] * size_scale * size);
            out.push_back(p);
        };
        if (emitter.burst > 0) {
            // A one-shot emitter spawns its whole burst at its start, spread over
            // burst_spread_ms; zoom density thins the burst.
            unsigned count = unsigned(std::max(1.0, std::nearbyint(emitter.burst * density)));
            for (unsigned ordinal = 0; ordinal < count; ++ordinal)
                spawned(ordinal, random01(key, ordinal, 7) * emitter.burst_spread_ms);
        } else {
            double interval = 1000.0 / (emitter.rate_per_second * density);
            long long first = std::max(0LL, (long long)std::floor((local - emitter.lifetime_ms) / interval) + 1);
            long long last = (long long)std::floor(local / interval);
            if (last - first + 1 > (long long)emitter.max_particles) first = last - emitter.max_particles + 1;
            for (long long ordinal = first; ordinal <= last; ++ordinal)
                spawned(unsigned(ordinal), ordinal * interval);
        }
    }
}

// Appends the live particles of one event at `time_ms` after its start.
inline void sample(Profile const& profile, std::string_view instance_id, long long time_ms, bool reduced,
                   std::vector<Particle>& out) {
    constexpr double tau = 6.283185307179586;
    if (time_ms < 0 || (!profile.loop && time_ms >= profile.duration_ms)) return;
    if (!profile.repeat_count) { sample_once(profile, instance_id, time_ms, reduced, out); return; }
    constexpr std::string_view separator("\0", 1);
    std::string id(instance_id);
    std::uint32_t jitter_key = fnv1a("jitter", fnv1a(separator, fnv1a(id + "#repeat", fnv1a(separator, fnv1a(profile.id)))));
    for (unsigned copy = 0; copy < profile.repeat_count; ++copy) {
        double angle = random01(jitter_key, copy, 0) * tau;
        double radius = std::sqrt(random01(jitter_key, copy, 1)) * profile.repeat_jitter;
        double centered = copy - (profile.repeat_count - 1) / 2.0;
        double offset[3] = {centered * profile.repeat_spacing[0] + std::cos(angle) * radius,
                            centered * profile.repeat_spacing[1] + std::sin(angle) * radius,
                            centered * profile.repeat_spacing[2]};
        long long local = time_ms - (long long)std::nearbyint(copy * double(profile.repeat_interval_ms));
        if (local < 0) continue;
        std::size_t first = out.size();
        sample_once(profile, id + "#" + std::to_string(copy), local, reduced, out);
        for (std::size_t i = first; i < out.size(); ++i) {
            for (int c = 0; c < 3; ++c) out[i].position[c] = float(out[i].position[c] + offset[c]);
            out[i].position[2] = std::max(0.f, out[i].position[2]);
        }
    }
}

} // namespace c3x_renderer::effects

#include "environment_runtime.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace c3x_renderer {
namespace {

constexpr float pi = 3.14159265358979323846f;

float clamp01(float value) {
    return std::clamp(value, 0.0f, 1.0f);
}

float smoothstep(float low, float high, float value) {
    if (low == high)
        return value >= high ? 1.0f : 0.0f;
    float t = clamp01((value - low) / (high - low));
    return t * t * (3.0f - 2.0f * t);
}

float wrap_hour(float hour) {
    if (!std::isfinite(hour))
        return 12.0f;
    hour = std::fmod(hour, 24.0f);
    return hour < 0.0f ? hour + 24.0f : hour;
}

void normalize(float vector[3]) {
    float length = std::sqrt(vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2]);
    if (length <= std::numeric_limits<float>::epsilon()) {
        vector[0] = 0.0f;
        vector[1] = 0.0f;
        vector[2] = 1.0f;
        return;
    }
    vector[0] /= length;
    vector[1] /= length;
    vector[2] /= length;
}

} // namespace

EnvironmentState evaluate_environment(float requested_hour, int season) {
    float hour = wrap_hour(requested_hour);
    float sunrise = smoothstep(5.0f, 7.0f, hour);
    float sunset = 1.0f - smoothstep(17.0f, 19.0f, hour);
    float daylight = clamp01(sunrise * sunset);
    // The canonical Civ VI phases retain a low directional sun at 06:00 and
    // 18:00. Use a slightly wider visual daylight arc than the clock's
    // activation window so those exact six-hour samples cast long shadows.
    float sun_elevation = std::max(0.0f, std::sin((hour - 5.0f) * pi / 14.0f));
    float warm = clamp01(1.0f - std::abs(sun_elevation - 0.18f) / 0.32f) *
                 smoothstep(0.02f, 0.16f, sun_elevation);
    float dusk_mix = warm * smoothstep(12.0f, 18.0f, hour);
    float dawn_mix = warm * (1.0f - smoothstep(12.0f, 18.0f, hour));

    EnvironmentState result = {};
    float sun_angle = (hour - 6.0f) * pi / 12.0f;
    result.sun_direction[0] = std::cos(sun_angle) * 0.78f;
    // In the Civ III isometric basis a negative world-Y light component casts
    // noon shadows down-screen, matching the canonical reference instead of
    // burying them up-screen beneath feature crowns.
    result.sun_direction[1] = -0.42f;
    result.sun_direction[2] = std::max(0.08f, sun_elevation);
    normalize(result.sun_direction);
    result.sun_color[0] = 1.0f;
    result.sun_color[1] = 0.96f - 0.30f * dusk_mix - 0.16f * dawn_mix;
    result.sun_color[2] = 0.88f - 0.46f * dusk_mix - 0.28f * dawn_mix;
    // Keep low-angle phases bright enough to read like the canonical Civ VI
    // 06:00/18:00 frames.  Elevation still controls shadow length; a daylight
    // fill term prevents twilight from looking like a prematurely dark night.
    result.sun_intensity = clamp01(sun_elevation * (0.45f + 0.35f * daylight) +
                                   0.20f * daylight);

    result.moon_direction[0] = -result.sun_direction[0];
    result.moon_direction[1] = -result.sun_direction[1];
    result.moon_direction[2] = std::max(0.18f, 1.0f - sun_elevation);
    normalize(result.moon_direction);
    result.moon_color[0] = 0.42f;
    result.moon_color[1] = 0.56f;
    result.moon_color[2] = 1.00f;
    result.night_activation = clamp01(1.0f - daylight);
    result.moon_intensity = result.night_activation *
                            (0.18f + 0.16f * (1.0f - daylight));

    // Match the canonical four-panel reference: a distinctly golden 18:00,
    // blue but legible midnight, and a cooler/greener 06:00.  Noon totals
    // remain unchanged so this calibration cannot wash out the approved
    // daylight terrain materials.
    result.ambient_color[0] = 0.18f + 0.55f * daylight +
                              0.20f * dusk_mix + 0.040f * dawn_mix;
    result.ambient_color[1] = 0.23f + 0.55f * daylight +
                              0.070f * dusk_mix + 0.020f * dawn_mix;
    result.ambient_color[2] = 0.42f + 0.40f * daylight -
                              0.12f * dusk_mix + 0.030f * dawn_mix;
    if (season == 1) {
        result.ambient_color[0] *= 1.04f;
        result.ambient_color[1] *= 0.94f;
    } else if (season == 2) {
        result.ambient_color[0] *= 0.88f;
        result.ambient_color[2] *= 1.08f;
    }
    result.exposure = 0.79f + 0.21f * daylight;
    result.shadow_strength = clamp01(0.16f + 0.68f * result.sun_intensity +
                                     0.10f * result.moon_intensity);
    result.emissive_scale = 0.25f + 1.10f * result.night_activation;
    result.water_fresnel = 0.04f + 0.08f * result.night_activation;
    result.water_specular = clamp01(0.20f + 0.42f * result.sun_intensity + 0.30f * result.moon_intensity);
    return result;
}

float evaluate_activation(ActivationPolicy policy, float requested_hour, float range_start, float range_end,
                          EnvironmentState const & environment) {
    float hour = wrap_hour(requested_hour);
    if (policy == ActivationPolicy::always)
        return 1.0f;
    if (policy == ActivationPolicy::night)
        return environment.night_activation;
    if (policy == ActivationPolicy::twilight_and_night)
        return clamp01(environment.night_activation * 1.35f);
    range_start = wrap_hour(range_start);
    range_end = wrap_hour(range_end);
    if (range_start <= range_end)
        return hour >= range_start && hour <= range_end ? 1.0f : 0.0f;
    return hour >= range_start || hour <= range_end ? 1.0f : 0.0f;
}

float evaluate_emissive(EmissiveChannel const & channel, EnvironmentState const & environment, float hour) {
    float amount = evaluate_activation(channel.activation_policy, hour, channel.active_hour_start,
                                       channel.active_hour_end, environment);
    return std::max(0.0f, channel.intensity) * environment.emissive_scale * amount;
}

float evaluate_analytic_light(AnalyticLight const & light, EnvironmentState const & environment, float hour,
                              std::uint32_t current_state_mask) {
    if (light.bounds.radius <= 0.0f ||
        (current_state_mask & light.required_state_mask) != light.required_state_mask)
        return 0.0f;
    return std::max(0.0f, light.intensity) * evaluate_activation(
        light.activation_policy, hour, light.active_hour_start, light.active_hour_end, environment);
}

AttachmentState evaluate_attachment(AmbientAttachment const & attachment, AttachmentInput const & input,
                                    EnvironmentState const & environment, float hour) {
    AttachmentState result = {};
    result.fallback = !input.resources_available || !input.owner_replaced;
    if (result.fallback || !input.visible || attachment.bounds.radius <= 0.0f ||
        (input.current_state_mask & attachment.required_state_mask) != attachment.required_state_mask)
        return result;
    result.activation = evaluate_activation(attachment.activation_policy, hour,
                                            attachment.active_hour_start, attachment.active_hour_end, environment);
    result.active = result.activation > 0.0001f;
    if (!result.active || !attachment.animated || attachment.period_ticks <= 0)
        return result;
    std::uint64_t period = static_cast<std::uint64_t>(attachment.period_ticks);
    std::uint64_t now = input.presentation_time_ticks > 0
        ? static_cast<std::uint64_t>(input.presentation_time_ticks) : 0u;
    std::uint64_t seeded = (now % period + static_cast<std::uint64_t>(attachment.stable_phase_seed) % period) % period;
    result.phase_millionths = static_cast<std::uint32_t>((seeded * 1000000ull) / period);
    result.visible_animation_count = 1;
    return result;
}

void shade_terrain(EnvironmentState const & environment, int terrain_type, int tile_x, int tile_y,
                   float variation, float output_rgb[3]) {
    float diffuse = 0.42f + 0.58f * environment.sun_intensity;
    for (int channel = 0; channel < 3; ++channel) {
        float direct = environment.sun_color[channel] * diffuse;
        float moon = environment.moon_color[channel] * environment.moon_intensity;
        output_rgb[channel] = variation * environment.exposure *
            (environment.ambient_color[channel] * 0.72f + direct * 0.58f + moon * 0.30f);
    }
    if (terrain_type >= 11 && terrain_type <= 13) {
        float direction = 0.5f + 0.5f * std::sin((static_cast<float>(tile_x) * environment.moon_direction[0] +
            static_cast<float>(tile_y) * environment.moon_direction[1]) * 1.73f);
        float moon_highlight = std::min(0.22f, direction * environment.moon_intensity *
            environment.water_specular * (0.55f + environment.water_fresnel));
        output_rgb[0] += moon_highlight * environment.moon_color[0];
        output_rgb[1] += moon_highlight * environment.moon_color[1];
        output_rgb[2] += moon_highlight * environment.moon_color[2];
    }
    for (int channel = 0; channel < 3; ++channel)
        output_rgb[channel] = std::clamp(output_rgb[channel], 0.0f, 1.35f);
}

} // namespace c3x_renderer

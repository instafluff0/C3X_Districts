#include "environment_runtime.h"

#include <algorithm>
#include <array>
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

// Civ VI's GameLighting.artdef (DEFAULT_LIGHTING) keys its sun colour at
// 03:50, 06:00, 12:02, 18:33 and 23:00, and blends noon, twilight (one rig
// serves sunrise and sunset) and night light rigs whose weight curves turn at
// 04:00, 06:00, 09:00, 15:00, 18:35 and 23:02 (civ6_lighting_findings.md).
// Only those times are source data. The values are C3X calibrations against
// the four-panel Civ VI reference (12, 18, 00, 06) relative to the approved
// noon: the source's cooked rigs and colour-grading keys are not decoded.
// Sunrise departs from Civ VI's dim lavender by the user's choice: a rose-gold
// golden hour as bright as sunset, so units stay easy to see.

// Direct light {hour, r, g, b, moonlight share} in units of the approved noon
// sun. Moonlight is the night share, so the sun fades as it takes over.
constexpr float night_direct[3] = {0.09f, 0.30f, 1.50f};
constexpr float direct_keys[][5] = {
    {3.825f, 0.0f, 0.0f, 0.0f, 1.0f},
    {6.0f, 0.97f, 0.56f, 0.40f, 0.0f},     // rose-gold sunrise
    {12.036f, 1.0f, 0.96f, 0.88f, 0.0f},   // approved noon
    {18.55f, 0.97f, 0.47f, 0.17f, 0.0f},   // golden sunset
    {23.0f, 0.0f, 0.0f, 0.0f, 1.0f},
};
// Sky {hour, r, g, b, display saturation, exposure}. Blue twilight and night
// fill keep shadows cool beside a warm sun; Civ VI also lifts exposure and
// grades colour by hour.
constexpr float sky_keys[][6] = {
    {4.0f, 0.34f, 0.66f, 2.50f, 0.70f, 0.79f},
    {6.0f, 0.45f, 0.46f, 1.05f, 0.95f, 1.14f},
    {9.0f, 0.73f, 0.78f, 0.82f, 1.0f, 1.0f},
    {15.0f, 0.73f, 0.78f, 0.82f, 1.0f, 1.0f},
    {18.575f, 0.50f, 0.50f, 1.15f, 0.88f, 1.18f},
    {23.04f, 0.34f, 0.66f, 2.50f, 0.70f, 0.79f},
};
// Snow is a bright neutral everywhere, so Winter's blue night glared (91% of
// noon's brightness against Summer's 74%). Its night light and colour scale
// by these at full night.
constexpr float winter_night_light = 0.42f;
constexpr float winter_night_saturation = 0.75f;
// Autumn's tawny ground under the gold sunrise and sunset turned one strong
// brown. At the sun's warmest Autumn pulls it this share back toward noon's
// colour and scales display saturation by this.
constexpr float autumn_golden_neutral = 0.35f;
constexpr float autumn_golden_saturation = 0.85f;

// The values at an hour, eased between neighbouring keys (wrapping
// midnight) so a continuous clock never kinks at a key.
template <std::size_t count, std::size_t width>
std::array<float, width - 1> interpolate(float const (&keys)[count][width], float hour) {
    std::size_t next = 0;
    while (next < count && keys[next][0] <= hour)
        ++next;
    float const * a = keys[(next + count - 1) % count];
    float const * b = keys[next % count];
    float span = b[0] - a[0], into = hour - a[0];
    if (span <= 0.0f)
        span += 24.0f;
    if (into < 0.0f)
        into += 24.0f;
    float t = smoothstep(0.0f, 1.0f, into / span);
    std::array<float, width - 1> result = {};
    for (std::size_t i = 0; i + 1 < width; ++i)
        result[i] = a[i + 1] + (b[i + 1] - a[i + 1]) * t;
    return result;
}

} // namespace

EnvironmentState evaluate_environment(float requested_hour, int season) {
    float hour = wrap_hour(requested_hour);
    float sunrise = smoothstep(5.0f, 7.0f, hour);
    float sunset = 1.0f - smoothstep(17.0f, 19.0f, hour);
    float daylight = clamp01(sunrise * sunset);
    // Elevation only tilts the Lab's sun vector; the shared scene light uses
    // a fixed shadow slope so clock phase never changes an object's
    // height-to-length scale.
    float sun_elevation = std::max(0.0f, std::sin((hour - 5.0f) * pi / 14.0f));

    EnvironmentState result = {};
    // User-selected Civ VI visual convention, expressed in screen space:
    // 12 W, 18 S, 00 E, 06 N. Invert the 2:1 isometric ground projection,
    // then negate the cast vector to obtain the direction toward the light.
    // This is an authored C3X cycle, not recovered Civ VI engine behavior.
    float phase = hour * pi / 12.f;
    float cast_x = std::cos(phase), cast_y = -std::sin(phase);
    float world_x = cast_x + 2.f * cast_y;
    float world_y = cast_x - 2.f * cast_y;
    float horizontal = std::hypot(world_x, world_y);
    result.sun_direction[0] = -world_x / horizontal;
    result.sun_direction[1] = -world_y / horizontal;
    result.sun_direction[2] = std::max(0.08f, sun_elevation);
    normalize(result.sun_direction);
    auto direct = interpolate(direct_keys, hour);
    result.sun_intensity = std::max({direct[0], direct[1], direct[2]});
    for (int channel = 0; channel < 3; ++channel)
        result.sun_color[channel] = result.sun_intensity > 0.0f
            ? direct[channel] / result.sun_intensity : direct_keys[2][channel + 1];

    // One continuous presentation orbit crosses the sun/moon handover. The
    // active night light must not reverse the intended midnight cast direction.
    result.moon_direction[0] = -world_x / horizontal;
    result.moon_direction[1] = -world_y / horizontal;
    result.moon_direction[2] = std::max(0.18f, 1.0f - sun_elevation);
    normalize(result.moon_direction);
    float moon_peak = std::max({night_direct[0], night_direct[1], night_direct[2]});
    for (int channel = 0; channel < 3; ++channel)
        result.moon_color[channel] = night_direct[channel] / moon_peak;
    // Every receiver weights moonlight by moon_response, so night_direct is
    // in the same noon-sun units as the sun keys.
    result.moon_intensity = direct[3] * moon_peak / moon_response;
    result.night_activation = clamp01(1.0f - daylight);

    auto sky = interpolate(sky_keys, hour);
    std::copy(sky.begin(), sky.begin() + 3, std::begin(result.ambient_color));
    result.exposure = sky[4];
    result.saturation = sky[3];
    if (season == 1) {
        result.ambient_color[0] *= 1.04f;
        result.ambient_color[1] *= 0.94f;
        // Warmth: 0 at noon's blue-to-red balance, near 0.8 at sunset.
        float noon_balance = direct_keys[2][3] / direct_keys[2][1];
        float warmth = result.sun_intensity > 0.0f
            ? clamp01(1.0f - result.sun_color[2] / result.sun_color[0] / noon_balance) : 0.0f;
        float neutral = autumn_golden_neutral * warmth;
        for (int channel = 0; channel < 3; ++channel)
            result.sun_color[channel] += (direct_keys[2][channel + 1] / direct_keys[2][1] -
                                          result.sun_color[channel]) * neutral;
        result.saturation *= 1.0f + (autumn_golden_saturation - 1.0f) * warmth;
    } else if (season == 2) {
        result.ambient_color[0] *= 0.88f;
        result.ambient_color[2] *= 1.08f;
        float night_light = 1.0f + (winter_night_light - 1.0f) * result.night_activation;
        for (float & channel : result.ambient_color)
            channel *= night_light;
        result.moon_intensity *= night_light;
        result.saturation *= 1.0f + (winter_night_saturation - 1.0f) * result.night_activation;
    }
    result.shadow_strength = clamp01(0.16f + 0.68f * result.sun_intensity +
                                     0.10f * result.moon_intensity);
    result.emissive_scale = 0.25f + 1.10f * result.night_activation;
    result.water_fresnel = 0.04f + 0.08f * result.night_activation;
    result.water_specular = clamp01(0.20f + 0.42f * result.sun_intensity + 0.30f * result.moon_intensity);
    result.season = static_cast<float>(season >= 0 && season < 4 ? season : 0);
    result.season_progress = 1.0f;
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
    // Compress like the display's max-channel Reinhard, scaled to keep noon,
    // instead of clipping: the blue night sky would flatten moonlit water
    // into land at a clipped blue channel.
    float peak = std::max({output_rgb[0], output_rgb[1], output_rgb[2], 0.0f});
    for (int channel = 0; channel < 3; ++channel)
        output_rgb[channel] = std::max(0.0f, output_rgb[channel]) * 2.12f / (1.0f + peak);
}

} // namespace c3x_renderer

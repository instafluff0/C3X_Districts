#ifndef C3X_ENVIRONMENT_RUNTIME_H
#define C3X_ENVIRONMENT_RUNTIME_H

#include <cstdint>

namespace c3x_renderer {

enum class ActivationPolicy : std::uint32_t {
    always = 0,
    night = 1,
    twilight_and_night = 2,
    hour_range = 3
};

enum class AnalyticLightType : std::uint32_t {
    point = 0,
    spot = 1,
    directional = 2
};

struct EnvironmentState {
    float sun_direction[3];
    float sun_color[3];
    float sun_intensity;
    float moon_direction[3];
    float moon_color[3];
    float moon_intensity;
    float ambient_color[3];
    float exposure;
    float shadow_strength;
    float night_activation;
    float emissive_scale;
    float water_fresnel;
    float water_specular;
    // C3X season (0 Summer, 1 Fall, 2 Winter, 3 Spring) for seasonal shading;
    // progress is reserved for staged seasons and is 1 today.
    float season;
    float season_progress;
};

struct LocalTransform {
    float translation[3];
    float rotation_degrees[3];
    float scale[3];
};

struct AttachmentBounds {
    float center[3];
    float radius;
};

struct EmissiveChannel {
    float color[3];
    float intensity;
    ActivationPolicy activation_policy;
    float active_hour_start;
    float active_hour_end;
};

struct AnalyticLight {
    std::uint64_t stable_id;
    AnalyticLightType type;
    LocalTransform local_transform;
    AttachmentBounds bounds;
    float color[3];
    float intensity;
    ActivationPolicy activation_policy;
    float active_hour_start;
    float active_hour_end;
    std::uint32_t required_state_mask;
};

struct AmbientAttachment {
    std::uint64_t stable_id;
    std::uint64_t analytic_light_id;
    LocalTransform local_transform;
    AttachmentBounds bounds;
    ActivationPolicy activation_policy;
    float active_hour_start;
    float active_hour_end;
    std::uint32_t required_state_mask;
    std::uint32_t stable_phase_seed;
    std::int64_t period_ticks;
    bool animated;
};

struct AttachmentInput {
    std::int64_t presentation_time_ticks;
    std::uint32_t current_state_mask;
    bool visible;
    bool resources_available;
    bool owner_replaced;
};

struct AttachmentState {
    float activation;
    std::uint32_t phase_millionths;
    std::uint32_t visible_animation_count;
    bool active;
    bool fallback;
};

EnvironmentState evaluate_environment(float hour, int season);
// C3X's continuous and user-time clocks: the frame's hour plus whole game
// minutes, advanced from when its minute began (QPC) at its rate, at most an
// hour past it (C3X captures every new hour).
inline float clock_hour(int hour, int minute, int rate, std::int64_t since_qpc,
                        std::int64_t now_qpc, std::int64_t frequency) {
    double minutes = minute;
    if (rate > 0 && since_qpc > 0 && frequency > 0 && now_qpc > since_qpc) {
        double passed = double(now_qpc - since_qpc) * rate / (double(frequency) * 60.0);
        minutes += double(static_cast<std::int64_t>(passed < 60.0 ? passed : 60.0));
    }
    float result = float(hour) + float(minutes / 60.0);
    return result >= 24.0f ? result - 24.0f : result;
}
// A running C3X clock (continuous or user-time) relights at most this often,
// in real seconds, adopting the latest minute: under Parallels each change
// costs about 6 of 60 presented frames (static re-raster, mirror, shadows).
constexpr double clock_step_seconds = 10.0;
// C3X's continuous day/night clock: whether a new clock time (same season)
// replaces the adopted lighting now. Each change re-rasterises the static
// layers and rebuilds shadow pages, so it waits until the previous change has
// fully refined, the camera is still and a step is due. The first frame and a
// season change apply at once.
inline bool adopt_clock_time(float adopted_hour, int adopted_season, float hour, int season,
                             bool refinement_pending, bool camera_still, bool step_due) {
    if (adopted_hour < 0.0f || season != adopted_season) return true;
    return hour != adopted_hour && !refinement_pending && camera_still && step_due;
}
float evaluate_activation(ActivationPolicy policy, float hour, float range_start, float range_end,
                          EnvironmentState const & environment);
float evaluate_emissive(EmissiveChannel const & channel, EnvironmentState const & environment, float hour);
float evaluate_analytic_light(AnalyticLight const & light, EnvironmentState const & environment, float hour,
                              std::uint32_t current_state_mask);
AttachmentState evaluate_attachment(AmbientAttachment const & attachment, AttachmentInput const & input,
                                    EnvironmentState const & environment, float hour);
void shade_terrain(EnvironmentState const & environment, int terrain_type, int tile_x, int tile_y,
                   float variation, float output_rgb[3]);

} // namespace c3x_renderer

#endif

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

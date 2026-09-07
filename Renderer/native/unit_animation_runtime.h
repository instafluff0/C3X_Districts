#ifndef C3X_UNIT_ANIMATION_RUNTIME_H
#define C3X_UNIT_ANIMATION_RUNTIME_H

#include <algorithm>
#include <cstdint>

namespace c3x_renderer {

// Captured at the native FLC body call, after Civ III's visibility/stack checks.
// Opaque identities are compared only: the renderer never dereferences them.
struct NativeUnitDraw {
    std::uintptr_t expected_sprite = 0, expected_canvas = 0;
    std::uintptr_t sprite = 0, canvas = 0;
    int unit_id = -1, action = 0, direction = 0;
    int action_cursor = 0, frame_count = 0;
    int body_x = 0, body_y = 0, sprite_width = 0, sprite_height = 0;
    bool reduced = false;
};

struct UnitAnimationPose {
    char const * action = nullptr;
    double phase = 0;
    int anchor_x = 0, anchor_y = 0, direction = 0;
    float projection_scale = 1;
};

// Native action numbers are the existing AnimationType values. Do not infer a
// queued action or disguise unsupported worker jobs as another action.
inline char const * native_unit_action(int action) {
    switch (action) {
    case 1: return "idle";
    case 2: return "move";
    case 3: case 4: case 5: return "attack";
    case 6: return "death";
    case 7: return "fortify";
    case 8: return "fidget";
    case 9: return "victory";
    case 10: return "capture";
    case 11: return "fortress";
    case 12: return "build";
    case 13: return "road";
    case 14: return "mine";
    case 15: return "irrigate";
    case 16: return "jungle";
    case 17: return "forest";
    case 18: return "plant";
    default: return nullptr;
    }
}

inline bool prepare_native_unit_pose(NativeUnitDraw const & draw, bool clip_loops,
                                     UnitAnimationPose & output) {
    char const * action = native_unit_action(draw.action);
    if (!action || draw.unit_id < 0 || !draw.expected_sprite || !draw.expected_canvas ||
        draw.sprite != draw.expected_sprite || draw.canvas != draw.expected_canvas ||
        draw.direction < 1 || draw.direction > 8 || draw.action_cursor < 0 ||
        draw.frame_count < 1 || draw.frame_count > 65536 ||
        draw.sprite_width < 1 || draw.sprite_height < 1 ||
        draw.sprite_width > 4096 || draw.sprite_height > 4096) return false;
    // Reconstruct the exact center used by Unit::tick_anim. In particular, odd
    // reduced dimensions use Width/4, not round(Width/2)*0.5. Offscreen anchors
    // are legitimate; checked wide arithmetic avoids overflow on bad captures.
    int divisor = draw.reduced ? 4 : 2;
    auto x = std::int64_t(draw.body_x)+draw.sprite_width/divisor;
    auto y = std::int64_t(draw.body_y)+draw.sprite_height/divisor;
    if (x < INT32_MIN || x > INT32_MAX || y < INT32_MIN || y > INT32_MAX) return false;
    UnitAnimationPose pose;
    pose.action = action;
    // The native cursor already advances, wraps or holds. Repeated calls and
    // skipped callbacks sample it directly. No wall-clock restart, travel
    // interpolation, gameplay waits or target-derived positions enter here.
    pose.phase = clip_loops ? double(draw.action_cursor % draw.frame_count)/draw.frame_count :
        (draw.frame_count == 1 ? 1.0 : double(std::min(draw.action_cursor, draw.frame_count-1))/(draw.frame_count-1));
    pose.anchor_x = static_cast<int>(x); pose.anchor_y = static_cast<int>(y);
    pose.direction = draw.direction;
    pose.projection_scale = draw.reduced ? 0.5f : 1.0f;
    output = pose;
    return true;
}

} // namespace c3x_renderer
#endif

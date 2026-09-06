#include <algorithm>
#include <cstdint>

#include "c3x_renderer_api.h"

namespace {

bool valid(c3x_renderer_schedule_v1 const * input, c3x_renderer_schedule_result_v1 const * output) {
    return input != nullptr && output != nullptr &&
        input->api_version == C3X_RENDERER_API_VERSION && input->struct_size == sizeof(*input) &&
        output->struct_size == sizeof(*output) && input->now_ticks >= 0 && input->frequency > 0 &&
        input->cadence_ms > 0 && input->cadence_ms <= 1000 && input->event_start_ticks >= 0 &&
        input->event_duration_ticks >= 0 && input->event_duration_ticks <= INT64_MAX / 1000000;
}

c3x_renderer_u32 absolute_phase(c3x_renderer_schedule_v1 const & input) {
    if (input.event_duration_ticks <= 0 || input.now_ticks <= input.event_start_ticks)
        return 0;
    c3x_renderer_i64 elapsed = input.now_ticks - input.event_start_ticks;
    if (input.event_loops != 0)
        elapsed %= input.event_duration_ticks;
    else if (elapsed >= input.event_duration_ticks)
        return 1000000u;
    return static_cast<c3x_renderer_u32>((elapsed * 1000000) / input.event_duration_ticks);
}

} // namespace

extern "C" __declspec(dllexport) int c3x_renderer_schedule(
    c3x_renderer_schedule_v1 const * input, c3x_renderer_schedule_result_v1 * output) {
    if (!valid(input, output))
        return C3X_RENDERER_RESULT_BAD_ARGUMENT;

    *output = {};
    output->api_version = C3X_RENDERER_API_VERSION;
    output->struct_size = sizeof(*output);
    output->frame_timestamp_ticks = input->now_ticks;
    output->phase_millionths = absolute_phase(*input);

    if (input->visible_animation_count == 0)
        return C3X_RENDERER_RESULT_OK;

    bool const paused = (input->state_flags & C3X_RENDERER_SCHEDULER_MAP_VISIBLE) == 0 ||
        (input->state_flags & C3X_RENDERER_SCHEDULER_FOCUSED) == 0 ||
        (input->state_flags & C3X_RENDERER_SCHEDULER_MODAL) != 0;
    if (paused) {
        output->rebase_clock = 1;
        return C3X_RENDERER_RESULT_OK;
    }
    if ((input->state_flags & (C3X_RENDERER_SCHEDULER_DRAWING |
        C3X_RENDERER_SCHEDULER_REDRAW_PENDING)) != 0)
        return C3X_RENDERER_RESULT_OK;

    bool discontinuity = input->last_presented_ticks <= 0 || input->now_ticks < input->last_presented_ticks;
    c3x_renderer_i64 elapsed = discontinuity ? 0 : input->now_ticks - input->last_presented_ticks;
    if (!discontinuity && elapsed > input->frequency && elapsed - input->frequency > input->frequency)
        discontinuity = true;
    if (discontinuity) {
        output->rebase_clock = 1;
        return C3X_RENDERER_RESULT_OK;
    }

    c3x_renderer_i64 cadence_ticks = (input->frequency / 1000) * input->cadence_ms +
        ((input->frequency % 1000) * input->cadence_ms) / 1000;
    if (cadence_ticks <= 0)
        cadence_ticks = 1;
    if (elapsed < cadence_ticks)
        return C3X_RENDERER_RESULT_OK;

    c3x_renderer_i64 intervals = elapsed / cadence_ticks;
    output->request_redraw = 1;
    output->dirty_flags = C3X_RENDERER_DIRTY_DYNAMIC | C3X_RENDERER_DIRTY_COMPOSITE;
    output->skipped_frame_count = static_cast<c3x_renderer_u32>(
        std::min<c3x_renderer_i64>(intervals > 1 ? intervals - 1 : 0, UINT32_MAX));
    return C3X_RENDERER_RESULT_OK;
}

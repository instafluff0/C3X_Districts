#pragma once
#include "gpu_frame_api.h"

// A completion observer borrows this descriptor only during the callback. It
// may copy values, but must not reenter the renderer or adopt/retire an image.
using c3x_renderer_camera_completion_fn=void(*)(void*,c3x_renderer_i64,int,
    c3x_renderer_gpu_camera_view_v1 const*);
using c3x_renderer_observe_camera_completion_fn=int(*)(c3x_renderer_camera_completion_fn,void*);

namespace c3x_remote_scene {
constexpr unsigned camera_completion_version=1;
constexpr unsigned camera_completion_capacity=16u*1024u*1024u;
struct CameraCompletionSlot {
    unsigned version=0,code=0,size=0;
    alignas(8) c3x_renderer_i64 ticket=0;
    unsigned char payload[camera_completion_capacity];
};
}

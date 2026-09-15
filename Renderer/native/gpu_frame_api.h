#pragma once
#include "c3x_renderer_api.h"
/* Optional caller-driven GPU composition API, versioned by exact struct size.
   No COM/native pointers cross this boundary. Tickets identify one immutable map;
   a newer map, configuration or reset retires the ticket. CPU/unit requests
   preserve native image ownership.
   Image handles persist across GPU frames, but not across session retirement.
   Caller arrays are copied before submission. Success means ordered submission,
   not physical display completion. No renderer callback requests a redraw. */
#pragma pack(push, 4)
struct c3x_renderer_gpu_frame_v1 {
    unsigned struct_size;
    c3x_renderer_i64 ticket, map_image;
    int width,height;
    unsigned device_generation,map_readbacks;
    c3x_renderer_i64 content_revision,session; /* stable across maps and CPU/unit requests; retired on configuration/reset */
};
enum c3x_renderer_gpu_action {C3X_GPU_CREATE=1,C3X_GPU_UPLOAD,C3X_GPU_SUBMIT,C3X_GPU_DESTROY,C3X_GPU_READBACK};
enum c3x_renderer_gpu_format {C3X_GPU_BGRA32=0,C3X_GPU_RGB555=1,C3X_GPU_RGB565=2};
struct c3x_renderer_gpu_command_v1 {
    int kind; /* 0 copy, 1 fill, 2 color key, 3 invert, 4 ordered map quantization, 5 native UI expansion, 6 decoded native sprite */
    c3x_renderer_i64 destination,source;
    int area[4],clip[4],source_x,source_y;
    unsigned color; /* quantize: phase_x&7 | (phase_y&7)<<3; expand: native key, or 65536 for opaque; sprite: 0 native words, 1/2 expand 555/565 */
};
struct c3x_renderer_gpu_images_v1 {
    unsigned struct_size;
    int action;
    c3x_renderer_i64 ticket,image,revision;
    int width,height;
    int format; /* CREATE only; enum c3x_renderer_gpu_format */
    unsigned const* pixels;
    unsigned pixel_count;
    struct c3x_renderer_gpu_command_v1 const* commands;
    unsigned command_count;
};
struct c3x_renderer_gpu_result_v1 {
    unsigned struct_size;
    c3x_renderer_i64 image;
    unsigned pixel_count;
    c3x_renderer_i64 resident_bytes,uploads,commands,readbacks;
};
struct c3x_renderer_gpu_present_v1 {
    unsigned struct_size;
    int action; /* 0 present, 1 release native window ownership */
    c3x_renderer_i64 ticket,image;
    void* window; /* HWND identity only; must belong to the calling thread */
    int width,height,area[4];
};
#pragma pack(pop)
typedef int (*c3x_renderer_gpu_render_fn)(struct c3x_renderer_camera_request_v1 const*,struct c3x_renderer_gpu_frame_v1*,struct c3x_renderer_output_v1*);
/* READBACK alone writes caller storage, after worker completion. Other calls
   require a null readback pointer/capacity. Native GPU composition does not read
   back implicitly; this explicit barrier supports CPU fallback and the oracle. */
typedef int (*c3x_renderer_gpu_images_fn)(struct c3x_renderer_gpu_images_v1 const*,struct c3x_renderer_gpu_result_v1*,unsigned* readback,unsigned capacity);

typedef int (*c3x_renderer_gpu_present_fn)(struct c3x_renderer_gpu_present_v1 const*);

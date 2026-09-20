#include <stdio.h>
#include <stddef.h>
#include "gpu_frame_api.h"
typedef char frame_layout[sizeof(struct c3x_renderer_gpu_frame_v1)==64?1:-1];
typedef char command_layout[sizeof(struct c3x_renderer_gpu_command_v1)==104?1:-1];
typedef char request_layout[sizeof(struct c3x_renderer_gpu_images_v1)==64?1:-1];
typedef char result_layout[sizeof(struct c3x_renderer_gpu_result_v1)==48?1:-1];
typedef char ticket_offset[offsetof(struct c3x_renderer_gpu_frame_v1,ticket)==4?1:-1];
typedef char present_layout[sizeof(struct c3x_renderer_gpu_present_v1)==52?1:-1];
typedef char atomic_image_offset[offsetof(struct c3x_renderer_gpu_camera_view_v1,image)==8?1:-1];
typedef char atomic_camera_offset[offsetof(struct c3x_renderer_gpu_camera_view_v1,camera)==72?1:-1];
typedef char atomic_view_layout[sizeof(struct c3x_renderer_gpu_camera_view_v1)==80+sizeof(struct c3x_renderer_camera_view_v1)?1:-1];
int main(void){puts("PASS GPU frame C ABI: frame=64 command=104 request=64 result=48 ticket_offset=4");return 0;}
